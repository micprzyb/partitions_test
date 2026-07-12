// test_cyclic_partitions.cpp -- standalone correctness suite for the cyclic
// partition / cyclic offset partition extension of refactored_partitions.hpp
// (see cyclic_partitions.txt section 8 for the plan).
//
//   g++ -std=c++23 -O3 -march=native test_cyclic_partitions.cpp -o t && ./t
//   (also run at -O1 and under -fsanitize=address,undefined)
//
// Every kernel is exercised DIRECTLY (not just through the dispatcher), so a
// tier change cannot hide a broken kernel.  The oracle copies the logical
// range out flat; checks: below count, logical prefix/suffix classification,
// multiset preservation, untouched out-of-range canaries, and the boundary
// normalization convention of cyclic_partition.

#include "refactored_partitions.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;
long g_checks = 0;

#define FAIL(...)                                            \
    do {                                                     \
        std::printf("FAIL %s:%d: ", __FILE__, __LINE__);     \
        std::printf(__VA_ARGS__);                            \
        std::printf("\n");                                   \
        if (++g_failures > 20) {                             \
            std::printf("too many failures, aborting\n");    \
            std::exit(1);                                    \
        }                                                    \
    } while (0)

using i32 = std::int32_t;
using i64 = std::int64_t;
using p32 = std::pair<i32, i32>;
using p64 = std::pair<i64, i64>;
using pf32 = std::pair<float, i32>;
using pd64 = std::pair<double, i64>;

// -- comparator/projection sets ----------------------------------------------
struct lex_cfg {
    using comp_t = std::less<>;
    using proj_t = std::identity;
    static constexpr const char* name = "lex";
};
struct first_cfg {  // compare by .first only
    struct proj_t {
        template <class P>
        auto operator()(const P& p) const { return p.first; }
    };
    using comp_t = std::less<>;
    static constexpr const char* name = "first";
};

template <class T>
T make_val(std::uint64_t x);
template <> i32 make_val<i32>(std::uint64_t x) { return static_cast<i32>(x); }
template <> i64 make_val<i64>(std::uint64_t x) { return static_cast<i64>(x); }
template <> p32 make_val<p32>(std::uint64_t x) {
    return {static_cast<i32>(x >> 8), static_cast<i32>(x & 0xff)};
}
template <> p64 make_val<p64>(std::uint64_t x) {
    return {static_cast<i64>(x >> 8), static_cast<i64>(x & 0xff)};
}
// float/double firsts from small integers: exact values, real ties on .first
// (NaN-free by construction -- the strict-weak-ordering contract).
template <> pf32 make_val<pf32>(std::uint64_t x) {
    return {static_cast<float>(x >> 8), static_cast<i32>(x & 0xff)};
}
template <> pd64 make_val<pd64>(std::uint64_t x) {
    return {static_cast<double>(x >> 8), static_cast<i64>(x & 0xff)};
}

// -- the logical view ---------------------------------------------------------
// A test instance is a buffer of capacity cap with a logical range of length n
// starting at physical index f (wraps iff f + n > cap).
template <class T>
struct ring {
    std::vector<T> buf;      // the cyclic buffer
    std::vector<T> canary;   // copy, to check out-of-range cells untouched
    std::size_t f, n;
    T* first;
    T* last;
    T* bb;
    T* be;

    ring(std::size_t cap, std::size_t f_, std::size_t n_,
         const std::vector<T>& logical, T fill)
        : buf(cap, fill), f(f_), n(n_) {
        for (std::size_t k = 0; k < n; ++k) buf[(f + k) % cap] = logical[k];
        canary = buf;
        bb = buf.data();
        be = buf.data() + cap;
        first = bb + f;
        last = bb + (f + n) % cap;
        if (n > 0 && f + n == cap) last = be;  // flat range ending at buf end
        // NOTE: when n == 0, last == first (empty).  When wrapped, last < first.
    }
    T logical_at(std::size_t k) const {
        return buf[(f + k) % buf.size()];
    }
    bool wrapped() const { return n > 0 && f + n > buf.size(); }
};

template <class T, class Cfg>
bool is_below(const T& x, const T& key, Cfg) {
    typename Cfg::comp_t comp{};
    typename Cfg::proj_t proj{};
    return comp(proj(x), proj(key));
}

// Verify a finished instance against the pristine logical input.
template <class T, class Cfg>
void verify(const ring<T>& r, const std::vector<T>& input, const T& key,
            std::ptrdiff_t c_ret, const char* what, Cfg cfg) {
    ++g_checks;
    std::ptrdiff_t c_ref = 0;
    for (const T& x : input) c_ref += is_below(x, key, cfg);
    if (c_ret != c_ref) {
        FAIL("%s: returned count %td != ref %td (n=%zu f=%zu cap=%zu)", what,
             c_ret, c_ref, r.n, r.f, r.buf.size());
        return;
    }
    for (std::size_t k = 0; k < r.n; ++k) {
        bool b = is_below(r.logical_at(k), key, cfg);
        if (b != (static_cast<std::ptrdiff_t>(k) < c_ref)) {
            FAIL("%s: logical[%zu] misplaced (c=%td n=%zu f=%zu cap=%zu)", what,
                 k, c_ref, r.n, r.f, r.buf.size());
            return;
        }
    }
    // multiset preserved over the logical range
    std::vector<T> got;
    got.reserve(r.n);
    for (std::size_t k = 0; k < r.n; ++k) got.push_back(r.logical_at(k));
    std::vector<T> want = input;
    auto lt = [](const T& a, const T& b) {
        if constexpr (std::is_arithmetic_v<T>) return a < b;
        else return a < b;  // pairs: lex operator<
    };
    std::sort(got.begin(), got.end(), lt);
    std::sort(want.begin(), want.end(), lt);
    if (got != want) {
        FAIL("%s: multiset not preserved (n=%zu f=%zu cap=%zu)", what, r.n,
             r.f, r.buf.size());
        return;
    }
    // out-of-range cells untouched
    for (std::size_t p = 0; p < r.buf.size(); ++p) {
        std::size_t rel = (p + r.buf.size() - r.f) % r.buf.size();
        bool in_range = rel < r.n || (r.n > 0 && r.f + r.n == r.buf.size() &&
                                      p >= r.f);  // flat-to-end case
        if (!in_range && !(r.buf[p] == r.canary[p])) {
            FAIL("%s: out-of-range cell %zu modified (n=%zu f=%zu cap=%zu)",
                 what, p, r.n, r.f, r.buf.size());
            return;
        }
    }
}

// -- kernels under test --------------------------------------------------------
enum class kern {
    dispatcher_it,   // cyclic_partition (iterator form + convention check)
    dispatcher_cnt,  // cyclic_partition_count
    wrap_hoare,
    gap_cyclic,
    seg_bridge,
    seg_fill,
};
constexpr kern all_plain[] = {kern::dispatcher_it, kern::dispatcher_cnt,
                              kern::wrap_hoare, kern::gap_cyclic,
                              kern::seg_bridge, kern::seg_fill};

enum class okern {
    dispatcher,  // cyclic_offset_partition
    gap_cyclic,
    w_fill,  // off_w1_fill / off_w2_fill picked by offset regime
    w_seg,   // off_w1_seg  / off_w2_seg
};
constexpr okern all_off[] = {okern::dispatcher, okern::gap_cyclic,
                             okern::w_fill, okern::w_seg};

template <class T, class Cfg>
void run_plain(std::size_t cap, std::size_t f, std::size_t n,
               const std::vector<T>& input, T key, T fill, kern k, Cfg cfg) {
    // wrapped-only kernels need a genuinely wrapped, nonempty-both-sides range
    bool wrapped = n > 0 && f + n > cap;
    if ((k == kern::wrap_hoare || k == kern::gap_cyclic ||
         k == kern::seg_bridge || k == kern::seg_fill) &&
        !wrapped)
        return;
    ring<T> r(cap, f, n, input, fill);
    typename Cfg::comp_t comp{};
    typename Cfg::proj_t proj{};
    std::ptrdiff_t c = -1;
    const char* what = "?";
    switch (k) {
        case kern::dispatcher_it: {
            what = "cyclic_partition";
            T* m = partitions::cyclic_partition(r.first, r.last, r.bb, r.be,
                                                proj(key), comp, proj);
            // recover the count from the normalized boundary
            std::ptrdiff_t len1 = r.be - r.first;
            if (r.first <= r.last) c = m - r.first;
            else c = (m >= r.first) ? m - r.first : (m - r.bb) + len1;
            // convention check: m must be in [bb, be) unless flat end
            if (!(m >= r.bb && (m < r.be || (r.first <= r.last && m == r.last))))
                FAIL("cyclic_partition: boundary outside buffer");
            break;
        }
        case kern::dispatcher_cnt:
            what = "cyclic_partition_count";
            c = partitions::cyclic_partition_count(r.first, r.last, r.bb, r.be,
                                                   proj(key), comp, proj);
            break;
        case kern::wrap_hoare:
            what = "wrap_hoare";
            c = partitions::detail::cyclic::wrap_hoare(
                r.first, r.last, r.bb, r.be, proj(key), comp, proj);
            break;
        case kern::gap_cyclic:
            what = "gap_cyclic";
            c = partitions::detail::cyclic::gap_cyclic(
                r.first, r.last, r.bb, r.be, std::ptrdiff_t{0}, proj(key), comp,
                proj);
            break;
        case kern::seg_bridge:
            what = "seg_bridge";
            c = partitions::detail::cyclic::seg_bridge(
                r.first, r.last, r.bb, r.be, proj(key), comp, proj);
            break;
        case kern::seg_fill:
            what = "seg_fill";
            c = partitions::detail::cyclic::seg_fill(
                r.first, r.last, r.bb, r.be, proj(key), comp, proj);
            break;
    }
    verify(r, input, key, c, what, cfg);
}

template <class T, class Cfg>
void run_off(std::size_t cap, std::size_t f, std::size_t n, std::size_t off,
             std::vector<T> input, T key, T fill, okern k, Cfg cfg) {
    // enforce the precondition: prefix [0, off) all >= key
    typename Cfg::proj_t proj{};
    typename Cfg::comp_t comp{};
    for (std::size_t i = 0; i < off && i < n; ++i)
        if (comp(proj(input[i]), proj(key))) {
            // replace with something >= key: flip below to >= by taking key
            input[i] = key;
        }
    bool wrapped = n > 0 && f + n > cap;
    if (k != okern::dispatcher && !wrapped) return;
    if (k == okern::gap_cyclic && off >= n) return;  // kernel precondition
    std::ptrdiff_t len1 =
        wrapped ? static_cast<std::ptrdiff_t>(cap - f) : static_cast<std::ptrdiff_t>(n);
    ring<T> r(cap, f, n, input, fill);
    std::ptrdiff_t c = -1;
    const char* what = "?";
    auto o = static_cast<std::ptrdiff_t>(off);
    switch (k) {
        case okern::dispatcher:
            what = "cyclic_offset_partition";
            c = partitions::cyclic_offset_partition(r.first, r.last, r.bb, r.be,
                                                    o, proj(key), comp, proj);
            break;
        case okern::gap_cyclic:
            what = "gap_cyclic(off)";
            c = partitions::detail::cyclic::gap_cyclic(
                r.first, r.last, r.bb, r.be, o, proj(key), comp, proj);
            break;
        case okern::w_fill:
            what = o <= len1 ? "off_w1_fill" : "off_w2_fill";
            c = o <= len1
                    ? partitions::detail::cyclic::off_w1_fill(
                          r.first, r.last, r.bb, r.be, o, proj(key), comp, proj)
                    : partitions::detail::cyclic::off_w2_fill(
                          r.first, r.last, r.bb, r.be, o, proj(key), comp, proj);
            break;
        case okern::w_seg:
            what = o <= len1 ? "off_w1_seg" : "off_w2_seg";
            c = o <= len1
                    ? partitions::detail::cyclic::off_w1_seg(
                          r.first, r.last, r.bb, r.be, o, proj(key), comp, proj)
                    : partitions::detail::cyclic::off_w2_seg(
                          r.first, r.last, r.bb, r.be, o, proj(key), comp, proj);
            break;
    }
    verify(r, input, key, c, what, cfg);
}

// -- T2: exhaustive small 0/1 patterns ----------------------------------------
template <class T, class Cfg>
void exhaustive_small(Cfg cfg) {
    for (std::size_t n = 0; n <= 10; ++n) {
        std::size_t cap = n + 3;
        for (std::size_t f = 0; f < cap; ++f) {
            for (std::uint32_t bits = 0; bits < (1u << n); ++bits) {
                std::vector<T> input(n);
                for (std::size_t i = 0; i < n; ++i)
                    input[i] = make_val<T>((bits >> i) & 1u ? 0x101u : 0x000u);
                for (std::uint64_t kv : {0x000ull, 0x101ull, 0x202ull}) {
                    T key = make_val<T>(kv);
                    for (kern k : all_plain)
                        run_plain<T>(cap, f, n, input, key,
                                     make_val<T>(0xdeadull), k, cfg);
                    for (std::size_t off = 0; off <= n; ++off)
                        for (okern k : all_off)
                            run_off<T>(cap, f, n, off, input, key,
                                       make_val<T>(0xdeadull), k, cfg);
                }
            }
        }
    }
}

// -- T3: randomized larger -----------------------------------------------------
template <class T, class Cfg>
void randomized(Cfg cfg, std::mt19937_64& rng) {
    const std::size_t sizes[] = {11,  13,  16,  24,  25,   64,   100,  128,
                                 129, 200, 256, 512, 513,  777,  1024, 1500,
                                 2048, 4096, 8191, 16384, 50000};
    for (std::size_t n : sizes) {
        for (int rep = 0; rep < (n > 4096 ? 2 : 6); ++rep) {
            std::size_t cap = n + 1 + static_cast<std::size_t>(rng() % 64);
            // wrap split: cover flat, tiny S1, tiny S2, mid
            std::size_t splits[] = {0, 1, n / 2, n - 1,
                                    static_cast<std::size_t>(rng() % n)};
            for (std::size_t len2 : splits) {
                // f such that len1 = n - len2 fills to cap end
                std::size_t f = (cap - (n - len2)) % cap;
                if (len2 == 0) f = rng() % (cap - n + 1);  // flat placement
                std::vector<T> input(n);
                int mode = static_cast<int>(rng() % 4);
                for (auto& x : input) {
                    std::uint64_t v = mode == 0 ? rng() % 0x40000
                                    : mode == 1 ? rng() % 5    // heavy dups
                                    : mode == 2 ? 0x123        // all equal
                                                : rng();  // full width: negative
                                                          // members, sign-bias path
                    x = make_val<T>(v);
                }
                T key = mode == 2 ? make_val<T>(0x123)
                                  : input[rng() % n];  // in-range percentile
                for (kern k : all_plain)
                    run_plain<T>(cap, f, n, input, key, make_val<T>(0xfeedull),
                                 k, cfg);
                std::size_t offs[] = {0, 1, n / 8, n / 2, (7 * n) / 8, n};
                // include offsets straddling len1 (W1/W2 boundary)
                std::size_t len1 = n - len2;
                std::size_t offs2[] = {len1 > 0 ? len1 - 1 : 0, len1,
                                       len1 + 1 <= n ? len1 + 1 : n};
                for (std::size_t off : offs)
                    for (okern k : all_off)
                        run_off<T>(cap, f, n, off, input, key,
                                   make_val<T>(0xfeedull), k, cfg);
                for (std::size_t off : offs2)
                    for (okern k : all_off)
                        run_off<T>(cap, f, n, off, input, key,
                                   make_val<T>(0xfeedull), k, cfg);
            }
        }
    }
}

// -- flat entry points (sized_partition / sized_partition_at /
// offset_partition / median_partition) against the same oracle: these now
// perform the pair-lex comparator substitution, so they need direct coverage,
// not just coverage through the cyclic dispatchers.
template <class T, class Cfg>
void flat_entries(Cfg cfg, std::mt19937_64& rng) {
    typename Cfg::comp_t comp{};
    typename Cfg::proj_t proj{};
    const std::size_t sizes[] = {0, 1, 2, 3, 7, 8, 16, 24, 25, 63, 64, 100,
                                 512, 513, 1024, 4097, 20000};
    for (std::size_t n : sizes) {
        for (int rep = 0; rep < 8; ++rep) {
            std::vector<T> input(n);
            int mode = static_cast<int>(rng() % 4);
            for (auto& x : input) {
                std::uint64_t v = mode == 0   ? rng() % 0x40000
                                  : mode == 1 ? rng() % 5
                                  : mode == 2 ? 0x123
                                              : rng();
                x = make_val<T>(v);
            }
            T key = n && mode != 2 ? input[rng() % n] : make_val<T>(0x123);
            auto check = [&](std::vector<T>& v, std::ptrdiff_t c,
                             const T& k, const char* what) {
                ++g_checks;
                std::ptrdiff_t c_ref = 0;
                for (const T& x : input) c_ref += is_below(x, k, cfg);
                if (c != c_ref) {
                    FAIL("%s: count %td != ref %td (n=%zu)", what, c, c_ref, n);
                    return;
                }
                for (std::size_t i = 0; i < n; ++i)
                    if (is_below(v[i], k, cfg) !=
                        (static_cast<std::ptrdiff_t>(i) < c)) {
                        FAIL("%s: misplaced at %zu (n=%zu)", what, i, n);
                        return;
                    }
                auto s1 = v, s2 = input;
                std::sort(s1.begin(), s1.end());
                std::sort(s2.begin(), s2.end());
                if (s1 != s2) FAIL("%s: multiset changed (n=%zu)", what, n);
            };
            {  // sized_partition
                auto v = input;
                T* m = partitions::sized_partition(v.data(), v.data() + n,
                                                   proj(key), comp, proj);
                check(v, m - v.data(), key, "sized_partition");
            }
            if (n > 0) {  // sized_partition_at (key = an in-block element)
                auto v = input;
                std::size_t pi = rng() % n;
                T pk = v[pi];
                T* m = partitions::sized_partition_at(
                    v.data(), v.data() + n, v.data() + pi, comp, proj);
                check(v, m - v.data(), pk, "sized_partition_at");
            }
            {  // offset_partition: prefix forced >= key
                auto in2 = input;
                std::size_t offs[] = {0, 1, n / 2, n ? n - 1 : 0, n};
                for (std::size_t off : offs) {
                    auto v = in2;
                    for (std::size_t i = 0; i < off && i < n; ++i)
                        if (comp(proj(v[i]), proj(key))) v[i] = key;
                    auto saved = input;
                    input = v;  // oracle counts over the fixed-up input
                    auto c = partitions::offset_partition(
                        v.data(), v.data() + n,
                        static_cast<std::ptrdiff_t>(std::min(off, n)),
                        proj(key), comp, proj);
                    check(v, c, key, "offset_partition");
                    input = saved;
                }
            }
            if (n > 0) {  // median_partition contract
                auto v = input;
                T* m = partitions::median_partition(v.data(), v.data() + n,
                                                    comp, proj);
                ++g_checks;
                // split contract: no right element comp-less than any left
                bool ok = true;
                if (n > 24 && m < v.data() + n) {
                    // *m at final rank: [first,m) < proj(*m), [m+1,last) >=
                    for (T* p = v.data(); p < m && ok; ++p)
                        if (!comp(proj(*p), proj(*m))) ok = false;
                    for (T* p = m + 1; p < v.data() + n && ok; ++p)
                        if (comp(proj(*p), proj(*m))) ok = false;
                } else if (n <= 24) {
                    if (m != v.data() + n / 2) ok = false;
                    for (T* a = v.data(); a < m && ok; ++a)
                        for (T* b = m; b < v.data() + n && ok; ++b)
                            if (comp(proj(*b), proj(*a))) ok = false;
                }
                if (!ok) FAIL("median_partition: contract violated (n=%zu)", n);
                auto s1 = v, s2 = input;
                std::sort(s1.begin(), s1.end());
                std::sort(s2.begin(), s2.end());
                if (s1 != s2)
                    FAIL("median_partition: multiset changed (n=%zu)", n);
            }
        }
    }
}

// -- find_min / cyclic_find_min: any-minimal contract --------------------------
template <class T, class Cfg>
void find_min_tests([[maybe_unused]] Cfg cfg, std::mt19937_64& rng) {
    typename Cfg::comp_t comp{};
    typename Cfg::proj_t proj{};
    auto run_one = [&](std::size_t cap, std::size_t f, std::size_t n,
                       const std::vector<T>& logical) {
        ring<T> r(cap, f, n, logical, make_val<T>(0xdeadull));
        // cyclic entry
        T* m = partitions::cyclic_find_min(r.first, r.last, r.bb, r.be, comp,
                                           proj);
        ++g_checks;
        if (n == 0) {
            if (m != r.first) FAIL("cyclic_find_min: empty must return first");
            return;
        }
        // m must lie in the logical range
        std::ptrdiff_t li = -1;
        for (std::size_t k = 0; k < n; ++k)
            if (r.bb + (r.f + k) % cap == m) li = (std::ptrdiff_t)k;
        if (li < 0) {
            FAIL("cyclic_find_min: result outside range (n=%zu f=%zu)", n, f);
            return;
        }
        for (std::size_t k = 0; k < n; ++k)
            if (comp(proj(r.logical_at(k)), proj(*m))) {
                FAIL("cyclic_find_min: not minimal (n=%zu f=%zu)", n, f);
                return;
            }
        // flat entry cross-check on the same logical data (value equality
        // with std::min_element under the same comp/proj)
        std::vector<T> flat = logical;
        if (n > 0) {
            T* fm = partitions::find_min(flat.data(), flat.data() + n, comp,
                                         proj);
            auto sm = std::min_element(
                flat.begin(), flat.end(), [&](const T& a, const T& b) {
                    return comp(proj(a), proj(b));
                });
            ++g_checks;
            if (comp(proj(*fm), proj(*sm)) || comp(proj(*sm), proj(*fm)))
                FAIL("find_min: value differs from std::min_element (n=%zu)",
                     n);
        }
    };
    // exhaustive tiny n, all wrap splits, dup-heavy domains (ties exercised)
    for (std::size_t n = 0; n <= 12; ++n) {
        std::size_t cap = n + 3;
        for (std::size_t f = 0; f < cap; ++f) {
            for (int rep = 0; rep < 6; ++rep) {
                std::vector<T> logical(n);
                int dom = rep % 3 == 0 ? 2 : rep % 3 == 1 ? 5 : 0x40000;
                for (auto& x : logical)
                    x = make_val<T>(rng() % (std::uint64_t)dom + (rep >= 3
                                        ? (rng() << 40)  // negatives/full width
                                        : 0));
                run_one(cap, f, n, logical);
            }
        }
    }
    // randomized larger
    for (std::size_t n : {13ul, 16ul, 24ul, 100ul, 999ul, 4096ul}) {
        for (int rep = 0; rep < 8; ++rep) {
            std::size_t cap = n + 1 + rng() % 32;
            std::size_t f = rng() % cap;
            std::vector<T> logical(n);
            for (auto& x : logical) x = make_val<T>(rng() % 1000);
            run_one(cap, f, n, logical);
        }
    }
}

}  // namespace

int main() {
    std::mt19937_64 rng(0xc0ffee123ULL);
    std::printf("exhaustive small (i64 lex)...\n");
    exhaustive_small<i64>(lex_cfg{});
    std::printf("exhaustive small (i32 lex)...\n");
    exhaustive_small<i32>(lex_cfg{});
    std::printf("exhaustive small (p64 lex)...\n");
    exhaustive_small<p64>(lex_cfg{});
    std::printf("exhaustive small (p64 first)...\n");
    exhaustive_small<p64>(first_cfg{});
    std::printf("exhaustive small (pf32 lex)...\n");
    exhaustive_small<pf32>(lex_cfg{});
    std::printf("exhaustive small (pd64 lex)...\n");
    exhaustive_small<pd64>(lex_cfg{});
    std::printf("randomized (i64 lex)...\n");
    randomized<i64>(lex_cfg{}, rng);
    std::printf("randomized (i32 lex)...\n");
    randomized<i32>(lex_cfg{}, rng);
    std::printf("randomized (p32 lex)...\n");
    randomized<p32>(lex_cfg{}, rng);
    std::printf("randomized (p64 lex)...\n");
    randomized<p64>(lex_cfg{}, rng);
    std::printf("randomized (p32 first)...\n");
    randomized<p32>(first_cfg{}, rng);
    std::printf("randomized (p64 first)...\n");
    randomized<p64>(first_cfg{}, rng);
    std::printf("randomized (pf32 lex)...\n");
    randomized<pf32>(lex_cfg{}, rng);
    std::printf("randomized (pd64 lex)...\n");
    randomized<pd64>(lex_cfg{}, rng);
    std::printf("randomized (pf32 first)...\n");
    randomized<pf32>(first_cfg{}, rng);
    std::printf("randomized (pd64 first)...\n");
    randomized<pd64>(first_cfg{}, rng);
    std::printf("find_min (all types)...\n");
    find_min_tests<i32>(lex_cfg{}, rng);
    find_min_tests<i64>(lex_cfg{}, rng);
    find_min_tests<p32>(lex_cfg{}, rng);
    find_min_tests<p64>(lex_cfg{}, rng);
    find_min_tests<pf32>(lex_cfg{}, rng);
    find_min_tests<pd64>(lex_cfg{}, rng);
    find_min_tests<p32>(first_cfg{}, rng);
    find_min_tests<p64>(first_cfg{}, rng);
    find_min_tests<pf32>(first_cfg{}, rng);
    find_min_tests<pd64>(first_cfg{}, rng);
    std::printf("flat entries (all types)...\n");
    flat_entries<i32>(lex_cfg{}, rng);
    flat_entries<i64>(lex_cfg{}, rng);
    flat_entries<p32>(lex_cfg{}, rng);
    flat_entries<p64>(lex_cfg{}, rng);
    flat_entries<pf32>(lex_cfg{}, rng);
    flat_entries<pd64>(lex_cfg{}, rng);
    flat_entries<p32>(first_cfg{}, rng);
    flat_entries<p64>(first_cfg{}, rng);
    flat_entries<pf32>(first_cfg{}, rng);
    flat_entries<pd64>(first_cfg{}, rng);
    if (g_failures == 0)
        std::printf("OK: %ld checks passed\n", g_checks);
    else
        std::printf("FAILED: %d failures out of %ld checks\n", g_failures,
                    g_checks);
    return g_failures != 0;
}
