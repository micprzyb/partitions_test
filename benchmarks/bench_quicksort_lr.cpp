// Benchmarks for the left-to-right selection-leaf quicksort (quicksort_lr).
//
// Studies, each emitting CSV; pass a mode on argv[1] (default "all"), optional
// max_size on argv[2].  Run several times and take per-row medians (the repo's
// noise protocol; absolute ns drift ±10% between sessions, so the per-row ratio
// within ONE run is the robust signal).
//
//   min    : the "improved find-minimum" study -- selection sort of random
//            small blocks under several minimum-finders (textbook reload vs
//            register-key, branchy vs branchless, 1/2/4 accumulators, plus an
//            OUT-OF-SPEC min+max double selection for reference).
//   leaf   : selection leaf vs the halver leaf (quicksort's) vs insertion, with
//            comparison + move COUNTS -- shows the selection leaf is latency-bound
//            (serial min-reduction) vs the halver's throughput-bound parallel
//            network: 2x slower on cheap-move i64, a tie on 16-byte pair64.
//   ab     : pivot A/B -- inline branchless ninther_pos vs library pivot::ninther
//            in ONE session (same driver) so frequency drift cancels in the ratio.
//   part   : partitioner A/B in the threshold driver (sized vs Lomuto vs boost vs
//            Lomuto-then-boost) -- is boost's setup worth it on small nodes?
//   pivcut : median-of-3 size-tier cutoff sweep (m3 small / ninther large), incl.
//            a median_of_3_killer probe at small n that exposes the O(n^2) blow-up
//            of an over-large cutoff.  (Both ab/part/pivcut documented REJECTED or
//            neutral results -- see docs/quicksort_lr.md.)
//   thresh : selection-leaf threshold sweep for the full sort.
//   vs     : head-to-head vs partitions::quicksort, std::sort, boost::pdqsort.
//   depth  : validates the explicit-stack capacity -- max left-spine depth over
//            every (type,distribution) at large n.
//
// CSV columns vary per study; the first token identifies the study.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#if __has_include(<boost/sort/sort.hpp>)
#include <boost/sort/sort.hpp>
#define HAVE_BOOST 1
#endif

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/quicksort.hpp"
#include "partitions/quicksort_lr.hpp"

using namespace partitions;

namespace {

struct first_key {
    template <class P>
    auto operator()(const P& p) const {
        return p.first;
    }
};

// Generate `n` elements of type T for distribution `d`, FAIRLY across types.
//
// The trap (see the perf-methodology memo): dist::random_uniform draws ranks in
// [0,n] and make_value<pair64>(rank) = {rank>>8, rank&0xff}.  So the i64 key and
// the pair64 LEXICOGRAPHIC key are both `rank` (~63% distinct, high cardinality),
// but the pair64-BY-FIRST key `.first = rank>>8` has only ~n/256 distinct values
// (0.4% -- low cardinality).  Comparing a low-cardinality sort against high-
// cardinality ones is apples-to-oranges and produces nonsensical cross-type
// numbers (the no-3-way quicksorts degrade on the dups; std::sort/pdqsort speed
// up on fewer distinct values).  Fix: for the by-first case put the FULL rank in
// .first so its key has the SAME cardinality as the i64 key, isolating the real
// difference (16-byte vs 8-byte element, identical i64 compare).
template <class T, class Dist, class Proj>
std::vector<T> gen_data(Dist d, std::size_t n, std::uint64_t seed, Proj) {
    if constexpr (std::is_same_v<T, pair64> &&
                  !std::is_same_v<std::decay_t<Proj>, std::identity>) {
        const auto ranks = d.template operator()<i64>(n, seed);
        std::vector<pair64> out(n);
        for (std::size_t i = 0; i < n; ++i) out[i] = pair64{ranks[i], 0};
        return out;
    } else {
        return d.template operator()<T>(n, seed);
    }
}

// ===========================================================================
// Minimum-finders under study.  Each is a stateless functor:
//     It operator()(It first, It last, Comp comp, Proj proj) const;  // last>first
// returning an iterator to *a* minimum of [first,last).
// ===========================================================================

// (0) Textbook: best kept in MEMORY (reload *m each step), branchy update.
struct F_textbook {
    template <class It, class Comp, class Proj>
    It operator()(It first, It last, Comp comp, Proj proj) const {
        It m = first;
        for (It it = first + 1; it != last; ++it)
            if (std::invoke(comp, std::invoke(proj, *it), std::invoke(proj, *m)))
                m = it;
        return m;
    }
};

// (1) Register-key, branchy update, single accumulator.
struct F_branchy_reg {
    template <class It, class Comp, class Proj>
    It operator()(It first, It last, Comp comp, Proj proj) const {
        It m = first;
        auto best = std::invoke(proj, *first);
        for (It it = first + 1; it != last; ++it) {
            auto k = std::invoke(proj, *it);
            if (std::invoke(comp, k, best)) {
                best = k;
                m = it;
            }
        }
        return m;
    }
};

// (2) Register-key, BRANCHLESS update, single accumulator.
struct F_bl1 {
    template <class It, class Comp, class Proj>
    It operator()(It first, It last, Comp comp, Proj proj) const {
        using D = std::iter_difference_t<It>;
        const D n = last - first;
        auto b = std::invoke(proj, first[0]);
        D bi = 0;
        for (D j = 1; j < n; ++j) {
            auto k = std::invoke(proj, first[j]);
            const bool l = static_cast<bool>(std::invoke(comp, k, b));
            detail::cond_assign(b, k, l);
            bi = l ? j : bi;
        }
        return first + bi;
    }
};

// (3) BRANCHLESS, TWO accumulators (detail::min_scan_narrow, the narrow-key
// path of detail::find_min).
struct F_bl2 {
    template <class It, class Comp, class Proj>
    It operator()(It first, It last, Comp comp, Proj proj) const {
        return detail::min_scan_narrow(first, last, comp, proj);
    }
};

// (4) BRANCHLESS, FOUR accumulators.
struct F_bl4 {
    template <class It, class Comp, class Proj>
    It operator()(It first, It last, Comp comp, Proj proj) const {
        using D = std::iter_difference_t<It>;
        const D n = last - first;
        auto b0 = std::invoke(proj, first[0]);
        auto b1 = b0, b2 = b0, b3 = b0;
        D i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        D j = 1;
        for (; j + 3 < n; j += 4) {
            auto k0 = std::invoke(proj, first[j]);
            auto k1 = std::invoke(proj, first[j + 1]);
            auto k2 = std::invoke(proj, first[j + 2]);
            auto k3 = std::invoke(proj, first[j + 3]);
            const bool l0 = static_cast<bool>(std::invoke(comp, k0, b0));
            const bool l1 = static_cast<bool>(std::invoke(comp, k1, b1));
            const bool l2 = static_cast<bool>(std::invoke(comp, k2, b2));
            const bool l3 = static_cast<bool>(std::invoke(comp, k3, b3));
            detail::cond_assign(b0, k0, l0); i0 = l0 ? j : i0;
            detail::cond_assign(b1, k1, l1); i1 = l1 ? j + 1 : i1;
            detail::cond_assign(b2, k2, l2); i2 = l2 ? j + 2 : i2;
            detail::cond_assign(b3, k3, l3); i3 = l3 ? j + 3 : i3;
        }
        for (; j < n; ++j) {  // tail (< 4)
            auto k = std::invoke(proj, first[j]);
            const bool l = static_cast<bool>(std::invoke(comp, k, b0));
            detail::cond_assign(b0, k, l); i0 = l ? j : i0;
        }
        // merge 4 -> 1 (strict <, so ties keep the lower lane).
        const bool p1 = static_cast<bool>(std::invoke(comp, b1, b0));
        detail::cond_assign(b0, b1, p1); i0 = p1 ? i1 : i0;
        const bool p3 = static_cast<bool>(std::invoke(comp, b3, b2));
        detail::cond_assign(b2, b3, p3); i2 = p3 ? i3 : i2;
        const bool p2 = static_cast<bool>(std::invoke(comp, b2, b0));
        i0 = p2 ? i2 : i0;
        return first + i0;
    }
};

// (5) Register-key, branchy, TWO accumulators (does ILP help the branchy form?).
struct F_branchy_reg2 {
    template <class It, class Comp, class Proj>
    It operator()(It first, It last, Comp comp, Proj proj) const {
        using D = std::iter_difference_t<It>;
        const D n = last - first;
        It m0 = first, m1 = first;
        auto b0 = std::invoke(proj, first[0]);
        auto b1 = b0;
        D j = 1;
        for (; j + 1 < n; j += 2) {
            auto k0 = std::invoke(proj, first[j]);
            auto k1 = std::invoke(proj, first[j + 1]);
            if (std::invoke(comp, k0, b0)) { b0 = k0; m0 = first + j; }
            if (std::invoke(comp, k1, b1)) { b1 = k1; m1 = first + j + 1; }
        }
        if (j < n) {
            auto k0 = std::invoke(proj, first[j]);
            if (std::invoke(comp, k0, b0)) { b0 = k0; m0 = first + j; }
        }
        return static_cast<bool>(std::invoke(comp, b1, b0)) ? m1 : m0;
    }
};

// Selection sort driven by a minimum-finder F (constraint-faithful: find min,
// swap to front, proceed).
template <class F, class It, class Comp, class Proj>
void sel_sort(It first, It last, Comp comp, Proj proj) {
    F f;
    for (It i = first; last - i > 1; ++i) {
        It m = f(i, last, comp, proj);
        if (m != i) std::iter_swap(i, m);
    }
}

// OUT-OF-SPEC reference: double-ended selection sort -- one pass finds BOTH the
// minimum and the maximum of the suffix, placing min at the front and max at the
// back, halving the number of passes.  This is the textbook "better way to use a
// min-scan" and bounds what constraint 1 (find-min-only) costs; NOT a deliverable.
template <class It, class Comp, class Proj>
void sel_sort_minmax(It first, It last, Comp comp, Proj proj) {
    using D = std::iter_difference_t<It>;
    while (last - first > 1) {
        const D n = last - first;
        auto bmin = std::invoke(proj, first[0]);
        auto bmax = bmin;
        D imin = 0, imax = 0;
        for (D j = 1; j < n; ++j) {
            auto k = std::invoke(proj, first[j]);
            const bool lo = static_cast<bool>(std::invoke(comp, k, bmin));
            const bool hi = static_cast<bool>(std::invoke(comp, bmax, k));
            detail::cond_assign(bmin, k, lo); imin = lo ? j : imin;
            detail::cond_assign(bmax, k, hi); imax = hi ? j : imax;
        }
        It fmin = first + imin, fmax = first + imax;
        std::iter_swap(first, fmin);
        if (fmax == first) fmax = fmin;       // max was at the front we just moved
        std::iter_swap(last - 1, fmax);
        ++first;
        --last;
    }
}

// ---- Part A: minimum-finder micro (selection sort over random small blocks) --

template <class T, class Run>
double time_blocks(std::size_t total, const std::vector<T>& master, Run run) {
    std::vector<T> work;
    auto setup = [&] { work = master; };
    auto do_work = [&] {
        run(work);
        bench::do_not_optimize(work[0]);
    };
    std::uint64_t reps = std::min<std::uint64_t>(
        std::max<std::uint64_t>((1u << 24) / total + 1, 8), 200);
    auto r = bench::measure(reps, setup, do_work, 3);
    return r.min_ns / static_cast<double>(total);  // ns per element sorted
}

template <class T, class Proj>
bool blocks_sorted(const std::vector<T>& v, std::size_t L, Proj proj) {
    auto comp = std::less<>{};
    for (std::size_t s = 0; s + L <= v.size(); s += L)
        for (std::size_t i = s + 1; i < s + L; ++i)
            if (comp(std::invoke(proj, v[i]), std::invoke(proj, v[i - 1])))
                return false;
    return true;
}

template <class T, class Proj>
void study_min(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    const std::size_t total = 1u << 18;
    if (total > max_size) return;
    auto master = gen_data<T>(dist::random_uniform{}, total, 0x51A7Eu, proj);

    for (std::size_t L : {std::size_t(4), std::size_t(8), std::size_t(12),
                          std::size_t(16), std::size_t(20), std::size_t(24),
                          std::size_t(32), std::size_t(48), std::size_t(64)}) {
        auto emit = [&](const char* fn, auto run) {
            // correctness gate
            std::vector<T> chk = master;
            run(chk);
            if (!blocks_sorted(chk, L, proj)) {
                std::fprintf(stderr, "BLOCKS NOT SORTED %s L=%zu %s\n", tname, L, fn);
                std::abort();
            }
            double ns = time_blocks<T>(total, master, run);
            std::printf("min,%s,blocks,%zu,%s,%.4f\n", tname, L, fn, ns);
            std::fflush(stdout);
        };
        auto blocks = [&](auto sorter) {
            return [&, sorter](std::vector<T>& w) {
                for (std::size_t s = 0; s + L <= w.size(); s += L)
                    sorter(w.begin() + s, w.begin() + s + L, comp, proj);
            };
        };
        emit("textbook",     blocks([](auto f, auto l, auto c, auto p) { sel_sort<F_textbook>(f, l, c, p); }));
        emit("branchy_reg",  blocks([](auto f, auto l, auto c, auto p) { sel_sort<F_branchy_reg>(f, l, c, p); }));
        emit("branchy_reg2", blocks([](auto f, auto l, auto c, auto p) { sel_sort<F_branchy_reg2>(f, l, c, p); }));
        emit("bl1",          blocks([](auto f, auto l, auto c, auto p) { sel_sort<F_bl1>(f, l, c, p); }));
        emit("bl2",          blocks([](auto f, auto l, auto c, auto p) { sel_sort<F_bl2>(f, l, c, p); }));
        emit("bl4",          blocks([](auto f, auto l, auto c, auto p) { sel_sort<F_bl4>(f, l, c, p); }));
        emit("minmax_oos",   blocks([](auto f, auto l, auto c, auto p) { sel_sort_minmax(f, l, c, p); }));
    }
}

// ---- Part A2: LEAF head-to-head -- selection vs the halver leaf vs insertion,
// on random blocks of the leaf sizes, with comparison + move counts.  This is
// the direct test of "is the selection leaf actually cheaper than the halver
// leaf?": per L-block, selection does L(L-1)/2 comparisons + <=L swaps; the
// halver (detail::halve_sort) does Sum C(size) ~ L*logL compare-exchanges.

long g_cmps = 0, g_moves = 0;
struct count_cmp {
    template <class A, class B>
    bool operator()(const A& a, const B& b) const { ++g_cmps; return std::less<>{}(a, b); }
};
// A wrapper element that counts its own moves (copy/assign), so we see the move
// traffic difference (selection's few swaps vs the halver's compare-exchanges).
template <class T>
struct counted {
    T v{};
    counted() = default;
    counted(const counted& o) : v(o.v) { ++g_moves; }
    counted(counted&& o) noexcept : v(o.v) { ++g_moves; }
    counted& operator=(const counted& o) { v = o.v; ++g_moves; return *this; }
    counted& operator=(counted&& o) noexcept { v = o.v; ++g_moves; return *this; }
};

// Tournament / tree-selection sort: find the min by a log-depth tournament over
// element INDICES (not moving the data), then on each extraction recompute only
// the root path -> O(L log L) comparisons.  Winner tree with a -1 "removed"
// marker (no generic +inf needed).  Elements are emitted in order to a small
// buffer, then copied back (2L element moves total -- far fewer than the halver).
template <class It, class Comp, class Proj>
void tournament_sort(It first, It last, Comp comp, Proj proj) {
    using T = std::iter_value_t<It>;
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    const int nn = static_cast<int>(last - first);
    if (nn <= 1) return;
    int P = 1;
    while (P < nn) P <<= 1;
    K key[64];
    int tree[128];
    T buf[64];
    for (int i = 0; i < nn; ++i) key[i] = std::invoke(proj, first[i]);
    for (int i = 0; i < P; ++i) tree[P + i] = (i < nn) ? i : -1;
    auto match = [&](int a, int b) {
        if (a < 0) return b;
        if (b < 0) return a;
        return static_cast<bool>(std::invoke(comp, key[b], key[a])) ? b : a;
    };
    for (int j = P - 1; j >= 1; --j) tree[j] = match(tree[2 * j], tree[2 * j + 1]);
    for (int o = 0; o < nn; ++o) {
        const int w = tree[1];          // current global min leaf
        buf[o] = first[w];
        int node = (P + w) >> 1;
        tree[P + w] = -1;
        for (; node >= 1; node >>= 1)
            tree[node] = match(tree[2 * node], tree[2 * node + 1]);
    }
    for (int i = 0; i < nn; ++i) first[i] = buf[i];
}

// A large sentinel key for the supported types, so a "removed" leaf always loses
// WITHOUT a branch (no generic +inf, so fall back to a copy of the build-time max
// for unknown types -- only the three target types take the fast branchless path).
template <class K>
K key_inf() {
    if constexpr (std::is_arithmetic_v<K>) return std::numeric_limits<K>::max();
    else if constexpr (std::is_same_v<K, pair64>)
        return pair64{std::numeric_limits<i64>::max(), std::numeric_limits<i64>::max()};
    else return K{};
}
// Branchless tournament: removed leaves get a +inf key instead of a -1 index, so
// `match` is a single comp + cmov with no misprediction.  Isolates how much of the
// tournament's cost was the removed-index branches vs the indirection/serial sift.
template <class It, class Comp, class Proj>
void tournament_bl(It first, It last, Comp comp, Proj proj) {
    using T = std::iter_value_t<It>;
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    const int nn = static_cast<int>(last - first);
    if (nn <= 1) return;
    int P = 1;
    while (P < nn) P <<= 1;
    K key[64];
    int tree[128];
    T buf[64];
    const K inf = key_inf<K>();
    for (int i = 0; i < nn; ++i) key[i] = std::invoke(proj, first[i]);
    for (int i = nn; i < P; ++i) key[i] = inf;
    for (int i = 0; i < P; ++i) tree[P + i] = i;
    auto match = [&](int a, int b) {
        return static_cast<bool>(std::invoke(comp, key[b], key[a])) ? b : a;
    };
    for (int j = P - 1; j >= 1; --j) tree[j] = match(tree[2 * j], tree[2 * j + 1]);
    for (int o = 0; o < nn; ++o) {
        const int w = tree[1];
        buf[o] = first[w];
        key[w] = inf;
        int node = (P + w) >> 1;
        for (; node >= 1; node >>= 1)
            tree[node] = match(tree[2 * node], tree[2 * node + 1]);
    }
    for (int i = 0; i < nn; ++i) first[i] = buf[i];
}

// The "sweet spot" between a full tournament and a linear scan: a 2-LEVEL
// grouped selection.  Partition into groups of `g`; keep each group's running
// min (key+index).  Each extraction scans the G group-mins for the global min
// (sequential, cmov), emits it, marks it +inf, and recomputes ONLY that group's
// min (sequential).  ~L*(g + L/g) comparisons, minimised near g=sqrt(L), with
// far less indirection than the tree (two contiguous scans, no sift path).
template <int G_SIZE, class It, class Comp, class Proj>
void grouped_sort(It first, It last, Comp comp, Proj proj) {
    using T = std::iter_value_t<It>;
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    const int nn = static_cast<int>(last - first);
    if (nn <= 1) return;
    const K inf = key_inf<K>();
    K key[64];
    T buf[64];
    for (int i = 0; i < nn; ++i) key[i] = std::invoke(proj, first[i]);
    constexpr int g = G_SIZE;
    const int G = (nn + g - 1) / g;
    K gmk[64];
    int gmi[64];
    auto group_min = [&](int grp) {
        const int lo = grp * g, hi = std::min(lo + g, nn);
        K bk = key[lo];
        int bi = lo;
        for (int j = lo + 1; j < hi; ++j) {
            const bool l = static_cast<bool>(std::invoke(comp, key[j], bk));
            detail::cond_assign(bk, key[j], l);
            bi = l ? j : bi;
        }
        gmk[grp] = bk;
        gmi[grp] = bi;
    };
    for (int grp = 0; grp < G; ++grp) group_min(grp);
    for (int o = 0; o < nn; ++o) {
        K bk = gmk[0];
        int bg = 0;
        for (int grp = 1; grp < G; ++grp) {
            const bool l = static_cast<bool>(std::invoke(comp, gmk[grp], bk));
            detail::cond_assign(bk, gmk[grp], l);
            bg = l ? grp : bg;
        }
        const int w = gmi[bg];
        buf[o] = first[w];
        key[w] = inf;
        group_min(bg);
    }
    for (int i = 0; i < nn; ++i) first[i] = buf[i];
}

// =====================================================================
// MIN-FINDING NETWORKS (data-INDEPENDENT, like the halver -- the right answer).
// min_index<N>: a fixed, branchless, log-depth BALANCED-TREE reduction over the
// N register-resident keys, tracking the winner's index by cmov.  NO indexed
// memory (every access is a compile-time offset), NO branches -> max ILP, and
// depth log2(N) instead of the 2-accumulator scan's N/2.  selection_sort_n<L>
// unrolls the whole leaf from these networks; one switch dispatches the runtime
// leaf size.
// =====================================================================
namespace minnet {
template <int Lo, int Hi, class It, class Comp, class Proj, class K>
[[gnu::always_inline]] inline void rec(It first, Comp& comp, Proj& proj, K& bk, int& bi) {
    if constexpr (Hi - Lo == 1) {
        bk = std::invoke(proj, first[Lo]);
        bi = Lo;
    } else {
        constexpr int Mid = (Lo + Hi) / 2;
        K ka, kb;
        int ia, ib;
        rec<Lo, Mid>(first, comp, proj, ka, ia);
        rec<Mid, Hi>(first, comp, proj, kb, ib);
        const bool lt = static_cast<bool>(std::invoke(comp, kb, ka));  // kb < ka
        bk = ka;
        detail::cond_assign(bk, kb, lt);  // bk = lt ? kb : ka
        bi = lt ? ib : ia;
    }
}
template <int N, class It, class Comp, class Proj>
[[gnu::always_inline]] inline int min_index(It first, Comp comp, Proj proj) {
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    K bk;
    int bi;
    rec<0, N>(first, comp, proj, bk, bi);
    return bi;
}
}  // namespace minnet

template <int L, class It, class Comp, class Proj>
[[gnu::always_inline]] inline void selection_sort_n(It first, Comp comp, Proj proj) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (..., ([&] {
            constexpr int rem = L - I;
            if constexpr (rem > 1) {
                const int m = minnet::min_index<rem>(first + I, comp, proj);
                if (m != 0) std::iter_swap(first + I, first + I + m);
            }
        }()));
    }(std::make_integer_sequence<int, L>{});
}

template <class It, class Comp, class Proj>
[[gnu::noinline]] void selsort_net(It first, It last, Comp comp, Proj proj) {
    switch (static_cast<int>(last - first)) {
        case 0: case 1: return;
#define QSLR_NET_CASE(k) case k: selection_sort_n<k>(first, comp, proj); return;
        QSLR_NET_CASE(2)  QSLR_NET_CASE(3)  QSLR_NET_CASE(4)  QSLR_NET_CASE(5)
        QSLR_NET_CASE(6)  QSLR_NET_CASE(7)  QSLR_NET_CASE(8)  QSLR_NET_CASE(9)
        QSLR_NET_CASE(10) QSLR_NET_CASE(11) QSLR_NET_CASE(12) QSLR_NET_CASE(13)
        QSLR_NET_CASE(14) QSLR_NET_CASE(15) QSLR_NET_CASE(16) QSLR_NET_CASE(17)
        QSLR_NET_CASE(18) QSLR_NET_CASE(19) QSLR_NET_CASE(20) QSLR_NET_CASE(21)
        QSLR_NET_CASE(22) QSLR_NET_CASE(23) QSLR_NET_CASE(24)
#undef QSLR_NET_CASE
        default: detail::selection_sort(first, last, comp, proj);
    }
}

template <class T, class Proj>
void study_leaf(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    const std::size_t total = 1u << 18;
    if (total > max_size) return;
    auto master = gen_data<T>(dist::random_uniform{}, total, 0x1EAF1u, proj);

    for (std::size_t L : {std::size_t(4), std::size_t(6), std::size_t(8),
                          std::size_t(12), std::size_t(16), std::size_t(20),
                          std::size_t(24)}) {
        auto run_each = [&](const char* nm, auto timed_block, auto count_block) {
            // op counts (counting comparator + counted<T> elements, one pass)
            std::vector<counted<T>> cw(master.size());
            for (std::size_t i = 0; i < master.size(); ++i) cw[i].v = master[i];
            g_cmps = 0; g_moves = 0;
            for (std::size_t s = 0; s + L <= cw.size(); s += L)
                count_block(cw.begin() + s, cw.begin() + s + L);
            // correctness gate: every block must be sorted by (less, proj).
            for (std::size_t s = 0; s + L <= cw.size(); s += L)
                for (std::size_t i = s + 1; i < s + L; ++i)
                    if (std::less<>{}(std::invoke(proj, cw[i].v), std::invoke(proj, cw[i - 1].v))) {
                        std::fprintf(stderr, "LEAF NOT SORTED %s L=%zu %s\n", tname, L, nm);
                        std::abort();
                    }
            const std::size_t blocks = master.size() / L;
            double cpb = double(g_cmps) / blocks, mpb = double(g_moves) / blocks;
            // timing (plain comparator, real element)
            double ns = time_blocks<T>(total, master, [&](std::vector<T>& w) {
                for (std::size_t s = 0; s + L <= w.size(); s += L)
                    timed_block(w.begin() + s, w.begin() + s + L);
            });
            std::printf("leaf,%s,%zu,%s,%.4f,cmp=%.1f,mov=%.1f\n", tname, L, nm, ns, cpb, mpb);
            std::fflush(stdout);
        };
        // Projection that applies the real proj to the counted wrapper's value,
        // so the counted run compares in the SAME order as the timed run.
        auto cproj = [&proj](const counted<T>& c) { return std::invoke(proj, c.v); };
        // selection (our leaf), the halver leaf (quicksort's), insertion (ref).
        run_each("selection",
            [&](auto f, auto l) { detail::selection_sort(f, l, comp, proj); },
            [&](auto f, auto l) { detail::selection_sort(f, l, count_cmp{}, cproj); });
        run_each("halver",
            [&](auto f, auto l) { detail::halve_sort(f, l, comp, proj); },
            [&](auto f, auto l) { detail::halve_sort(f, l, count_cmp{}, cproj); });
        run_each("insertion",
            [&](auto f, auto l) { small_sort::insertion_sort(f, l, comp, proj); },
            [&](auto f, auto l) { small_sort::insertion_sort(f, l, count_cmp{}, cproj); });
        run_each("tournament",
            [&](auto f, auto l) { tournament_sort(f, l, comp, proj); },
            [&](auto f, auto l) { tournament_sort(f, l, count_cmp{}, cproj); });
        run_each("tourn_bl",
            [&](auto f, auto l) { tournament_bl(f, l, comp, proj); },
            [&](auto f, auto l) { tournament_bl(f, l, count_cmp{}, cproj); });
        run_each("group4",
            [&](auto f, auto l) { grouped_sort<4>(f, l, comp, proj); },
            [&](auto f, auto l) { grouped_sort<4>(f, l, count_cmp{}, cproj); });
        run_each("selnet",
            [&](auto f, auto l) { selsort_net(f, l, comp, proj); },
            [&](auto f, auto l) { selsort_net(f, l, count_cmp{}, cproj); });
    }
}

// ---- Part B: threshold sweep (full sort) -----------------------------------

template <class T, class Proj, class SortFn>
void time_full(const char* study, const char* tname, const char* dn, std::size_t n,
               const char* algo, Proj proj, const std::vector<T>& master, SortFn sortfn) {
    auto comp = std::less<>{};
    std::vector<T> work = master;
    sortfn(work);
    if (!std::is_sorted(work.begin(), work.end(), [&](const T& a, const T& b) {
            return comp(std::invoke(proj, a), std::invoke(proj, b));
        })) {
        std::fprintf(stderr, "NOT SORTED %s/%s/%s\n", tname, dn, algo);
        std::abort();
    }
    std::uint64_t reps = n >= (1u << 23) ? 3 : std::min<std::uint64_t>(
                             std::max<std::uint64_t>((1u << 23) / n, 5), 60);
    auto setup = [&] { work = master; };
    auto do_work = [&] { sortfn(work); bench::do_not_optimize(work[0]); };
    auto r = bench::measure(reps, setup, do_work, n >= (1u << 23) ? 1 : 3);
    std::printf("%s,%s,%s,%zu,%s,%.4f\n", study, tname, dn, n, algo,
                r.min_ns / static_cast<double>(n));
    std::fflush(stdout);
}

template <class T, class Proj>
void study_thresh(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    std::size_t n = 1u << 22;
    if (n > max_size) n = max_size;
    auto sweep = [&](const char* dn, const std::vector<T>& m) {
        auto run = [&](auto Tc) {
            constexpr std::ptrdiff_t Th = decltype(Tc)::value;
            char name[32];
            std::snprintf(name, sizeof name, "T%td", Th);
            time_full("thresh", tname, dn, m.size(), name, proj, m,
                      [&](std::vector<T>& w) { quicksort_lr<Th>(w.begin(), w.end(), comp, proj); });
        };
        run(std::integral_constant<std::ptrdiff_t, 4>{});
        run(std::integral_constant<std::ptrdiff_t, 6>{});
        run(std::integral_constant<std::ptrdiff_t, 8>{});
        run(std::integral_constant<std::ptrdiff_t, 10>{});
        run(std::integral_constant<std::ptrdiff_t, 12>{});
        run(std::integral_constant<std::ptrdiff_t, 16>{});
        run(std::integral_constant<std::ptrdiff_t, 20>{});
        run(std::integral_constant<std::ptrdiff_t, 24>{});
        run(std::integral_constant<std::ptrdiff_t, 32>{});
        run(std::integral_constant<std::ptrdiff_t, 48>{});
        run(std::integral_constant<std::ptrdiff_t, 64>{});
    };
    // random_uniform is the primary case; sorted_* are fast structured inputs
    // that are genuinely threshold-sensitive (a selection leaf is non-adaptive,
    // so a larger leaf costs more on already-ordered data).  few_unique is
    // omitted from the sweep: it is dominated by the no-3-way partition
    // recursion and is essentially threshold-independent (just slow).
    sweep("random_uniform", gen_data<T>(dist::random_uniform{}, n, 0x9501Du + n, proj));
    sweep("sorted_ascending", gen_data<T>(dist::sorted_ascending{}, n, 0, proj));
    sweep("sorted_descending", gen_data<T>(dist::sorted_descending{}, n, 0, proj));
}

// ---- Part B2: pivot-tier cutoff sweep (median-of-3 small / ninther large) ---
//
// Sweeps the node size below which the cheap inline median-of-3 pivot is used
// (above it: ninther).  A large cutoff = "median-of-3 nearly everywhere" (cheap
// per node but weaker balance / unsafe on the killer); a tiny cutoff = ninther
// almost everywhere (the old, expensive behaviour).  random/sorted at the sweep
// size; the median_of_3_killer is probed at a SMALL n so an O(n^2) blow-up of an
// over-large cutoff shows up as a huge ns/elem without stalling the run.

template <class T, class Proj>
void study_pivcut(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    std::size_t n = std::min<std::size_t>(1u << 22, max_size);
    auto sweep = [&](const char* dn, const std::vector<T>& m) {
        auto run = [&](auto Cc) {
            constexpr std::ptrdiff_t Cut = decltype(Cc)::value;
            char name[24];
            std::snprintf(name, sizeof name, "C%td", Cut);
            time_full("pivcut", tname, dn, m.size(), name, proj, m, [&](std::vector<T>& w) {
                detail::quicksort_lr_impl<16, Cut>(w.begin(), w.end(), comp, proj);
            });
        };
        run(std::integral_constant<std::ptrdiff_t, 16>{});
        run(std::integral_constant<std::ptrdiff_t, 32>{});
        run(std::integral_constant<std::ptrdiff_t, 64>{});
        run(std::integral_constant<std::ptrdiff_t, 128>{});
        // NOTE: cutoffs >= 256 are UNSAFE and are not swept -- "median-of-3 on
        // bigger nodes" degenerates on sorted_descending (the median-of-3 /
        // partition recursion interaction) to an O(n^2) deep LEFT spine that
        // overflows the fixed stack (a hard crash in release; observed at C256
        // for pair64).  This is exactly why the large nodes must use the ninther.
        // The sweep over the safe band 16..128 already shows the median-of-3 tier
        // is a wash on random and a growing penalty on sorted.
    };
    sweep("random_uniform", gen_data<T>(dist::random_uniform{}, n, 0x9501Du + n, proj));
    sweep("sorted_ascending", gen_data<T>(dist::sorted_ascending{}, n, 0, proj));
    sweep("sorted_descending", gen_data<T>(dist::sorted_descending{}, n, 0, proj));
    // Safety probe: median_of_3_killer at a SMALL n -- an over-large cutoff goes
    // O(n^2) here (ns/elem in the hundreds-thousands) while a safe cutoff stays
    // O(n log n) (tens).  n kept small so even the O(n^2) cells finish fast.
    std::size_t nk = std::min<std::size_t>(1u << 13, max_size);
    sweep("m3_killer_small", gen_data<T>(dist::median_of_3_killer{}, nk, 0, proj));
}

// ---- Part B3: pivot A/B -- isolate "inline ninther_pos vs library pivot::ninther"
// in ONE session (same driver, threshold, stack, leaf) so session-to-session
// frequency drift cancels in the ratio.

template <std::ptrdiff_t Threshold, class PivFn, class It, class Comp, class Proj>
[[gnu::noinline]] void ab_drv(It first, It last, PivFn piv, Comp comp, Proj proj) {
    It stack[256];
    int sp = 0;
    It lo = first, hi = last;
    while (true) {
        while (hi - lo > Threshold) {
            It p = piv(lo, hi, comp, proj);
            std::iter_swap(lo, p);
            auto key = std::invoke(proj, *lo);
            It m = algo::sized{}(lo + 1, hi, key, comp, proj);
            std::iter_swap(lo, m - 1);
            It pp = m - 1;
            stack[sp++] = pp + 1;
            hi = pp;
        }
        detail::selection_sort(lo, hi, comp, proj);
        if (sp == 0) break;
        lo = stack[--sp];
        hi = sp == 0 ? last : stack[sp - 1] - 1;
    }
}
struct piv_inline {
    template <class It, class C, class P>
    It operator()(It lo, It hi, C c, P p) const { return detail::ninther_pos(lo, hi, c, p); }
};
struct piv_lib {
    template <class It, class C, class P>
    It operator()(It lo, It hi, C c, P p) const { return pivot::ninther{}(lo, hi, c, p); }
};

template <class T, class Proj>
void study_ab(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    std::size_t n = std::min<std::size_t>(1u << 22, max_size);
    auto m = gen_data<T>(dist::random_uniform{}, n, 0x9501Du + n, proj);
    time_full("ab", tname, "random_uniform", n, "ninther_inline", proj, m,
              [&](std::vector<T>& w) { ab_drv<20>(w.begin(), w.end(), piv_inline{}, comp, proj); });
    time_full("ab", tname, "random_uniform", n, "ninther_lib", proj, m,
              [&](std::vector<T>& w) { ab_drv<20>(w.begin(), w.end(), piv_lib{}, comp, proj); });
}

// ---- Part B4: partitioner A/B in the threshold driver -----------------------
// Most nodes are small here, so a partitioner's fixed SETUP cost (boost_block
// cacheline-aligns two offset buffers) can outweigh its per-element edge.  Does
// a setup-free branchless Lomuto over a wider small-node band help (esp. pair64,
// where algo::sized switches to boost at n>24)?

template <std::ptrdiff_t Threshold, class PartFn, class It, class Comp, class Proj>
[[gnu::noinline]] void part_drv(It first, It last, PartFn part, Comp comp, Proj proj) {
    It stack[256];
    int sp = 0;
    It lo = first, hi = last;
    while (true) {
        while (hi - lo > Threshold) {
            It p = detail::ninther_pos(lo, hi, comp, proj);
            std::iter_swap(lo, p);
            auto key = std::invoke(proj, *lo);
            It m = part(lo + 1, hi, key, comp, proj);
            std::iter_swap(lo, m - 1);
            It pp = m - 1;
            stack[sp++] = pp + 1;
            hi = pp;
        }
        detail::selection_sort(lo, hi, comp, proj);
        if (sp == 0) break;
        lo = stack[--sp];
        hi = sp == 0 ? last : stack[sp - 1] - 1;
    }
}
struct part_sized { template <class I,class K,class C,class P> I operator()(I f,I l,K k,C c,P p) const { return algo::sized{}(f,l,k,c,p); } };
struct part_lomuto { template <class I,class K,class C,class P> I operator()(I f,I l,K k,C c,P p) const { return algo::lomuto_branchless{}(f,l,k,c,p); } };
struct part_boost  { template <class I,class K,class C,class P> I operator()(I f,I l,K k,C c,P p) const { return algo::boost_block{}(f,l,k,c,p); } };
template <std::ptrdiff_t Cut>
struct part_lom_boost { template <class I,class K,class C,class P> I operator()(I f,I l,K k,C c,P p) const {
    return (l - f) <= Cut ? algo::lomuto_branchless{}(f,l,k,c,p) : algo::boost_block{}(f,l,k,c,p); } };

template <class T, class Proj>
void study_part(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    std::size_t n = std::min<std::size_t>(1u << 22, max_size);
    auto m = gen_data<T>(dist::random_uniform{}, n, 0x9501Du + n, proj);
    auto one = [&](const char* nm, auto part) {
        time_full("part", tname, "random_uniform", n, nm, proj, m,
                  [&](std::vector<T>& w) { part_drv<20>(w.begin(), w.end(), part, comp, proj); });
    };
    one("sized", part_sized{});
    one("lomuto", part_lomuto{});
    one("boost", part_boost{});
    one("lom128_boost", part_lom_boost<128>{});
    one("lom256_boost", part_lom_boost<256>{});
}

// ---- Part C: head-to-head ---------------------------------------------------

template <class T, class Proj>
void study_vs(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    auto one = [&](const char* dn, std::size_t n, const std::vector<T>& m) {
        time_full("vs", tname, dn, n, "quicksort_lr", proj, m,
                  [&](std::vector<T>& w) { quicksort_lr(w.begin(), w.end(), comp, proj); });
        time_full("vs", tname, dn, n, "quicksort", proj, m,
                  [&](std::vector<T>& w) { quicksort(w.begin(), w.end(), comp, proj); });
        time_full("vs", tname, dn, n, "std_sort", proj, m, [&](std::vector<T>& w) {
            if constexpr (std::is_same_v<Proj, std::identity>) std::sort(w.begin(), w.end());
            else std::sort(w.begin(), w.end(), [&](const T& a, const T& b) {
                return comp(std::invoke(proj, a), std::invoke(proj, b)); });
        });
#ifdef HAVE_BOOST
        time_full("vs", tname, dn, n, "boost_pdqsort", proj, m, [&](std::vector<T>& w) {
            if constexpr (std::is_same_v<Proj, std::identity>) boost::sort::pdqsort(w.begin(), w.end());
            else boost::sort::pdqsort(w.begin(), w.end(), [&](const T& a, const T& b) {
                return comp(std::invoke(proj, a), std::invoke(proj, b)); });
        });
#endif
    };
    for (std::size_t n : {std::size_t(1u << 16), std::size_t(1u << 20), std::size_t(1u << 22)}) {
        if (n > max_size) continue;
        one("random_uniform", n, gen_data<T>(dist::random_uniform{}, n, 0x9501Du + n, proj));
    }
    std::size_t n = 1u << 20;
    if (n <= max_size) {
        one("sorted_ascending", n, gen_data<T>(dist::sorted_ascending{}, n, 0, proj));
        one("sorted_descending", n, gen_data<T>(dist::sorted_descending{}, n, 0, proj));
    }
}

// ---- Part D: stack-depth validation ----------------------------------------
//
// Replicates the quicksort_lr driver but only counts the maximum left-spine
// depth (no leaf work), to confirm the static stack capacity is never exceeded.

template <std::ptrdiff_t Threshold, class It, class Comp, class Proj>
int max_left_spine(It first, It last, Comp comp, Proj proj) {
    std::vector<It> stack;  // unbounded: we are measuring, not bounding
    int peak = 0;
    It lo = first, hi = last;
    while (true) {
        while (hi - lo > Threshold) {
            const auto n = hi - lo;
            It p = n <= detail::qslr_pivot_ninther_cutoff
                       ? detail::median3_pos(lo, lo + (n >> 1), hi - 1, comp, proj)
                       : detail::ninther_pos(lo, hi, comp, proj);
            std::iter_swap(lo, p);
            auto key = std::invoke(proj, *lo);
            It m = algo::sized{}(lo + 1, hi, key, comp, proj);
            std::iter_swap(lo, m - 1);
            It pp = m - 1;
            stack.push_back(pp + 1);
            peak = std::max(peak, static_cast<int>(stack.size()));
            hi = pp;
        }
        detail::selection_sort(lo, hi, comp, proj);
        if (stack.empty()) break;
        lo = stack.back();
        stack.pop_back();
        hi = stack.empty() ? last : stack.back() - 1;
    }
    return peak;
}

template <class T, class Proj>
void study_depth(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    int worst = 0;
    const char* worst_dist = "";
    auto probe = [&](const char* dn, std::vector<T> v) {  // by value: mutable copy
        int dpt = max_left_spine<detail::qslr_threshold>(v.begin(), v.end(), comp, proj);
        std::printf("depth,%s,%s,%zu,maxspine,%d\n", tname, dn, v.size(), dpt);
        if (dpt > worst) { worst = dpt; worst_dist = dn; }
    };
    // (a) ALL distributions at a small n (cheap even for the O(n^2) low-
    // cardinality ones), to confirm none builds a deep spine.
    {
        std::size_t n = std::min<std::size_t>(4096, max_size);
        for_each(default_distributions(), [&](auto d) {
            probe(d.name, gen_data<T>(d, n, 0x9501Du + n, proj));
        });
    }
    // (b) The non-pathological distributions at large n -- where log2(n), hence
    // the spine, is largest.  (The low-cardinality inputs are O(n^2) here and
    // have a trivially shallow spine anyway: an all-equal split leaves the left
    // empty every step.)
    {
        std::size_t n = std::min<std::size_t>(1u << 22, max_size);
        probe("random_uniform@big", gen_data<T>(dist::random_uniform{}, n, 1, proj));
        probe("sorted_ascending@big", gen_data<T>(dist::sorted_ascending{}, n, 0, proj));
        probe("sorted_descending@big", gen_data<T>(dist::sorted_descending{}, n, 0, proj));
        probe("nearly_sorted@big", gen_data<T>(dist::nearly_sorted{}, n, 2, proj));
        probe("organ_pipe@big", gen_data<T>(dist::organ_pipe{}, n, 0, proj));
        probe("reverse_organ_pipe@big", gen_data<T>(dist::reverse_organ_pipe{}, n, 0, proj));
        probe("shuffled_blocks@big", gen_data<T>(dist::shuffled_blocks{}, n, 3, proj));
        probe("median_of_3_killer@big", gen_data<T>(dist::median_of_3_killer{}, n, 0, proj));
        probe("single_outlier@big", gen_data<T>(dist::single_outlier{}, n, 0, proj));
    }
    std::printf("depth,%s,__WORST__,%zu,cap%d,%d\n", tname, std::size_t(1u << 22),
                detail::qslr_stack_cap, worst);
    std::fprintf(stderr, "[depth] %s worst left-spine = %d (%s), cap = %d\n",
                 tname, worst, worst_dist, detail::qslr_stack_cap);
    std::fflush(stdout);
}

template <class T, class Proj>
void run_type(const char* tname, Proj proj, std::size_t max_size, const std::string& mode) {
    if (mode == "all" || mode == "min") study_min<T>(tname, proj, max_size);
    if (mode == "all" || mode == "leaf") study_leaf<T>(tname, proj, max_size);
    if (mode == "all" || mode == "ab") study_ab<T>(tname, proj, max_size);
    if (mode == "all" || mode == "part") study_part<T>(tname, proj, max_size);
    if (mode == "all" || mode == "pivcut") study_pivcut<T>(tname, proj, max_size);
    if (mode == "all" || mode == "thresh") study_thresh<T>(tname, proj, max_size);
    if (mode == "all" || mode == "vs") study_vs<T>(tname, proj, max_size);
    if (mode == "all" || mode == "depth") study_depth<T>(tname, proj, max_size);
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = "all";
    std::size_t max_size = 1u << 26;
    if (argc > 1) mode = argv[1];
    if (argc > 2) max_size = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    std::printf("study,type,dist_or_L,n,algo,ns_or_depth\n");
    run_type<i64>("i64", std::identity{}, max_size, mode);
    run_type<pair64>("pair64", std::identity{}, max_size, mode);
    run_type<pair64>("pair64f", first_key{}, max_size, mode);
    return 0;
}
