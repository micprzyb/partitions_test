// Benchmark: OFFSET partition -- forward partition with a known all->= prefix.
//
// SPEC.  [first, first+offset) is all >= key (precondition); put every element
// strictly below `key` at the FRONT and return the count.  See
// include/partitions/offset_partition.hpp.
//
// AXES.  The cost of a candidate is governed by three parameters:
//   n  : total array length;
//   f  : offset/n, the prefix fraction (how much of the array is pre-known);
//   p  : the below fraction OF THE SUFFIX (the only compared region) --
//        c = p * (n - offset) elements end up at the front.
// The interesting interactions: the epilogue move of part_swap is
// min(c, offset) and so peaks when both f and p are large; scan_swap's branch
// is predictable at p ~ 0 or 1 and hostile at p ~ 0.5; `whole` wastes exactly
// the prefix compares, i.e. a factor 1/(1-f) of useful work.
//
// ALGOS.  scan_swap / gap_off / part_swap / fused_block / sized_off (the
// shipped dispatcher) from the header, plus `whole` = algo::sized over the
// entire range IGNORING the precondition -- the "don't exploit it" reference.
//
// MODES.  default/large: single calls, n in {2^16, 2^20, 2^22} (capped by
// argv[2]), pair64-by-first FIRST (the primary case); `byfirst`: the
// pair-of-i64-by-first deep dive (projection vs comparator formulation A/B,
// plus low-cardinality firsts -- see run_byfirst); `quick`/`small`: batched
// tiny blocks n in {64 .. 4096} (one call is below clock resolution).
//
// CSV: type,n,f,p,algo,ns_per_elem,ns_per_suffix_elem
//   ns_per_elem        = min ns / n           (whole-array view)
//   ns_per_suffix_elem = min ns / (n-offset)  (per COMPARED element -- the fair
//                        cross-f metric, since the prefix is free by contract)
//
// Inputs are random high-cardinality keys (mt19937_64), suffix shuffled, the
// prefix drawn from the >= side -- construction below.  Every algo/cell pair
// is verified (boundary + multiset) before timing.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/offset_partition.hpp"
#include "partitions/types.hpp"

using namespace partitions;

namespace {

struct first_key {
    template <class P>
    auto operator()(const P& p) const {
        return p.first;
    }
};

// `whole`: ignore the precondition, partition everything (reference).
struct whole_off {
    static constexpr const char* name = "whole";
    template <class I, class S, class K, class Comp, class Proj>
    I operator()(I first, S last, std::iter_difference_t<I>, K key, Comp comp,
                 Proj proj) const {
        return algo::sized{}(first, last, key, comp, proj);
    }
};

// Build one master input: n elements, `off` prefix elements >= key, suffix a
// shuffle of the remaining >= elements and exactly c = round(p*(n-off))
// below-key elements.  Returns {data, key}.  Ranks are unique (a shuffled
// 0..n-1), so counts are exact and cardinality is high.
template <class T, class Proj>
std::pair<std::vector<T>, std::remove_cvref_t<
                              decltype(std::invoke(Proj{}, T{}))>>
make_input(std::size_t n, double f, double p, std::uint64_t seed, Proj) {
    const auto off = static_cast<std::ptrdiff_t>(f * static_cast<double>(n));
    const auto c = static_cast<std::ptrdiff_t>(
        p * static_cast<double>(static_cast<std::ptrdiff_t>(n) - off));
    std::mt19937_64 rng(seed);
    std::vector<i64> ranks(n);
    for (std::size_t i = 0; i < n; ++i) ranks[i] = static_cast<i64>(i);
    std::shuffle(ranks.begin(), ranks.end(), rng);
    // key rank: exactly c elements have rank < c.
    const i64 key_rank = c;
    // Layout: prefix = `off` ranks >= key_rank; suffix = the rest, shuffled.
    std::vector<i64> lay;
    lay.reserve(n);
    std::vector<i64> rest;
    rest.reserve(n);
    for (i64 r : ranks)
        if (r >= key_rank && static_cast<std::ptrdiff_t>(lay.size()) < off)
            lay.push_back(r);
        else
            rest.push_back(r);
    std::shuffle(rest.begin(), rest.end(), rng);
    lay.insert(lay.end(), rest.begin(), rest.end());

    using K = std::remove_cvref_t<decltype(std::invoke(Proj{}, T{}))>;
    std::vector<T> out(n);
    K key;
    if constexpr (std::is_same_v<T, pair64> && std::is_same_v<K, i64>) {
        // by-first projection: rank in .first, payload in .second.
        for (std::size_t i = 0; i < n; ++i)
            out[i] = pair64{lay[i], static_cast<i64>(i)};
        key = key_rank;
    } else {
        for (std::size_t i = 0; i < n; ++i) out[i] = make_value<T>(lay[i]);
        key = std::invoke(Proj{}, make_value<T>(key_rank));
    }
    return {std::move(out), key};
}

template <class T, class K, class Comp, class Proj>
bool verify(const std::vector<T>& after, const std::vector<T>& before,
            std::ptrdiff_t m, K key, Comp comp, Proj proj) {
    for (std::ptrdiff_t i = 0; i < m; ++i)
        if (!static_cast<bool>(comp(std::invoke(proj, after[i]), key)))
            return false;
    for (std::size_t i = static_cast<std::size_t>(m); i < after.size(); ++i)
        if (static_cast<bool>(comp(std::invoke(proj, after[i]), key)))
            return false;
    auto x = after;
    auto y = before;
    auto less = [](const T& a, const T& b) { return std::less<>{}(a, b); };
    std::sort(x.begin(), x.end(), less);
    std::sort(y.begin(), y.end(), less);
    return x == y;
}

// Time every algorithm on one prepared cell (single calls, min over reps).
template <class T, class K, class Comp, class Proj>
void bench_cell(const char* tn, std::size_t n, double f, double p,
                std::ptrdiff_t off, const std::vector<T>& master, K key,
                Comp comp, Proj proj) {
    auto run = [&](const char* nm, auto alg) {
        std::vector<T> work = master;
        auto m = alg(work.begin(), work.end(), off, key, comp, proj) -
                 work.begin();
        if (!verify(work, master, m, key, comp, proj)) {
            std::fprintf(stderr, "WRONG %s n=%zu f=%.2f p=%.2f %s\n", tn, n, f,
                         p, nm);
            std::abort();
        }
        std::uint64_t reps =
            n >= (1u << 21)
                ? 9
                : std::min<std::uint64_t>(
                      std::max<std::uint64_t>((1u << 24) / n, 9), 64);
        auto setup = [&] { work = master; };
        auto do_work = [&] {
            auto mm = alg(work.begin(), work.end(), off, key, comp, proj);
            bench::do_not_optimize(mm);
            bench::do_not_optimize(work[0]);
        };
        auto r = bench::measure(reps, setup, do_work, 3);
        const double suf = static_cast<double>(n) - static_cast<double>(off);
        std::printf("%s,%zu,%.2f,%.2f,%s,%.4f,%.4f\n", tn, n, f, p, nm,
                    r.min_ns / static_cast<double>(n), r.min_ns / suf);
        std::fflush(stdout);
    };
    run("scan_swap", algo_off::scan_swap_off{});
    run("gap_off", algo_off::gap_off{});
    run("part_swap", algo_off::part_swap_off{});
    run("fused_block", algo_off::fused_block_off{});
    run("prefix_fill", algo_off::prefix_fill_off{});
    run("whole", whole_off{});
    run("sized_off", algo_off::sized_off{});
}

template <class T, class Proj>
void run_large(const char* tn, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    for (std::size_t n : {std::size_t(1u << 16), std::size_t(1u << 20),
                          std::size_t(1u << 22)}) {
        if (n > max_size) continue;
        for (double f : {0.10, 0.50, 0.90}) {
            for (double p : {0.10, 0.50, 0.90}) {
                auto [master, key] =
                    make_input<T>(n, f, p, 0xA11CEu + n + std::size_t(f * 100) * 31 +
                                               std::size_t(p * 100) * 1009,
                                  proj);
                const auto off =
                    static_cast<std::ptrdiff_t>(f * static_cast<double>(n));
                bench_cell(tn, n, f, p, off, master, key, comp, proj);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mode "byfirst": the pair-of-i64-compared-by-FIRST deep dive (the primary
// production case).  Three variants per (n, f, p) cell, same data geometry:
//
//   p64f_proj   : the projection formulation -- proj = first_key, key is the
//                 8-byte first coordinate (what `large` mode also runs).
//   p64f_comp   : the comparator formulation -- comp compares .first only,
//                 the key is a full 16-byte pair64 PASSED BY VALUE.  An A/B
//                 against p64f_proj: same compares, wider pivot -- tests the
//                 by-value pivot/register-pressure claim for wide keys.
//   p64f_dup256 : projection formulation on LOW-CARDINALITY firsts (256
//                 distinct values, ties everywhere) -- the regime where the
//                 below-test branch turns predictable in runs and the
//                 branchy/branchless trade can flip (cf. boost_block's
//                 low-cardinality caveat).
// ---------------------------------------------------------------------------

struct first_lt {
    bool operator()(const pair64& a, const pair64& b) const {
        return a.first < b.first;
    }
};

// pair64-by-first input with selectable first-coordinate cardinality.
// card == 0: unique firsts (a shuffled 0..n-1); card > 0: firsts drawn
// uniformly from [0, card).  The key is the c-th smallest first (c =
// round(p*(n-off))), so the nominal p is exact for unique firsts and accurate
// to within one tie-block for low cardinality.  Prefix = `off` elements with
// first >= key; suffix shuffled.
inline std::pair<std::vector<pair64>, i64> make_byfirst(std::size_t n, double f,
                                                        double p, int card,
                                                        std::uint64_t seed) {
    const auto off = static_cast<std::ptrdiff_t>(f * static_cast<double>(n));
    const auto c = static_cast<std::ptrdiff_t>(
        p * static_cast<double>(static_cast<std::ptrdiff_t>(n) - off));
    std::mt19937_64 rng(seed);
    std::vector<i64> firsts(n);
    if (card == 0) {
        for (std::size_t i = 0; i < n; ++i) firsts[i] = static_cast<i64>(i);
        std::shuffle(firsts.begin(), firsts.end(), rng);
    } else {
        for (auto& v : firsts)
            v = static_cast<i64>(rng() % static_cast<std::uint64_t>(card));
    }
    std::vector<i64> sorted = firsts;
    std::sort(sorted.begin(), sorted.end());
    const i64 key = sorted[static_cast<std::size_t>(c)];

    std::vector<i64> lay, rest;
    lay.reserve(n);
    rest.reserve(n);
    for (i64 v : firsts)
        if (v >= key && static_cast<std::ptrdiff_t>(lay.size()) < off)
            lay.push_back(v);
        else
            rest.push_back(v);
    std::shuffle(rest.begin(), rest.end(), rng);
    lay.insert(lay.end(), rest.begin(), rest.end());

    std::vector<pair64> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = pair64{lay[i], static_cast<i64>(i)};
    return {std::move(out), key};
}

// ---------------------------------------------------------------------------
// Mode "zero": the ZERO-OFFSET OVERHEAD measurement.  offset == 0 is, by
// contract, the ordinary forward partition, so the shipped dispatcher
// `sized_off` must match raw `algo::sized` on the same data -- any gap is
// pure overhead of the offset machinery.  This mode times EXACTLY that
// head-to-head (nothing else: comparing raw against a variant the dispatcher
// would not route to is meaningless) and prints one row per (type, n, p):
//
//   type,n,p,routed,raw_ns_per_elem,off0_ns_per_elem,overhead_pct,control_pct
//
// control_pct is the measured "overhead" of a BYTE-IDENTICAL second copy of
// the raw kernel -- the harness noise floor (code placement of equal copies);
// off0 overhead is only real where it exceeds the control's scatter.
//
// `routed` names the variant sized_off selects at offset 0 for this (T, n)
// (gap_off below the size cutoff, prefix_fill above -- which immediately
// falls through to algo::sized, see the header).  n < 2^16 is batched over a
// 2^20-element pool (a single call is below clock resolution), larger sizes
// are single calls; both sides of the comparison use identical data, the
// same harness, and min-of-reps.
// ---------------------------------------------------------------------------

// The two zero-mode kernels are NOINLINE: inlined into the timing lambdas,
// each side gets its own copy of the (now identical, given sized_off's
// offset==0 early exit) partition code at a different address, and loop
// alignment / uop-cache luck then produces +-10-27% per-cell scatter IN BOTH
// DIRECTIONS -- visible as off0 "beating" raw, which is impossible for real
// overhead.  Function-aligned out-of-line copies cut that variance; the
// remaining jitter is reported as such.
template <class It, class K, class Comp, class Proj>
[[gnu::noinline]] It zero_raw_call(It b, It e, K key, Comp comp, Proj proj) {
    return algo::sized{}(b, e, key, comp, proj);
}
template <class It, class K, class Comp, class Proj>
[[gnu::noinline]] It zero_off0_call(It b, It e, K key, Comp comp, Proj proj) {
    return algo_off::sized_off{}(b, e, 0, key, comp, proj);
}
// CONTROL: a byte-identical second copy of the raw kernel.  Its measured
// "overhead" vs zero_raw_call is pure harness noise (code placement /
// alignment of an equal copy), i.e. the per-cell noise floor against which
// the off0 column must be read.
template <class It, class K, class Comp, class Proj>
[[gnu::noinline]] It zero_raw2_call(It b, It e, K key, Comp comp, Proj proj) {
    return algo::sized{}(b, e, key, comp, proj);
}

// Element from a unique rank for the zero study; pair_li is bench-local
// (types.hpp's make_value does not cover it).
template <class T>
T zero_elem(i64 rank, i64 idx) {
    if constexpr (std::is_same_v<T, pair_li>) {
        return pair_li{static_cast<long>(rank),
                       static_cast<int>(idx & 0x7fffffff)};
    } else if constexpr (std::is_same_v<T, pair64>) {
        return pair64{rank, idx};  // by-first AND lex are monotone in rank
    } else {
        return static_cast<T>(rank);
    }
}

template <class T, class Proj>
void run_zero(const char* tn, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    using K = std::remove_cvref_t<decltype(std::invoke(proj, T{}))>;
    constexpr std::size_t kPool = 1u << 20;
    for (std::size_t n :
         {std::size_t(64), std::size_t(256), std::size_t(1024),
          std::size_t(4096), std::size_t(16384), std::size_t(1u << 16),
          std::size_t(1u << 18), std::size_t(1u << 20), std::size_t(1u << 22)}) {
        if (n > max_size) continue;
        const bool batched = n < (1u << 16);
        const std::size_t blocks = batched ? kPool / n : 1;
        for (double p : {0.10, 0.50, 0.90}) {
            // Each block: a fresh shuffle of ranks 0..n-1, so the key rank
            // c = p*n (and thus the key value) is the same for every block.
            const auto c = static_cast<i64>(p * static_cast<double>(n));
            std::mt19937_64 rng(0x0FF5E70u + n + std::size_t(p * 100) * 1009);
            std::vector<T> master(blocks * n);
            std::vector<i64> ranks(n);
            for (std::size_t b = 0; b < blocks; ++b) {
                for (std::size_t i = 0; i < n; ++i)
                    ranks[i] = static_cast<i64>(i);
                std::shuffle(ranks.begin(), ranks.end(), rng);
                for (std::size_t i = 0; i < n; ++i)
                    master[b * n + i] =
                        zero_elem<T>(ranks[i], static_cast<i64>(i));
            }
            const K key = std::invoke(proj, zero_elem<T>(c, 0));

            // One timing pass over the pool for a given per-block callable.
            auto time_one = [&](auto fn) {
                std::vector<T> work = master;
                {  // verify every block once: boundary == c, multiset kept
                    for (std::size_t b = 0; b < blocks; ++b) {
                        auto bb = work.begin() +
                                  static_cast<std::ptrdiff_t>(b * n);
                        std::vector<T> orig(bb, bb + n);
                        std::ptrdiff_t m = fn(bb, bb + n) - bb;
                        if (m != c ||
                            !verify(std::vector<T>(bb, bb + n), orig, m, key,
                                    comp, proj)) {
                            std::fprintf(stderr, "WRONG zero %s n=%zu p=%.2f\n",
                                         tn, n, p);
                            std::abort();
                        }
                    }
                }
                auto setup = [&] { work = master; };
                auto do_work = [&] {
                    for (std::size_t b = 0; b < blocks; ++b) {
                        auto bb = work.begin() +
                                  static_cast<std::ptrdiff_t>(b * n);
                        auto m = fn(bb, bb + n);
                        bench::do_not_optimize(m);  // anti-fusion barrier
                    }
                    bench::do_not_optimize(work[0]);
                };
                std::uint64_t reps =
                    batched ? 16
                            : std::min<std::uint64_t>(
                                  std::max<std::uint64_t>((1u << 24) / n, 9),
                                  64);
                auto r = bench::measure(reps, setup, do_work, 3);
                return r.min_ns / static_cast<double>(blocks * n);
            };

            // A/B/A/B: two alternated rounds per side, min of mins, so slow
            // drift (frequency, thermal) cannot bias one side.
            auto raw_fn = [&](auto b, auto e) {
                return zero_raw_call(b, e, key, comp, proj);
            };
            auto off0_fn = [&](auto b, auto e) {
                return zero_off0_call(b, e, key, comp, proj);
            };
            auto raw2_fn = [&](auto b, auto e) {
                return zero_raw2_call(b, e, key, comp, proj);
            };
            double raw = time_one(raw_fn);
            double off0 = time_one(off0_fn);
            double raw2 = time_one(raw2_fn);
            raw = std::min(raw, time_one(raw_fn));
            off0 = std::min(off0, time_one(off0_fn));
            raw2 = std::min(raw2, time_one(raw2_fn));
            // What algo::sized itself runs at this size (both sides take the
            // same path: sized_off's offset==0 early exit IS algo::sized).
            const char* routed =
                static_cast<std::ptrdiff_t>(n) <=
                        algo::detail::sized_cutoff<T>
                    ? "lomuto_branchless"
                    : "boost_block";
            std::printf("%s,%zu,%.2f,%s,%.4f,%.4f,%+.2f,%+.2f\n", tn, n, p,
                        routed, raw, off0, (off0 / raw - 1.0) * 100.0,
                        (raw2 / raw - 1.0) * 100.0);
            std::fflush(stdout);
        }
    }
}

void run_byfirst(std::size_t max_size) {
    for (std::size_t n : {std::size_t(1u << 16), std::size_t(1u << 20),
                          std::size_t(1u << 22)}) {
        if (n > max_size) continue;
        for (double f : {0.10, 0.50, 0.90}) {
            for (double p : {0.10, 0.50, 0.90}) {
                const auto off =
                    static_cast<std::ptrdiff_t>(f * static_cast<double>(n));
                const std::uint64_t seed = 0xF1257u + n +
                                           std::size_t(f * 100) * 31 +
                                           std::size_t(p * 100) * 1009;
                {
                    auto [master, key] = make_byfirst(n, f, p, 0, seed);
                    bench_cell("p64f_proj", n, f, p, off, master, key,
                               std::less<>{}, first_key{});
                    bench_cell("p64f_comp", n, f, p, off, master,
                               pair64{key, 0}, first_lt{}, std::identity{});
                }
                {
                    auto [master, key] = make_byfirst(n, f, p, 256, seed);
                    bench_cell("p64f_dup256", n, f, p, off, master, key,
                               std::less<>{}, first_key{});
                }
            }
        }
    }
}

// Batched tiny blocks: split a pool, run the algorithm on each size-n block;
// one cell's key/offset geometry is identical across blocks.  do_not_optimize
// per call is the anti-fusion barrier (see bench_move_low_half).
template <class T, class Proj>
void run_small(const char* tn, Proj proj, double fixed_f = 0.50) {
    auto comp = std::less<>{};
    const std::size_t pool = 1u << 20;
    for (std::size_t n : {std::size_t(64), std::size_t(256), std::size_t(1024),
                          std::size_t(4096)}) {
        const std::size_t blocks = pool / n;
        for (double f : {fixed_f}) {
            for (double p : {0.10, 0.50, 0.90}) {
                const auto off =
                    static_cast<std::ptrdiff_t>(f * static_cast<double>(n));
                // One independent block input per slot, same geometry.
                std::vector<T> master;
                master.reserve(pool);
                std::remove_cvref_t<decltype(std::invoke(proj, T{}))> key{};
                for (std::size_t b = 0; b < blocks; ++b) {
                    auto [blk, k] = make_input<T>(n, f, p, 0xBEEFu + b, proj);
                    key = k;  // same n/f/p -> same key rank for every block
                    master.insert(master.end(), blk.begin(), blk.end());
                }
                auto run = [&](const char* nm, auto alg) {
                    {  // verify every block once
                        std::vector<T> chk = master;
                        for (std::size_t b = 0; b < blocks; ++b) {
                            auto bb = chk.begin() +
                                      static_cast<std::ptrdiff_t>(b * n);
                            std::vector<T> orig(bb, bb + n);
                            auto m = alg(bb, bb + n, off, key, comp, proj) - bb;
                            if (!verify(std::vector<T>(bb, bb + n), orig, m,
                                        key, comp, proj)) {
                                std::fprintf(stderr,
                                             "WRONG small %s n=%zu p=%.2f %s\n",
                                             tn, n, p, nm);
                                std::abort();
                            }
                        }
                    }
                    std::vector<T> work = master;
                    auto setup = [&] { work = master; };
                    auto do_work = [&] {
                        for (std::size_t b = 0; b < blocks; ++b) {
                            auto bb = work.begin() +
                                      static_cast<std::ptrdiff_t>(b * n);
                            auto m = alg(bb, bb + n, off, key, comp, proj);
                            bench::do_not_optimize(m);
                        }
                        bench::do_not_optimize(work[0]);
                    };
                    auto r = bench::measure(16, setup, do_work, 3);
                    const double suf =
                        static_cast<double>(blocks) *
                        (static_cast<double>(n) - static_cast<double>(off));
                    std::printf("%s,%zu,%.2f,%.2f,%s,%.4f,%.4f\n", tn, n, f, p,
                                nm,
                                r.min_ns / static_cast<double>(blocks * n),
                                r.min_ns / suf);
                    std::fflush(stdout);
                };
                run("scan_swap", algo_off::scan_swap_off{});
                run("gap_off", algo_off::gap_off{});
                run("part_swap", algo_off::part_swap_off{});
                run("fused_block", algo_off::fused_block_off{});
                run("prefix_fill", algo_off::prefix_fill_off{});
                run("whole", whole_off{});
                run("sized_off", algo_off::sized_off{});
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "large";
    if (mode != "zero")
        std::printf("type,n,f,p,algo,ns_per_elem,ns_per_suffix_elem\n");
    if (mode == "small" || mode == "quick") {  // quick == the batched small study
        run_small<i64>("i64", std::identity{});
        run_small<pair64>("pair64", std::identity{});
        run_small<pair64>("pair64f", first_key{});
        return 0;
    }
    std::size_t max_size = 1u << 22;
    if (argc > 2) max_size = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (mode == "byfirst") {  // the primary case: pair of i64 compared by .first
        run_byfirst(max_size);
        return 0;
    }
    if (mode == "zero") {  // offset == 0 overhead vs the raw partition
        std::printf("type,n,p,routed,raw_ns_per_elem,off0_ns_per_elem,overhead_pct,control_pct\n");
        run_zero<pair64>("pair64f", first_key{}, max_size);
        run_zero<i64>("i64", std::identity{}, max_size);
        run_zero<pair64>("pair64", std::identity{}, max_size);
        run_zero<i32>("i32", std::identity{}, max_size);
        run_zero<pair_li>("pair_li", std::identity{}, max_size);
        return 0;
    }
    // pair64-by-first runs FIRST: it is the most important production case.
    run_large<pair64>("pair64f", first_key{}, max_size);
    run_large<i64>("i64", std::identity{}, max_size);
    run_large<pair64>("pair64", std::identity{}, max_size);
    return 0;
}
