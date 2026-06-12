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
// argv[2]); `quick`: sizes capped at 4096 (fast smoke run); `small`: batched
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
                auto run = [&](const char* nm, auto alg) {
                    std::vector<T> work = master;
                    auto m = alg(work.begin(), work.end(), off, key, comp, proj) -
                             work.begin();
                    if (!verify(work, master, m, key, comp, proj)) {
                        std::fprintf(stderr, "WRONG %s n=%zu f=%.2f p=%.2f %s\n",
                                     tn, n, f, p, nm);
                        std::abort();
                    }
                    std::uint64_t reps =
                        n >= (1u << 21)
                            ? 9
                            : std::min<std::uint64_t>(
                                  std::max<std::uint64_t>((1u << 24) / n, 9), 64);
                    auto setup = [&] { work = master; };
                    auto do_work = [&] {
                        auto mm = alg(work.begin(), work.end(), off, key, comp,
                                      proj);
                        bench::do_not_optimize(mm);
                        bench::do_not_optimize(work[0]);
                    };
                    auto r = bench::measure(reps, setup, do_work, 3);
                    const double suf = static_cast<double>(n) -
                                       static_cast<double>(off);
                    std::printf("%s,%zu,%.2f,%.2f,%s,%.4f,%.4f\n", tn, n, f, p,
                                nm, r.min_ns / static_cast<double>(n),
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

// Batched tiny blocks: split a pool, run the algorithm on each size-n block;
// one cell's key/offset geometry is identical across blocks.  do_not_optimize
// per call is the anti-fusion barrier (see bench_move_low_half).
template <class T, class Proj>
void run_small(const char* tn, Proj proj) {
    auto comp = std::less<>{};
    const std::size_t pool = 1u << 20;
    for (std::size_t n : {std::size_t(64), std::size_t(256), std::size_t(1024),
                          std::size_t(4096)}) {
        const std::size_t blocks = pool / n;
        for (double f : {0.50}) {
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
    std::printf("type,n,f,p,algo,ns_per_elem,ns_per_suffix_elem\n");
    if (mode == "small" || mode == "quick") {  // quick == the batched small study
        run_small<i64>("i64", std::identity{});
        run_small<pair64>("pair64", std::identity{});
        run_small<pair64>("pair64f", first_key{});
        return 0;
    }
    std::size_t max_size = 1u << 22;
    if (argc > 2) max_size = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    run_large<i64>("i64", std::identity{}, max_size);
    run_large<pair64>("pair64", std::identity{}, max_size);
    run_large<pair64>("pair64f", first_key{}, max_size);
    return 0;
}
