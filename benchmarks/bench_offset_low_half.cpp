// Benchmark: move the bottom HALF of the below-key elements to the FRONT,
// given a known all->= prefix (offset_low_half / include/partitions/
// offset_low_half.hpp).
//
// STRATEGIES (S = #below in the whole array, all in the suffix; target
// k ~= S/2, take-all when S/2 < 16):
//
//   ref          : oracle -- count S, std::nth_element at rank k (k exact).
//   exact        : offset_partition by `key` (all S belows to the front),
//                  then ONE quickselect of the front block at rank S/2.
//                  k exact.  (shipped offset_low_half_exact)
//   two_phase    : the naive plan -- sample-estimate the below-median t,
//                  partition the SUFFIX around t with the best forward
//                  partitioner, then block-swap the left side to the front
//                  (= algo_off::part_swap_off by t).  Each delivered element
//                  moves twice.  Fallback t >= key: exact.
//   pivot_first  : test t BEFORE partitioning -- t < key (guaranteed for a
//                  sampled t): ONE offset_partition(first,last,offset,t)
//                  pass (prefix_fill: belows-to-t land at the front during
//                  the sweep, moved once).  (shipped offset_low_half)
//   ninther_first: pivot_first with a cheap ninther-of-the-suffix pivot
//                  instead of the 512-point sample: tests whether the sample
//                  costs anything (it shouldn't at large n) and what the
//                  ninther's k-accuracy looks like (it tracks the SUFFIX
//                  median, not the below-median -> k ~ (n-off)/2, biased by
//                  1/p).  The t >= key branch is REAL here (p < .5 makes the
//                  suffix median land above key).
//
// CSV: type,n,f,p,algo,ns_per_suffix_elem,k_ratio
//   k_ratio = 2k/S, the accuracy of the half split (1.0 = perfect; the
//   take-all base case reports 2.0 by construction and is fine).
//
// Axes as bench_offset_partition: f = offset/n, p = S/(n-offset).  Every
// algo/cell output is verified (bottom-k + multiset) before timing.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/offset_low_half.hpp"
#include "partitions/quicksort_lr.hpp"  // detail::ninther_pos
#include "partitions/types.hpp"

using namespace partitions;

namespace {

struct first_key {
    template <class P>
    auto operator()(const P& p) const {
        return p.first;
    }
};

// ---- ref: count + nth_element (the correctness/perf oracle) ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_ref(It first, It last, std::ptrdiff_t offset, K key,
                     Comp comp, Proj proj) {
    std::ptrdiff_t S = 0;
    for (It it = first + offset; it != last; ++it)
        S += static_cast<bool>(comp(std::invoke(proj, *it), key));
    const std::ptrdiff_t k = (S / 2 < 16) ? S : S / 2;
    if (k > 0 && k < last - first)
        std::nth_element(first, first + k, last,
                         [&](const auto& a, const auto& b) {
                             return static_cast<bool>(comp(
                                 std::invoke(proj, a), std::invoke(proj, b)));
                         });
    return k;
}

// Shared sampled below-median estimate of the SUFFIX; returns {ok, t}.
template <class It, class K, class Comp, class Proj>
auto sample_t(It first, It last, std::ptrdiff_t offset, K key, Comp comp,
              Proj proj) {
    using Key = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    struct R { bool ok; Key t; };
    const std::ptrdiff_t suffix = (last - first) - offset;
    constexpr int kSample = 512;
    if (suffix < 1024) return R{false, Key{}};
    const int m = static_cast<int>(std::min<std::ptrdiff_t>(suffix / 4, kSample));
    const std::ptrdiff_t stride = suffix / m;
    It base = first + offset;
    Key samp[kSample];
    std::ptrdiff_t below = 0;
    for (int i = 0; i < m; ++i) {
        samp[i] = std::invoke(proj, base[i * stride]);
        below += static_cast<bool>(comp(samp[i], key));
    }
    if (below < 16) return R{false, Key{}};
    const int r = static_cast<int>(below / 2);
    detail::quickselect(samp, samp + m, samp + r, comp, std::identity{});
    return R{true, samp[r]};
}

// ---- two_phase: suffix partition around t, then block swap to the front ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_two_phase(It first, It last, std::ptrdiff_t offset, K key,
                           Comp comp, Proj proj) {
    auto [ok, t] = sample_t(first, last, offset, key, comp, proj);
    if (!ok)
        return offset_low_half_exact(first, last, offset, key, comp, proj);
    It m = algo_off::part_swap_off{}(first, last, offset, t, comp, proj);
    if (m != first) return m - first;
    return offset_low_half_exact(first, last, offset, key, comp, proj);
}

// ---- pivot_first: ONE offset partition around t (the shipped path) ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_pivot_first(It first, It last, std::ptrdiff_t offset, K key,
                             Comp comp, Proj proj) {
    return offset_low_half(first, last, offset, key, comp, proj);
}

// ---- ninther_first: cheap suffix-ninther pivot, REAL t >= key branch ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_ninther_first(It first, It last, std::ptrdiff_t offset,
                               K key, Comp comp, Proj proj) {
    const std::ptrdiff_t suffix = (last - first) - offset;
    if (suffix < 1024)
        return offset_low_half_exact(first, last, offset, key, comp, proj);
    It p = detail::ninther_pos(first + offset, last, comp, proj);
    auto t = std::invoke(proj, *p);
    if (!static_cast<bool>(comp(t, key)))  // t >= key: bottom half not below t
        return offset_low_half_exact(first, last, offset, key, comp, proj);
    It m = algo_off::sized_off{}(first, last, offset, t, comp, proj);
    return m - first;
}

// ---------------------------------------------------------------------------

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
    const i64 key_rank = c;
    std::vector<i64> lay, rest;
    lay.reserve(n);
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
        for (std::size_t i = 0; i < n; ++i)
            out[i] = pair64{lay[i], static_cast<i64>(i)};
        key = key_rank;
    } else {
        for (std::size_t i = 0; i < n; ++i) out[i] = make_value<T>(lay[i]);
        key = std::invoke(Proj{}, make_value<T>(key_rank));
    }
    return {std::move(out), key};
}

template <class T, class Comp, class Proj>
bool bottom_k_ok(const std::vector<T>& v, std::ptrdiff_t k, Comp comp,
                 Proj proj) {
    const auto n = static_cast<std::ptrdiff_t>(v.size());
    if (k < 0 || k > n) return false;
    if (k == 0 || k == n) return true;
    auto kof = [&](std::ptrdiff_t i) {
        return std::invoke(proj, v[static_cast<std::size_t>(i)]);
    };
    auto maxfront = kof(0);
    for (std::ptrdiff_t i = 1; i < k; ++i)
        if (comp(maxfront, kof(i))) maxfront = kof(i);
    for (std::ptrdiff_t i = k; i < n; ++i)
        if (comp(kof(i), maxfront)) return false;
    return true;
}

template <class T>
bool same_multiset(std::vector<T> a, std::vector<T> b) {
    auto less = [](const T& x, const T& y) { return std::less<>{}(x, y); };
    std::sort(a.begin(), a.end(), less);
    std::sort(b.begin(), b.end(), less);
    return a == b;
}

template <class T, class Proj>
void run_type(const char* tn, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    for (std::size_t n : {std::size_t(1u << 16), std::size_t(1u << 20),
                          std::size_t(1u << 22)}) {
        if (n > max_size) continue;
        for (double f : {0.10, 0.50, 0.90}) {
            for (double p : {0.10, 0.50, 0.90}) {
                auto [master, key] = make_input<T>(
                    n, f, p,
                    0x10A5u + n + std::size_t(f * 100) * 31 +
                        std::size_t(p * 100) * 1009,
                    proj);
                const auto off =
                    static_cast<std::ptrdiff_t>(f * static_cast<double>(n));
                std::ptrdiff_t S = 0;
                for (std::size_t i = static_cast<std::size_t>(off); i < n; ++i)
                    S += static_cast<bool>(
                        comp(std::invoke(proj, master[i]), key));
                auto run = [&](const char* nm, auto fn) {
                    std::vector<T> work = master;
                    std::ptrdiff_t k = fn(work.begin(), work.end(), off, key,
                                          comp, proj);
                    if (!bottom_k_ok(work, k, comp, proj) ||
                        !same_multiset(work, master) || k > S) {
                        std::fprintf(stderr, "WRONG %s n=%zu f=%.2f p=%.2f %s k=%td\n",
                                     tn, n, f, p, nm, k);
                        std::abort();
                    }
                    std::uint64_t reps =
                        n >= (1u << 21)
                            ? 9
                            : std::min<std::uint64_t>(
                                  std::max<std::uint64_t>((1u << 24) / n, 9),
                                  48);
                    auto setup = [&] { work = master; };
                    auto do_work = [&] {
                        auto kk = fn(work.begin(), work.end(), off, key, comp,
                                     proj);
                        bench::do_not_optimize(kk);
                        bench::do_not_optimize(work[0]);
                    };
                    auto r = bench::measure(reps, setup, do_work, 3);
                    const double suf = static_cast<double>(n) -
                                       static_cast<double>(off);
                    const double kr =
                        S > 0 ? 2.0 * static_cast<double>(k) /
                                    static_cast<double>(S)
                              : 0.0;
                    std::printf("%s,%zu,%.2f,%.2f,%s,%.4f,%.3f\n", tn, n, f, p,
                                nm, r.min_ns / suf, kr);
                    std::fflush(stdout);
                };
                run("ref", [](auto a, auto e, auto o, auto k, auto c, auto pr) {
                    return v_ref(a, e, o, k, c, pr);
                });
                run("exact", [](auto a, auto e, auto o, auto k, auto c, auto pr) {
                    return offset_low_half_exact(a, e, o, k, c, pr);
                });
                run("two_phase", [](auto a, auto e, auto o, auto k, auto c, auto pr) {
                    return v_two_phase(a, e, o, k, c, pr);
                });
                run("pivot_first", [](auto a, auto e, auto o, auto k, auto c, auto pr) {
                    return v_pivot_first(a, e, o, k, c, pr);
                });
                run("ninther_first", [](auto a, auto e, auto o, auto k, auto c, auto pr) {
                    return v_ninther_first(a, e, o, k, c, pr);
                });
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 22;
    if (argc > 1) max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,n,f,p,algo,ns_per_suffix_elem,k_ratio\n");
    run_type<pair64>("pair64f", first_key{}, max_size);
    run_type<i64>("i64", std::identity{}, max_size);
    run_type<pair64>("pair64", std::identity{}, max_size);
    return 0;
}
