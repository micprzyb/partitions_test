// Partition EFFICIENCY study.
//
// Efficiency metric (lower = better):
//
//     E = (pivot-find time + partition time) / |smaller part|
//
// where |smaller part| = min(m - first, last - m) for the partition point m.
// This is "nanoseconds spent per element of guaranteed split" -- it rewards both
// SPEED (numerator) and BALANCE (a perfect split maximises the denominator at
// n/2; a 3:1 split halves it).  Aggregated as a ratio of totals over a batch:
// E = (total time over all blocks) / (sum of |smaller part| over all blocks).
//
// We also print the raw ns/elem and the mean balance (fraction < pivot) for
// context.  At large n the O(1) pivot cost vanishes and E -> partition ns/elem
// divided by the balance fraction, so the metric isolates "fast partition x
// well-centred pivot".
//
// SMALL ARRAYS are compared against a small SORTER, which gives a PERFECT
// partition (sort the block, split at the middle: |smaller| = n/2 exactly) at
// the cost of a full branchless sorting network.  The question is whether that
// perfect balance beats a cheaper-but-less-balanced find+partition under E.
//
// HAND-CRAFTED small partitioners are included: a sorting-network partition
// (perfect), a cheap pseudomedian-pivot + branchless partition, and a
// median-of-3-of-3 pivot in FIND vs EXCHANGE form.  (Note: the pseudo15/pseudo9
// approximate-median *networks* are value-computing DAGs, not permutation
// networks, so they cannot be run in place as compare-exchanges to make a
// partition; the realizable "compare-exchange network -> perfect partition" is
// the sorting network, i.e. sort_mid.)
//
// Element types: i64, pair64 (lexicographic), pair64f (compare .first only).
//
// Usage: bench_partition_efficiency [maxsize]
// CSV: type,n,strategy,E_ns_per_smaller,raw_ns_per_elem,mean_balance,smaller_frac

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/small_sort.hpp"

using namespace partitions;

namespace {

// ---- strategies: each maps [first,last) -> partition point m ---------------

template <class I, class C, class P>
I m_sortmid(I f, I l, C c, P pr) {
    small_sort::sort(f, l, c, pr);
    auto key = std::invoke(pr, *(f + (l - f) / 2));
    return std::lower_bound(f, l, key, [&](const auto& x, const auto& k) {
        return static_cast<bool>(std::invoke(c, std::invoke(pr, x), k));
    });
}
template <class I, class C, class P>
I m_m3_boost(I f, I l, C c, P pr) {
    auto p = pivot::median_of_3{}(f, l, c, pr);
    return partition_by_position(algo::boost_block{}, f, l, p, c, pr);
}
template <class I, class C, class P>
I m_ninther_boost(I f, I l, C c, P pr) {
    auto p = pivot::ninther{}(f, l, c, pr);  // m3m3 find
    return partition_by_position(algo::boost_block{}, f, l, p, c, pr);
}
template <class I, class C, class P>
I m_m5m5_boost(I f, I l, C c, P pr) {
    auto p = pivot::median_of_5_medians_of_5{}(f, l, c, pr);
    return partition_by_position(algo::boost_block{}, f, l, p, c, pr);
}
template <class I, class C, class P>
I m_mom_boost(I f, I l, C c, P pr) {  // exact median (BFPRT quickselect, O(n))
    auto p = pivot::median_of_medians_5_inplace{}(f, l, c, pr);
    return partition_by_position(algo::boost_block{}, f, l, p, c, pr);
}
template <class I, class C, class P>
I m_pseudo9_boost(I f, I l, C c, P pr) {  // cheap branchless network pseudomedian
    auto r = pivot::pseudo9{}(f, l, c, pr);  // value pivot
    return partition_by_key(algo::boost_block{}, f, l, r.key, c, pr);
}
template <class I, class C, class P>
I m_ninther_lomuto(I f, I l, C c, P pr) {  // hand-crafted: branchless Lomuto
    auto p = pivot::ninther{}(f, l, c, pr);
    auto key = std::invoke(pr, *p);
    return algo::lomuto_branchless{}(f, l, key, c, pr);
}

// m3m3 EXCHANGE pivot (reorders the 9 sampled elements) + boost partition.
template <class Less, class It>
void isort_swap(Less less, It* p, int k) {
    for (int i = 1; i < k; ++i)
        for (int j = i; j > 0 && less(p[j], p[j - 1]); --j) std::iter_swap(p[j - 1], p[j]);
}
template <class I, class C, class P>
I m_m3m3swap_boost(I f, I l, C c, P pr) {
    const auto n = l - f;
    if (n < 9) return m_m3_boost(f, l, c, pr);
    auto less = pivot::make_less(c, pr);
    const auto e = n / 8;
    I a[3] = {f, f + e, f + 2 * e};
    I b[3] = {f + n / 2 - e, f + n / 2, f + n / 2 + e};
    I cc[3] = {f + (n - 1) - 2 * e, f + (n - 1) - e, f + (n - 1)};
    isort_swap(less, a, 3);
    isort_swap(less, b, 3);
    isort_swap(less, cc, 3);
    I med[3] = {a[1], b[1], cc[1]};
    isort_swap(less, med, 3);
    return partition_by_position(algo::boost_block{}, f, l, med[1], c, pr);
}

constexpr std::size_t kElemsPerIter = 1u << 16;

template <class T, class Proj, class Gen, class Strat>
void run_one(const char* tname, Proj proj, Gen gen, const char* sname,
             Strat strat, std::size_t n) {
    auto comp = std::less<>{};
    const std::size_t batch = n < kElemsPerIter ? kElemsPerIter / n : 1;
    const std::size_t total = n * batch;
    std::vector<T> master;
    master.reserve(total);
    for (std::size_t b = 0; b < batch; ++b) {
        auto blk = gen(n, 0x1234u + b + n);
        master.insert(master.end(), blk.begin(), blk.end());
    }
    std::vector<T> work(total);

    // Smaller-part sum + balance: deterministic, measured once (untimed).
    work = master;
    std::uint64_t sum_smaller = 0;
    double bal_sum = 0;
    for (std::size_t b = 0; b < batch; ++b) {
        auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
        auto end = beg + static_cast<std::ptrdiff_t>(n);
        auto m = strat(beg, end, comp, proj);
        auto left = static_cast<std::uint64_t>(m - beg);
        sum_smaller += std::min(left, static_cast<std::uint64_t>(n) - left);
        bal_sum += static_cast<double>(left) / static_cast<double>(n);
    }
    if (sum_smaller == 0) sum_smaller = 1;  // avoid div by 0 on degenerate splits

    std::uint64_t reps = static_cast<std::uint64_t>((1u << 22) / total);
    reps = std::min<std::uint64_t>(std::max<std::uint64_t>(reps, 5), 200);
    auto setup = [&] { work = master; };
    auto do_work = [&] {
        std::ptrdiff_t s = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            s += strat(beg, beg + static_cast<std::ptrdiff_t>(n), comp, proj) - beg;
        }
        bench::do_not_optimize(s);
    };
    auto r = bench::measure(reps, setup, do_work);

    double E = r.min_ns / static_cast<double>(sum_smaller);
    double raw = r.min_ns / static_cast<double>(total);
    double mean_bal = bal_sum / static_cast<double>(batch);
    double smaller_frac = static_cast<double>(sum_smaller) / static_cast<double>(total);
    std::printf("%s,%zu,%s,%.4f,%.4f,%.4f,%.4f\n", tname, n, sname, E, raw,
                mean_bal, smaller_frac);
    std::fflush(stdout);
}

// strategy dispatch by name set (small adds sort_mid + hand-crafted).  sort_mid
// is only valid for n <= 24 (small_sort falls back to O(n^2) above that).
template <class T, class Proj, class Gen>
void run_type(const char* tname, Proj proj, Gen gen, std::size_t max_size) {
    constexpr std::size_t small_n[] = {8, 12, 16, 21, 24, 32, 48, 64};
    constexpr std::size_t large_n[] = {256,      1024,    4096,     1u << 16,
                                       1u << 18, 1u << 20, 1u << 22};
    auto S = [&](const char* nm, auto fn, std::size_t n) {
        run_one<T>(tname, proj, gen, nm, fn, n);
    };
    for (std::size_t n : small_n) {
        if (n > max_size) continue;
        if (n <= 24)
            S("sort_mid", [](auto f, auto l, auto c, auto p) { return m_sortmid(f, l, c, p); }, n);
        S("m3_boost", [](auto f, auto l, auto c, auto p) { return m_m3_boost(f, l, c, p); }, n);
        S("ninther_boost", [](auto f, auto l, auto c, auto p) { return m_ninther_boost(f, l, c, p); }, n);
        S("ninther_lomuto", [](auto f, auto l, auto c, auto p) { return m_ninther_lomuto(f, l, c, p); }, n);
        S("pseudo9_boost", [](auto f, auto l, auto c, auto p) { return m_pseudo9_boost(f, l, c, p); }, n);
        S("m3m3swap_boost", [](auto f, auto l, auto c, auto p) { return m_m3m3swap_boost(f, l, c, p); }, n);
    }
    for (std::size_t n : large_n) {
        if (n > max_size) continue;
        S("m3_boost", [](auto f, auto l, auto c, auto p) { return m_m3_boost(f, l, c, p); }, n);
        S("ninther_boost", [](auto f, auto l, auto c, auto p) { return m_ninther_boost(f, l, c, p); }, n);
        S("m5m5_boost", [](auto f, auto l, auto c, auto p) { return m_m5m5_boost(f, l, c, p); }, n);
        S("mom_boost", [](auto f, auto l, auto c, auto p) { return m_mom_boost(f, l, c, p); }, n);
        S("pseudo9_boost", [](auto f, auto l, auto c, auto p) { return m_pseudo9_boost(f, l, c, p); }, n);
    }
}

struct first_key {
    template <class P>
    auto operator()(const P& p) const { return p.first; }
};

// Generators.  i64 and pair64(lex) use random_uniform directly.  pair64f
// (compare .first only) must have FULL-ENTROPY first keys -- random_uniform's
// pair64 packs rank>>8 into .first, so for n<256 every .first is 0 (a degenerate
// all-equal-key block).  Instead build pair64{x, 0} from a full-range random x.
std::vector<pair64> gen_pair64f(std::size_t n, std::uint64_t seed) {
    auto k = dist::random_uniform{}.operator()<i64>(n, seed);
    std::vector<pair64> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = pair64{k[i], 0};
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 22;
    if (argc > 1) max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,n,strategy,E_ns_per_smaller,raw_ns_per_elem,mean_balance,smaller_frac\n");
    run_type<i64>("i64", std::identity{},
                  [](std::size_t n, std::uint64_t s) {
                      return dist::random_uniform{}.operator()<i64>(n, s);
                  },
                  max_size);
    run_type<pair64>("pair64", std::identity{},
                     [](std::size_t n, std::uint64_t s) {
                         return dist::random_uniform{}.operator()<pair64>(n, s);
                     },
                     max_size);
    run_type<pair64>("pair64f", first_key{}, gen_pair64f, max_size);
    return 0;
}
