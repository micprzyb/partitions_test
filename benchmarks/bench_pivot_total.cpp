// Study: FIND vs EXCHANGE pivot selection for a SINGLE partition, judged on
// TOTAL cost = (cost of finding the pivot) + (cost of the partition).
//
// Two flavours of the same sample-based pseudo-median pivot, isolated so they
// differ in ONE thing only -- what the in-place insertion sort of the samples
// swaps:
//   * find     -- sorts the array of sampled ITERATORS (std::swap on pointers);
//                 the block is left UNCHANGED.  Returns the median's position.
//   * exchange -- sorts the sampled BLOCK ELEMENTS (std::iter_swap); physically
//                 reorders the block.  Returns the median's position.
// Same sample positions, same comparisons, same selection algorithm: the only
// difference is whether the swaps move 8-byte iterators in a hot local array or
// move the (8- or 16-byte) block elements at scattered addresses.
//
// Hypothesis under test (user's): exchanging elements while finding the pivot
// might pay for itself.  Classically it does for SCALAR partitions -- a
// median-of-3 that sorts first/mid/last yields sentinels that let the inner
// loops drop bound checks.  This asks whether that survives the branchless BLOCK
// partition (boost_block, via the position fast path -- identical for find and
// exchange, so the partition cost is held constant and only selection differs).
//
// Strategies: m3m3 = 9 samples (3 medians-of-3, Tukey ninther shape);
//             m5m5 = 25 samples (5 medians-of-5).  Each in find / swap form.
//
// Per (type, n, strategy): select, total(select+partition), partition(=total-
// select) ns/elem, and balance (fraction < pivot).
//
// FINDINGS (Meteor Lake, GCC -O3 -march=native, random_uniform).  Exchanging
// elements while finding the pivot is NOT beneficial for a pure partition:
//   * Large n (>= 2^16): selection is O(1) -> ~0 ns/elem; total == partition
//     cost; find and exchange are equal.  The block partition cannot exploit the
//     reordering -- unlike a scalar Hoare loop it has no per-element bound checks
//     for a sentinel to remove -- so the swaps buy nothing.
//   * Small/mid n: exchange is a slight LOSS, worst for wide elements.  Holding
//     the selection algorithm fixed and changing ONLY the swap target (pointers
//     vs block elements), pair64 m5m5 select costs ~0.36 (find) vs ~0.50 (swap)
//     ns/elem at n=256 -- the reorder adds scattered 16-byte writes the find
//     avoids.  For i64 the 8-byte swap is cheap, so it is ~neutral.
//   * m3m3 (9 samples) <= m5m5 (25 samples) on total at small/mid n (cheaper
//     selection, ~equal balance on random data); equal at large n.
// So the lowest total = cheapest NON-reordering find (m3m3 / ninther), paired
// with the block partition.  (This corrects an earlier mis-measurement whose
// "find" control used iter_swap and thus secretly reordered.)
//
// Usage: bench_pivot_total [maxsize]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"

using namespace partitions;

namespace {

// FIND: insertion-sort the iterator array (pointer swaps; block untouched).
template <class Less, class It>
void isort_find(Less less, It* p, int k) {
    for (int i = 1; i < k; ++i)
        for (int j = i; j > 0 && less(p[j], p[j - 1]); --j) std::swap(p[j - 1], p[j]);
}
// EXCHANGE: insertion-sort by swapping the block elements (physical reorder).
template <class Less, class It>
void isort_swap(Less less, It* p, int k) {
    for (int i = 1; i < k; ++i)
        for (int j = i; j > 0 && less(p[j], p[j - 1]); --j) std::iter_swap(p[j - 1], p[j]);
}

template <bool Exchange, int G, int Grp, class I, class S, class Comp, class Proj>
I select_mNmN(I first, S last, Comp comp, Proj proj) {
    constexpr int NS = G * Grp;
    const auto n = last - first;
    auto less = pivot::make_less(comp, proj);
    const auto step = n - 1;
    I samp[NS];
    for (int k = 0; k < NS; ++k)
        samp[k] = first + (static_cast<decltype(n)>(k) * step) / (NS - 1);
    I med[G];
    for (int g = 0; g < G; ++g) {
        if constexpr (Exchange) isort_swap(less, samp + g * Grp, Grp);
        else isort_find(less, samp + g * Grp, Grp);
        med[g] = samp[g * Grp + Grp / 2];
    }
    if constexpr (Exchange) isort_swap(less, med, G);
    else isort_find(less, med, G);
    return med[G / 2];
}

struct m3m3_find {
    static constexpr const char* name = "m3m3_find";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        if (last - first < 9) return pivot::median_of_3{}(first, last, comp, proj);
        return select_mNmN<false, 3, 3>(first, last, comp, proj);
    }
};
struct m3m3_swap {
    static constexpr const char* name = "m3m3_swap";
    static constexpr bool reorders = true;
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        if (last - first < 9) return pivot::median_of_3{}(first, last, comp, proj);
        return select_mNmN<true, 3, 3>(first, last, comp, proj);
    }
};
struct m5m5_find {
    static constexpr const char* name = "m5m5_find";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        if (last - first < 25) return pivot::median_of_5{}(first, last, comp, proj);
        return select_mNmN<false, 5, 5>(first, last, comp, proj);
    }
};
struct m5m5_swap {
    static constexpr const char* name = "m5m5_swap";
    static constexpr bool reorders = true;
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        if (last - first < 25) return pivot::median_of_5{}(first, last, comp, proj);
        return select_mNmN<true, 5, 5>(first, last, comp, proj);
    }
};

constexpr std::size_t kSizes[] = {64,      256,     1024,     4096,
                                  1u << 16, 1u << 18, 1u << 20, 1u << 22};
constexpr std::size_t kElemsPerIter = 1u << 16;

template <class T, class Pivot>
void run_single(Pivot pv, std::size_t n) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    const std::size_t batch = n < kElemsPerIter ? kElemsPerIter / n : 1;
    const std::size_t total = n * batch;
    std::vector<T> master;
    master.reserve(total);
    for (std::size_t b = 0; b < batch; ++b) {
        auto blk = dist::random_uniform{}.template operator()<T>(n, 0x1234u + b + n);
        master.insert(master.end(), blk.begin(), blk.end());
    }
    std::vector<T> work(total);
    std::uint64_t reps = static_cast<std::uint64_t>((1u << 22) / total);
    reps = std::min<std::uint64_t>(std::max<std::uint64_t>(reps, 5), 200);
    auto setup = [&] { work = master; };

    auto sel = [&] {
        std::ptrdiff_t s = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            s += pv(beg, beg + static_cast<std::ptrdiff_t>(n), comp, proj) - beg;
        }
        bench::do_not_optimize(s);
    };
    auto tot = [&] {
        std::ptrdiff_t s = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            auto end = beg + static_cast<std::ptrdiff_t>(n);
            auto p = pv(beg, end, comp, proj);
            s += partition_by_position(algo::boost_block{}, beg, end, p, comp, proj) - beg;
        }
        bench::do_not_optimize(s);
    };
    auto rs = bench::measure(reps, setup, sel);
    auto rt = bench::measure(reps, setup, tot);
    double sel_npe = rs.min_ns / static_cast<double>(total);
    double tot_npe = rt.min_ns / static_cast<double>(total);

    auto bcopy = master;
    auto pp = pv(bcopy.begin(), bcopy.begin() + static_cast<std::ptrdiff_t>(n), comp, proj);
    auto bal = stat::measure_balance(bcopy.begin(),
                                     bcopy.begin() + static_cast<std::ptrdiff_t>(n),
                                     std::invoke(proj, *pp), comp, proj);
    std::printf("%s,%zu,%s,%.4f,%.4f,%.4f,%.4f\n", type_name<T>(), n, pv.name,
                sel_npe, tot_npe, tot_npe - sel_npe, bal.smaller_fraction());
    std::fflush(stdout);
}

template <class T>
void run_type(std::size_t max_size) {
    for (std::size_t n : kSizes) {
        if (n > max_size) continue;
        run_single<T>(m3m3_find{}, n);
        run_single<T>(m3m3_swap{}, n);
        run_single<T>(m5m5_find{}, n);
        run_single<T>(m5m5_swap{}, n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 22;
    if (argc > 1) max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,n,strategy,select_npe,total_npe,partition_npe,balance\n");
    run_type<i64>(max_size);
    run_type<pair64>(max_size);
    return 0;
}
