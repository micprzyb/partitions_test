#ifndef PARTITIONS_QUICKSORT_HPP
#define PARTITIONS_QUICKSORT_HPP

// A pure-partition, fully size-adaptive quicksort.
//
// It is built ONLY from partitioning -- there is no insertion-sort or
// sorting-network leaf.  Both the **partition algorithm** and the **pivot
// strategy** are chosen by the size of the block being partitioned, and the
// smallest blocks are finished by recursive HALVING (a balanced rank-split,
// itself a branchless partition primitive -- not a leaf sort):
//
//   block size n        pivot                 partition primitive
//   ----------------    ------------------    ----------------------------------
//   n <= 16             (none)                recursive halver (balanced split)
//   16 < n <= 2048      median_of_3 (cheap)   algo::sized  (lomuto small / boost)
//   n  > 2048           ninther (9 samples)   algo::sized  (-> boost_block)
//
// Rationale (all measured; see docs/pure_quicksort.md and the size sweep in
// benchmarks/bench_quicksort.cpp):
//   * the pivot is chosen at every node, so for SMALL nodes the cheapest pivot
//     (median_of_3) wins on per-node cost; for the few BIG nodes the ninther's
//     better balance is worth its sampling (amortised over O(n) partition work);
//   * `algo::sized` already adapts the partitioner by size (branchless Lomuto for
//     small/narrow blocks, the pdqsort branchless block partition for large ones);
//   * below 16 a recursive halver beats continuing to pivot-partition (no pivot
//     selection, deterministic balance, ~20-33% fewer compare-exchanges).
//
// The pivot is swapped to the front and placed at its final rank, so it is
// excluded from both recursive calls -- guaranteeing progress on heavy-duplicate
// input.  Recursion is on the smaller side with a loop on the larger (bounded
// stack).  Verified by tests/test_quicksort.cpp.

#include <functional>
#include <iterator>
#include <utility>

#include "algorithms.hpp"
#include "pivot.hpp"
#include "small_halve.hpp"

namespace partitions {

namespace detail {

// Blocks this size and below are finished by recursive halving (halve_sort).
// 24 is the largest size with a hand-built halver; a careful per-type sweep (5+3
// runs, see docs/quicksort_pivot_tiers.md) shows raising the cutoff from 16 to 24
// is ~1.0-1.6% faster on the 16-byte pair types and neutral on i64 -- the
// branchless balanced split beats ninther+partition on those nodes.  (A smaller
// cutoff, and an m5/m3 small-node pivot tier, were tested and lose.)
inline constexpr std::ptrdiff_t qs_halve_cutoff = 24;
// Above this size use the pricier median-of-5-of-5 (25 samples): on the few huge
// top-of-recursion nodes its cost amortises fully over an O(n) partition while its
// better balance shaves work off every level below.  Measured ~2-3% faster on
// i64/pair64f at n>=2^20, neutral on pair64, with NO penalty on sorted input.
inline constexpr std::ptrdiff_t qs_m5m5_cutoff = 1 << 16;  // 65536

// Sort a small block (<= qs_halve_cutoff) by recursive balanced halving.
template <class It, class Comp, class Proj>
void halve_sort(It first, It last, Comp comp, Proj proj) {
    const auto n = last - first;
    if (n <= 1) return;
    It mid = small_halve::halve(first, last, comp, proj);  // first + n/2, balanced
    halve_sort(first, mid, comp, proj);
    halve_sort(mid, last, comp, proj);
}

}  // namespace detail

template <std::random_access_iterator It, class Comp = std::less<>,
          class Proj = std::identity>
void quicksort(It first, It last, Comp comp = {}, Proj proj = {}) {
    using D = std::iter_difference_t<It>;
    while (true) {
        const D n = last - first;
        if (n <= detail::qs_halve_cutoff) {
            detail::halve_sort(first, last, comp, proj);
            return;
        }
        // ninther (median-of-9) at every node, median-of-5-of-5 on the huge ones.
        // A cheaper median_of_3 is faster on random data but is the textbook
        // median-of-3 killer: on sorted/reverse input the partitioner's
        // deterministic output feeds it near-EXTREME pivots, collapsing to O(n^2)
        // (measured: reverse-sorted work/elem 131073 = n/8 with m3, vs 16 with
        // ninther).  Ninther's 9 spread samples resist that; m5m5's 25 samples buy
        // a little extra balance on the few large nodes where it is free.
        It p = n > detail::qs_m5m5_cutoff
                   ? pivot::median_of_5_medians_of_5{}(first, last, comp, proj)
                   : pivot::ninther{}(first, last, comp, proj);
        std::iter_swap(first, p);                      // pivot to front
        auto key = std::invoke(proj, *first);
        // size-adaptive partitioner (algo::sized: lomuto small / boost large)
        It m = algo::sized{}(first + 1, last, key, comp, proj);
        std::iter_swap(first, m - 1);                  // place pivot at its rank
        It pp = m - 1;
        if (pp - first < last - (pp + 1)) {            // recurse smaller, loop larger
            quicksort(first, pp, comp, proj);
            first = pp + 1;
        } else {
            quicksort(pp + 1, last, comp, proj);
            last = pp;
        }
    }
}

}  // namespace partitions

#endif  // PARTITIONS_QUICKSORT_HPP
