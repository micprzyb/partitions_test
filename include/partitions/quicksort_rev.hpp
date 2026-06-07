#ifndef PARTITIONS_QUICKSORT_REV_HPP
#define PARTITIONS_QUICKSORT_REV_HPP

// Reversed (DESCENDING) pure-partition quicksort -- the mirror of quicksort.hpp.
//
// Identical size-adaptive structure, but every partition step puts elements
// `>= pivot` on the LEFT and strictly `< pivot` on the RIGHT, and the small-block
// leaf is finished by DESCENDING recursive halving.  The net result is a
// descending sort.  The pivot strategies are unchanged (a median is
// direction-agnostic); only the partition primitive and the leaf halver flip:
//
//   block size n        pivot                 partition primitive
//   ----------------    ------------------    ----------------------------------
//   n <= 24             (none)                reversed halver (descending split)
//   24 < n <= 65536     ninther (9 samples)   algo_rev::sized_rev
//   n  > 65536          m5-of-5 (25 samples)  algo_rev::sized_rev (-> boost_block_rev)
//
// Pivot to front, reverse-partition the rest into [>=key | <key], place the
// pivot at the boundary (m-1) and exclude it from both recursive calls --
// mirroring quicksort.hpp's progress guarantee and smaller-side recursion.
// Verified by tests/test_quicksort_rev.cpp.

#include <functional>
#include <iterator>
#include <utility>

#include "pivot.hpp"
#include "quicksort.hpp"          // detail::qs_halve_cutoff, qs_m5m5_cutoff
#include "reverse_partition.hpp"  // algo_rev::sized_rev
#include "small_halve_rev.hpp"    // small_halve_rev::halve_rev

namespace partitions {

namespace detail {

// Sort a small block (<= qs_halve_cutoff) DESCENDING by recursive balanced
// halving -- the reversed mirror of detail::halve_sort.
template <class It, class Comp, class Proj>
void halve_sort_rev(It first, It last, Comp comp, Proj proj) {
    const auto n = last - first;
    if (n <= 1) return;
    It mid = small_halve_rev::halve_rev(first, last, comp, proj);  // largest half low
    halve_sort_rev(first, mid, comp, proj);
    halve_sort_rev(mid, last, comp, proj);
}

}  // namespace detail

// Sorts [first, last) into DESCENDING order by (comp, proj):
//   comp(proj(v[i+1]), proj(v[i])) holds for adjacent elements (each <= prev).
template <std::random_access_iterator It, class Comp = std::less<>,
          class Proj = std::identity>
void quicksort_rev(It first, It last, Comp comp = {}, Proj proj = {}) {
    using D = std::iter_difference_t<It>;
    while (true) {
        const D n = last - first;
        if (n <= detail::qs_halve_cutoff) {
            detail::halve_sort_rev(first, last, comp, proj);
            return;
        }
        It p = n > detail::qs_m5m5_cutoff
                   ? pivot::median_of_5_medians_of_5{}(first, last, comp, proj)
                   : pivot::ninther{}(first, last, comp, proj);
        std::iter_swap(first, p);                      // pivot to front
        auto key = std::invoke(proj, *first);
        // reversed size-adaptive partitioner: [first+1, m) >= key, [m, last) < key
        It m = algo_rev::sized_rev{}(first + 1, last, key, comp, proj);
        std::iter_swap(first, m - 1);                  // place pivot at its rank
        It pp = m - 1;                                 // [first,pp) >= key, [pp+1,last) < key
        if (pp - first < last - (pp + 1)) {            // recurse smaller, loop larger
            quicksort_rev(first, pp, comp, proj);
            first = pp + 1;
        } else {
            quicksort_rev(pp + 1, last, comp, proj);
            last = pp;
        }
    }
}

}  // namespace partitions

#endif  // PARTITIONS_QUICKSORT_REV_HPP
