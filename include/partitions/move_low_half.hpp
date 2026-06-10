#ifndef PARTITIONS_MOVE_LOW_HALF_HPP
#define PARTITIONS_MOVE_LOW_HALF_HPP

// move_low_half: move the k SMALLEST elements of [first, last) to the END, where
// k is approximately HALF of the elements below `key` (S = #{x : comp(proj(x),
// key)}).  Those k smallest are exactly the lower half of the below-key elements
// (all < key).  Order within the two parts is irrelevant; the call returns k.
//
//   * `move_low_half`        -- FAST, k ~= S/2 (approximate, within a few %).
//   * `move_low_half_exact`  -- k == floor(S/2) exactly.
//
// Base case (both): if the target half S/2 would be < 16, take ALL S below-key
// elements (k = S) -- not worth selecting a half of a tiny set.
//
// The two strategies were benchmarked against each other and against several
// others (count+select, partition-by-key+select, descend-until, a single
// median-partition) in benchmarks/bench_move_low_half.cpp; see that file for the
// full study.  The key results (Meteor Lake, GCC, -O3 -march=native, random
// high-cardinality keys, n=2^22, ns/elem; p = S/n = key percentile):
//
//                       p=0.10   p=0.50   p=0.90
//     move_low_half      0.46     0.60     0.75    (k/S ~ 0.48-0.52)
//     move_low_half_exact 0.62    1.34     1.76    (k/S = 0.50)
//
//   move_low_half is ~n (a cheap stride sample + ONE partition) for EVERY key
//   percentile and stays accurate, because it samples the ORIGINAL (unsorted ->
//   unbiased) array to estimate the (S/2)-th value, then partitions once around
//   it.  move_low_half_exact partitions by `key` (-> S) then quickselects the
//   below-key half; it is exact and fastest when the key is LOW (small select),
//   but its select cost grows with S.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "algorithms.hpp"     // algo::sized (the branchless partitioner)
#include "quicksort_lr.hpp"   // detail::ninther_pos (inline branchless pivot)
#include "small_sort.hpp"     // small_sort::sort (quickselect leaf)

namespace partitions {

namespace detail {

// Quickselect: reorder [first, last) so [first, nth) <= *nth <= [nth, last).
template <class It, class Comp, class Proj>
void quickselect(It first, It last, It nth, Comp comp, Proj proj) {
    while (last - first > 24) {
        It p = ninther_pos(first, last, comp, proj);
        std::iter_swap(first, p);
        auto key = std::invoke(proj, *first);
        It m = algo::sized{}(first + 1, last, key, comp, proj);
        std::iter_swap(first, m - 1);
        It pp = m - 1;
        if (nth == pp) return;
        if (nth < pp) last = pp; else first = pp + 1;
    }
    small_sort::sort(first, last, comp, proj);
}

// Move the front-k (a valid bottom-k) to the last k positions; order-agnostic,
// O(min(k, n-k)), no overlap even for k > n/2.
template <class It>
void move_front_to_end(It first, It last, std::ptrdiff_t k) {
    const std::ptrdiff_t n = last - first;
    if (k <= 0 || k >= n) return;
    if (k <= n - k) std::swap_ranges(first, first + k, last - k);
    else std::swap_ranges(first, first + (n - k), first + k);
}

}  // namespace detail

// k == floor(S/2) exactly (or S if that is < 16): partition by `key` to isolate
// the S below-key elements, quickselect their lower half, move it to the end.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t move_low_half_exact(It first, It last, K key, Comp comp = {},
                                   Proj proj = {}) {
    It m = algo::sized{}(first, last, key, comp, proj);  // [first, m) < key
    const std::ptrdiff_t S = m - first;
    const std::ptrdiff_t k = (S / 2 < 16) ? S : S / 2;
    if (k > 0 && k < S) detail::quickselect(first, m, first + k, comp, proj);
    detail::move_front_to_end(first, last, k);
    return k;
}

// k ~= S/2 (approximate): estimate the (S/2)-th value from a stride sample of the
// ORIGINAL array (unbiased -- the array is not yet partitioned), then ONE
// partition around it.  ~n work for any key percentile.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t move_low_half(It first, It last, K key, Comp comp = {},
                             Proj proj = {}) {
    using Key = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    const std::ptrdiff_t n = last - first;
    constexpr int kSample = 512;
    // Small / few-below: the sample would be unreliable and the exact path is
    // already cheap there (and handles the take-all base case).
    if (n <= 4 * kSample) return move_low_half_exact(first, last, key, comp, proj);

    const int m = kSample;
    const std::ptrdiff_t stride = n / m;
    Key samp[kSample];
    std::ptrdiff_t below = 0;
    for (int i = 0; i < m; ++i) {
        samp[i] = std::invoke(proj, first[i * stride]);
        below += static_cast<bool>(std::invoke(comp, samp[i], key));
    }
    if (below < 16)  // S is small relative to n -> be exact (cheap, correct base case)
        return move_low_half_exact(first, last, key, comp, proj);

    const int r = static_cast<int>(below / 2);  // sample rank ~ the (S/2)-th overall
    std::nth_element(samp, samp + r, samp + m, [&](const Key& x, const Key& y) {
        return static_cast<bool>(std::invoke(comp, x, y));
    });
    const Key t = samp[r];  // estimate of the (S/2)-th value (< key w.h.p.)
    It mid = algo::sized{}(first, last, t, comp, proj);  // [first, mid) < t
    const std::ptrdiff_t k = mid - first;
    detail::move_front_to_end(first, last, k);
    return k;
}

}  // namespace partitions

#endif  // PARTITIONS_MOVE_LOW_HALF_HPP
