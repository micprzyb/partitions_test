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
// DESIGN: both functions are built on the REVERSED partition family
// (`algo_rev::sized_rev`, reverse_partition.hpp), which routes the strictly-
// below elements to the RIGHT ([m, last) < pivot) at the SAME cost as the
// forward partition (role-exchanged scans, identical op count -- see that
// header).  The wanted bottom set therefore lands AT THE END as a side effect
// of the one partition pass, and the `move_front_to_end` epilogue the forward
// formulation needed (an extra min(k, n-k) swaps -- up to ~45% of n in 16-byte
// traffic at high key percentiles) disappears entirely.  Likewise the exact
// path's quickselect runs REVERSED (`detail::quickselect_rev`): every partition
// of the select keeps the small side glued to the array end, so the answer is
// in place when the recursion stops.  Measured against the forward+move
// formulation this is up to ~1.8x at high percentiles and never slower (see
// benchmarks/bench_move_low_half.cpp and docs/move_low_half.md).
//
// The two strategies were benchmarked against each other and against several
// others (count+select, descend-until, a single median-partition) in
// benchmarks/bench_move_low_half.cpp; see that file for the full study.  Key
// results (Meteor Lake, GCC, -O3 -march=native, random high-cardinality keys,
// n=2^22, ns/elem; p = S/n = key percentile):
//
//                       p=0.10   p=0.50   p=0.90
//     move_low_half      0.43     0.44     0.48    (k/S ~ 0.48-0.52)
//     move_low_half_exact 0.53    0.90     1.34    (k/S = 0.50)
//     (the superseded forward+move versions: 0.42/0.60/0.73 and 0.58/1.25/1.65)
//
//   move_low_half is ~n (a cheap stride sample + ONE reversed partition) for
//   EVERY key percentile and stays accurate, because it samples the ORIGINAL
//   (unsorted -> unbiased) array to estimate the (S/2)-th value, then
//   partitions once around it.  move_low_half_exact partitions by `key`
//   (-> S) then reverse-quickselects the below-key half in place; it is exact
//   and fastest when the key is LOW (small select), but its select cost grows
//   with S.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "algorithms.hpp"         // algo::sized (forward partitioner; sample-rank select)
#include "quicksort_lr.hpp"       // detail::ninther_pos (inline branchless pivot)
#include "reverse_partition.hpp"  // algo_rev::sized_rev (>= left | < right)
#include "small_sort.hpp"         // small_sort::sort (quickselect leaf)

namespace partitions {

namespace detail {

// Quickselect: reorder [first, last) so [first, nth) <= *nth <= [nth, last).
// (Forward; used on the small flat sample buffer, where nothing needs to land
// at an array end.)
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

// REVERSED quickselect: reorder [first, last) so that every element of
// [first, nth) is >= every element of [nth, last) -- i.e. the (last - nth)
// smallest elements end up in the TAIL, which is exactly this file's contract.
// Built on `sized_rev` ([lo, m) >= pivot | [m, hi) < pivot), so each partition
// pushes the small side toward the end and no final move is ever needed.
//
// Pivot bookkeeping mirror-imaged: the pivot element (== its key) belongs on
// the LEFT (>=) side here, so after partitioning the tail [first+1, last) it is
// parked at m-1 -- the last slot of the >= run -- giving
//     [first, pp) >= key | *pp == key | [pp+1, last) < key,   pp = m-1.
// Both pp and pp+1 are then valid split points (left min == key >= right max),
// so the loop terminates on either; recursing left EXCLUDES the parked pivot,
// which guarantees progress on duplicate-heavy input.
//
// Leaf: a descending small_sort network via the ARGUMENT-SWAPPED comparator
// (a strict weak order, unlike the negation !comp which is reflexive and would
// break the network -- see small_halve_rev.hpp).  Descending order puts the
// leaf's smallest at its end, adjacent to the already-finalised smaller tail.
template <class It, class Comp, class Proj>
void quickselect_rev(It first, It last, It nth, Comp comp, Proj proj) {
    while (last - first > 24) {
        It p = ninther_pos(first, last, comp, proj);
        std::iter_swap(first, p);
        auto key = std::invoke(proj, *first);
        It m = algo_rev::sized_rev{}(first + 1, last, key, comp, proj);
        std::iter_swap(first, m - 1);
        It pp = m - 1;
        if (nth == pp || nth == pp + 1) return;
        if (nth < pp) last = pp; else first = pp + 1;
    }
    small_sort::sort(
        first, last,
        [&comp](const auto& a, const auto& b) {
            return static_cast<bool>(std::invoke(comp, b, a));
        },
        proj);
}

}  // namespace detail

// k == floor(S/2) exactly (or S if that is < 16): ONE reversed partition by
// `key` lands all S below-key elements in the tail [m, last); the take-all base
// case is then already done, and otherwise a reversed quickselect splits the
// tail in place around its median.  No element is ever moved twice.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t move_low_half_exact(It first, It last, K key, Comp comp = {},
                                   Proj proj = {}) {
    It m = algo_rev::sized_rev{}(first, last, key, comp, proj);  // [m,last) < key
    const std::ptrdiff_t S = last - m;
    const std::ptrdiff_t k = (S / 2 < 16) ? S : S / 2;
    if (k > 0 && k < S) detail::quickselect_rev(m, last, last - k, comp, proj);
    return k;
}

// k ~= S/2 (approximate): estimate the (S/2)-th value from a stride sample of
// the ORIGINAL array (unbiased -- the array is not yet partitioned), then ONE
// reversed partition around it puts exactly the elements below the estimate at
// the end.  ~n work for any key percentile, zero post-processing.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t move_low_half(It first, It last, K key, Comp comp = {},
                             Proj proj = {}) {
    using Key = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    const std::ptrdiff_t n = last - first;
    constexpr int kSample = 512;
    // Small n: the best affordable sample (m = n/4) is too noisy -- measured
    // per-call k/S 5th-95th percentile spread is ~+-25% at n=256 / +-15% at
    // n=512 (see docs/move_low_half.md) -- and the exact path is already cheap
    // there (and handles the take-all base case).  From n=1024 (m >= 256) the
    // spread at the healthy percentiles matches the large-n path (~+-10% or
    // tighter), so sampling starts paying from there: 10-35% faster than exact
    // at n=1024-2048, mid/high percentiles.
    if (n < 1024) return move_low_half_exact(first, last, key, comp, proj);

    const int m = static_cast<int>(std::min<std::ptrdiff_t>(n / 4, kSample));
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
    // Rank-r of the sample via the repo's own branchless quickselect --
    // std::nth_element's branchy introselect is 4-8x slower at scale and
    // measurably slower even on this 512-key buffer.
    detail::quickselect(samp, samp + m, samp + r, comp, std::identity{});
    const Key t = samp[r];  // estimate of the (S/2)-th value (< key, guaranteed:
                            // r < below = #samples below key)
    It mid = algo_rev::sized_rev{}(first, last, t, comp, proj);  // [mid,last) < t
    return last - mid;
}

}  // namespace partitions

#endif  // PARTITIONS_MOVE_LOW_HALF_HPP
