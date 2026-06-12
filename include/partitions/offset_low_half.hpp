#ifndef PARTITIONS_OFFSET_LOW_HALF_HPP
#define PARTITIONS_OFFSET_LOW_HALF_HPP

// OFFSET low-half: move the bottom HALF of the below-key elements to the
// FRONT, given a known all->= prefix.
//
// SPEC.  Given [first, last), a key and `offset` with the PRECONDITION that
// every element of [first, first + offset) is >= key, let
// S = #{x : comp(proj(x), key)} (all in the suffix, by the precondition).
// Rearrange so the FIRST k positions hold the k smallest elements of the
// whole range (any order), with k ~= S/2, and return k.  Those k smallest
// are exactly the lower half of the below-key elements.  Base case: if the
// target half S/2 < 16, take ALL S below-key elements (k = S) -- selecting a
// half of a tiny set is not worth a pass (same rule as move_low_half).
//
//   * offset_low_half        -- FAST, k ~= S/2 (sampled estimate, within a
//                               few % at healthy sizes).
//   * offset_low_half_exact  -- k == floor(S/2) exactly (or S, base case).
//
// DESIGN (the PIVOT-FIRST strategy; full study in benchmarks/
// bench_offset_low_half.cpp and docs/offset_low_half.md):
//
//   The naive two-phase plan -- partition the suffix around an estimated
//   below-median t, then block-swap the left side to the front -- pays the
//   part_swap double move (each delivered element moved twice).  Testing the
//   pivot BEFORE partitioning collapses it to ONE pass: t is an estimate of
//   the (S/2)-th value; if t < key, then offset_partition(first, last,
//   offset, t) -- the prefix_fill machinery -- routes exactly the bottom
//   ~S/2 elements into the front slots during the single partition sweep
//   (each moved once, the prefix never compared).  With a SAMPLED t the test
//   is vacuous by construction (the chosen sample rank lies strictly below
//   the sample's below-key count, so t < key always); the t >= key branch
//   exists for cheap unguarded pivots (e.g. a suffix ninther -- see the
//   bench's ninther_first variant, which trades k-accuracy for nothing
//   measurable and is therefore NOT shipped).
//
//   SUFFIX-ONLY SAMPLING (refinement over move_low_half's sampler): the
//   prefix is all >= key by contract, so sampling it would only dilute the
//   below-count signal -- m samples over the whole array estimate S with
//   m*(suffix/n) useful points.  Sampling [first+offset, last) directly
//   keeps the estimator's variance independent of f = offset/n.
//
//   The exact variant is offset_partition by `key` (all S belows to the
//   front -- the take-all base case is then already done) followed by ONE
//   in-place quickselect of the front block around rank S/2; the repo's
//   branchless quickselect (detail::quickselect, move_low_half.hpp) is used
//   for both this and the sample-rank selection.
//
// Like move_low_half, the routing threshold between the exact and sampled
// paths is the sample budget: below suffix length 1024 the affordable
// sample (suffix/4) is too noisy and exact is already cheap.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "move_low_half.hpp"      // detail::quickselect (branchless select)
#include "offset_partition.hpp"   // algo_off::sized_off (gap_off / prefix_fill)

namespace partitions {

// k == floor(S/2) exactly (or S if S/2 < 16): one offset partition by `key`
// lands all S below-key elements at the front; the take-all base case is
// then already done, otherwise one quickselect splits the front block in
// place around its median.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t offset_low_half_exact(It first, It last,
                                     std::iter_difference_t<It> offset, K key,
                                     Comp comp = {}, Proj proj = {}) {
    It m = algo_off::sized_off{}(first, last, offset, key, comp, proj);
    const std::ptrdiff_t S = m - first;
    const std::ptrdiff_t k = (S / 2 < 16) ? S : S / 2;
    if (k > 0 && k < S)
        detail::quickselect(first, m, first + k, comp, proj);
    return k;
}

// k ~= S/2 (sampled): estimate the (S/2)-th value from a stride sample of
// the SUFFIX (the prefix carries no information about the belows), then ONE
// offset partition around the estimate -- the bottom ~S/2 land at the front
// in a single prefix_fill sweep.  ~(n - offset) work at every key
// percentile, zero post-processing.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t offset_low_half(It first, It last,
                               std::iter_difference_t<It> offset, K key,
                               Comp comp = {}, Proj proj = {}) {
    using Key = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    const std::ptrdiff_t suffix = (last - first) - offset;
    constexpr int kSample = 512;
    // Small suffix: the affordable sample (suffix/4) is too noisy and the
    // exact path is already cheap (and owns the take-all base case).
    if (suffix < 1024)
        return offset_low_half_exact(first, last, offset, key, comp, proj);

    const int m = static_cast<int>(std::min<std::ptrdiff_t>(suffix / 4, kSample));
    const std::ptrdiff_t stride = suffix / m;
    It base = first + offset;
    Key samp[kSample];
    std::ptrdiff_t below = 0;
    for (int i = 0; i < m; ++i) {
        samp[i] = std::invoke(proj, base[i * stride]);
        below += static_cast<bool>(std::invoke(comp, samp[i], key));
    }
    if (below < 16)  // S is small relative to the suffix -> be exact
        return offset_low_half_exact(first, last, offset, key, comp, proj);

    const int r = static_cast<int>(below / 2);  // sample rank ~ the (S/2)-th
    detail::quickselect(samp, samp + m, samp + r, comp, std::identity{});
    const Key t = samp[r];  // estimate of the (S/2)-th value; t < key is
                            // GUARANTEED (r < below = #samples below key)
    // Route the t-partition to prefix_fill DIRECTLY, not through sized_off:
    // the dispatcher cannot know the below fraction, so for narrow types
    // with offset >= suffix it picks the p-flat gap_off (~0.5 ns/elem) --
    // but HERE the effective fraction vs t is ~p/2 by construction (t is
    // the below-median), exactly the regime where prefix_fill's
    // per-below-only moves win.  Measured: i64 f=.5 p=.1 2^20 drops 0.67 ->
    // ~0.40 ns/suffix-elem (the benchmark's two_phase variant exposed this).
    It mid = algo_off::prefix_fill_off{}(first, last, offset, t, comp, proj);
    if (mid != first) return mid - first;
    // mid == first: nothing is strictly below t, i.e. the below-key
    // population is dominated by a single value (t is its minimum).  No
    // VALUE split can separate equal elements -- only rank selection can --
    // so fall back to the exact path.  (Duplicate-heavy data; the wasted
    // partition pass is the price of detecting it.)
    return offset_low_half_exact(first, last, offset, key, comp, proj);
}

}  // namespace partitions

#endif  // PARTITIONS_OFFSET_LOW_HALF_HPP
