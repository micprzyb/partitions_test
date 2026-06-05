#ifndef PARTITIONS_SMALL_MERGE_HPP
#define PARTITIONS_SMALL_MERGE_HPP

// Specialised small-array completion routines for the case where most of the
// block is already sorted (n - k small).  Two shapes:
//
//   (a) extend_sorted  : [0,k) sorted, [k,n) UNSORTED -> sort the whole block by
//                        inserting each tail element into the sorted prefix.
//   (b) merge_sorted   : [0,k) sorted AND [k,n) sorted -> merge in place.
//
// Both target n - k = 1..4 (the partial-sort / introsort-tail regime) but are
// correct for any 0 <= k <= n <= 24.  They are compared in bench_small_merge
// against re-sorting from scratch with a sorting network and against library
// mergers (std::inplace_merge, std::merge).  Raw throughput is the only goal.
//
// Design notes (see docs/small_sort_findings.md):
//   * Insertion/merge are *branchy* (data-dependent trip counts) but do work
//     proportional to how unsorted the input is -- O(n-k) comparisons/moves when
//     the tail lands near the top, versus a network's fixed ~n*log^2(n) CEs.
//     For n-k small that is a large win; the branch mispredicts (one per
//     inserted element) are cheap by comparison.
//   * merge_sorted copies the SMALL run (the tail) to a stack buffer and merges
//     right-to-left with an early exit once the tail is exhausted -- the same
//     trick timsort uses ("copy the smaller run").  Moves = (prefix elements
//     above the smallest tail key) + (n-k); best case n-k, worst case n.
//   * A fully branchless fixed-trip merge (merge_sorted_branchless) is also
//     provided: exactly n compare-selects, no mispredicts, but no early exit.
//     It wins when the tail is large; the branchy form wins when n-k is tiny.

#include <cstddef>
#include <cstring>
#include <functional>
#include <iterator>
#include <type_traits>

namespace partitions::small_merge {

// ---------------------------------------------------------------------------
// (a) [0,k) sorted, [k,n) unsorted -> fully sorted.  Shift-based insertion
//     started at i = k (the prefix is already in order).
// ---------------------------------------------------------------------------
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void extend_sorted(It first, std::size_t k, std::size_t n, Cmp comp = {},
                          Proj proj = {}) {
    using T = typename std::iterator_traits<It>::value_type;
    if (k < 1) k = 1;  // a single element is trivially sorted
    for (std::size_t i = k; i < n; ++i) {
        T key = first[i];
        const auto kp = proj(key);
        std::size_t j = i;
        while (j > 0 && comp(kp, proj(first[j - 1]))) {
            first[j] = first[j - 1];
            --j;
        }
        first[j] = key;
    }
}

// Investigated, but slower -- documented negative result.  Binary-search the
// insertion point (<= log2(i) compares vs ~i/2 for the linear scan) then memmove
// the gap.  It LOSES to plain `extend_sorted`: the linear scan's shift loop is a
// *monotone, well-predicted* branch (one mispredict at the exit, no matter how
// far it scans), whereas binary search makes ~log2(i) *unpredictable* branches
// (~50% mispredict each).  For n<=24 those mispredicts cost far more than the
// extra cheap moves the linear scan does.  See docs/small_sort_findings.md.
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void extend_sorted_bsearch(It first, std::size_t k, std::size_t n,
                                  Cmp comp = {}, Proj proj = {}) {
    using T = typename std::iterator_traits<It>::value_type;
    static_assert(std::is_trivially_copyable_v<T>);
    if (k < 1) k = 1;
    for (std::size_t i = k; i < n; ++i) {
        T key = first[i];
        const auto kp = proj(key);
        // upper_bound: first p in [0,i) with key < first[p] (stable: ties land
        // after the existing equal run).
        std::size_t lo = 0, hi = i;
        while (lo < hi) {
            const std::size_t mid = (lo + hi) >> 1;
            if (comp(kp, proj(first[mid])))
                hi = mid;
            else
                lo = mid + 1;
        }
        // shift [lo, i) up by one, then drop key at lo.
        std::memmove(&first[lo + 1], &first[lo], (i - lo) * sizeof(T));
        first[lo] = key;
    }
}

// Investigated, but slower -- kept as a documented negative result from the
// assembler study.  Idea: keep the well-predicted LINEAR scan but don't move
// during it; find the landing position with bare compares, then shift the gap
// with a SINGLE memmove instead of i/2 individual element moves.  It LOSES to
// the plain interleaved `extend_sorted` because (1) the shift width is a
// *runtime* value, so the compiler emits a generic memmove (size dispatch +
// `rep movs` startup) rather than a fixed-size SIMD copy, and (2) in the
// interleaved form the element moves already execute underneath the compares
// via out-of-order ILP, so there is no serial move cost to eliminate.  See
// docs/small_sort_findings.md.
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void extend_sorted_scan(It first, std::size_t k, std::size_t n,
                               Cmp comp = {}, Proj proj = {}) {
    using T = typename std::iterator_traits<It>::value_type;
    static_assert(std::is_trivially_copyable_v<T>);
    if (k < 1) k = 1;
    for (std::size_t i = k; i < n; ++i) {
        T key = first[i];
        const auto kp = proj(key);
        std::size_t j = i;
        while (j > 0 && comp(kp, proj(first[j - 1]))) --j;  // scan only, no moves
        std::memmove(&first[j + 1], &first[j], (i - j) * sizeof(T));
        first[j] = key;
    }
}

// ---------------------------------------------------------------------------
// (b) [0,k) and [k,n) both sorted -> merged in place.  Optimised for a small
//     tail: copy the tail out, merge right-to-left, exit when the tail drains.
//     Stable (equal keys keep prefix-before-tail order).
// ---------------------------------------------------------------------------
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void merge_sorted(It first, std::size_t k, std::size_t n, Cmp comp = {},
                         Proj proj = {}) {
    using T = typename std::iterator_traits<It>::value_type;
    const std::size_t m = n - k;
    if (m == 0 || k == 0) return;
    T tmp[24];
    for (std::size_t x = 0; x < m; ++x) tmp[x] = first[k + x];
    std::size_t i = k, t = m, dst = n;
    while (t > 0) {
        // Place the larger of first[i-1] / tmp[t-1] at dst-1.  On a tie take
        // tmp (the later run) so the merge stays stable.
        if (i > 0 && comp(proj(tmp[t - 1]), proj(first[i - 1]))) {
            first[--dst] = first[--i];
        } else {
            first[--dst] = tmp[--t];
        }
    }
    // [0,i) of the prefix is already in place.
}

// Fully branchless fixed-trip merge: exactly n compare-selects, no early exit.
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void merge_sorted_branchless(It first, std::size_t k, std::size_t n,
                                    Cmp comp = {}, Proj proj = {}) {
    using T = typename std::iterator_traits<It>::value_type;
    const std::size_t m = n - k;
    if (m == 0 || k == 0) return;
    T tmp[24];
    for (std::size_t x = 0; x < m; ++x) tmp[x] = first[k + x];
    std::ptrdiff_t i = static_cast<std::ptrdiff_t>(k) - 1;
    std::ptrdiff_t t = static_cast<std::ptrdiff_t>(m) - 1;
    std::ptrdiff_t dst = static_cast<std::ptrdiff_t>(n) - 1;
    for (std::size_t s = 0; s < n; ++s) {
        const bool pre_done = (i < 0);
        const bool tmp_done = (t < 0);
        // clamp indices so the (unused) loads never go out of bounds
        const T vp = first[i < 0 ? 0 : i];
        const T vt = tmp[t < 0 ? 0 : t];
        // take the prefix element if the tail is drained, or it is the larger
        // (>=) of the two; take tmp when the prefix is drained.
        const bool take_prefix =
            tmp_done | (!pre_done & !comp(proj(vp), proj(vt)));
        first[dst] = take_prefix ? vp : vt;
        i -= take_prefix;
        t -= !take_prefix;
        --dst;
    }
}

}  // namespace partitions::small_merge

#endif  // PARTITIONS_SMALL_MERGE_HPP
