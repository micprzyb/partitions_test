#ifndef PARTITIONS_QUICKSORT_LR_HPP
#define PARTITIONS_QUICKSORT_LR_HPP

// A left-to-right, NON-RECURSIVE quicksort whose small-block leaf is a
// SELECTION sort.  Three hard constraints shape it (everything else is tuned
// for raw throughput on the three target key types -- i64, pair64 lexicographic
// and pair64-by-first):
//
//   1. SELECTION-SORT LEAF.  Blocks of size <= `threshold` are finished by
//      selection sort -- repeatedly find the minimum of the unsorted suffix and
//      swap it to the front -- and by NO other method (no insertion sort, no
//      sorting network, no halver).  The only freedom left is *how fast we find
//      the minimum*; see `detail::find_min` and the variant study in
//      benchmarks/bench_quicksort_lr.cpp.
//
//   2. LEFT-TO-RIGHT.  At every partition step the LEFT side is sorted to
//      completion before the right side is touched, so the array is finalised
//      strictly front-to-back.  We achieve this by always *continuing* with the
//      left sub-block and *deferring* the right onto an explicit stack; the
//      stack therefore holds, bottom-to-top, the pending right siblings along
//      the current left spine, and popping yields them in increasing position
//      order.
//
//   3. NO RECURSION.  The recursion stack is materialised by hand as a small
//      array storing ONLY the left boundary of each deferred right (its hi is
//      derivable -- see the driver).  Because constraint 2 forbids the usual
//      "recurse into the smaller side" trick (that would sort the smaller --
//      possibly right -- side first), the stack depth is the height of the left
//      spine.  An all-inline branchless `ninther` pivot keeps that logarithmic
//      on every distribution in the test/bench matrix; `qslr_stack_cap` is sized
//      with a large margin over the logarithm and asserted (measured worst depth
//      26 at n=2^22).  Like the sibling `partitions::quicksort`, this sort is tuned
//      for -- and only makes progress guarantees on -- non-adversarial key
//      cardinality (no 3-way equal partition; heavy-duplicate input degrades
//      toward O(n^2), exactly as the recursive version does).
//
// The pivot is swapped to the front and placed at its final rank so it is
// excluded from both children (progress on duplicates), and the partitioner is
// the size-adaptive `algo::sized` (measured best here -- see the `part` study).
// Versus `partitions::quicksort` this differs in the leaf (selection vs halver),
// the driver (hand-rolled left-first stack vs recursion) AND the pivot
// (all-inline branchless ninther vs the library `pivot::ninther` whose median-3
// GCC outlines).  Verified by tests/test_quicksort_lr.cpp.

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "algorithms.hpp"  // algo::sized (the partitioner; pivots are inline here)

namespace partitions {

namespace detail {

// ---------------------------------------------------------------------------
// Branchless conditional assignment: `dst = c ? src : dst`, guaranteed to
// lower without a data-dependent branch for every target key type.
//
// For a register-width integral key this is a single CMOV.  But for a 16-byte
// aggregate (pair64) `c ? src : dst` is lowered by GCC as a BRANCH + 16-byte
// copy -- there is no 128-bit CMOV on x86-64 -- which then mispredicts on the
// random keys a min-scan sees.  We reinterpret the 16 bytes as two integers and
// blend each half with an arithmetic mask (0 / all-ones), forcing two GP-
// register CMOV-free XOR blends.  This mirrors small_sort::half_swap; it is the
// same reason the small-sort compare-exchange decomposes the pair swap.
// `std::bit_cast` keeps the value in registers (no class-memaccess, no spill).
// ---------------------------------------------------------------------------
struct two_words {
    std::uint64_t a, b;
};

template <class K>
[[gnu::always_inline]] inline void cond_assign(K& dst, const K& src, bool c) {
    if constexpr (std::is_trivially_copyable_v<K> && sizeof(K) == 16) {
        auto d = std::bit_cast<two_words>(dst);
        const auto s = std::bit_cast<two_words>(src);
        const std::uint64_t mask = 0ull - static_cast<std::uint64_t>(c);
        d.a ^= (d.a ^ s.a) & mask;
        d.b ^= (d.b ^ s.b) & mask;
        dst = std::bit_cast<K>(d);
    } else if constexpr (std::is_trivially_copyable_v<K> && sizeof(K) == 8 &&
                         !std::is_integral_v<K>) {
        // 8-byte non-integral key (e.g. a double first-key): force a GP blend so
        // the value is not kept in an XMM register where the conditional move
        // becomes a branch.
        auto d = std::bit_cast<std::uint64_t>(dst);
        const auto s = std::bit_cast<std::uint64_t>(src);
        const std::uint64_t mask = 0ull - static_cast<std::uint64_t>(c);
        d ^= (d ^ s) & mask;
        dst = std::bit_cast<K>(d);
    } else {
        dst = c ? src : dst;  // CMOV for register-width integrals
    }
}

// ---------------------------------------------------------------------------
// Find the minimum of [first, last) under (comp, proj); returns an iterator to
// it.  Requires last - first >= 1.  This is the "improved find-minimum" the
// whole leaf rests on.  The best implementation is KEY-WIDTH dependent (every
// claim below is measured in bench_quicksort_lr.cpp's `min` study and confirmed
// by disassembly -- see docs/quicksort_lr.md):
//
//   * Both forms keep the running minimum KEY in a register, so *m is never
//     reloaded.  Crucially we write the update in the BRANCHY form
//     `if (comp(k,best)) { best = k; idx = j; }`: for a register-width key GCC
//     already lowers it to `cmp + cmovg + cmovg` (ONE compare feeding both the
//     value and index conditional-moves), whereas the "obviously branchless"
//     `best = c?k:best; idx = c?j:idx;` clobbers `best` before the index
//     predicate is taken and so compiles to a wasted SECOND compare.
//
//   * NARROW key (sizeof(K) <= 8: i64, pair64-by-first, ...): TWO independent
//     accumulators over the even/odd interleave.  A single-accumulator scan is
//     latency-bound on the loop-carried `cmp -> cmov(best) -> cmp` chain (~2
//     cycles/element); two lanes put two such chains in flight and win from
//     L ~ 16 up (e.g. i64 L=64: 13.0 vs 18.6 ns/elem).  For these keys the
//     index is tracked branchlessly and the merged key-blend is a cheap CMOV.
//
//   * WIDE key (16-byte pair64 lexicographic): a SINGLE accumulator.  The lex
//     `operator<` already short-circuits to a (well-predicted on diverse keys)
//     branch + CMOV, and the rare 16-byte `best = k` copy only runs on an
//     actual minimum update.  Forcing the two-lane / branchless-blend form
//     instead pays a 16-byte conditional blend on EVERY element (updates are
//     rare) and is 2-3x SLOWER (L=16: 23.6 vs 8.2 ns/elem) -- the exact mirror
//     of why algo::boost_block keeps the short-circuiting partition compare.
//
// (An OUT-OF-SPEC double-ended min+max selection -- one scan placing both ends
// -- is ~1.5-1.9x faster still, bounding what constraint 1 costs; see the
// `minmax_oos` column.  It finds the maximum too, so it is not "find the
// minimum and swap to the first", and is excluded.)
// ---------------------------------------------------------------------------

// NARROW key: branchless two-accumulator scan (index as a CMOV-able integer
// offset, not an iterator wrapper GCC may branch on).
template <class It, class Comp, class Proj>
[[gnu::always_inline]] inline It min_scan_narrow(It first, It last, Comp comp,
                                                 Proj proj) {
    using D = std::iter_difference_t<It>;
    const D n = last - first;
    auto b0 = std::invoke(proj, first[0]);
    auto b1 = b0;
    D i0 = 0, i1 = 0;
    D j = 1;
    for (; j + 1 < n; j += 2) {
        auto k0 = std::invoke(proj, first[j]);
        auto k1 = std::invoke(proj, first[j + 1]);
        const bool l0 = static_cast<bool>(std::invoke(comp, k0, b0));
        const bool l1 = static_cast<bool>(std::invoke(comp, k1, b1));
        cond_assign(b0, k0, l0);
        i0 = l0 ? j : i0;
        cond_assign(b1, k1, l1);
        i1 = l1 ? j + 1 : i1;
    }
    if (j < n) {  // odd tail
        auto k0 = std::invoke(proj, first[j]);
        const bool l0 = static_cast<bool>(std::invoke(comp, k0, b0));
        cond_assign(b0, k0, l0);
        i0 = l0 ? j : i0;
    }
    const bool pick1 = static_cast<bool>(std::invoke(comp, b1, b0));
    return first + (pick1 ? i1 : i0);
}

// WIDE key: single-accumulator, register-resident, branchy update.
template <class It, class Comp, class Proj>
[[gnu::always_inline]] inline It min_scan_wide(It first, It last, Comp comp,
                                               Proj proj) {
    It m = first;
    auto best = std::invoke(proj, *first);
    for (It it = first + 1; it != last; ++it) {
        auto k = std::invoke(proj, *it);
        if (static_cast<bool>(std::invoke(comp, k, best))) {
            best = k;
            m = it;
        }
    }
    return m;
}

// Dispatch on the projected-key width (the same sizeof(K) rationale algo::sized
// uses to switch partitioners).
template <class It, class Comp, class Proj>
[[gnu::always_inline]] inline It find_min(It first, It last, Comp comp, Proj proj) {
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    if constexpr (sizeof(K) <= 8)
        return min_scan_narrow(first, last, comp, proj);
    else
        return min_scan_wide(first, last, comp, proj);
}

// Selection sort of [first, last): find the minimum of the unsorted suffix and
// swap it to the front, repeatedly (constraint 1).  The `m != i` guard skips a
// self-swap and is well predicted (for a random suffix of length L the minimum
// sits at the front with probability only 1/L, so the swap almost always runs).
template <class It, class Comp, class Proj>
[[gnu::always_inline]] inline void selection_sort(It first, It last, Comp comp,
                                                  Proj proj) {
    for (It i = first; last - i > 1; ++i) {
        It m = find_min(i, last, comp, proj);
        if (m != i) std::iter_swap(i, m);
    }
}

// ---------------------------------------------------------------------------
// Pivot: an all-INLINE, BRANCHLESS ninther.  A threshold quicksort selects the
// pivot ~n/Threshold times, so the per-node pivot COST -- not its statistical
// quality -- dominates there, and the pivot was the one real per-node overhead
// the old version paid (`pivot::ninther` builds its median-of-3 through a
// `make_less` lambda that GCC OUTLINES into an .isra clone and CALLS 4x/node for
// the 16-byte pair -- confirmed by disassembly).  Two fixes, both measured:
//
//   * INLINE the ninther from a local median-of-3 (no lambda -> no call).
//   * make that median-of-3 BRANCHLESS (cmp + cmov, no ~50% mispredict).  Net of
//     session noise this is +3-4% on the full i64 / pair64-lex sort vs the
//     library ninther; pair64-by-first (the low-cardinality generator case) is
//     ~4% slower, but summed over the three target types branchless still nets
//     ahead, so it is the raw-throughput choice (the `ab` study).
//
// A median-of-3 SIZE TIER (cheap pivot for small nodes, ninther for large) was
// measured (`pivcut`) to be a WASH on random AND unsafe (large cutoffs let m3
// degenerate sorted_descending into an O(n^2) deep spine -> stack overflow), so
// it is disabled by default (qslr_pivot_ninther_cutoff = 0) and kept only as a
// reproducible study.  A median-of-5-of-5 huge-node tier was dropped (it pulled
// in std::sort for <1%).
// ---------------------------------------------------------------------------

// Iterator to the median of the values at a, b, c (strict `comp` on the key),
// computed BRANCHLESSLY as max(min(a,b), min(max(a,b), c)) with the iterator
// carried alongside its key.  The textbook nested-if median-of-3 compiles to
// data-dependent branches that mispredict ~50% on random keys; this form lowers
// to `cmp + cmov` (the iterator picks are pointer CMOVs, the key picks reuse the
// leaf's `cond_assign` so the 16-byte pair blends without a branch too).  ~3
// comparisons either way, but no misprediction -- measurably better for the
// scalar-key types where the median compares are cheap and 50/50.
template <class It, class Comp, class Proj>
[[gnu::always_inline]] inline It median3_pos(It a, It b, It c, Comp comp,
                                             Proj proj) {
    auto ka = std::invoke(proj, *a);
    const auto kb = std::invoke(proj, *b);
    const auto kc = std::invoke(proj, *c);
    const bool ab = static_cast<bool>(std::invoke(comp, ka, kb));  // ka < kb
    It loi = ab ? a : b;          // iterator of min(a,b)
    It hii = ab ? b : a;          // iterator of max(a,b)
    auto klo = kb, khi = ka;      // klo = min key, khi = max key
    cond_assign(klo, ka, ab);     // klo = ab ? ka : kb
    cond_assign(khi, kb, ab);     // khi = ab ? kb : ka
    const bool hc = static_cast<bool>(std::invoke(comp, khi, kc));  // max(a,b) < c
    It m1 = hc ? hii : c;         // iterator of min(max(a,b), c)
    auto km1 = kc;
    cond_assign(km1, khi, hc);    // km1 = hc ? khi : kc
    const bool u = static_cast<bool>(std::invoke(comp, klo, km1));  // min(a,b) < m1
    return u ? m1 : loi;          // max(min(a,b), min(max(a,b), c))
}

// Tukey's ninther (median of three medians-of-3 over nine spread samples), built
// from the inline median3_pos.  Requires n = hi - lo >= 9 (guaranteed: only
// called for n > NintherCutoff).
template <class It, class Comp, class Proj>
[[gnu::always_inline]] inline It ninther_pos(It lo, It hi, Comp comp, Proj proj) {
    const auto n = hi - lo;
    const auto e = n / 8;
    It a = median3_pos(lo, lo + e, lo + 2 * e, comp, proj);
    It b = median3_pos(lo + n / 2 - e, lo + n / 2, lo + n / 2 + e, comp, proj);
    It c = median3_pos(hi - 1 - 2 * e, hi - 1 - e, hi - 1, comp, proj);
    return median3_pos(a, b, c, comp, proj);
}

// Selection-leaf threshold (template-exposed for tuning; default from the
// `thresh` sweep, median of 3): a flat plateau over T16..T24 on random, i64
// peaking at 20 and pair64 tied 16-20, so 20 is within ~1% of every healthy
// type's optimum.  Selection sort is O(L^2) and non-adaptive, so sorted input
// prefers a smaller leaf -- 20 keeps that penalty bounded.
inline constexpr std::ptrdiff_t qslr_threshold = 20;
// Nodes <= this take the cheap median-of-3 pivot; larger nodes take the ninther.
// DEFAULT 0 == ninther for every partitioned node.  The `pivcut` sweep
// (bench_quicksort_lr.cpp) measured the median-of-3 tier to be a WASH on random
// (the inline ninther_pos is already cheap -- it was the OUTLINED median3 in the
// old pivot::ninther, ~4 calls/node for the 16-byte pair, that hurt) while large
// cutoffs are UNSAFE: median-of-3 degenerates on sorted_descending (the m3 /
// partition recursion interaction) to an O(n^2) deep left spine that overflows
// the fixed stack.  So the tier is kept only for that reproducible study and is
// off by default.
inline constexpr std::ptrdiff_t qslr_pivot_ninther_cutoff = 0;

// Explicit-stack capacity: an upper bound on the left-spine height.  The ninther
// keeps the height ~log2(n) -- measured peak 26 at n=2^22 (the `depth` study;
// ~1.1*log2 n) -- so 192 (1.5 KiB now that we store one iterator per entry)
// clears n up to ~2^60 of balanced data by a wide margin.  Asserted at push.
inline constexpr int qslr_stack_cap = 192;

// Driver, templated on BOTH tunables so the benchmark can sweep them.
template <std::ptrdiff_t Threshold, std::ptrdiff_t NintherCutoff,
          std::random_access_iterator It, class Comp, class Proj>
void quicksort_lr_impl(It first, It last, Comp comp, Proj proj) {
    // The stack holds ONLY the left boundary of each deferred right.  Going
    // always-left, the deferred rights tile the suffix adjacently (separated by
    // the in-place pivots), so a right's hi == (lo of the entry below it) - 1,
    // or `last` at the bottom -- derived on pop.  Storing one iterator per entry
    // instead of two halves the stack traffic.
    It stack[qslr_stack_cap];
    int sp = 0;
    It lo = first, hi = last;
    while (true) {
        // Descend the left spine: partition, defer the right, continue left.
        while (hi - lo > Threshold) {
            const auto n = hi - lo;
            It p = n <= NintherCutoff
                       ? median3_pos(lo, lo + (n >> 1), hi - 1, comp, proj)
                       : ninther_pos(lo, hi, comp, proj);
            std::iter_swap(lo, p);                 // pivot to front
            auto key = std::invoke(proj, *lo);
            It m = algo::sized{}(lo + 1, hi, key, comp, proj);
            std::iter_swap(lo, m - 1);             // place pivot at its rank
            It pp = m - 1;             // [lo,pp) < key | {pp}=pivot | [pp+1,hi) >= key
            assert(sp < qslr_stack_cap && "left-spine deeper than stack");
            stack[sp++] = pp + 1;                  // defer right [pp+1,hi); lo only
            hi = pp;                               // continue left (lo unchanged)
        }
        selection_sort(lo, hi, comp, proj);

        if (sp == 0) break;                        // array finalised front-to-back
        lo = stack[--sp];
        hi = (sp == 0) ? last : stack[sp - 1] - 1;  // derive hi (adjacent tiling)
    }
}

}  // namespace detail

// Sorts [first, last) ascending by (comp, proj).  `Threshold` (the selection-leaf
// size) is template-exposed for tuning; the pivot tier cutoff uses the tuned
// default.
template <std::ptrdiff_t Threshold = detail::qslr_threshold,
          std::random_access_iterator It, class Comp = std::less<>,
          class Proj = std::identity>
void quicksort_lr(It first, It last, Comp comp = {}, Proj proj = {}) {
    detail::quicksort_lr_impl<Threshold, detail::qslr_pivot_ninther_cutoff>(
        first, last, comp, proj);
}

}  // namespace partitions

#endif  // PARTITIONS_QUICKSORT_LR_HPP
