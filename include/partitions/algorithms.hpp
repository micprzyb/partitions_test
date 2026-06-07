#ifndef PARTITIONS_ALGORITHMS_HPP
#define PARTITIONS_ALGORITHMS_HPP

// Reference partition-around-a-pivot implementations.
//
// Each is a stateless function object modelling `PivotPartitioner`
// (concepts.hpp):
//
//     I operator()(I first, S last, K pivot, Comp comp, Proj proj) const;
//
// reordering [first, last) so that elements for which `comp(proj(x), pivot)` is
// true come first and returning the partition point.  The comparator and
// projection are threaded directly (as in `small_sort.hpp`); during a partition
// `comp` is only ever evaluated against the single fixed `pivot`, so it is used
// purely as a unary predicate `below(x) = comp(proj(x), pivot)`.  These serve
// both as ready-to-use algorithms and as worked examples for plugging in your
// own.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>

#include "concepts.hpp"

namespace partitions::algo {

// ---------------------------------------------------------------------------
// std::partition wrapper -- the correctness oracle.  Always correct by
// definition; used to cross-check the others and as a performance baseline.
// ---------------------------------------------------------------------------
struct std_partition {
    static constexpr const char* name = "std_partition";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        return std::partition(first, last, [&](const auto& x) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, x), pivot));
        });
    }
};

// ---------------------------------------------------------------------------
// Lomuto -- single forward scan, one swap per kept element.  Simple, but on
// all-equal input (when `below` is constantly true) it swaps every element
// with itself: a classic pathology.
// ---------------------------------------------------------------------------
struct lomuto {
    static constexpr const char* name = "lomuto";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I store = first;
        for (I cur = first; cur != last; ++cur) {
            if (static_cast<bool>(
                    std::invoke(comp, std::invoke(proj, *cur), pivot))) {
                std::iter_swap(store, cur);
                ++store;
            }
        }
        return store;
    }
};

// ---------------------------------------------------------------------------
// Branchless Lomuto (orlp.net "gap" method).
//
// A single element is lifted out to open a "gap"; each iteration performs two
// moves into and out of the gap and advances the store position by the
// *value* of the predicate (0 or 1) instead of branching on it.  The compiler
// lowers `i += below(...)` to a conditional add, eliminating the mispredicted
// branch that dominates Lomuto's cost on random data.
//
// Reference: Orson Peters, "Branchless Lomuto partitioning", orlp.net.
// ---------------------------------------------------------------------------
struct lomuto_branchless {
    static constexpr const char* name = "lomuto_branchless";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        using D = std::iter_difference_t<I>;
        const D n = last - first;
        if (n == 0) return first;
        auto below = [&](D idx) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, first[idx]), pivot));
        };

        auto v = first;  // index with v[k]
        std::iter_value_t<I> tmp = std::move(v[0]);
        D i = 0;
        for (D j = 0; j < n - 1; ++j) {
            v[j] = std::move(v[i]);
            // gap is now at j; pull the next element into the gap at i.
            v[i] = std::move(v[j + 1]);
            i += static_cast<D>(below(i));
        }
        v[n - 1] = std::move(v[i]);
        v[i] = std::move(tmp);
        i += static_cast<D>(below(i));
        return first + i;
    }
};

// ---------------------------------------------------------------------------
// Hoare -- two pointers converging from the ends.  Performs roughly half as
// many swaps as Lomuto and, crucially, splits all-equal input down the middle
// (each side advances symmetrically), avoiding Lomuto's degenerate behaviour.
// ---------------------------------------------------------------------------
struct hoare {
    static constexpr const char* name = "hoare";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        auto below = [&](I it) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, *it), pivot));
        };
        I lo = first;
        I hi = last;
        while (true) {
            while (lo != hi && below(lo)) ++lo;
            do {
                if (lo == hi) return lo;
                --hi;
            } while (!below(hi));
            std::iter_swap(lo, hi);
            ++lo;
        }
    }
};

// ---------------------------------------------------------------------------
// Guarded Hoare -- a *position-aware* partitioner.
//
// Given the pivot POSITION (not just its value), it moves the pivot element to
// the end of the range to act as a sentinel: because that element equals the
// pivot it is never "< pivot", so the left-to-right scan is guaranteed to stop
// there and needs NO per-element bound check.  After each swap the just-placed
// (>= pivot) element becomes the sentinel for the next ascending scan, so the
// guard holds throughout.  This is the classic "move pivot to an end, use it as
// a guard" optimisation, and it is only sound when the pivot is a real element
// of the block -- hence it lives behind the position interface.
//
// It also provides the plain value-pivot `operator()` as the fallback used for
// key pivots (which may be absent, so no sentinel can be assumed).
// ---------------------------------------------------------------------------
struct hoare_guarded {
    static constexpr const char* name = "hoare_guarded";

    // Fallback (value pivot / no position): ordinary bounds-checked Hoare.
    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        return hoare{}(first, last, pivot, comp, proj);
    }

    // Position-aware forward partition: [first, m) < pivot, [m, last) >= pivot.
    template <std::random_access_iterator I, std::sentinel_for<I> S,
              class Comp = std::less<>, class Proj = std::identity>
    I at(I first, S last_s, I pivot, Comp comp = {}, Proj proj = {}) const {
        I last = first + (last_s - first);
        I sentinel = last - 1;
        std::iter_swap(pivot, sentinel);             // pivot now guards the < scan
        auto pivot_key = std::invoke(proj, *sentinel);  // stable copy
        auto below = [&](I it) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, *it), pivot_key));
        };
        I lo = first;
        I hi = sentinel;  // the sentinel sits at hi and stops the ascending scan
        while (true) {
            while (below(lo)) ++lo;                  // GUARDED: no bound check
            --hi;
            while (hi > lo && !below(hi)) --hi;       // bounds-checked descending
            if (lo >= hi) break;
            std::iter_swap(lo, hi);
            ++lo;
        }
        return lo;  // [first,lo) < pivot; [lo,last) >= pivot (incl. the sentinel)
    }
};

// ---------------------------------------------------------------------------
// Block (BlockQuicksort) partitioning.
//
// Hoare's two-pointer scheme, but the comparison results for a block of
// elements are first written into small offset buffers without branching;
// only then are the at-most-min(left,right) necessary swaps performed.  This
// is the partitioning core of pattern-defeating quicksort and of the C++
// standard libraries' introsort.
//
// Reference: Edelkamp & Weiss, "BlockQuicksort: Avoiding Branch
// Mispredictions in Quicksort", ESA 2016.
// ---------------------------------------------------------------------------
struct block {
    static constexpr const char* name = "block";
    static constexpr int B = 128;

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        using D = std::iter_difference_t<I>;
        auto below = [&](I it) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, *it), pivot));
        };
        I l = first;
        I r = first + (last - first);  // one past the last unprocessed element

        unsigned char off_l[B];  // positions (from l) of elements going right
        unsigned char off_r[B];  // positions (back from r) of elements going left
        int num_l = 0, num_r = 0;
        int start_l = 0, start_r = 0;

        while (r - l > 2 * static_cast<D>(B)) {
            // Refill the left offset buffer: record misplaced elements
            // (those that should move right, i.e. !below).
            if (num_l == 0) {
                start_l = 0;
                for (int k = 0; k < B; ++k) {
                    off_l[num_l] = static_cast<unsigned char>(k);
                    num_l += !below(l + k);
                }
            }
            // Refill the right offset buffer: elements that should move left.
            if (num_r == 0) {
                start_r = 0;
                for (int k = 0; k < B; ++k) {
                    off_r[num_r] = static_cast<unsigned char>(k + 1);
                    num_r += below(r - (k + 1));
                }
            }
            const int num = std::min(num_l, num_r);
            for (int k = 0; k < num; ++k) {
                std::iter_swap(l + off_l[start_l + k], r - off_r[start_r + k]);
            }
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;
            if (num_l == 0) l += B;
            if (num_r == 0) r -= B;
        }

        // Loop invariant on exit: every element in [first, l) is below pivot and
        // every element in [r, last) is not -- l only advances a full block once
        // all its misplaced elements have been swapped out, and symmetrically
        // for r.  Any partially-consumed block therefore lies entirely within
        // [l, r), so finishing that window with the scalar Hoare scheme yields
        // the correct global partition point.  (Re-scanning at most ~2B elements
        // is a negligible constant.)
        return hoare{}(l, r, pivot, comp, proj);
    }
};

// ---------------------------------------------------------------------------
// Boost / pdqsort branchless block partition.
//
// This is Orson Peters' heavily micro-optimised refinement of BlockQuicksort
// (as shipped in boost::sort::pdqsort and the reference pdqsort), adapted to
// this library's pivot+comparator+projection primitive.  Versus `algo::block`
// it adds the three tricks that make it the fastest scalar partition known:
//
//   * CACHE: the two offset buffers are over-allocated and `align_cacheline`d,
//     so each sits on its own 64-byte line -- the dense sequential writes never
//     straddle a line boundary or false-share with the other buffer.
//
//   * BRANCH PREDICTION: the offset-fill loops are fully branchless.  Each step
//     does `offsets[num] = i; num += predicate;` -- the misplaced index is
//     written unconditionally and the counter advances by the predicate's value
//     (0/1), which the compiler lowers to `setcc`/conditional-add with no
//     data-dependent jump.  On random data this removes the ~50%-mispredict
//     branch that dominates a naive Hoare partition.  The loops are unrolled 8x
//     (block_size is a multiple of 8) to amortise the loop counter.
//
//   * REGISTER PRESSURE / MOVES: when both buffers carry the same count the
//     swaps are a pure rotation (`swap_offsets` with use_swaps=false) -- one
//     temporary and 2N moves instead of 3N for N independent swaps.  The
//     `use_swaps=true` path is kept for the case the counts differ, which keeps
//     the descending distribution O(n).
//
// The initial left scan is templated on `Guarded`: for the position fast path
// (`at`) the pivot is a real in-block element that is itself >= pivot and thus
// stops the scan, so the bound check is compiled out (sentinel elision); the
// value-pivot `operator()` keeps the bound because the key may be absent.
//
// FINDINGS (Meteor Lake, GCC, -O3 -march=native, random_uniform, ns/elem; full
// sweep via benchmarks/bench_partition).  This is the fastest partitioner here:
//
//                       n=24    n=256   n=4096   n=2^16   n=2^22
//     i64    Hoare       3.01    2.58     2.27     2.84     2.38
//     i64    boost_block 1.36    0.69     0.49     0.49     0.56   (2.2-5.8x)
//     pair64 Hoare       3.39    2.71     2.32     2.41     2.12
//     pair64 boost_block 2.07    1.02     0.58     0.57     0.75   (1.6-4.5x)
//
//   Disassembly confirms the fill loop is the intended branchless
//   `cmp; setge/setl; add` (i64, pair64-by-.first) with only the per-8 loop
//   back-edge as a branch.  Two micro-optimisations were tried and REJECTED by
//   measurement (negative results worth keeping):
//     - a fully branchless lexicographic `below` (small_sort's trick): SLOWER
//       for a fixed-pivot partition -- see the `below` lambda comment below.
//     - branchless 16-byte swap: the swaps are min(num_l,num_r) per block and
//       far rarer than the compares, so the compare path dominates; the plain
//       move-based `swap_offsets` rotation is already optimal.
//   The hardware prefetcher handles the sequential block scans, so explicit
//   prefetch was not pursued.
//
// INTERFACE STUDY (does the value-pivot + comp/proj primitive cost anything?).
// Three drop-in variants of this exact block algorithm were benchmarked:
//   - value  : the current design -- a local key copy, comp/proj threaded;
//   - pred   : the old `Pred keep_left` interface (opaque predicate);
//   - pos    : faithful pdqsort -- pivot physically swapped to an end and used
//              as a sentinel, then placed at the boundary.
//   Findings (random_uniform, min ns/elem, stable over runs):
//     * value vs pred: equal at small n; value is FASTER at large n (pair64
//       2^22: 0.83 vs 1.13; i64 2^22: 0.64 vs 0.71) -- the opaque predicate
//       inhibits codegen.  The comp/proj interface is not a regression.
//     * value vs pos: within a few % either way; pos does two extra swaps and
//       places the pivot for no consistent win.  The guard/sentinel benefit is
//       fully retained by the UNGUARDED scan (Guarded=false), so passing the
//       pivot by value rather than by position does NOT degrade performance.
//
// SORT-AS-PARTITION (could a branchless small-sort network beat partitioning at
// small n?).  Only for a CHEAP-TO-SWAP element: sorting then taking lower_bound
// beats this partitioner for i64 at n<=~16 (n=12: 1.61 vs 1.87 ns/elem), but
// LOSES badly for pair64 / pair64-by-first at every n (its O(n log^2 n) 16-byte
// compare-exchanges dwarf a partition's O(n) few swaps).  So it is a narrow,
// element-type-dependent win not worth a special path here; the branchless block
// is the better single default down to the cutoff.
// ---------------------------------------------------------------------------
namespace detail {

// Tuned empirically on i64/pair64 random_uniform (Meteor Lake, GCC, -O3
// -march=native; see the findings block above boost_block).  block_size=128
// (vs the reference pdqsort's 64) is a stable ~17% win for i64 at n=2^22
// (0.63 -> 0.53 ns/elem): the larger offset block runs fewer outer iterations
// and amortises the boundary bookkeeping; it is neutral for pair64.  It must
// stay a multiple of 8 (8x unrolled fill) and < 256 (offsets are unsigned char,
// and the right buffer stores values up to block_size).
inline constexpr std::size_t boost_block_size = 128;
inline constexpr std::size_t boost_cacheline_size = 64;  // power of two

// Below this length the offset-buffer machinery's fixed setup cost (cacheline
// alignment, block bookkeeping) outweighs the branch-misprediction it saves, so
// the value/position entry points fall back to scalar Hoare.  A head-to-head
// sweep (bench_partition + the methods investigation) shows the branchless block
// already beats scalar Hoare from n=8 on i64 and n=12 on pair64 lexicographic
// (n=16: i64 1.66 vs 2.93, pair64 2.52 vs 3.31 ns/elem) on DIVERSE keys, so the
// crossover is well below the n>21 one might guess -- we set the cutoff at 16.
//
// CAVEAT (documented, not fixed): for a NEAR-FREE, well-predicted compare on
// LOW-CARDINALITY keys (e.g. pair64 by .first when many first keys are equal --
// the degenerate all_equal-ish small-n regime), the scalar Hoare scan has no
// branch to mispredict and beats the offset machinery (n=16: ~0.26 vs ~0.45
// ns/elem).  A single generic cutoff cannot serve both; 16 optimises for diverse
// keys (the norm and the large-n regime, where this type wins 3-4.7x too).  The
// absolute cost in the degenerate regime is sub-0.5 ns/elem, far smaller than
// the 1+ ns/elem won on the expensive-compare types.
inline constexpr std::ptrdiff_t boost_scalar_cutoff = 16;

template <class T>
[[gnu::always_inline]] inline T* align_cacheline(T* p) {
    auto ip = reinterpret_cast<std::uintptr_t>(p);
    ip = (ip + (boost_cacheline_size - 1)) &
         ~(static_cast<std::uintptr_t>(boost_cacheline_size) - 1);
    return reinterpret_cast<T*>(ip);
}

// Apply the at-most-`num` cross swaps recorded in the offset buffers.  With
// equal counts (use_swaps=false) this is a cyclic rotation: one temporary,
// two moves per element.  Otherwise fall back to independent iter_swaps.
template <class Iter>
[[gnu::always_inline]] inline void swap_offsets(Iter first, Iter last,
                                                const unsigned char* offsets_l,
                                                const unsigned char* offsets_r,
                                                std::size_t num, bool use_swaps) {
    using T = typename std::iterator_traits<Iter>::value_type;
    if (use_swaps) {
        for (std::size_t i = 0; i < num; ++i)
            std::iter_swap(first + offsets_l[i], last - offsets_r[i]);
    } else if (num > 0) {
        Iter l = first + offsets_l[0];
        Iter r = last - offsets_r[0];
        T tmp(std::move(*l));
        *l = std::move(*r);
        for (std::size_t i = 1; i < num; ++i) {
            l = first + offsets_l[i];
            *r = std::move(*l);
            r = last - offsets_r[i];
            *l = std::move(*r);
        }
        *r = std::move(tmp);
    }
}

// Partition [begin, end) around the value `pivot`.  Elements for which
// `comp(proj(x), pivot)` holds go left, the rest (incl. equal) go right;
// returns the boundary.  `Guarded` keeps the `first < last` bound on the
// initial left scan (value pivot, possibly absent); pass false only when an
// in-block element is guaranteed to stop it (position pivot sentinel).
template <bool Guarded, class Iter, class K, class Compare, class Proj>
inline Iter branchless_partition(Iter begin, Iter end, K pivot,
                                 Compare comp, Proj proj) {
    // below(x) = comp(proj(x), pivot), the fixed-pivot partition predicate.
    //
    // NEGATIVE RESULT (disassembly + A/B, kept deliberately): for the pair64
    // lexicographic order the default `operator<` short-circuits, so GCC emits
    // `cmp first; je <second-key handler>` -- a data-dependent branch on
    // first-key EQUALITY.  Replacing it with small_sort's fully branchless lex
    // predicate `(b0<a0) | (b0==a0 & b1<a1)` (computed off one compare) made the
    // partition ~30-47% SLOWER at large n (pair64 2^16: 0.58 -> 0.86 ns/elem).
    // The reason is the mirror image of the small-sort case: a partition
    // compares every element against ONE fixed pivot, and on random data the
    // first keys are mostly distinct from the pivot's, so the `je` is rarely
    // taken and the branch predictor nails it -- the short-circuit then skips
    // the second-key load+compare entirely.  The branchless form pays for the
    // second key on every element for a predictor win that does not exist here.
    // So we keep the plain short-circuiting compare.
    auto below = [&](Iter it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };

    Iter first = begin;
    Iter last = end;

    // Find the first element >= pivot.
    if constexpr (Guarded) {
        while (first < last && below(first)) ++first;
    } else {
        while (below(first)) ++first;  // in-block pivot is the sentinel
    }

    // Find the first element strictly below the pivot from the right.  Guard
    // the search if there was no element before *first.
    if (first == begin) {
        while (begin < last && !below(--last)) {
        }
    } else {
        while (!below(--last)) {
        }
    }

    // If the first pair that should be swapped is the same element, the input
    // was already partitioned.
    if (first < last) {
        std::iter_swap(first, last);
        ++first;

        alignas(boost_cacheline_size)
            unsigned char offsets_l_storage[boost_block_size + boost_cacheline_size];
        alignas(boost_cacheline_size)
            unsigned char offsets_r_storage[boost_block_size + boost_cacheline_size];
        unsigned char* offsets_l = align_cacheline(offsets_l_storage);
        unsigned char* offsets_r = align_cacheline(offsets_r_storage);

        Iter offsets_l_base = first;
        Iter offsets_r_base = last;
        std::size_t num_l = 0, num_r = 0, start_l = 0, start_r = 0;

        while (first < last) {
            // How many elements to scan into each offset block this round.
            std::size_t num_unknown = static_cast<std::size_t>(last - first);
            std::size_t left_split =
                num_l == 0 ? (num_r == 0 ? num_unknown / 2 : num_unknown) : 0;
            std::size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

            // Fill the left offset block with elements that belong on the right.
            if (left_split >= boost_block_size) {
                for (std::size_t i = 0; i < boost_block_size;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                }
            } else {
                for (std::size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                }
            }

            // Fill the right offset block with elements that belong on the left.
            if (right_split >= boost_block_size) {
                for (std::size_t i = 0; i < boost_block_size;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                }
            } else {
                for (std::size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                }
            }

            // Swap the identified out-of-place pairs and advance the boundaries.
            std::size_t num = std::min(num_l, num_r);
            swap_offsets(offsets_l_base, offsets_r_base, offsets_l + start_l,
                         offsets_r + start_r, num, num_l == num_r);
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;
            if (num_l == 0) {
                start_l = 0;
                offsets_l_base = first;
            }
            if (num_r == 0) {
                start_r = 0;
                offsets_r_base = last;
            }
        }

        // Drain any leftover offsets against the opposite end.
        if (num_l) {
            offsets_l += start_l;
            while (num_l--)
                std::iter_swap(offsets_l_base + offsets_l[num_l], --last);
            first = last;
        }
        if (num_r) {
            offsets_r += start_r;
            while (num_r--) {
                std::iter_swap(offsets_r_base - offsets_r[num_r], first);
                ++first;
            }
            last = first;
        }
    }
    return first;  // [begin,first) < pivot; [first,end) >= pivot
}

}  // namespace detail

struct boost_block {
    static constexpr const char* name = "boost_block";

    // Value pivot (possibly absent): guarded scan, scalar fallback for small n.
    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I end = first + (last - first);
        if (end - first < detail::boost_scalar_cutoff)
            return hoare{}(first, end, pivot, comp, proj);
        return detail::branchless_partition<true>(first, end, pivot, comp, proj);
    }

    // Position pivot (a real in-block element): unguarded scan -- the pivot is
    // its own sentinel, so the per-element bound check is compiled out.
    template <std::random_access_iterator I, std::sentinel_for<I> S,
              class Comp = std::less<>, class Proj = std::identity>
    I at(I first, S last_s, I pivot, Comp comp = {}, Proj proj = {}) const {
        I last = first + (last_s - first);
        auto key = std::invoke(proj, *pivot);  // stable copy
        if (last - first < detail::boost_scalar_cutoff)
            return hoare{}(first, last, key, comp, proj);
        return detail::branchless_partition<false>(first, last, key, comp, proj);
    }
};

// ---------------------------------------------------------------------------
// Fulcrum partition (crumsort, Igor van den Hoven) -- the Hoare+Lomuto hybrid.
//
// BIDIRECTIONAL like Hoare (it streams the block reading from both ends), but
// BRANCHLESS like Lomuto via a gap, using the "double write" trick: each element
// is written UNCONDITIONALLY to a left slot `ptl[m]` (= array[m]) AND a right
// slot `ptr[m]`; the running count `m` advances only when the element belongs
// left, so exactly one of the two writes is committed and the other is
// overwritten on a later iteration -- no data-dependent branch, and only one
// store survives per element (≈ Hoare's move count, not Lomuto's two).
//
// To keep the unconditional writes from clobbering elements not yet read, the
// first 32 and last 32 elements are copied into a stack `swap` buffer up front;
// the middle is then streamed (reads stay ahead of writes by >= 16, maintained
// by the `(pta-ptl)-m` slack tests), and the 64 saved elements are processed
// last.  Reference: scandum/crumsort `fulcrum_default_partition`.
// ---------------------------------------------------------------------------
namespace detail {

inline constexpr std::ptrdiff_t fulcrum_cutoff = 80;  // needs n >= 2*32 + slack

template <class I, class K, class Comp, class Proj>
I fulcrum_partition(I first, std::iter_difference_t<I> n, K piv, Comp comp,
                    Proj proj) {
    using T = std::iter_value_t<I>;
    using D = std::iter_difference_t<I>;
    constexpr D B = 32;
    T swap[2 * B];
    for (D k = 0; k < B; ++k) swap[k] = first[k];
    for (D k = 0; k < B; ++k) swap[B + k] = first[n - B + k];

    // `piv` is taken BY VALUE (not const K& / K&&): a reference of any kind
    // forces the compiler to assume the pivot may alias the array being permuted,
    // so it is reloaded from memory on every compare (`cmp (%rcx),...`); a
    // by-value parameter is provably non-aliasing and stays in a register across
    // the whole hot loop (~5% on i64).  See the note on the PivotPartitioner
    // primitive in concepts.hpp.

    I ptl = first;                  // left write base (committed left -> ptl[m])
    I ptr = first + (n - 1);        // right write base (descends each step)
    I pta = first + B;              // left read cursor (middle, ascending)
    I tpa = first + (n - 1 - B);    // right read cursor (middle, descending)
    D m = 0;

    // Place one element via crumsort's branchless DOUBLE WRITE: store x to BOTH
    // ptl[m] and ptr[m]; `m += v` commits exactly one (the other is overwritten
    // on a later iteration).  Both destination addresses (ptl+m, ptr+m) depend
    // only on m, so they are ready BEFORE the comparison resolves and the two
    // stores issue early on the two store ports.
    //
    // REJECTED alternative (kept as a note): a cmov single-store
    // `*(v ? ptl+m : ptr+m) = x` halves the store count and is correct (the
    // committed writes alone tile [0,n)), and it wins in an isolated microbench;
    // but in this templated comp/proj context it was ~5-15% SLOWER, because the
    // store address then depends on v (the compare), serialising the store behind
    // the comparison instead of letting the v-independent addresses pipeline.
    auto place = [&](T x) {
        const D v = static_cast<D>(
            static_cast<bool>(std::invoke(comp, std::invoke(proj, x), piv)));
        ptl[m] = x;
        ptr[m] = x;
        m += v;
        --ptr;
    };

    D cnt = n / 16 - 4;
    while (true) {
        if ((pta - ptl) - m <= 48) {
            if (cnt-- == 0) break;
            for (int i = 16; i; --i) place(*pta++);
        }
        if ((pta - ptl) - m >= 16) {
            if (cnt-- == 0) break;
            for (int i = 16; i; --i) place(*tpa--);
        }
    }
    if ((pta - ptl) - m <= 48) {
        for (D c = n % 16; c; --c) place(*pta++);
    } else {
        for (D c = n % 16; c; --c) place(*tpa--);
    }
    const T* ps = swap;
    for (D c = 0; c < 2 * B; ++c) place(*ps++);
    return first + m;  // [first, first+m) < pivot; [first+m, end) >= pivot
}

}  // namespace detail

struct fulcrum {
    static constexpr const char* name = "fulcrum";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I end = first + (last - first);
        const auto n = end - first;
        if (n < detail::fulcrum_cutoff)
            return boost_block{}(first, end, pivot, comp, proj);
        return detail::fulcrum_partition(first, n, pivot, comp, proj);
    }
};

// ---------------------------------------------------------------------------
// Size-dispatching partitioner.  Routes to the sub-partitioner that wins at the
// given block size: a cheap branchless Lomuto for small blocks, the pdqsort
// branchless block partition for large ones.  The branchless Lomuto does two
// moves per element, so it only pays for narrow, cheap-to-move elements -- hence
// the array-size threshold is itself chosen by sizeof(T) (the same rationale as
// Rust ipnsort's size_of<T> rule): a generous cutoff for register-width keys,
// almost none for wide (>8 byte) ones, which go straight to the block partition.
// boost_block already has its own small-n scalar fallback, so this only adds the
// Lomuto fast path where it actually wins.
// ---------------------------------------------------------------------------
namespace detail {
template <class T>
inline constexpr std::ptrdiff_t sized_cutoff = sizeof(T) <= 8 ? 512 : 24;
}

struct sized {
    static constexpr const char* name = "sized";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I end = first + (last - first);
        if (end - first <= detail::sized_cutoff<std::iter_value_t<I>>)
            return lomuto_branchless{}(first, end, pivot, comp, proj);
        return boost_block{}(first, end, pivot, comp, proj);
    }

    template <std::random_access_iterator I, std::sentinel_for<I> S,
              class Comp = std::less<>, class Proj = std::identity>
    I at(I first, S last_s, I pivot, Comp comp = {}, Proj proj = {}) const {
        I last = first + (last_s - first);
        if (last - first <= detail::sized_cutoff<std::iter_value_t<I>>) {
            auto key = std::invoke(proj, *pivot);  // stable copy (no sentinel path)
            return lomuto_branchless{}(first, last, key, comp, proj);
        }
        return boost_block{}.at(first, last, pivot, comp, proj);
    }
};

}  // namespace partitions::algo

#endif  // PARTITIONS_ALGORITHMS_HPP
