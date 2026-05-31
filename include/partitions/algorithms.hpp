#ifndef PARTITIONS_ALGORITHMS_HPP
#define PARTITIONS_ALGORITHMS_HPP

// Reference partition-by-predicate implementations.
//
// Each is a stateless function object modelling `PredicatePartitioner`:
//
//     I operator()(I first, S last, Pred keep_left) const;
//
// reordering [first, last) so that elements satisfying `keep_left` come first
// and returning the partition point.  These serve both as ready-to-use
// algorithms and as worked examples for plugging in your own.

#include <algorithm>
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

    template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
    I operator()(I first, S last, Pred keep_left) const {
        return std::partition(first, last, std::move(keep_left));
    }
};

// ---------------------------------------------------------------------------
// Lomuto -- single forward scan, one swap per kept element.  Simple, but on
// all-equal input (when `keep_left` is constantly true) it swaps every
// element with itself: a classic pathology.
// ---------------------------------------------------------------------------
struct lomuto {
    static constexpr const char* name = "lomuto";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
    I operator()(I first, S last, Pred keep_left) const {
        I store = first;
        for (I cur = first; cur != last; ++cur) {
            if (keep_left(*cur)) {
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
// lowers `i += pred(...)` to a conditional add, eliminating the mispredicted
// branch that dominates Lomuto's cost on random data.
//
// Reference: Orson Peters, "Branchless Lomuto partitioning", orlp.net.
// ---------------------------------------------------------------------------
struct lomuto_branchless {
    static constexpr const char* name = "lomuto_branchless";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
    I operator()(I first, S last, Pred keep_left) const {
        using D = std::iter_difference_t<I>;
        const D n = last - first;
        if (n == 0) return first;

        auto v = first;  // index with v[k]
        std::iter_value_t<I> tmp = std::move(v[0]);
        D i = 0;
        for (D j = 0; j < n - 1; ++j) {
            v[j] = std::move(v[i]);
            // gap is now at j; pull the next element into the gap at i.
            v[i] = std::move(v[j + 1]);
            i += static_cast<D>(static_cast<bool>(keep_left(v[i])));
        }
        v[n - 1] = std::move(v[i]);
        v[i] = std::move(tmp);
        i += static_cast<D>(static_cast<bool>(keep_left(v[i])));
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

    template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
    I operator()(I first, S last, Pred keep_left) const {
        I lo = first;
        I hi = last;
        while (true) {
            while (lo != hi && keep_left(*lo)) ++lo;
            do {
                if (lo == hi) return lo;
                --hi;
            } while (!keep_left(*hi));
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
// It also provides the plain predicate `operator()` as the fallback used for
// key pivots (which may be absent, so no sentinel can be assumed).
// ---------------------------------------------------------------------------
struct hoare_guarded {
    static constexpr const char* name = "hoare_guarded";

    // Fallback (key pivots / no position): ordinary bounds-checked Hoare.
    template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
    I operator()(I first, S last, Pred keep_left) const {
        return hoare{}(first, last, std::move(keep_left));
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

    template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
    I operator()(I first, S last, Pred keep_left) const {
        using D = std::iter_difference_t<I>;
        I l = first;
        I r = last;  // one past the last unprocessed element

        unsigned char off_l[B];  // positions (from l) of elements going right
        unsigned char off_r[B];  // positions (back from r) of elements going left
        int num_l = 0, num_r = 0;
        int start_l = 0, start_r = 0;

        while (r - l > 2 * static_cast<D>(B)) {
            // Refill the left offset buffer: record misplaced elements
            // (those that should move right, i.e. !keep_left).
            if (num_l == 0) {
                start_l = 0;
                for (int k = 0; k < B; ++k) {
                    off_l[num_l] = static_cast<unsigned char>(k);
                    num_l += !keep_left(l[k]);
                }
            }
            // Refill the right offset buffer: elements that should move left.
            if (num_r == 0) {
                start_r = 0;
                for (int k = 0; k < B; ++k) {
                    off_r[num_r] = static_cast<unsigned char>(k + 1);
                    num_r += keep_left(r[-(k + 1)]);
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

        // Loop invariant on exit: every element in [first, l) satisfies
        // keep_left and every element in [r, last) does not -- l only advances
        // a full block once all its misplaced elements have been swapped out,
        // and symmetrically for r.  Any partially-consumed block therefore
        // lies entirely within [l, r), so finishing that window with the
        // scalar Hoare scheme yields the correct global partition point.
        // (Re-scanning at most ~2B elements is a negligible constant.)
        return hoare{}(l, r, std::move(keep_left));
    }
};

}  // namespace partitions::algo

#endif  // PARTITIONS_ALGORITHMS_HPP
