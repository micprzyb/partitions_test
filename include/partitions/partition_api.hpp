#ifndef PARTITIONS_PARTITION_API_HPP
#define PARTITIONS_PARTITION_API_HPP

// Public partition API.
//
// RETURN VALUE -- the "middle".  Every form returns an iterator `m` to the
// middle of the partition:
//
//     [first, m)  : strictly < pivot          (the left group)
//     [m,   last) : >= pivot                   (m itself and everything right)
//
// `m` is the boundary index, NOT the resting place of any particular pivot
// element.  This is well-defined even when the pivot key does not occur in the
// block: `m == first` means nothing was smaller (empty left), `m == last` means
// everything was smaller (empty right), and in general `m - first` equals the
// number of elements strictly less than the pivot.  Because the contract is
// stated purely in terms of the key, the by-key form needs no element to
// actually equal the pivot.
//
// All four forms reduce to a single partition-by-predicate call on the supplied
// algorithm.  The only thing that varies is the predicate:
//
//                       left group              returns (middle m)
//   forward    : keep_left(x) =  comp(proj(x), pivot)   [first,m) <  pivot
//   reverse    : keep_left(x) = !comp(proj(x), pivot)   [first,m) >= pivot
//
// "Pivot by key" takes the pivot value directly (it need not occur in the
// block).  "Pivot by position" takes an iterator into the block and reads the
// key once, up front, into a stable local copy -- so the element it points at
// may be moved freely during partitioning (and the position may come from a
// reordering pivot strategy).
//
// comp is a strict-weak-ordering "less".  Note that ">= pivot" is expressed as
// "not (x < pivot)", so a single comparator drives both the strict (<) and
// non-strict (>=) sides correctly.

#include <functional>
#include <iterator>
#include <utility>

#include "concepts.hpp"

namespace partitions {

// ----- forward partition: [smaller than pivot | >= pivot] ------------------

// Pivot given as a key value (may be absent from the block).
template <class Alg, std::random_access_iterator I, std::sentinel_for<I> S,
          class Key, class Comp = std::less<>, class Proj = std::identity>
I partition_by_key(Alg alg, I first, S last, const Key& pivot, Comp comp = {},
                   Proj proj = {}) {
    return alg(first, last, [&](const auto& x) {
        return static_cast<bool>(std::invoke(comp, std::invoke(proj, x), pivot));
    });
}

// Pivot given as a position.  If the algorithm offers a position-aware fast
// path (`alg.at(...)`, e.g. one that uses the pivot element as a sentinel), it
// is used; otherwise the key is captured up front and the predicate path runs.
template <class Alg, std::random_access_iterator I, std::sentinel_for<I> S,
          class Comp = std::less<>, class Proj = std::identity>
I partition_by_position(Alg alg, I first, S last, I pivot, Comp comp = {},
                        Proj proj = {}) {
    if constexpr (requires { alg.at(first, last, pivot, comp, proj); }) {
        return alg.at(first, last, pivot, comp, proj);
    } else {
        auto pivot_key = std::invoke(proj, *pivot);  // stable copy
        return alg(first, last, [&](const auto& x) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, x), pivot_key));
        });
    }
}

// ----- reverse partition: [>= pivot | smaller than pivot] ------------------

template <class Alg, std::random_access_iterator I, std::sentinel_for<I> S,
          class Key, class Comp = std::less<>, class Proj = std::identity>
I reverse_partition_by_key(Alg alg, I first, S last, const Key& pivot,
                           Comp comp = {}, Proj proj = {}) {
    return alg(first, last, [&](const auto& x) {
        return !static_cast<bool>(
            std::invoke(comp, std::invoke(proj, x), pivot));
    });
}

// Reverse partition by position.  A reverse partition is a forward partition
// performed over the reversed range, so a position-aware `alg.at` is reused on
// reverse iterators and the boundary is mapped back with `.base()`; otherwise
// the predicate path runs.
template <class Alg, std::random_access_iterator I, std::sentinel_for<I> S,
          class Comp = std::less<>, class Proj = std::identity>
I reverse_partition_by_position(Alg alg, I first, S last, I pivot,
                                Comp comp = {}, Proj proj = {}) {
    I end = first + (last - first);
    auto rfirst = std::make_reverse_iterator(end);
    if constexpr (requires {
                      alg.at(rfirst, std::make_reverse_iterator(first),
                             std::make_reverse_iterator(pivot + 1), comp, proj);
                  }) {
        auto rp = alg.at(rfirst, std::make_reverse_iterator(first),
                         std::make_reverse_iterator(pivot + 1), comp, proj);
        return rp.base();
    } else {
        auto pivot_key = std::invoke(proj, *pivot);
        return alg(first, last, [&](const auto& x) {
            return !static_cast<bool>(
                std::invoke(comp, std::invoke(proj, x), pivot_key));
        });
    }
}

}  // namespace partitions

#endif  // PARTITIONS_PARTITION_API_HPP
