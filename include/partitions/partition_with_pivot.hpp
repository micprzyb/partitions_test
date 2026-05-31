#ifndef PARTITIONS_PARTITION_WITH_PIVOT_HPP
#define PARTITIONS_PARTITION_WITH_PIVOT_HPP

// Glue between a pivot-strategy result and the partition API.
//
// A strategy may return a *position* (an iterator into the block) or a *value*
// (`pivot::value_pivot<K>`, possibly absent from the block).  These helpers
// dispatch to `partition_by_position` or `partition_by_key` accordingly, so
// callers (tests, benchmarks, reports) can stay convention-agnostic:
//
//     auto r = strategy(first, last, comp, proj);       // position OR value
//     auto m = partition_with_pivot(alg, first, last, r, comp, proj);
//
// `r` must outlive the call.  For a position result, the pivot key is read from
// the block at call time (so a preceding reordering strategy is fine); for a
// value result, the stored key is used directly.

#include <functional>
#include <iterator>

#include "partition_api.hpp"
#include "pivot.hpp"

namespace partitions {

template <class Alg, std::random_access_iterator I, std::sentinel_for<I> S,
          class R, class Comp = std::less<>, class Proj = std::identity>
I partition_with_pivot(Alg alg, I first, S last, const R& r, Comp comp = {},
                       Proj proj = {}) {
    if constexpr (pivot::is_value_pivot_v<R>)
        return partition_by_key(alg, first, last, r.key, comp, proj);
    else
        return partition_by_position(alg, first, last, r, comp, proj);
}

template <class Alg, std::random_access_iterator I, std::sentinel_for<I> S,
          class R, class Comp = std::less<>, class Proj = std::identity>
I reverse_partition_with_pivot(Alg alg, I first, S last, const R& r,
                               Comp comp = {}, Proj proj = {}) {
    if constexpr (pivot::is_value_pivot_v<R>)
        return reverse_partition_by_key(alg, first, last, r.key, comp, proj);
    else
        return reverse_partition_by_position(alg, first, last, r, comp, proj);
}

}  // namespace partitions

#endif  // PARTITIONS_PARTITION_WITH_PIVOT_HPP
