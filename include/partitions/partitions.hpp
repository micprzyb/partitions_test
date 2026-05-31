#ifndef PARTITIONS_PARTITIONS_HPP
#define PARTITIONS_PARTITIONS_HPP

// Umbrella header + registries.
//
// The registries are plain std::tuples of stateless function objects, walked
// with `for_each`.  Add an algorithm / pivot / distribution by appending it to
// the corresponding tuple here and it is picked up by every test, benchmark
// and report automatically.

#include <tuple>
#include <utility>

#include "algorithms.hpp"
#include "concepts.hpp"
#include "distributions.hpp"
#include "partition_api.hpp"
#include "partition_with_pivot.hpp"
#include "pivot.hpp"
#include "statistics.hpp"
#include "types.hpp"

namespace partitions {

// Apply `f` to each element of a tuple.
template <class Tuple, class F>
constexpr void for_each(Tuple&& t, F&& f) {
    std::apply([&](auto&&... xs) { (f(xs), ...); }, std::forward<Tuple>(t));
}

// The partition algorithms under test.  All model PredicatePartitioner;
// hoare_guarded additionally offers a position-aware fast path (`.at`) that
// uses the pivot element as a sentinel.
inline auto default_partitioners() {
    return std::tuple{algo::std_partition{}, algo::lomuto{},
                      algo::lomuto_branchless{}, algo::hoare{},
                      algo::hoare_guarded{}, algo::block{}};
}

// Pivot-selection strategies.  Most return a POSITION; midpoint_* return a
// VALUE that need not be in the block; the last two REORDER the block.
inline auto default_pivots() {
    return std::tuple{pivot::first_element{},
                      pivot::middle_element{},
                      pivot::last_element{},
                      pivot::median_of_3{},
                      pivot::median_of_5{},
                      pivot::ninther{},
                      pivot::median_of_medians_5{},
                      pivot::random_pivot{},
                      pivot::midpoint_min_max{},
                      pivot::midpoint_first_last{},
                      pivot::median_of_3_inplace{},
                      pivot::median_of_medians_5_inplace{}};
}

// Input distributions.
inline auto default_distributions() {
    return std::tuple{
        dist::random_uniform{}, dist::few_unique{},      dist::all_equal{},
        dist::binary{},         dist::sorted_ascending{}, dist::sorted_descending{},
        dist::nearly_sorted{},  dist::organ_pipe{},      dist::reverse_organ_pipe{},
        dist::sawtooth{},       dist::median_of_3_killer{}, dist::single_outlier{},
        dist::shuffled_blocks{}};
}

}  // namespace partitions

#endif  // PARTITIONS_PARTITIONS_HPP
