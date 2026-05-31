#ifndef PARTITIONS_CONCEPTS_HPP
#define PARTITIONS_CONCEPTS_HPP

// The single extension point of this library.
//
// Every partition algorithm is expressed as a *partition-by-predicate*:
//
//     I operator()(I first, S last, Pred keep_left) const;
//
// where `keep_left(x)` returns true for elements that belong on the left.
// The function reorders [first, last) so that every element for which
// `keep_left` is true precedes every element for which it is false, and
// returns the partition point `p` (the first element of the right group).
//
// From this one primitive the public API derives all four user-facing forms
// (forward/reverse partition, pivot-by-key/pivot-by-position) by supplying
// different predicates.  To test a custom partition function you only need to
// model `PredicatePartitioner` -- everything else (the full distribution
// matrix, both pivot-passing conventions, the reverse variant, the balance
// statistics and the benchmarks) then applies to it automatically.

#include <functional>
#include <iterator>

namespace partitions {

// A unary predicate over the *projected* element value.
template <class Pred, class I, class Proj = std::identity>
concept ElementPredicate =
    std::indirect_unary_predicate<Pred, std::projected<I, Proj>>;

// A partition algorithm expressed over a raw element predicate.  We require
// random access: every scheme of interest (Hoare, Lomuto, block partitioning)
// and the benchmarks assume it, and it keeps the contract simple.
template <class Alg, class I, class S = I, class Pred = std::identity>
concept PredicatePartitioner =
    std::random_access_iterator<I> && std::sentinel_for<S, I> &&
    requires(Alg alg, I first, S last, Pred pred) {
        { alg(first, last, pred) } -> std::same_as<I>;
    };

}  // namespace partitions

#endif  // PARTITIONS_CONCEPTS_HPP
