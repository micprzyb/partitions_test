#ifndef PARTITIONS_CONCEPTS_HPP
#define PARTITIONS_CONCEPTS_HPP

// The single extension point of this library.
//
// Every partition algorithm is expressed as a *partition-around-a-pivot*,
// threading the comparator and projection directly (exactly like the small
// sorters in `small_sort.hpp`, whose compare-exchange takes `comp, proj`):
//
//     I operator()(I first, S last, K pivot, Comp comp, Proj proj) const;
//
// The pivot is taken BY VALUE, not by `const K&` (nor a forwarding `K&&`).  A
// partition compares every element against this one fixed pivot while
// simultaneously PERMUTING the block; if the pivot were a reference the compiler
// must assume it might alias the elements being written and reloads it from
// memory on every comparison (`cmp (%rcx),...`).  A by-value parameter is
// provably non-aliasing, so the pivot (small for every key type here -- i64,
// pair64) stays in a register across the whole hot loop.  Confirmed by
// disassembly: const K& and K&& both reload per element, by value does not.
//
// It reorders [first, last) so that every element for which
// `comp(proj(x), pivot)` is true precedes every element for which it is false,
// and returns the partition point `m` (the first element of the right group):
//
//     [first, m)  : comp(proj(x), pivot)         (the left group)
//     [m,   last) : !comp(proj(x), pivot)        (m and everything to its right)
//
// IMPORTANT -- why this primitive can take a bare comparator rather than a
// strict-weak-ordering predicate object:  during a *partition* the comparator
// is only ever evaluated as `comp(proj(x), pivot)` for a SINGLE fixed `pivot`.
// It is therefore used purely as a unary predicate `x |-> comp(proj(x), pivot)`;
// the partition step never compares two arbitrary elements, so strict-weak
// ordering is irrelevant here (it matters only for pivot *selection* and for
// sorting).  This is what lets `partition_api.hpp` derive all four user-facing
// forms from this one primitive by only varying `comp`:
//
//                       left group                 returns (middle m)
//   forward    : comp                    (x < pivot)        [first,m) <  pivot
//   reverse    : ge = !comp              (x >= pivot)       [first,m) >= pivot
//
// To test a custom partition function you only need to model
// `PivotPartitioner` -- everything else (the full distribution matrix, both
// pivot-passing conventions, the reverse variant, the balance statistics and
// the benchmarks) then applies to it automatically.
//
// An algorithm MAY additionally provide a position-aware fast path
//
//     I at(I first, S last, I pivot, Comp comp, Proj proj) const;
//
// taking the pivot ELEMENT (an iterator into the block), which is then a real
// element of the block and can serve as a sentinel -- see `algo::hoare_guarded`
// and `algo::boost_block`.

#include <functional>
#include <iterator>

namespace partitions {

// A unary predicate over the *projected* element value.
template <class Pred, class I, class Proj = std::identity>
concept ElementPredicate =
    std::indirect_unary_predicate<Pred, std::projected<I, Proj>>;

// A partition algorithm expressed over a pivot key + comparator + projection.
// We require random access: every scheme of interest (Hoare, Lomuto, block
// partitioning) and the benchmarks assume it, and it keeps the contract simple.
template <class Alg, class I, class S = I, class K = std::iter_value_t<I>,
          class Comp = std::less<>, class Proj = std::identity>
concept PivotPartitioner =
    std::random_access_iterator<I> && std::sentinel_for<S, I> &&
    requires(Alg alg, I first, S last, K pivot, Comp comp, Proj proj) {
        { alg(first, last, pivot, comp, proj) } -> std::same_as<I>;
    };

}  // namespace partitions

#endif  // PARTITIONS_CONCEPTS_HPP
