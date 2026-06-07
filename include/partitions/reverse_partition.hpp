#ifndef PARTITIONS_REVERSE_PARTITION_HPP
#define PARTITIONS_REVERSE_PARTITION_HPP

// REVERSED partition family -- the mirror image of `algorithms.hpp`.
//
// Every algorithm here reorders [first, last) so that
//
//     [first, m)  : proj(x) >= pivot       (the left group -- equal goes LEFT)
//     [m,   last) : proj(x) <  pivot        (strictly smaller goes RIGHT)
//
// and returns the boundary `m` (= the count of elements that are >= pivot).
// This is exactly the `ptv::reverse_ok` contract in tests/verify.hpp.
//
// WHY A SEPARATE FILE AND NOT JUST A NEGATED COMPARATOR.  A forward partitioner
// uses the comparator purely as a *unary* predicate `below(x) = comp(proj(x),
// pivot)` (= x < pivot) and sends `below`-true elements left.  To get the
// reversed split we want the *complementary* predicate
//
//     keep(x) = !comp(proj(x), pivot)   (= x >= pivot)
//
// to drive the "goes left" decision.  For the scalar/block partitioners
// (lomuto, hoare, block) that is a one-character change -- the predicate is a
// unary test against a single fixed pivot, so flipping it is sound and FREE:
// the compiler lowers `!(x < piv)` to `setae`/`cmovae` instead of `setb`,
// identical instruction count and latency (verified by disassembly + the
// head-to-head benchmark in bench_reverse_partition.cpp).  Strict-weak ordering
// of the comparator is irrelevant here because it is never used to compare two
// arbitrary elements -- only x against the fixed pivot.
//
// This is the SAME transformation `partition_api.hpp::reverse_partition_by_key`
// performs via `negate_comp`; the rewritten forms below exist so the reversed
// predicate is hard-wired into the hot loop (no comparator-wrapping closure at
// all) and so the reversal cost can be measured against the forward original
// with zero interface noise.  They are deliberate copies of the `algorithms.hpp`
// originals with `below` replaced by `keep`; per the project brief code
// duplication is accepted in exchange for guaranteed zero overhead.
//
// NB: these do NOT model the forward `PivotPartitioner` contract (their middle
// is the >=/< boundary, not </>=), so they are intentionally NOT added to
// `default_partitioners()` -- the forward test/stat matrix would reject them.
// They live in `partitions::algo_rev` and are exercised by their own tests and
// benchmark.  No position-aware `at` fast path is provided: the forward sentinel
// trick needs an in-block element that STOPS the left scan, i.e. one that is
// `!keep` (< pivot); the pivot element itself is `keep` (== pivot, >=), so it
// cannot serve as the reversed sentinel.  quicksort_rev (like the forward
// quicksort) drives the partition by VALUE, so `at` is never needed.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>

#include "algorithms.hpp"  // reuse algo::detail (cacheline align, swap_offsets, cutoffs)
#include "concepts.hpp"

namespace partitions::algo_rev {

// ---------------------------------------------------------------------------
// Reversed Hoare -- two converging pointers.  `keep` (>=) elements stay left.
// Mirror of algo::hoare.  Used as the small-n / drain fallback below.
// ---------------------------------------------------------------------------
struct hoare_rev {
    static constexpr const char* name = "hoare_rev";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        auto keep = [&](I it) {
            return !static_cast<bool>(
                std::invoke(comp, std::invoke(proj, *it), pivot));
        };
        I lo = first;
        I hi = last;
        while (true) {
            while (lo != hi && keep(lo)) ++lo;
            do {
                if (lo == hi) return lo;
                --hi;
            } while (!keep(hi));
            std::iter_swap(lo, hi);
            ++lo;
        }
    }
};

// ---------------------------------------------------------------------------
// Reversed branchless Lomuto (orlp gap method).  Mirror of
// algo::lomuto_branchless with `below` -> `keep`.
// ---------------------------------------------------------------------------
struct lomuto_branchless_rev {
    static constexpr const char* name = "lomuto_branchless_rev";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        using D = std::iter_difference_t<I>;
        const D n = last - first;
        if (n == 0) return first;
        auto keep = [&](D idx) {
            return !static_cast<bool>(
                std::invoke(comp, std::invoke(proj, first[idx]), pivot));
        };

        auto v = first;
        std::iter_value_t<I> tmp = std::move(v[0]);
        D i = 0;
        for (D j = 0; j < n - 1; ++j) {
            v[j] = std::move(v[i]);
            v[i] = std::move(v[j + 1]);
            i += static_cast<D>(keep(i));
        }
        v[n - 1] = std::move(v[i]);
        v[i] = std::move(tmp);
        i += static_cast<D>(keep(i));
        return first + i;
    }
};

// ---------------------------------------------------------------------------
// Reversed pdqsort branchless block partition.  Mirror of
// algo::detail::branchless_partition, `below` -> `keep`.  Elements that are
// `keep` (>= pivot) belong LEFT; the rest belong RIGHT.
// ---------------------------------------------------------------------------
namespace detail {

using algo::detail::align_cacheline;
using algo::detail::boost_block_size;
using algo::detail::boost_cacheline_size;
using algo::detail::boost_scalar_cutoff;
using algo::detail::swap_offsets;

template <class Iter, class K, class Compare, class Proj>
inline Iter branchless_partition_rev(Iter begin, Iter end, K pivot,
                                     Compare comp, Proj proj) {
    // keep(x) = !comp(proj(x), pivot) = (x >= pivot): the reversed "goes left"
    // predicate.  This is the ONLY change from the forward branchless_partition;
    // see the file header for why flipping the unary predicate is sound + free.
    auto keep = [&](Iter it) {
        return !static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };

    Iter first = begin;
    Iter last = end;

    // Find the first element that does NOT belong left (first x < pivot).
    while (first < last && keep(first)) ++first;

    // Find, from the right, the first element that DOES belong left (x >= pivot).
    if (first == begin) {
        while (begin < last && !keep(--last)) {
        }
    } else {
        while (!keep(--last)) {
        }
    }

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
            std::size_t num_unknown = static_cast<std::size_t>(last - first);
            std::size_t left_split =
                num_l == 0 ? (num_r == 0 ? num_unknown / 2 : num_unknown) : 0;
            std::size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

            // Left offset block: elements that belong on the RIGHT (x < pivot).
            if (left_split >= boost_block_size) {
                for (std::size_t i = 0; i < boost_block_size;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                }
            } else {
                for (std::size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !keep(first); ++first;
                }
            }

            // Right offset block: elements that belong on the LEFT (x >= pivot).
            if (right_split >= boost_block_size) {
                for (std::size_t i = 0; i < boost_block_size;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                }
            } else {
                for (std::size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += keep(--last);
                }
            }

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
    return first;  // [begin,first) >= pivot; [first,end) < pivot
}

}  // namespace detail

struct boost_block_rev {
    static constexpr const char* name = "boost_block_rev";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I end = first + (last - first);
        if (end - first < detail::boost_scalar_cutoff)
            return hoare_rev{}(first, end, pivot, comp, proj);
        return detail::branchless_partition_rev(first, end, pivot, comp, proj);
    }
};

// ---------------------------------------------------------------------------
// Reversed size-dispatching partitioner.  Mirror of algo::sized: branchless
// reversed Lomuto for small/narrow blocks, reversed block partition for large.
// ---------------------------------------------------------------------------
struct sized_rev {
    static constexpr const char* name = "sized_rev";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I end = first + (last - first);
        if (end - first <= algo::detail::sized_cutoff<std::iter_value_t<I>>)
            return lomuto_branchless_rev{}(first, end, pivot, comp, proj);
        return boost_block_rev{}(first, end, pivot, comp, proj);
    }
};

}  // namespace partitions::algo_rev

#endif  // PARTITIONS_REVERSE_PARTITION_HPP
