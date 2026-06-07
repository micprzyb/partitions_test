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
// HOW THE REVERSAL IS DONE -- by EXCHANGING THE ROLES OF THE TWO SCANS, using
// the SAME comparison as forward (not a negated comparator, not reverse
// iteration).  A partition decides, per element, which side it goes to.  The
// forward partitioners already use BOTH `below(x) = comp(proj(x), pivot)`
// (= x < pivot) and its complement: e.g. Hoare's left scan advances over
// `below` and its right scan over `!below`; the block partition's right fill
// counts `below` and its left fill `!below`.  The reversed split is obtained by
// swapping which scan plays which role -- the left scan now advances over
// `!below` and the right over `below`, the block's left fill counts `below` and
// its right fill `!below`.  So forward and reversed perform the *identical*
// work: one `below` test and one `!below` test each.  There is NO inherent extra
// operation and NO different comparison; written with the bare `below` and a
// single `!` they compile to the same machine code (scalar: only the `setcc`
// condition flips, `setg`<->`setle`; vectorised block: one `vpcmpgtq`, no
// mask-invert -- verified by disassembly).  Equal elements land LEFT because the
// strict `below` (x < pivot) is false for them, so they are never routed right.
//
// (Two earlier WRONG approaches, kept out: negating the comparator -- breaks the
// halver networks, which need a strict order; and reverse-iterating the block --
// GCC fails to auto-vectorise a reverse_iterator scan, ~3.5x more, scalar code.
// The role-exchange below avoids both: forward memory streaming, strict `<`,
// fully vectorised, measured at parity with the forward partition.)
//
// NB: these do NOT model the forward `PivotPartitioner` contract (their middle
// is the >=/< boundary, not </>=), so they are intentionally NOT added to
// `default_partitioners()` -- the forward test/stat matrix would reject them.
// They live in `partitions::algo_rev` and are exercised by their own tests and
// benchmark.  No position-aware `at` fast path is provided: the forward sentinel
// trick needs an in-block element that STOPS the left scan (one that is
// < pivot); the pivot element is == pivot, so it cannot serve as the reversed
// sentinel.  quicksort_rev (like the forward quicksort) drives the partition by
// VALUE, so `at` is never needed.

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
// algo::detail::branchless_partition, obtained by EXCHANGING THE ROLES of the
// two scans -- the comparison stays the bare strict `below(x) = (x < pivot)`,
// exactly as forward, so the reversed partition compiles to the same vector code
// (one `vpcmpgtq`, no mask-invert).  Elements >= pivot end up LEFT, < pivot RIGHT.
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
    // The SAME strict comparison as the forward partition -- below(x) = (x <
    // pivot).  The reversal is achieved purely by EXCHANGING THE ROLES of the two
    // scans (not by negating the comparator, and not by reverse iteration): the
    // forward left fill collects `!below` and the right fill `below`; here the
    // left fill collects `below` and the right fill `!below`.  So forward and
    // reversed compute exactly one `below` fill and one `!below` fill each --
    // identical operation count.  Written with the BARE `below` and a single `!`
    // (no `keep`/`!keep` double negation), GCC compiles both to the same vector
    // code: one `vpcmpgtq`, NO mask-invert (verified by disassembly).
    auto below = [&](Iter it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };

    Iter first = begin;
    Iter last = end;

    // Skip elements already on the correct (left) side: those >= pivot (!below).
    // Stop at the first element strictly < pivot (it belongs on the right).
    while (first < last && !below(first)) ++first;

    // From the right, find the first element that belongs LEFT (>= pivot, !below),
    // skipping the correctly-placed strictly-< ones (below).
    if (first == begin) {
        while (begin < last && below(--last)) {
        }
    } else {
        while (below(--last)) {
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

            // Left offset block: elements that belong on the RIGHT (x < pivot,
            // i.e. `below`) -- the role forward gives to its RIGHT fill.
            if (left_split >= boost_block_size) {
                for (std::size_t i = 0; i < boost_block_size;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                }
            } else {
                for (std::size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += below(first); ++first;
                }
            }

            // Right offset block: elements that belong on the LEFT (x >= pivot,
            // i.e. `!below`) -- the role forward gives to its LEFT fill.  Whether
            // GCC vectorises this `!below` fill (lowering `>=` as `vpcmpgtq` + a
            // free mask-invert, since AVX2 has no `>=` predicate) or the direct
            // `below` fill is a context-dependent codegen choice; either way it
            // benchmarks at parity with forward (the fill is store/swap-bound).
            if (right_split >= boost_block_size) {
                for (std::size_t i = 0; i < boost_block_size;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
                }
            } else {
                for (std::size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += !below(--last);
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
