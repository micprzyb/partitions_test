#ifndef PARTITIONS_OFFSET_PARTITION_HPP
#define PARTITIONS_OFFSET_PARTITION_HPP

// OFFSET partition -- forward partition with a known all->= prefix.
//
// SPEC.  Given [first, last), a key and an `offset` with the PRECONDITION that
// every element of [first, first + offset) is >= key (!comp(proj(x), key)),
// reorder the range so that
//
//     [first, m)  : proj(x) <  key      (all strictly-below elements, front)
//     [m,   last) : proj(x) >= key
//
// and return the boundary `m` (= first + count of below-key elements).  Only
// the suffix [first + offset, last) ever needs a comparison: the prefix
// elements are known to belong on the right, so they are pure swap *targets*.
// offset == 0 degenerates to the ordinary forward partition; a below-count
// c > offset exhausts the prefix and the remainder is a normal partition of
// what is left -- both fall out of every algorithm here without a special
// case.
//
// The variants (all return the boundary iterator; the convenience entry point
// `partitions::offset_partition` returns the COUNT and size-dispatches):
//
//   * scan_swap_off : trivial branchy Lomuto-from-offset.  Store cursor at
//     `first`, scan cursor at `first + offset` -- the precondition is exactly
//     Lomuto's invariant ([store, cur) all >= key) after `offset` virtual
//     iterations.  One ~50%-mispredicted branch per suffix element on random
//     data; kept as the readable baseline.
//
//   * gap_off : branchless Lomuto (orlp gap method, algo::lomuto_branchless)
//     started mid-stream: lift v[offset] into the temporary, scan j from
//     `offset`, store index i from 0.  The gap invariant ("[0,i) below,
//     [i,j) not-below, gap at j") is established by the precondition instead
//     of by iterating, so the prefix costs literally nothing.  2 moves and
//     1 compare per suffix element, no data-dependent branch.
//
//   * part_swap_off : partition ONLY the suffix with the repo's best forward
//     partitioner (algo::sized), then ONE block swap_ranges moves the below
//     block to the front -- min(c, offset) sequential, branch-free, vectorised
//     16-byte swaps.  Elements can move twice, but both passes are optimal.
//
//   * fused_block_off : the pdqsort branchless block partition
//     (algo::detail::branchless_partition) over the WHOLE range, with one
//     change: left offset blocks that lie entirely inside the known prefix
//     are filled with a compare-free identity pattern.  KEPT AS A MEASURED
//     NEGATIVE RESULT: the left cursor of a forward block partition only
//     advances to the boundary c, so when c < offset the identity fill never
//     fires past c and the RIGHT scan walks down through the prefix COMPARING
//     it from the wrong side -- at f = offset/n = 0.9 it is as slow as
//     ignoring the precondition entirely (i64 2^22 p=.1: 3.75 vs `whole`
//     3.73 ns/suf-elem, vs prefix_fill 0.45).
//
//   * prefix_fill_off : the offset-aware algorithm proper -- see its comment.
//
// FINDINGS (Meteor Lake, GCC, -O3 -march=native, random high-cardinality
// suffix, ns per SUFFIX element = min ns / (n - offset); f = offset/n,
// p = below fraction of the suffix; full sweep in
// benchmarks/bench_offset_partition.cpp, study in docs/offset_partition.md):
//
//   * prefix_fill is the best single choice at large n: i64 2^22 it wins
//     every p at f=.1 (0.45-0.53) and the p=.1 cells at any f (0.45 vs
//     part_swap 0.50); 16-byte types it wins nearly everywhere (pair64 2^22
//     f=.5 p=.5: 1.03 vs part_swap 1.33, gap 1.06, whole 1.46).  vs `whole`
//     (ignore the precondition) the win reaches 8x at f=.9 (0.45 vs 3.73).
//   * gap_off is p-FLAT (1 compare + 2 sequential moves regardless): it wins
//     the high-p cells (i64 2^22 f=.9 p=.9: 0.61 vs prefix_fill 0.79) and,
//     averaged over p, narrow-type cells with a dominating prefix -- hence
//     the dispatcher's narrow-type offset >= suffix rule.  For 16-byte
//     elements its two wide moves per element are never competitive (4-6
//     ns/suf-elem on pair64).
//   * part_swap ties prefix_fill at low p / low f (its epilogue is
//     min(c, offset) AVX-vectorised swaps -- bounded by the SMALLER of the
//     two), and is the piece prefix_fill's phase 2 reuses; standalone it
//     loses up to 1.5x where both f and p are mid/high (the double move).
//   * scan_swap mispredicts ~once per suffix element on random data (3-6x
//     slower at p=.5) but is fastest when the predicate is PREDICTABLE
//     (f=.9 p=.9 cache-resident: 0.44 vs 0.49 gap) -- same caveat as
//     algo::boost_block's low-cardinality note; not worth a special path.
//
// Like algo_rev, these do NOT model the 5-argument PivotPartitioner concept,
// so they are not in default_partitioners(); they have their own test suite
// (tests/test_offset_partition.cpp) and benchmark.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <utility>

#include "algorithms.hpp"  // algo::sized, algo::detail (block tunables, swap_offsets)

namespace partitions::algo_off {

// ---------------------------------------------------------------------------
// Branchy Lomuto-from-offset -- the baseline.  [first, store) < key,
// [store, cur) >= key holds initially with store = first, cur = first+offset
// by the precondition.
// ---------------------------------------------------------------------------
struct scan_swap_off {
    static constexpr const char* name = "scan_swap";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last_s, std::iter_difference_t<I> offset, K key,
                 Comp comp, Proj proj) const {
        I last = first + (last_s - first);
        I store = first;
        for (I cur = first + offset; cur != last; ++cur) {
            if (static_cast<bool>(
                    std::invoke(comp, std::invoke(proj, *cur), key))) {
                std::iter_swap(store, cur);
                ++store;
            }
        }
        return store;
    }
};

// ---------------------------------------------------------------------------
// Branchless Lomuto gap method started at `offset`.  algo::lomuto_branchless
// with the warm-up replaced by the precondition: lifting v[offset] opens the
// gap exactly in the state the standard loop would reach after `offset`
// iterations that found nothing below (i = 0, [0, offset) not-below).
// ---------------------------------------------------------------------------
struct gap_off {
    static constexpr const char* name = "gap_off";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last_s, std::iter_difference_t<I> offset, K key,
                 Comp comp, Proj proj) const {
        using D = std::iter_difference_t<I>;
        const D n = last_s - first;
        if (offset >= n) return first;  // empty suffix: no below elements exist
        auto below = [&](D idx) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, first[idx]), key));
        };

        auto v = first;
        std::iter_value_t<I> tmp = std::move(v[offset]);
        D i = 0;
        for (D j = offset; j < n - 1; ++j) {
            v[j] = std::move(v[i]);
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
// Partition the suffix, then block-move the below run to the front.
//
// After algo::sized on [first+offset, last) the layout is
//     [ >=key : offset ][ <key : c ][ >=key ]
// and ONE swap_ranges of min(c, offset) elements yields the contract:
//   c <= offset: swap the below run with the FIRST c prefix slots;
//   c >  offset: swap the whole prefix with the LAST offset elements of the
//                below run (the runs are disjoint since c > offset).
// Both arms are two sequential streams -- branch-free and vectorised.
// ---------------------------------------------------------------------------
struct part_swap_off {
    static constexpr const char* name = "part_swap";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last_s, std::iter_difference_t<I> offset, K key,
                 Comp comp, Proj proj) const {
        I last = first + (last_s - first);
        I base = first + offset;
        I m = algo::sized{}(base, last, key, comp, proj);  // [base, m) < key
        const auto c = m - base;
        if (c <= offset)
            std::swap_ranges(first, first + c, base);
        else
            std::swap_ranges(first, base, m - offset);
        return first + c;
    }
};

// ---------------------------------------------------------------------------
// Prefix-aware pdqsort branchless block partition.
//
// Identical to algo::detail::branchless_partition except for the left fill:
// while the left cursor is inside [begin, pfx_end) the elements are known
// not-below by PRECONDITION, so the offset block is filled with the identity
// pattern 0..split-1 and num_l advances unconditionally -- no element access,
// no comparison; GCC vectorises the fill into plain byte stores.  A block
// straddling pfx_end (at most one per call) is split into an identity part
// and a compared part.  Everything else -- right fill, swap_offsets rotation,
// drains -- is byte-for-byte the forward algorithm's.
// ---------------------------------------------------------------------------
namespace detail {

using algo::detail::align_cacheline;
using algo::detail::boost_block_size;
using algo::detail::boost_cacheline_size;
using algo::detail::swap_offsets;

// Suffix lengths below this go to scan_swap_off: same rationale (and value
// shape) as algo::detail::boost_scalar_cutoff -- the offset-buffer setup does
// not amortise over a handful of compared elements.
inline constexpr std::ptrdiff_t fused_scalar_cutoff = 16;

template <class Iter, class K, class Compare, class Proj>
inline Iter branchless_partition_off(Iter begin, Iter end, Iter pfx_end,
                                     K key, Compare comp, Proj proj) {
    auto below = [&](Iter it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), key));
    };

    Iter first = begin;
    Iter last = end;

    // Initial left scan (skip elements already on the correct side).  With a
    // non-empty prefix *begin is >= key by precondition and stops the scan at
    // begin immediately, so the scan only runs in the offset == 0 case.
    if (first == pfx_end) {
        while (first < last && below(first)) ++first;
    }

    // From the right, find the last below element.  `first == begin` (always
    // true when the prefix is non-empty) means no below element guards the
    // descent, so keep the bound.
    if (first == begin) {
        while (begin < last && !below(--last)) {
        }
    } else {
        while (!below(--last)) {
        }
    }

    if (first < last) {
        std::iter_swap(first, last);
        ++first;
        // The swap wrote a below element over the first prefix slot; the
        // remaining prefix [first, pfx_end) is untouched and still all >= key.

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

            // Left fill: positions of elements that belong right (!below).
            // Inside the prefix EVERY position qualifies by precondition, so
            // the fill is the identity pattern with no compare.  num_l == 0
            // here (left_split > 0 only then), so writes start at offset 0.
            if (left_split >= boost_block_size) {
                const std::size_t pfx =
                    first < pfx_end ? std::min<std::size_t>(
                                          boost_block_size,
                                          static_cast<std::size_t>(pfx_end - first))
                                    : 0;
                if (pfx == boost_block_size) {  // whole block in prefix
                    for (std::size_t i = 0; i < boost_block_size; ++i)
                        offsets_l[i] = static_cast<unsigned char>(i);
                    num_l = boost_block_size;
                    first += boost_block_size;
                } else if (pfx == 0) {  // whole block past prefix: forward fill
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
                } else {  // straddling block: at most once per call
                    std::size_t i = 0;
                    for (; i < pfx;) {
                        offsets_l[num_l++] = static_cast<unsigned char>(i++); ++first;
                    }
                    for (; i < boost_block_size;) {
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    }
                }
            } else {
                const std::size_t pfx =
                    first < pfx_end ? std::min<std::size_t>(
                                          left_split,
                                          static_cast<std::size_t>(pfx_end - first))
                                    : 0;
                std::size_t i = 0;
                for (; i < pfx;) {
                    offsets_l[num_l++] = static_cast<unsigned char>(i++); ++first;
                }
                for (; i < left_split;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                }
            }

            // Right fill: positions of elements that belong left (below).
            // Unchanged from the forward algorithm.
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
    return first;  // [begin, first) < key; [first, end) >= key
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Prefix-fill -- the offset-aware algorithm proper.
//
// INSIGHT (from measurement of fused_block_off, kept as a negative result):
// in a forward block partition the LEFT cursor only advances as far as the
// boundary c, so when c < offset the identity-fill optimisation above touches
// almost nothing -- the bulk of the traversal is the RIGHT scan, which then
// walks down through the prefix COMPARING it from the other side
// (fused_block ~= whole at f=0.9).  The prefix must be treated as what it is:
// a supply of swap TARGETS, not scan input.
//
// Phase 1: scan the suffix from the right in branchless blocks (the pdqsort
// offset fill: unconditional byte write + setcc add, 8x unrolled); each found
// below element is swapped DIRECTLY into the next front slot.  The left side
// of the pairing is consecutive (lo, lo+1, ...), so a constant identity
// offset table feeds the same 2-moves-per-element `swap_offsets` rotation the
// block partition uses -- no compare ever touches the prefix, and each below
// element is moved exactly once.  The consumed right block becomes all->= and
// is settled.
//
// Phase 2 (slots nearly exhausted -- c approaching offset -- or suffix
// exhausted): finish with the part_swap scheme on the remaining window:
// algo::sized on [pfx_end, last), then one swap_ranges bridges the leftover
// [lo, pfx_end) gap.  Loop granularity guarantees phase 1 only runs whole
// blocks, so no element is ever rescanned.
//
// Net cost: exactly (n - offset) compares; min(c, offset)-ish elements moved
// once (rotation), the rest as part_swap's optimal remainder.
// ---------------------------------------------------------------------------
namespace detail {

// Move the `num` below elements recorded in the right offset buffer into the
// consecutive slots [lo, lo + num), displacing the slot contents to the
// vacated right positions.  This is swap_offsets' use_swaps=false cyclic
// rotation (one temporary, two moves per element) specialised to a
// CONSECUTIVE left side: indexing lo[i] directly instead of through an
// identity offset table saves a byte load and an address dependency per
// element (the generic form's `movzbl identity[i]` survived in the
// disassembly -- GCC does not fold a constant identity table).
template <class Iter>
[[gnu::always_inline]] inline void fill_slots(Iter lo, Iter r_base,
                                              const unsigned char* offsets_r,
                                              std::size_t num) {
    using T = typename std::iterator_traits<Iter>::value_type;
    if (num == 0) return;
    Iter r = r_base - offsets_r[0];
    T tmp(std::move(*lo));
    *lo = std::move(*r);
    for (std::size_t i = 1; i < num; ++i) {
        *r = std::move(lo[static_cast<std::ptrdiff_t>(i)]);
        Iter r2 = r_base - offsets_r[i];
        lo[static_cast<std::ptrdiff_t>(i)] = std::move(*r2);
        r = r2;
    }
    *r = std::move(tmp);
}

}  // namespace detail

struct prefix_fill_off {
    static constexpr const char* name = "prefix_fill";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last_s, std::iter_difference_t<I> offset, K key,
                 Comp comp, Proj proj) const {
        using D = std::iter_difference_t<I>;
        constexpr D B = static_cast<D>(detail::boost_block_size);
        I last = first + (last_s - first);
        I lo = first;             // next front slot to fill (always < pfx_end)
        I pfx_end = first + offset;
        auto below = [&](I it) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, *it), key));
        };

        alignas(detail::boost_cacheline_size)
            unsigned char offsets_r_storage[detail::boost_block_size +
                                            detail::boost_cacheline_size];
        unsigned char* offsets_r = detail::align_cacheline(offsets_r_storage);

        // Phase 1: whole right blocks while both slots and suffix last.
        while (pfx_end - lo >= B && last - pfx_end >= B) {
            std::size_t num_r = 0;
            for (std::size_t i = 0; i < detail::boost_block_size;) {
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            }
            // Pair the found belows with consecutive front slots: a
            // 2-moves-per-element cyclic rotation (see fill_slots).
            detail::fill_slots(lo, last + B, offsets_r, num_r);
            lo += static_cast<D>(num_r);
        }

        // Phase 2: partition what is left of the suffix, then bridge the
        // remaining [lo, pfx_end) gap with one block swap (cf. part_swap).
        I m = algo::sized{}(pfx_end, last, key, comp, proj);  // [pfx_end, m) < key
        const D c_rem = m - pfx_end;
        const D slots = pfx_end - lo;
        if (c_rem <= slots)
            std::swap_ranges(lo, lo + c_rem, pfx_end);
        else
            std::swap_ranges(lo, pfx_end, m - slots);
        return lo + c_rem;
    }
};

struct fused_block_off {
    static constexpr const char* name = "fused_block";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last_s, std::iter_difference_t<I> offset, K key,
                 Comp comp, Proj proj) const {
        I last = first + (last_s - first);
        if ((last - first) - offset < detail::fused_scalar_cutoff)
            return scan_swap_off{}(first, last, offset, key, comp, proj);
        return detail::branchless_partition_off(first, last, first + offset,
                                                key, comp, proj);
    }
};

// ---------------------------------------------------------------------------
// Size-dispatching variant -- mirror of algo::sized, thresholds from the
// bench_offset_partition sweep:
//
//   * tiny suffix (<= algo::detail::sized_cutoff<T>: 512 narrow / 24 wide):
//     gap_off -- no setup, branch-free, same trade as algo::sized's Lomuto
//     fast path, over the COMPARED length (the suffix);
//   * narrow T (<= 8 bytes) with offset >= suffix: gap_off -- its flat
//     2-moves-per-element stream beats prefix_fill's per-below scattered
//     rotation ON AVERAGE OVER p when the prefix dominates (i64 2^22 f=0.5:
//     0.58 vs 0.62 ns/suf-elem averaged over p in {.1,.5,.9}; the rule loses
//     only the p=0.1 cell, 0.52 vs 0.45, and wins p>=0.5 by 10-25%).  p is
//     unknowable at call time, so this is an expected-cost choice;
//   * everything else: prefix_fill (for 16-byte elements gap_off's double
//     16-byte moves are 4-6 ns/elem -- never competitive above the cutoff).
// ---------------------------------------------------------------------------
struct sized_off {
    static constexpr const char* name = "sized_off";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last_s, std::iter_difference_t<I> offset, K key,
                 Comp comp, Proj proj) const {
        I last = first + (last_s - first);
        const auto suffix = (last - first) - offset;
        if (suffix <= algo::detail::sized_cutoff<std::iter_value_t<I>>)
            return gap_off{}(first, last, offset, key, comp, proj);
        if constexpr (sizeof(std::iter_value_t<I>) <= 8) {
            if (offset >= suffix)
                return gap_off{}(first, last, offset, key, comp, proj);
        }
        return prefix_fill_off{}(first, last, offset, key, comp, proj);
    }
};

}  // namespace partitions::algo_off

namespace partitions {

// Convenience entry point: returns the COUNT of below-key elements (now at
// the front).  Precondition: [first, first + offset) all >= key.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t offset_partition(It first, It last,
                                std::iter_difference_t<It> offset, K key,
                                Comp comp = {}, Proj proj = {}) {
    return algo_off::sized_off{}(first, last, offset, key, comp, proj) - first;
}

}  // namespace partitions

#endif  // PARTITIONS_OFFSET_PARTITION_HPP
