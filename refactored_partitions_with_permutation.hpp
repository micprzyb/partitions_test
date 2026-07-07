#ifndef REFACTORED_PARTITIONS_WITH_PERMUTATION_HPP
#define REFACTORED_PARTITIONS_WITH_PERMUTATION_HPP

// refactored_partitions_with_permutation.hpp -- partition and offset-partition
// that maintain an inverse-position ("permutation") array through the reorder.
// Builds on refactored_partitions.hpp (same directory); requires C++20.
//
// PROBLEM.  The caller owns, next to the data array, a `permutation` array and
// an `id` map with the invariant
//
//     permutation[id(element)] == the element's current position
//                                 (an iterator, or an index into the array)
//
// i.e. `permutation` is the INVERSE of the layout: given an element's identity
// (a stable key carried by the element itself, returned by `id`), it tells
// where that element currently lives.  A partition permutes the elements, so
// it must also update `permutation` to keep the invariant.
//
// PUBLIC ENTRY POINTS (namespace partitions):
//
//   * sized_partition_perm(first, last, key, perm, id, comp, proj) -> iterator
//         Forward partition around `key` ([first,m) < key <= [m,last)), same
//         contract as sized_partition, preserving the permutation invariant.
//   * offset_partition_perm(first, last, offset, key, perm, id, comp, proj)
//         -> count.  Offset partition (PRECONDITION: [first, first+offset)
//         all >= key), same contract as offset_partition, preserving the
//         invariant.
//
//   `perm` is a random-access iterator to the permutation array.  Its value
//   type is either
//     - the data iterator type itself (entries are iterators), or
//     - any integral type (entries are indices).
//   For integral entries the index origin is DERIVED from the invariant
//   itself before partitioning (origin = first - perm[id(*first)]), so
//   subranges of a larger array work without extra parameters.  `id` receives
//   a const element reference and must return its permutation slot, unique
//   per element.  `comp` must be a strict weak ordering (dispatch may sort
//   tiny blocks).
//
// THE TWO MAINTENANCE STRATEGIES (both implemented; the dispatcher picks the
// measured winner -- see the findings block below):
//
//   * REPAIR: partition at full, unmodified speed, then one pass
//         for p in [first,last): perm[id(*p)] = entry(p)
//     Cost: n sequential element loads + n *scattered* stores into perm,
//     regardless of how many elements actually moved.
//
//   * MIRROR: the swap-based kernels know exactly which positions they
//     relocated (the pdqsort block partition records them in its offset
//     buffers), so after each block's swaps are applied, write the entries of
//     just the MOVED positions:
//         for each moved position q:  perm[id(*q)] = entry(q)
//     That is ~2*p*(1-p)*n scattered stores for a below-fraction p -- at most
//     half of repair's traffic at p = 1/2 and almost none for skewed splits.
//     The branchless fills themselves are untouched (they only READ
//     elements), so the partition loop keeps its structure; the
//     AVX2-vectorised fill variant mirrors identically.
//
//     (A tempting alternative mirror -- swapping the two perm SLOTS after an
//     element swap, `swap(perm[id(a)], perm[id(b)])`, which would need no
//     entry encoding at all -- is only correct for genuinely pairwise swaps.
//     The block partition's fast path applies its swaps as a cyclic ROTATION
//     (one temporary, 2N moves; its net permutation is a cycle, NOT the
//     pairwise swaps), and the offset kernel's fill_slots likewise, so slot
//     swapping mirrors the wrong permutation there; direct entry writes are
//     also cheaper -- one store per moved element instead of a load + store.)
//
//   Move-based kernels (branchless-Lomuto "gap", AVX2 compress) relocate
//   EVERY element, so mirroring them costs >= n scattered updates anyway --
//   for them repair is the mirror, and the only question is which kernel +
//   repair wins.  (Two more alternatives were analysed and rejected: (1)
//   partitioning the permutation array as an array of handles and then
//   applying the resulting permutation to the data -- turns every comparison
//   into a dependent random load, strictly more scattered traffic than
//   mirror; (2) fusing scattered perm stores into the AVX2 compress loop --
//   saves only the sequential element reload of repair, measured < 5% of the
//   repair pass, not worth the kernel complexity.)
//
// MEASURED FINDINGS (Zen 3 / GCC 16 -O3 -march=native; ns/elem, min over
// reps, batched independent blocks each with its own perm slice; p = below
// fraction; element types: i64 with id = value (AVX2-eligible) and 16-byte
// {key,id} records; i64 entries unless noted):
//
//   * The scattered perm stores dominate beyond cache-resident sizes: the
//     plain partition costs 0.25-0.9 ns/elem, the invariant maintenance
//     0.5-4 ns/elem.  HOW the invariant is maintained matters far more than
//     which partition kernel runs.
//
//   * repair vs mirror is decided by p (i64 / rec at n = 262144):
//         p=0.02:  best repair 1.84/2.49    mirror 0.71/0.89   (2.6-2.8x)
//         p=0.50:  best repair 1.85/2.76    mirror 2.20/3.00
//         p=0.98:  best repair 1.82/2.54    mirror 0.72/0.94   (2.5-2.7x)
//     Mirror's ~2p(1-p)n scattered stores win whenever the split is skewed;
//     at balanced splits repair's clean sequential-load + store stream wins
//     despite touching 2x the entries.
//
//   * Small n (block + perm slice cache-resident): gap-Lomuto + repair wins
//     at every p (n=1024, p=.5: 1.54/2.16 vs mirrored block 1.75/2.23), and
//     probing is not worth its cache pollution below ~2048.
//
//   * Balanced splits on narrow (<= 8-byte) elements: gap-Lomuto + repair
//     keeps winning to ~2^17 (n=16384: 1.65 vs plain+repair 2.06 vs mirror
//     2.06) -- its sequential store stream primes the exact lines the repair
//     pass reloads.  Past L2 the plain dispatcher + repair edges ahead
//     (262144: 1.85 vs 1.99).
//
//   * Very skewed non-SIMD blocks: mirrored HOARE beats the mirrored block
//     partition (records, p=.02/.98, 4096..2^22: 0.86-1.02 vs 1.10-1.22
//     ns/elem) -- few swaps, and its scan branches almost always fall
//     through.  The dispatcher uses it when the probe shows p <= ~10%.
//
//   * The plain dispatcher's AVX2 compress tier is DEAD here: compress moves
//     every element, forcing full repair, and never beats the alternatives
//     (2^20, p=.5: 2.49 vs 2.09 for lomuto+repair).
//
//   * Entry width: i32 entries help at DRAM sizes (i64 data, 2^22, p=.5:
//     dispatcher 1.68 vs 2.54 with i64 entries) -- half the scattered-line
//     footprint.  Both widths (and iterator entries) are supported.
//
//   * UNSIGNED PAIRS (pair<u32,u32> and pair<u64,u64>, id = .second),
//     compared BY FIRST or LEXICOGRAPHICALLY:
//       - std::less<> on the pair (lex) is a short-circuit two-field compare
//         whose second branch is data-dependent: it turns the branchless
//         kernels branchy and loses 2-2.8x at balanced splits (gap-Lomuto,
//         pair<u32,u32>, n=4096, p=.5: 5.91 vs 2.41 ns/elem by-first).  For
//         UNSIGNED members lex order equals the numeric order of the
//         concatenation first:second, so the entry points REWRITE
//         (std::less<>, identity, key) into a wide-compare projection (u64
//         for 4+4-byte pairs; unsigned __int128 for 8+8, GCC/Clang) before
//         dispatching -- verified branchless (cmp+adc / cmp+sbb per element)
//         and the lex penalty collapses to 3-10% (pair<u32,u32>, n=1024,
//         p=.5: 5.51 -> 2.08).  Note key = {k, 0} makes the lex API an exact
//         (and faster) replacement for a by-first partition.
//       - 8-byte pairs are their own dispatch class: the mirrored block
//         partition beats every repair variant from ~192 elements at EVERY
//         below-fraction (p=.5: 1.46-2.88 vs repair 1.87-3.53; the pair
//         field extraction taxes the gap kernel's dependent move chain), so
//         they skip the Lomuto tier and the probe up to 2048 and take the
//         mirrored block partition when balanced.
//       - 16-byte pairs behave like the 16-byte records above: repair when
//         balanced until ~2^20, mirror beyond, mirrored Hoare when very
//         skewed (0.89-0.99 ns/elem at p=.02/.98).
//     Dispatcher end-to-end, by-first / lex (ns/elem): pair<u32,u32> p=.02:
//     0.69-1.93 / 0.70-2.11; p=.5: 1.51-2.99 / 1.70-2.75.  pair<u64,u64>
//     p=.02: 0.92-2.17 / 1.00-2.22; p=.5: 1.63-3.68 / 1.70-3.98.
//
//   * Offset partition: the SUFFIX below-fraction decides.  Low p:
//     prefix-fill + mirror (2^20, offset=.75n, p=.1: 0.68 vs 2.36 for
//     kernel+full-repair -- 3.5x).  Mid/high p: gap + NARROWED repair
//     (repairing only [first, first+min(c+1,offset)) and the suffix -- the
//     only ranges the gap kernel can touch) wins or ties everywhere
//     (offset=.75n, p=.5/.9: 1.22-1.64 vs 1.75-2.54).
//
// ASSEMBLY NOTES (verified on the generated code): the repair loop compiles
// to 6 instructions/element (GCC narrows the record load to the 8-byte id
// field and folds the entry origin into address arithmetic: load, lea+sar,
// scattered store, loop); the mirror's entry-write loop is 4 instructions
// per moved element; the probe loop is branchless (cmp+setg+add per sample)
// and its two skew thresholds compile to two unsigned range checks.
//
// DISPATCH POLICY (implemented below; when n > 2048 a branchless pass of 64
// strided probes estimates p first -- <= 2% cost at the sizes where it runs):
//
//   sized_partition_perm:
//     unsigned pair + std::less<> + identity -> wide-compare rewrite first
//     8-byte pairs: n <= 192 -> gap-Lomuto + repair; n <= 2048 -> mirrored
//                   block partition (no probe)
//     others:       n <= 2048 -> gap-Lomuto + repair
//     p^ outside [1/4, 3/4]        -> mirrored block partition (AVX2 fills
//                                     when eligible; very skewed non-SIMD
//                                     -> mirrored Hoare)
//     else (balanced):
//       8-byte pair                -> mirrored block partition
//       sizeof(T) <= 8, n <= 2^17  -> gap-Lomuto + repair
//       sizeof(T) >  8, n >= 2^21  -> mirrored block partition
//       otherwise                  -> plain sized_partition + full repair
//
//   offset_partition_perm:
//     suffix <= 2048               -> gap_off + narrowed repair
//     suffix p^ <= 1/4             -> prefix-fill + mirror
//     else                         -> gap_off + narrowed repair
//
// Dispatcher end-to-end (i64/rec, ns/elem): within ~6% of the per-cell
// winner across the whole grid -- e.g. n=262144: 0.75/0.89 at p=.02,
// 1.88/2.72 at p=.5, 0.76/0.89 at p=.98; n=2^22, p=.5: 2.61/3.81.
//
// The dispatcher below implements exactly the measured policy.

#include "refactored_partitions.hpp"

#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER) && !defined(__clang__)
#define PARTITIONS_PERM_ALWAYS_INLINE __forceinline
#else
#define PARTITIONS_PERM_ALWAYS_INLINE [[gnu::always_inline]] inline
#endif

namespace partitions::detail::perm {

// ---------------------------------------------------------------------------
// Lexicographic unsigned-pair rewrite.  std::less<> on a std::pair is a
// short-circuit two-field compare; GCC lowers the below-key predicate built
// from it to a data-dependent BRANCH (verified in the disassembly of the
// gap-Lomuto loop: cmp/je + cmp/jae per element), which mispredicts on
// balanced splits -- measured 2.8x slower than the by-first projection at
// p = 0.5 on pair<u32,u32>.  For UNSIGNED members, lexicographic order over
// (first, second) is EXACTLY the numeric order of the concatenation
// first:second, so the dispatcher rewrites (std::less<>, identity, key) into
// (std::less<>, concat-projection, concat(key)) before running: one wide
// unsigned compare per element, branchless in every kernel, zero kernel
// changes.  pairs up to 4+4 bytes concatenate into u64; 8+8 into unsigned
// __int128 (GCC/Clang; elsewhere the rewrite simply does not fire).
// ---------------------------------------------------------------------------
template <class T>
concept unsigned_pair = std::is_class_v<T> && requires(T t) {
    t.first;
    t.second;
} && std::is_unsigned_v<decltype(std::declval<T&>().first)> &&
    std::is_unsigned_v<decltype(std::declval<T&>().second)> &&
    sizeof(T) == sizeof(std::declval<T&>().first) +
                     sizeof(std::declval<T&>().second);

#if defined(__GNUC__) || defined(__clang__)
#define PARTITIONS_PERM_HAS_U128 1
__extension__ typedef unsigned __int128 u128;
#else
#define PARTITIONS_PERM_HAS_U128 0
#endif

// NOTE: the member accesses live in constrained partial specialisations, not
// in the primary template: a variable-template initialiser is not a SFINAE
// context, so `unsigned_pair<T> && sizeof(declval<T&>().first) <= 4` would
// HARD-ERROR for any non-pair T even though && short-circuits its value.
template <class T>
inline constexpr bool lex_rewrite_64 = false;
template <unsigned_pair T>
inline constexpr bool lex_rewrite_64<T> =
    sizeof(std::declval<T&>().first) <= 4 &&
    sizeof(std::declval<T&>().second) <= 4;

template <class T>
inline constexpr bool lex_rewrite_128 = false;
#if PARTITIONS_PERM_HAS_U128
template <unsigned_pair T>
inline constexpr bool lex_rewrite_128<T> =
    !lex_rewrite_64<T> && sizeof(std::declval<T&>().first) <= 8 &&
    sizeof(std::declval<T&>().second) <= 8;
#endif

struct lex_concat64 {
    template <class T>
    PARTITIONS_PERM_ALWAYS_INLINE std::uint64_t operator()(const T& x) const {
        constexpr unsigned sb = 8 * sizeof(x.second);
        return (static_cast<std::uint64_t>(x.first) << sb) |
               static_cast<std::uint64_t>(x.second);
    }
};
#if PARTITIONS_PERM_HAS_U128
struct lex_concat128 {
    template <class T>
    PARTITIONS_PERM_ALWAYS_INLINE u128 operator()(const T& x) const {
        constexpr unsigned sb = 8 * sizeof(x.second);
        return (static_cast<u128>(x.first) << sb) |
               static_cast<u128>(x.second);
    }
};
#endif

template <class T, class Comp, class Proj>
inline constexpr bool lex_rewritable =
    (std::is_same_v<Comp, std::less<>> || std::is_same_v<Comp, std::less<T>>)&&
    std::is_same_v<Proj, std::identity> &&
    (lex_rewrite_64<T> || lex_rewrite_128<T>);

// ---------------------------------------------------------------------------
// Entry representation.  Entries are either the data iterator itself or an
// integral index; for indices the origin is recovered from the invariant
// (must be constructed BEFORE the elements move).
// ---------------------------------------------------------------------------
template <class I, class PermIt>
struct entry_maker {
    using P = std::iter_value_t<PermIt>;
    static constexpr bool integral = std::is_integral_v<P>;
    I base{};

    PARTITIONS_PERM_ALWAYS_INLINE P operator()(I pos) const {
        if constexpr (integral)
            return static_cast<P>(pos - base);
        else
            return pos;
    }
};

template <class I, class PermIt, class Id>
PARTITIONS_PERM_ALWAYS_INLINE entry_maker<I, PermIt> make_entry(I first, I last,
                                                                PermIt perm,
                                                                Id& id) {
    entry_maker<I, PermIt> em{};
    if constexpr (entry_maker<I, PermIt>::integral) {
        if (first != last)
            em.base = first - static_cast<std::iter_difference_t<I>>(
                                  perm[std::invoke(id, *first)]);
    }
    (void)perm;
    (void)id;
    (void)last;
    return em;
}

// ---------------------------------------------------------------------------
// Invariant-maintenance primitives.
// ---------------------------------------------------------------------------

// Write the entry for position p (the element must already be in place):
// one id extraction + one scattered store.  This is the mirror primitive.
//
// (A representation-agnostic alternative -- swapping the two perm slots after
// an element swap, `swap(perm[id(a)], perm[id(b)])`, needing no entry
// construction at all -- works ONLY for genuinely pairwise swaps.  The block
// partition's fast path applies its swaps as a cyclic ROTATION (one
// temporary, 2N moves), whose net permutation is a cycle, not the pairwise
// swaps; mirroring it with slot swaps writes wrong entries.  Direct entry
// writes are also cheaper: 1 store per moved element instead of a
// load+store, so they are used everywhere.)
template <class I, class PermIt, class Id, class EM>
PARTITIONS_PERM_ALWAYS_INLINE void fix_at(PermIt perm, Id& id, const EM& em,
                                          I p) {
    perm[std::invoke(id, *p)] = em(p);
}

// Rebuild entries for every position in [first, last): one sequential element
// load + one scattered perm store per position.
template <class I, class PermIt, class Id, class EM>
inline void repair(I first, I last, PermIt perm, Id& id, const EM& em) {
    for (I p = first; p != last; ++p) perm[std::invoke(id, *p)] = em(p);
}

// ---------------------------------------------------------------------------
// Mirrored swap application for the block partition: identical to
// scalar::swap_offsets, plus the entry slot-swap per swapped pair.  The
// cyclic-rotation arm has the same net effect as the pairwise swaps, so its
// mirror is a separate slot-swap pass over the (L1-hot) swapped positions.
// ---------------------------------------------------------------------------
template <class Iter, class PermIt, class Id, class EM>
PARTITIONS_PERM_ALWAYS_INLINE void swap_offsets_perm(
    Iter first, Iter last, const unsigned char* offsets_l,
    const unsigned char* offsets_r, std::size_t num, bool use_swaps,
    PermIt perm, Id& id, const EM& em) {
    using T = typename std::iterator_traits<Iter>::value_type;
    if (use_swaps) {
        for (std::size_t i = 0; i < num; ++i) {
            Iter l = first + offsets_l[i];
            Iter r = last - offsets_r[i];
            std::iter_swap(l, r);
            fix_at(perm, id, em, l);
            fix_at(perm, id, em, r);
        }
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
        // The rotation's net permutation is a CYCLE over the L/R positions
        // (not the pairwise swaps); write the entries of every moved
        // position directly.
        for (std::size_t i = 0; i < num; ++i) {
            fix_at(perm, id, em, first + offsets_l[i]);
            fix_at(perm, id, em, last - offsets_r[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// FULCRUM partition with FUSED permutation writes.
//
// The fulcrum scheme (Peters / van den Hoven; the scalar ancestor of the AVX2
// compress kernel) buys two properties no swap- or gap-based kernel has:
//
//   * STRUCTURALLY branchless data movement: every element is written
//     unconditionally to BOTH write cursors (the losing copy lands in the
//     open gap and is later overwritten); the predicate feeds only integer
//     cursor arithmetic (l += b; r -= !b).  There is no value select and no
//     conditional store for the compiler to re-branch -- the recurring
//     if-conversion hazard has no attack surface in the mover itself (the
//     below-PREDICATE can still be branchy for short-circuit comparators;
//     the entry points' unsigned-pair rewrite handles that).
//
//   * every element's FINAL position is known at the instant it is placed
//     (the winning cursor), so the permutation entry can be written IN the
//     same pass -- fusing what repair does in a second sweep.  No other
//     kernel here can fuse: Lomuto moves elements repeatedly, the block
//     partition knows final positions only per swapped pair.
//
// Cost per element: 1 sequential load + 2 sequential stores + 1 scattered
// perm store, ONE pass over the data (lomuto+repair streams it twice).
// Requires n >= 2*fulcrum_buf.
//
// MEASURED VERDICT (Zen 3): kept as a reference kernel, NOT dispatched.  With
// chunked reads it reaches within ~15-40% of the adaptive dispatcher at
// balanced splits on <= 8-byte elements (i64 n=65536, p=.5: 2.87 vs 1.80;
// it even wins one cell, 16-byte records at n=4096, p=.5: 2.21 vs 2.41), but
// it can never compete at skewed splits (it relocates every element where
// mirror touches ~2p(1-p)n: 2.8-5.7 vs 0.7-1.0 at p=.02), 16-byte pairs pay
// double 16-byte stores (5.2-7.3 vs 2.5-3.8), and past L3 its bidirectional
// read + three write streams defeat the prefetchers (2^22: 5.3 vs 2.5).  The
// single-pass fusion advantage never outweighs losing the O(moved-only)
// option.  Structurally it remains the most if-conversion-robust mover: the
// predicate feeds only cursor arithmetic, there is no value select to
// re-branch (see check_branchless.py).
// ---------------------------------------------------------------------------
// Buffer per side and read-chunk size.  Chunked reads are crumsort's trick:
// a PER-ELEMENT read-side selection is a data-dependent branch that follows
// the below pattern and mispredicts at balanced splits (measured 2x); one
// side decision per chunk of 16 amortises it away.  Safety: slack(L)+slack(R)
// is invariant 2*fulcrum_buf = 4 chunks, and a chunk is read from a side with
// slack < chunk only if the other has >= 3 chunks, so both slacks stay >= 1
// at every write.
inline constexpr std::ptrdiff_t fulcrum_buf = 32;
inline constexpr std::ptrdiff_t fulcrum_chunk = 16;

template <std::random_access_iterator I, class K, class PermIt, class Id,
          class EM, class Comp, class Proj>
inline I fulcrum_perm(I first, I last, K key, PermIt perm, Id& id,
                      const EM& em, Comp comp, Proj proj) {
    using T = std::iter_value_t<I>;
    using D = std::iter_difference_t<I>;
    constexpr D B = fulcrum_buf;
    constexpr D C = fulcrum_chunk;
    T tmp[2 * B];
    for (D k = 0; k < B; ++k) tmp[k] = first[k];
    for (D k = 0; k < B; ++k) tmp[B + k] = last[k - B];

    I l = first, r = last;        // write cursors (belows at l, aboves at --r)
    I i = first + B, j = last - B;  // read cursors over the unread middle
    auto place = [&](const T& v) {
        const bool b = static_cast<bool>(
            std::invoke(comp, std::invoke(proj, v), key));
        *l = v;
        *(r - 1) = v;
        I pos = b ? l : r - 1;    // final resting place, known NOW
        perm[std::invoke(id, v)] = em(pos);
        l += static_cast<D>(b);
        r -= static_cast<D>(!b);
    };
    // Chunk reads: LEFT chunks are consumed ascending (the ascending below
    // writes trail the reads), RIGHT chunks DESCENDING (the descending above
    // writes trail the reads) -- consuming a right chunk bottom-up would let
    // an above write at r-1 clobber a not-yet-read element of the chunk.
    while (j - i >= C) {
        bool from_left;
        if (r - j < C)
            from_left = false;    // right gap low -> replenish from the right
        else if (i - l < C)
            from_left = true;     // left gap low -> replenish from the left
        else
            from_left = i - l <= r - j;  // both fine: smaller-slack side
        if (from_left) {
            const I src = i;
            i += C;
            for (D k = 0; k < C; ++k) place(src[k]);
        } else {
            j -= C;
            for (D k = C; k-- > 0;) place(j[k]);
        }
    }
    while (i < j) {               // < C leftovers: per-element side rule
        T v;
        if (i - l <= r - j) {
            v = *i;
            ++i;
        } else {
            --j;
            v = *j;
        }
        place(v);
    }
    for (D k = 0; k < 2 * B; ++k) place(tmp[k]);
    return l;  // l == r: the boundary
}

// ---------------------------------------------------------------------------
// Hoare partition with mirrored swaps (the swap-based scalar small kernel).
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class PermIt, class Id, class EM, class Comp, class Proj>
inline I hoare_perm(I first, S last, K pivot, PermIt perm, Id& id,
                    const EM& em, Comp comp, Proj proj) {
    auto below = [&](I it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };
    I lo = first;
    I hi = first + (last - first);
    while (true) {
        while (lo != hi && below(lo)) ++lo;
        do {
            if (lo == hi) return lo;
            --hi;
        } while (!below(hi));
        std::iter_swap(lo, hi);
        fix_at(perm, id, em, lo);
        fix_at(perm, id, em, hi);
        ++lo;
    }
}

// ---------------------------------------------------------------------------
// Block partition with mirrored swaps.  Identical control flow and fills as
// scalar::branchless_partition / avx2::branchless_partition_amd (the fills
// only READ elements, so they need no mirroring and keep their branchless /
// vectorised form); every swap application is mirrored.  UseSimd vectorises
// the whole-block fills on the AVX2 fast-path types.
// ---------------------------------------------------------------------------
template <bool Guarded, bool UseSimd, std::size_t BS, class Iter, class K,
          class PermIt, class Id, class EM, class Compare, class Proj>
inline Iter branchless_partition_perm(Iter begin, Iter end, K pivot,
                                      PermIt perm, Id& id, const EM& em,
                                      Compare comp, Proj proj) {
    using T [[maybe_unused]] = std::iter_value_t<Iter>;
    auto below = [&](Iter it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };
#if defined(__AVX2__)
    constexpr bool kSimd = UseSimd && std::contiguous_iterator<Iter> &&
                           avx2::simd_eligible<T, Compare, Proj>;
#else
    constexpr bool kSimd = false;
#endif

    Iter first = begin, last = end;
    if constexpr (Guarded) {
        while (first < last && below(first)) ++first;
    } else {
        while (below(first)) ++first;
    }
    if (first == begin) {
        while (begin < last && !below(--last)) {
        }
    } else {
        while (!below(--last)) {
        }
    }

    if (first < last) {
        std::iter_swap(first, last);
        fix_at(perm, id, em, first);
        fix_at(perm, id, em, last);
        ++first;

        alignas(tuning::cacheline_bytes)
            unsigned char off_l_store[BS + 2 * tuning::cacheline_bytes];
        alignas(tuning::cacheline_bytes)
            unsigned char off_r_store[BS + 2 * tuning::cacheline_bytes];
        unsigned char* offsets_l = scalar::align_cacheline(off_l_store);
        unsigned char* offsets_r = scalar::align_cacheline(off_r_store);

        Iter offsets_l_base = first;
        Iter offsets_r_base = last;
        std::size_t num_l = 0, num_r = 0, start_l = 0, start_r = 0;

        while (first < last) {
            std::size_t num_unknown = static_cast<std::size_t>(last - first);
            std::size_t left_split =
                num_l == 0 ? (num_r == 0 ? num_unknown / 2 : num_unknown) : 0;
            std::size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

            if (left_split >= BS) {
                if constexpr (kSimd) {
#if defined(__AVX2__)
                    avx2::vfill_left<T>(&*first, pivot, offsets_l, num_l, BS);
                    first += BS;
#endif
                } else {
                    for (std::size_t i = 0; i < BS;) {
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                        offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    }
                }
            } else {
                for (std::size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                }
            }

            if (right_split >= BS) {
                if constexpr (kSimd) {
#if defined(__AVX2__)
                    avx2::vfill_right<T>(&*last, pivot, offsets_r, num_r, BS);
                    last -= BS;
#endif
                } else {
                    for (std::size_t i = 0; i < BS;) {
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                        offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                    }
                }
            } else {
                for (std::size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                }
            }

            std::size_t num = std::min(num_l, num_r);
            swap_offsets_perm(offsets_l_base, offsets_r_base,
                              offsets_l + start_l, offsets_r + start_r, num,
                              num_l == num_r, perm, id, em);
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;
            if (num_l == 0) { start_l = 0; offsets_l_base = first; }
            if (num_r == 0) { start_r = 0; offsets_r_base = last; }
        }

        if (num_l) {
            offsets_l += start_l;
            while (num_l--) {
                --last;
                Iter l = offsets_l_base + offsets_l[num_l];
                std::iter_swap(l, last);
                fix_at(perm, id, em, l);
                fix_at(perm, id, em, last);
            }
            first = last;
        }
        if (num_r) {
            offsets_r += start_r;
            while (num_r--) {
                Iter r = offsets_r_base - offsets_r[num_r];
                std::iter_swap(r, first);
                fix_at(perm, id, em, r);
                fix_at(perm, id, em, first);
                ++first;
            }
            last = first;
        }
    }
    return first;
}

// Guarded block partition + mirror with the scalar-cutoff fallback, for both
// the scalar and (compile-time opt-in) AVX2-vectorised fills.
template <bool UseSimd, std::random_access_iterator I, std::sentinel_for<I> S,
          class K, class PermIt, class Id, class EM, class Comp, class Proj>
inline I block_perm(I first, S last, K pivot, PermIt perm, Id& id,
                    const EM& em, Comp comp, Proj proj) {
    I end = first + (last - first);
    if (end - first < tuning::block_scalar_cutoff)
        return hoare_perm(first, end, pivot, perm, id, em, comp, proj);
    return branchless_partition_perm<true, UseSimd, tuning::block_size>(
        first, end, pivot, perm, id, em, comp, proj);
}

}  // namespace partitions::detail::perm

// ===========================================================================
// Keyed forward partition preserving the permutation invariant.
// ===========================================================================
namespace partitions {

namespace detail::perm {
// Below this many elements the whole working set (block + its perm slice) is
// cache-resident and the gap-Lomuto kernel + repair pass wins over every
// mirrored swap-based kernel at every measured below-fraction (see the file
// header); above it the winner depends on the below-fraction, which the
// dispatcher ESTIMATES from strided probes.
inline constexpr std::ptrdiff_t perm_small_cutoff = 2048;
// Balanced splits on narrow (<= 8-byte) elements keep preferring the gap
// kernel + repair well past the small cutoff (its sequential store stream
// primes the cache the repair pass then hits); the plain dispatcher's
// AVX2 kernels only catch up once the block spills L2.
inline constexpr std::ptrdiff_t perm_lomuto_mid_cutoff = std::ptrdiff_t(1)
                                                         << 17;
inline constexpr std::ptrdiff_t perm_probe_count = 64;
// Estimated below-count thresholds (out of perm_probe_count): outside
// [1/4, 3/4] counts as skewed (mirror pays); within, repair pays.
inline constexpr std::ptrdiff_t perm_skew_lo = perm_probe_count / 4;
// A very skewed non-SIMD block is fastest under mirrored Hoare (few swaps,
// scan branches almost always fall through): measured 0.86-1.02 vs block's
// 1.10-1.22 ns/elem on 16-byte records at p = 0.02/0.98.
inline constexpr std::ptrdiff_t perm_vskew_lo = 6;

// Branchless strided probe: count of below-key elements among
// perm_probe_count evenly spaced samples.
template <class I, class K, class Comp, class Proj>
inline std::ptrdiff_t probe_below(I first, std::ptrdiff_t n, K key, Comp& comp,
                                  Proj& proj) {
    const std::ptrdiff_t stride = n / perm_probe_count;
    std::ptrdiff_t cnt = 0;
    for (std::ptrdiff_t k = 0; k < perm_probe_count; ++k)
        cnt += static_cast<std::ptrdiff_t>(static_cast<bool>(std::invoke(
            comp, std::invoke(proj, first[k * stride]), key)));
    return cnt;
}
}  // namespace detail::perm

template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class PermIt, class Id, class Comp = std::less<>,
          class Proj = std::identity>
I sized_partition_perm(I first, S last, K key, PermIt perm, Id id,
                       Comp comp = {}, Proj proj = {}) {
    namespace dp = detail::perm;
    // Unsigned-pair lexicographic rewrite: one wide branchless compare per
    // element instead of std::less<>'s short-circuit two-field compare (which
    // GCC lowers to a mispredicting branch; measured 2.8x at p = 0.5).
    if constexpr (dp::lex_rewritable<std::iter_value_t<I>, Comp, Proj> &&
                  std::is_same_v<std::remove_cvref_t<K>,
                                 std::iter_value_t<I>>) {
        if constexpr (dp::lex_rewrite_64<std::iter_value_t<I>>) {
            return sized_partition_perm(first, last, dp::lex_concat64{}(key),
                                        perm, id, std::less<>{},
                                        dp::lex_concat64{});
        }
#if PARTITIONS_PERM_HAS_U128
        else {
            return sized_partition_perm(first, last, dp::lex_concat128{}(key),
                                        perm, id, std::less<>{},
                                        dp::lex_concat128{});
        }
#endif
    }
    I end = first + (last - first);
    const auto n = end - first;
    if (n <= 0) return end;

    auto em = dp::make_entry(first, end, perm, id);
    // Small 8-byte PAIRS (e.g. pair<u32,u32>) are a class of their own: the
    // mirrored block partition beats every repair variant from ~192 elements
    // at every below-fraction (the pairs' field extraction taxes the gap
    // kernel's dependent-move chain), so they skip both the Lomuto tier and
    // the probe earlier.
    constexpr bool small_pair8 =
        detail::net::pair_like_swappable<std::iter_value_t<I>> &&
        sizeof(std::iter_value_t<I>) <= 8;
    constexpr std::ptrdiff_t small_cut =
        small_pair8 ? 192 : dp::perm_small_cutoff;
    if (n <= small_cut) {
        // gap kernel moves everything; repair the (cache-resident) range.
        I m = detail::scalar::lomuto_branchless(first, end, key, comp, proj);
        dp::repair(first, end, perm, id, em);
        return m;
    }
    if constexpr (small_pair8) {
        if (n <= dp::perm_small_cutoff)
            return dp::block_perm<false>(first, end, key, perm, id, em, comp,
                                         proj);
    }
    // Estimate the below-fraction: it decides whether the mirrored block
    // partition (scattered perm traffic ~ 2p(1-p)n) or a full-speed plain
    // partition + full repair (traffic = n, but a cleaner store stream) wins.
    const auto cnt = dp::probe_below(first, n, key, comp, proj);
    const bool skewed = cnt <= dp::perm_skew_lo ||
                        cnt >= dp::perm_probe_count - dp::perm_skew_lo;
    if (skewed) {
        if constexpr (detail::simd_fast_path<I, Comp, Proj>) {
            return dp::block_perm<true>(first, end, key, perm, id, em, comp,
                                        proj);
        } else {
            if (cnt <= dp::perm_vskew_lo ||
                cnt >= dp::perm_probe_count - dp::perm_vskew_lo)
                return dp::hoare_perm(first, end, key, perm, id, em, comp,
                                      proj);
            return dp::block_perm<false>(first, end, key, perm, id, em, comp,
                                         proj);
        }
    }
    // Balanced split.  Narrow elements up to L2-ish sizes: gap kernel +
    // repair (measured fastest, see the file header).  Wide elements at DRAM
    // scale still prefer the mirror (the data stream itself is heavy;
    // measured 3.91 vs 4.80 at n = 2^22 on 16-byte records); everything
    // else: fastest plain kernel + full repair.
    if constexpr (small_pair8) {
        // Balanced 8-byte pairs keep preferring the mirrored block partition
        // at every size (measured: 1.9-2.9 vs repair 2.3-3.5 ns/elem).
        return dp::block_perm<false>(first, end, key, perm, id, em, comp,
                                     proj);
    } else if constexpr (sizeof(std::iter_value_t<I>) <= 8) {
        if (n <= dp::perm_lomuto_mid_cutoff) {
            I m = detail::scalar::lomuto_branchless(first, end, key, comp,
                                                    proj);
            dp::repair(first, end, perm, id, em);
            return m;
        }
    } else {
        if (n >= (std::ptrdiff_t(1) << 21))
            return dp::block_perm<false>(first, end, key, perm, id, em, comp,
                                         proj);
    }
    I m = sized_partition(first, end, key, comp, proj);
    dp::repair(first, end, perm, id, em);
    return m;
}

}  // namespace partitions

// ===========================================================================
// Offset partition preserving the permutation invariant.
// PRECONDITION: [first, first + offset) all >= key, and the permutation
// invariant holds over [first, last).
// ===========================================================================
namespace partitions::detail::perm {

// gap_off moves every touched element; positions modified are exactly
// [first, first + min(c + 1, offset)) and [first + offset, last), where c is
// the below count -- repair only those.
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class PermIt, class Id, class Comp, class Proj>
inline I gap_off_perm(I first, S last_s, std::iter_difference_t<I> offset,
                      K key, PermIt perm, Id& id, Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    I last = first + (last_s - first);
    const D n = last - first;
    if (offset >= n) return first;  // empty suffix: nothing moves
    auto em = make_entry(first, last, perm, id);
    I m = partitions::detail::offset::gap_off(first, last, offset, key, comp,
                                              proj);
    const D c = m - first;
    repair(first, first + std::min(c + 1, offset), perm, id, em);
    repair(first + offset, last, perm, id, em);
    return m;
}

// Prefix-fill with mirrored phase-1 rotations and bridge; phase 2 goes
// through the keyed perm dispatcher (mirrored block / repaired Lomuto).
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class PermIt, class Id, class Comp, class Proj>
inline I prefix_fill_perm(I first, S last_s, std::iter_difference_t<I> offset,
                          K key, PermIt perm, Id& id, Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    constexpr D B = static_cast<D>(tuning::block_size);
    I last = first + (last_s - first);
    I lo = first;
    I pfx_end = first + offset;
    auto em = make_entry(first, last, perm, id);
    auto below = [&](I it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), key));
    };

    alignas(tuning::cacheline_bytes)
        unsigned char offsets_r_storage[tuning::block_size +
                                        tuning::cacheline_bytes];
    unsigned char* offsets_r = scalar::align_cacheline(offsets_r_storage);

    // Phase 1: whole right blocks while both slots and suffix last.  The
    // fill_slots rotation's net effect is the pairwise swaps
    // (lo + i) <-> (last + B - offsets_r[i]); mirror them afterwards.
    while (pfx_end - lo >= B && last - pfx_end >= B) {
        std::size_t num_r = 0;
        for (std::size_t i = 0; i < tuning::block_size;) {
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
            offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
        }
        partitions::detail::offset::fill_slots(lo, last + B, offsets_r, num_r);
        // fill_slots is a cyclic rotation; write entries of all moved
        // positions directly (both the filled slots and the vacated right
        // positions).
        for (std::size_t i = 0; i < num_r; ++i) {
            fix_at(perm, id, em, lo + static_cast<D>(i));
            fix_at(perm, id, em, last + B - offsets_r[i]);
        }
        lo += static_cast<D>(num_r);
    }

    // Phase 2: keyed perm-partition of the remaining suffix, then one
    // mirrored bridge of the leftover [lo, pfx_end) gap.
    I m = partitions::sized_partition_perm(pfx_end, last, key, perm, id, comp,
                                           proj);
    const D c_rem = m - pfx_end;
    const D slots = pfx_end - lo;
    if (c_rem <= slots) {
        for (D i = 0; i < c_rem; ++i) {
            std::iter_swap(lo + i, pfx_end + i);
            fix_at(perm, id, em, lo + i);
            fix_at(perm, id, em, pfx_end + i);
        }
    } else {
        I src = m - slots;
        for (D i = 0; i < slots; ++i) {
            std::iter_swap(lo + i, src + i);
            fix_at(perm, id, em, lo + i);
            fix_at(perm, id, em, src + i);
        }
    }
    return lo + c_rem;
}

}  // namespace partitions::detail::perm

namespace partitions {

template <std::random_access_iterator It, class K, class PermIt, class Id,
          class Comp = std::less<>, class Proj = std::identity>
std::ptrdiff_t offset_partition_perm(It first, It last,
                                     std::iter_difference_t<It> offset, K key,
                                     PermIt perm, Id id, Comp comp = {},
                                     Proj proj = {}) {
    namespace dp = detail::perm;
    // Same unsigned-pair lexicographic rewrite as sized_partition_perm.
    if constexpr (dp::lex_rewritable<std::iter_value_t<It>, Comp, Proj> &&
                  std::is_same_v<std::remove_cvref_t<K>,
                                 std::iter_value_t<It>>) {
        if constexpr (dp::lex_rewrite_64<std::iter_value_t<It>>) {
            return offset_partition_perm(first, last, offset,
                                         dp::lex_concat64{}(key), perm, id,
                                         std::less<>{}, dp::lex_concat64{});
        }
#if PARTITIONS_PERM_HAS_U128
        else {
            return offset_partition_perm(first, last, offset,
                                         dp::lex_concat128{}(key), perm, id,
                                         std::less<>{}, dp::lex_concat128{});
        }
#endif
    }
    const auto suffix = (last - first) - offset;
    // Tiny suffix: the gap kernel + narrowed repair (its touched ranges are
    // cache-resident).
    if (suffix <= dp::perm_small_cutoff)
        return dp::gap_off_perm(first, last, offset, key, perm, id, comp,
                                proj) -
               first;
    // Estimate the SUFFIX below-fraction (only the suffix is ever compared).
    // Few belows -> prefix-fill + mirror (moves and touches only the below
    // elements: measured up to 3.5x over kernel+full-repair at p = 0.1).
    // Otherwise the gap kernel + narrowed repair wins or ties everywhere
    // measured (p = 0.5 / 0.9, offset fraction 0.25 / 0.75).
    const auto cnt =
        dp::probe_below(first + offset, suffix, key, comp, proj);
    if (cnt <= dp::perm_skew_lo)
        return dp::prefix_fill_perm(first, last, offset, key, perm, id, comp,
                                    proj) -
               first;
    return dp::gap_off_perm(first, last, offset, key, perm, id, comp, proj) -
           first;
}

}  // namespace partitions

#undef PARTITIONS_PERM_ALWAYS_INLINE

#endif  // REFACTORED_PARTITIONS_WITH_PERMUTATION_HPP
