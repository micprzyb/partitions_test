#ifndef REFACTORED_PARTITIONS_HPP
#define REFACTORED_PARTITIONS_HPP

// refactored_partitions.hpp -- self-contained extraction of this library's
// best general-purpose partition routines (a refactoring of
// extracted_partitions.hpp: same algorithms, same measured tiers, coherent
// namespaces).  Requires C++20; no dependencies beyond the standard library.
// AVX2 fast paths are compiled in automatically when the translation unit is
// built with -mavx2/-march=native (and fall away cleanly otherwise -- the
// portable scalar tiers remain).
//
// PUBLIC ENTRY POINTS (namespace partitions):
//
//   * sized_partition(first, last, key, comp, proj)          -> iterator
//         Forward partition around a key, dispatched by size / element width /
//         key cost / ISA to the fastest measured kernel (tier table below).
//   * sized_partition_at(first, last, pivot_it, comp, proj)  -> iterator
//         Position form: the pivot is an iterator INTO the block, so the
//         large-n scalar tier elides its left-scan bound check (the pivot
//         element is its own sentinel).  Same contract, keyed by proj(*pivot).
//   * offset_partition(first, last, offset, key, comp, proj) -> count
//         Same contract, plus a PRECONDITION that [first, first+offset) is
//         already all >= key; the prefix is never compared, only swapped into.
//   * median_partition(first, last, comp, proj)               -> iterator
//         NO key: partition around an ESTIMATED MEDIAN.  n <= 24 applies one
//         halver network (an exact n/2 rank split); larger blocks pick a
//         ninther / median-of-5-medians-of-5 pivot and run the keyed
//         dispatcher, returning the pivot placed at its final sorted rank.
//         Size tiers are the source repo's measured pure-quicksort settings.
//   * cyclic_partition(first, last, buf_begin, buf_end, key, comp, proj)
//     cyclic_partition_count(...)                     -> iterator / count
//     cyclic_offset_partition(first, last, buf_begin, buf_end, offset, key,
//                             comp, proj)             -> count
//         The same partition / offset partition over a possibly-WRAPPED range
//         of a cyclic buffer [buf_begin, buf_end): first > last means the
//         range spans [first, buf_end) ++ [buf_begin, last).  See the
//         detail::cyclic section header for the range convention, contract
//         and design (idea log + measurements: cyclic_partitions.txt).
//   * find_min(first, last, comp, proj)                -> iterator
//     cyclic_find_min(first, last, buf_begin, buf_end, comp, proj) -> iterator
//         An iterator to A minimal element (any-minimal tie rule) of a flat /
//         possibly-wrapped range; key-width-dispatched kernels from the
//         source repo's find-minimum study (see the detail::minsel section).
//
// DESIGN NOTES OF THIS REFACTORING
//
//   * Free functions, not function objects.  The source repo needs stateless
//     functor structs because algorithms are *registry entries*: they are
//     stored in tuples, iterated generically, and carry `name` strings for
//     reporting.  None of that exists here -- every call site in this file is
//     an immediate `X{}(...)` invocation -- so each `struct X { operator() }`
//     is replaced by a bare function template.  This is codegen-identical
//     (a stateless functor invoked in place and a function template inline to
//     the same thing under GCC and Clang) and removes a layer of syntax.  The
//     one struct-only feature used, the `boost_block{}.at(...)` second form,
//     becomes a sibling function (`boost_block_at`).
//
//   * Namespace map:
//         partitions            public entry points (above)
//         partitions::tuning    every measured threshold, in one place, with
//                               its provenance -- the per-architecture knobs
//         partitions::detail    internals:
//           ::net               branchless compare-exchange + halver networks
//           ::scalar            portable kernels (Hoare, branchless Lomuto,
//                               pdqsort block partition, size dispatch)
//           ::avx2              AVX2 kernels (only under __AVX2__)
//           ::pivot             median estimators (ninther, m5m5, ...)
//           ::offset            offset-partition kernels
//
//   * Portability.  The only compiler extensions used are the always-inline
//     attribute (wrapped in PARTITIONS_ALWAYS_INLINE: [[gnu::always_inline]]
//     on GCC/Clang, __forceinline on MSVC) and AVX2 intrinsics behind
//     #if defined(__AVX2__).  `std::popcount` replaces __builtin_popcount.
//     The performance-critical idioms -- bool + ternary compare-exchange,
//     `offsets[num]=i; num += pred;` setcc-add fills, XOR-mask blends --
//     are the patterns GCC *and* Clang lower branchlessly at -O2/-O3.
//     Thresholds in partitions::tuning were measured on Zen 3; the two most
//     architecture-sensitive ones are macro-overridable (below) without
//     touching the code.
//
// CONFIGURATION MACROS (define before including; all optional):
//
//   PARTITIONS_L2_BYTES        per-core L2 size used for the AVX2
//                              compress/simd crossover.  Default 512 KiB
//                              (Zen 3).
//   PARTITIONS_HALVE_MID_MAX   keyed-dispatch halver-tier gate, default 0
//                              (OFF -- see the measured table below); values
//                              above 24 are clamped by the network set.
//
// DISPATCH TIERS of the keyed forward partition (sized_partition):
//
//   n <= PARTITIONS_HALVE_MID_MAX (halver tier -- OFF by default, see below)
//       -> halve_mid: sort the block by recursive balanced HALVING (branchless
//          comparator networks, ~20-33% fewer compare-exchanges than a full
//          sorting network per split) and take the boundary by a branchless
//          count.
//   cheap-compare narrow integer key (i32/i64, identity proj, `<`),
//   contiguous iterator, AVX2:
//       n <  64                 -> lomuto_branchless (no setup)
//       n <= L2_bytes/sizeof(T) -> block_compress (single-pass AVX2
//                                  compaction; 2x store traffic -> wins only
//                                  while L2-resident)
//       else                    -> block_simd (AVX2 offset fill + few-write
//                                  swap; bandwidth-friendly once L2 spills)
//   everything else:
//       -> sized (branchless Lomuto small / pdqsort block large -- the
//          portable dispatcher, best for expensive-compare and wide elements)
//
// WHY THE HALVER TIER IS OFF BY DEFAULT (measured, Zen 3 / GCC -O3
// -march=native, batched independent random blocks, key = 0, min ns/elem over
// 300 reps; networks running the branchless member-wise compare-exchange --
// see pair_like_swappable below):
//
//                      n=8          n=16         n=24
//     i64        halve_mid 2.82 / lomuto 0.48;  3.24 / 0.48;  3.87 / 0.48
//     i32                  2.42 / 0.56;         2.74 / 0.52;  2.97 / 0.51
//     pair64 lex           5.01 / 3.65;         8.28 / 4.04; 10.59 / 4.12
//     pair64 .first        3.51 / 1.05;         4.90 / 1.03;  5.92 / 1.01
//
// A halver replaces PIVOT-SELECTION + partition in a quicksort leaf, where no
// key exists and a balanced rank-split is the goal; there it wins (see the
// source repo's pure quicksort).  This API is *given* the key, so a partition
// costs 1 compare + 2 cheap moves per element while a halve-sort costs ~5-8
// compare-exchanges per element -- the halver tier loses 3-12x at every
// (type, n) measured, so the default gate is 0.  The tier is kept compiled
// and correct; define PARTITIONS_HALVE_MID_MAX (<= 24) to route blocks of
// that size and below through it, e.g. to re-measure on another architecture.
// The halvers' natural home in this file is `median_partition`, where no key
// exists and the rank split is exactly what is asked for.
//
// CONTRACT.  Every keyed partition here reorders [first, last) so that
//
//     [first, m)  : comp(proj(x), key)   is true   ("below" -- strictly < key)
//     [m,   last) : comp(proj(x), key)   is false  (>= key)
//
// and returns the boundary `m`.  `m` is the boundary index, NOT the resting
// place of any pivot element: the key need not occur in the block (`m == first`
// means empty left group, `m == last` means empty right group, and `m - first`
// always equals the number of below-key elements).
//
// COMPARATOR REQUIREMENT.  `comp` must be a strict-weak-ordering "less" over
// the projected values (the halver tier and median_partition really SORT /
// rank-split blocks, so unlike a pure partition kernel they evaluate comp
// between two elements, not only against the key).  Do not pass a
// negated/reflexive comparator.
//
// PAIR-LEX SUBSTITUTION.  When the element is pair-like, the key is the
// element type, comp is the default less and proj is identity, every public
// entry point (flat and cyclic) transparently swaps comp for a pinned scan
// comparator of IDENTICAL order (detail::net: lex_less_packed for 4+4-byte
// integer pairs -- one u64 compare; lex_less_semibranch otherwise -- the
// short-circuit optimum pinned at source level).  Rationale and measured
// numbers: the section above detail::net's comparator block and
// cyclic_partitions.txt rounds 2-3.  NaN in float members is outside the
// strict-weak-ordering contract, as everywhere else in this file.
//
// THE PIVOT KEY IS TAKEN BY VALUE, not `const K&` or `K&&`: a partition
// permutes the block while comparing against the key, so a *reference* key
// makes the compiler assume aliasing and reload it from memory on every
// comparison; by value it stays in a register (verified by disassembly -- both
// `const K&` and `K&&` reload, by value does not).  Keys are expected to be
// small (<= 16 bytes), so the copy is free.
//
// NOTE.  This file duplicates entities of the full partitions library under
// the same `partitions` namespace; do not include it in the same translation
// unit as the library's own headers (or as extracted_partitions.hpp).

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Force-inline attribute for the small always-hot helpers (network
// compare-exchanges, offset fills).  These MUST inline: an outlined
// compare-exchange loses its cmov lowering and an outlined fill loses its
// vectorisation.
#if defined(_MSC_VER) && !defined(__clang__)
#define PARTITIONS_ALWAYS_INLINE __forceinline
#else
#define PARTITIONS_ALWAYS_INLINE [[gnu::always_inline]] inline
#endif

#if !defined(PARTITIONS_L2_BYTES)
#define PARTITIONS_L2_BYTES (512 * 1024)  // Zen 3 per-core L2
#endif
#if !defined(PARTITIONS_HALVE_MID_MAX)
#define PARTITIONS_HALVE_MID_MAX 0
#endif
// Cyclic-partition gap-tier gates (see tuning::cyclic_* below).  The gates
// are SOUND at any value -- every kernel is correct at every size -- only the
// crossover is architecture-dependent (defaults measured on Meteor Lake; the
// same portability contract as PARTITIONS_L2_BYTES).
#if !defined(PARTITIONS_CYCLIC_GAP_MAX)
#define PARTITIONS_CYCLIC_GAP_MAX 1024
#endif
#if !defined(PARTITIONS_CYCLIC_GAP_MAX_SIMD)
#define PARTITIONS_CYCLIC_GAP_MAX_SIMD 512
#endif
#if !defined(PARTITIONS_CYCLIC_OFF_GAP_MAX)
#define PARTITIONS_CYCLIC_OFF_GAP_MAX 1024
#endif
#if !defined(PARTITIONS_CYCLIC_OFF_GAP_MAX_SIMD)
#define PARTITIONS_CYCLIC_OFF_GAP_MAX_SIMD 256
#endif

// ===========================================================================
// Tuning -- every measured threshold, in one place.  All values were measured
// on Zen 3 (see the source repo's docs/ and the file header); they are sound
// on any architecture, just not necessarily optimal.
// ===========================================================================
namespace partitions::tuning {

// -- pdqsort block partition (detail::scalar) --------------------------------
// block_size=128 (vs the reference pdqsort's 64) is a stable ~17% win for i64
// at n=2^22 and neutral for pair64.  It must stay a multiple of 8 (8x unrolled
// fill) and < 256 (offsets are unsigned char, and the right buffer stores
// values up to block_size).
inline constexpr std::size_t block_size = 128;
inline constexpr std::size_t cacheline_bytes = 64;  // power of two
// Below this length the offset-buffer machinery's fixed setup cost outweighs
// the branch mispredictions it saves; fall back to scalar Hoare.
inline constexpr std::ptrdiff_t block_scalar_cutoff = 16;

// -- portable size dispatch (detail::scalar::sized) --------------------------
// The branchless Lomuto does two moves per element, so it only pays for
// narrow, cheap-to-move elements -- hence the array-size threshold is itself
// chosen by sizeof(T) (the same rationale as Rust ipnsort's size_of<T> rule).
template <class T>
inline constexpr std::ptrdiff_t lomuto_cutoff = sizeof(T) <= 8 ? 512 : 24;

// -- AVX2 fast path (sized_partition dispatch) -------------------------------
// The compaction partition double-stores every element (its vectorised
// "swap"); that is free while the block is L2-resident, but once it spills
// the extra writes hit the lower L3/DRAM bandwidth and the few-write
// fill+swap (block_simd) wins.  So the crossover is exactly "does the block
// fit L2?" -> compress_max = L2_bytes / sizeof(T).  This is both the measured
// Zen 3 crossover (i64 wins through 2^16 = 512 KiB = Zen 3 L2, cliffs at
// 2^18) and a portable rule: override PARTITIONS_L2_BYTES per target.
inline constexpr std::size_t l2_bytes = PARTITIONS_L2_BYTES;
inline constexpr std::ptrdiff_t simd_lomuto_cutoff = 64;  // below: lomuto
inline constexpr std::size_t simd_block_size = 128;       // vectorised fill
inline constexpr std::ptrdiff_t compress_min = 64;        // compress needs 2*W
template <class T>
inline constexpr std::ptrdiff_t compress_max =
    static_cast<std::ptrdiff_t>(l2_bytes / sizeof(T));

// -- keyed-dispatch halver-tier gate (OFF by default; see the file header) ---
inline constexpr std::ptrdiff_t halve_mid_max =
    PARTITIONS_HALVE_MID_MAX > 24 ? 24 : PARTITIONS_HALVE_MID_MAX;

// -- cyclic dispatch (cyclic_partition / cyclic_offset_partition) ------------
// Gate of the gap_cyclic tier (the wrapped branchless-Lomuto gap kernel: no
// setup, no bridge, ~1 extra cmov per element) versus the two-segment
// kernels.  Two comparator classes behave differently enough to need their
// own gates (measured, Meteor Lake, min ns/elem over all wrap fractions --
// regret tables in cyclic_partitions.txt):
//
//   * pair-lex (a pair-like element under the default `<`): the compare is a
//     short-circuit branch, so the gap walk mispredicts per element just like
//     everything else while the block kernels amortise better -> tiny gate
//     (24), seg_bridge above it (the fused fill's scattered rotation loses to
//     the vectorised bridge at every mid wrap for this class).
//   * AVX2-eligible integers: the gap walk holds ~0.48 ns/elem flat in the
//     wrap fraction, but the vectorised segment kernels pull ahead once their
//     setup amortises -> gate 512, seg_fill above it.
//   * everything else (cheap branchless compare, no SIMD kernels to amortise
//     against): gate 1024, seg_fill above it.
//
// For the OFFSET partition the gate is on the SUFFIX (compared) length; the
// AVX2 integer fast path scans with vfill_right, which overtakes the gap walk
// earlier (256).
inline constexpr std::ptrdiff_t cyclic_gap_max_lex = 24;
inline constexpr std::ptrdiff_t cyclic_gap_max_simd = PARTITIONS_CYCLIC_GAP_MAX_SIMD;
inline constexpr std::ptrdiff_t cyclic_gap_max = PARTITIONS_CYCLIC_GAP_MAX;
inline constexpr std::ptrdiff_t cyclic_off_gap_max_lex = 24;
inline constexpr std::ptrdiff_t cyclic_off_gap_max_simd = PARTITIONS_CYCLIC_OFF_GAP_MAX_SIMD;
inline constexpr std::ptrdiff_t cyclic_off_gap_max = PARTITIONS_CYCLIC_OFF_GAP_MAX;
// The MULTI-setcc branchless-lex comparator (lex_less_branchless, i.e. the
// pair-lex substitution) gets its own gates: its 3-setcc predicate is
// vulnerable to a GCC register-allocation hazard in SOME instantiation
// contexts (a setcc landing on a register that carries a store-forwarded
// reload puts ~6 extra cycles on the gap loop's carried dependency --
// measured 2.6x, context-dependent; forensics in cyclic_partitions.txt
// round 2).  The gates below are tuned to the DISPATCHER-context codegen:
// tiny gap tier, seg_bridge through mid sizes, seg_fill above.
inline constexpr std::ptrdiff_t cyclic_gap_max_mlex = 24;
inline constexpr std::ptrdiff_t cyclic_bridge_max_mlex = 1024;
inline constexpr std::ptrdiff_t cyclic_off_gap_max_mlex = 256;

// -- median_partition tiers (the source quicksort's measured bounds) ---------
// 24 is the largest hand-built halver (raising the halve cutoff from 16 to 24
// measured ~1.0-1.6% faster on 16-byte pair types, neutral on i64).  Above
// 2^16 the 25-sample pivot's cost amortises fully over the O(n) partition
// while its better balance shaves work off every level below (~2-3% on
// i64/pair64f at n >= 2^20, with NO penalty on sorted input).
inline constexpr std::ptrdiff_t median_halve_cutoff = 24;
inline constexpr std::ptrdiff_t median_m5m5_cutoff = 1 << 16;  // 65536

}  // namespace partitions::tuning

// ===========================================================================
// detail::net -- branchless compare-exchange and the halver networks.
//
// A *halver* is a comparator network that, unlike a sorter, only splits the
// block by RANK: after `halve_n<N>` the N/2 smallest elements occupy the
// bottom half [first, first+N/2) and the N/2 largest the top half (each half
// unordered).  A halver needs ~20-33% fewer compare-exchanges than a full
// sorting network (n=8: 14 vs 19; n=16: 44 vs 60; n=24: 80 vs 120).  The
// networks below were found by iterated-local-search / CEGAR optimisation and
// proved correct by exhaustive 0/1 enumeration (tools/verify_small_halve in
// the source repo); do not edit them by hand.
// ===========================================================================
namespace partitions::detail::net {

// Two issues need handling separately on x86-64 for 16-byte aggregates:
//
//   (1) The COMPARE: pair<i64,i64>::operator< (from default <=>) is a
//       short-circuit lex compare and GCC emits actual branches for it.
//       Branch-predictor misses cost ~15-20 cycles per cswap on random data.
//
//   (2) The BLEND: `pair64 = bool ? pair64 : pair64` is lowered by GCC as a
//       branch + xmm copy, NOT as two cmovs on the halves (there is no 128-bit
//       CMOV in x86-64).
//
// Fix: when T is a 16-byte trivial aggregate AND the user supplied no custom
// comparator, compute the lex compare branchlessly and decompose the swap into
// an integer XOR-mask blend over the two 64-bit halves.  For everything else
// the canonical ternary-on-T pattern is used; GCC and Clang lower it to a CMOV
// for register-sized arithmetic types.

template <class T>
constexpr bool is_16byte_trivial =
    sizeof(T) == 16 && std::is_trivially_copyable_v<T>;

template <class Cmp, class Proj>
constexpr bool is_default_less =
    std::is_same_v<std::decay_t<Cmp>, std::less<>> &&
    std::is_same_v<std::decay_t<Proj>, std::identity>;

// Branchless 16-byte swap via integer XOR-mask (mask = 0 or all-ones).  Must be
// XOR-mask, not a cmov form: a floating field keeps the value in XMM registers,
// where a conditional 16-byte copy is lowered back to a branch.
template <class T>
PARTITIONS_ALWAYS_INLINE void half_swap(T& a, T& b, bool swap) {
    std::uint64_t a0, a1, b0, b1;
    std::memcpy(&a0, &a, 8);
    std::memcpy(&a1, reinterpret_cast<const char*>(&a) + 8, 8);
    std::memcpy(&b0, &b, 8);
    std::memcpy(&b1, reinterpret_cast<const char*>(&b) + 8, 8);
    const std::uint64_t mask = 0ull - static_cast<std::uint64_t>(swap);
    const std::uint64_t d0 = (a0 ^ b0) & mask;
    const std::uint64_t d1 = (a1 ^ b1) & mask;
    a0 ^= d0; b0 ^= d0;
    a1 ^= d1; b1 ^= d1;
    std::memcpy(&a, &a0, 8);
    std::memcpy(reinterpret_cast<char*>(&a) + 8, &a1, 8);
    std::memcpy(&b, &b0, 8);
    std::memcpy(reinterpret_cast<char*>(&b) + 8, &b1, 8);
}

// XOR-mask swap for any trivially-copyable type of size <= 8.
template <class T>
PARTITIONS_ALWAYS_INLINE void word_swap(T& a, T& b, bool swap) {
    static_assert(sizeof(T) <= 8);
    std::uint64_t av = 0, bv = 0;
    std::memcpy(&av, &a, sizeof(T));
    std::memcpy(&bv, &b, sizeof(T));
    const std::uint64_t mask = 0ull - static_cast<std::uint64_t>(swap);
    const std::uint64_t d = (av ^ bv) & mask;
    av ^= d;
    bv ^= d;
    std::memcpy(&a, &av, sizeof(T));
    std::memcpy(&b, &bv, sizeof(T));
}

// "Looks like a lexicographic pair": a class with public `.first`/`.second`,
// ordered first-then-second (what a defaulted operator<=> produces).
//
// The requirements are MEMBER-wise (each member a non-reference,
// trivially-copyable, <= 8-byte field, and no padding), NOT whole-object
// std::is_trivially_copyable: libstdc++ 16 gave std::pair user-provided
// const-assignment overloads, so std::pair is no longer trivially copyable
// there, and a whole-object gate silently dropped std::pair elements onto the
// generic ternary path -- which GCC lowers to a data-dependent BRANCH per
// compare-exchange (there is no 16-byte cmov), measured 2.4-4.5x slower on
// random data (halve24 pair64-lex 8.0 -> 3.3 ns/elem after this fix; see the
// median_partition section).  Reading and writing the *members* through
// memcpy is legal whenever the members themselves are trivially copyable,
// which is all the XOR-mask blend needs; the no-padding requirement makes the
// member-wise swap equal to a whole-value swap.
template <class T>
concept pair_like_swappable =
    std::is_class_v<T> && requires(T t) {
        t.first;
        t.second;
    } && !std::is_reference_v<decltype(std::declval<T&>().first)> &&
    !std::is_reference_v<decltype(std::declval<T&>().second)> &&
    std::is_trivially_copyable_v<decltype(std::declval<T&>().first)> &&
    std::is_trivially_copyable_v<decltype(std::declval<T&>().second)> &&
    sizeof(std::declval<T&>().first) <= 8 &&
    sizeof(std::declval<T&>().second) <= 8 &&
    sizeof(T) == sizeof(std::declval<T&>().first) +
                     sizeof(std::declval<T&>().second);

template <class T>
concept lex_pair_like = pair_like_swappable<T>;

// Member-wise XOR-mask swap for pair_like_swappable types that are NOT
// whole-object trivially copyable (std::pair on libstdc++ 16): each member is
// blended through its own uint64 image, staying in the integer domain like
// half_swap/word_swap (a cmov form on a floating member would be lowered back
// to a branch in the XMM domain).  Types that ARE trivially copyable keep the
// original half_swap/word_swap byte-image paths, so their codegen is
// unchanged.
template <class T>
PARTITIONS_ALWAYS_INLINE void pair_swap(T& a, T& b, bool swap) {
    const std::uint64_t mask = 0ull - static_cast<std::uint64_t>(swap);
    std::uint64_t af = 0, bf = 0, as = 0, bs = 0;
    std::memcpy(&af, &a.first, sizeof(a.first));
    std::memcpy(&bf, &b.first, sizeof(b.first));
    std::memcpy(&as, &a.second, sizeof(a.second));
    std::memcpy(&bs, &b.second, sizeof(b.second));
    const std::uint64_t df = (af ^ bf) & mask;
    const std::uint64_t ds = (as ^ bs) & mask;
    af ^= df;
    bf ^= df;
    as ^= ds;
    bs ^= ds;
    std::memcpy(&a.first, &af, sizeof(a.first));
    std::memcpy(&b.first, &bf, sizeof(b.first));
    std::memcpy(&a.second, &as, sizeof(a.second));
    std::memcpy(&b.second, &bs, sizeof(b.second));
}

template <class T, class Cmp, class Proj>
PARTITIONS_ALWAYS_INLINE void cswap(T& a, T& b, Cmp& comp, Proj& proj) {
    if constexpr (lex_pair_like<T> && is_default_less<Cmp, Proj>) {
        // Branchless lexicographic compare-exchange.  `lt0` and `eq0` are
        // computed from the SAME operand pair so the compiler issues a single
        // compare and reads both predicates off its flags.
        const auto a0 = a.first, a1 = a.second;
        const auto b0 = b.first, b1 = b.second;
        const unsigned lt0 = static_cast<unsigned>(b0 < a0);
        const unsigned eq0 = static_cast<unsigned>(b0 == a0);
        const unsigned lt1 = static_cast<unsigned>(b1 < a1);
        const bool swap = (lt0 | (eq0 & lt1)) != 0;
        if constexpr (is_16byte_trivial<T>) {
            half_swap(a, b, swap);
        } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 8) {
            word_swap(a, b, swap);
        } else {
            pair_swap(a, b, swap);  // pair-like, not trivially copyable
        }
    } else if constexpr (is_16byte_trivial<T>) {
        const bool swap = comp(proj(b), proj(a));
        half_swap(a, b, swap);
    } else if constexpr (pair_like_swappable<T>) {
        // Pair-like but not whole-object trivially copyable (std::pair on
        // libstdc++ 16) under a custom comparator/projection: the same
        // member-wise branchless blend, with the comparator deciding.
        const bool swap = comp(proj(b), proj(a));
        pair_swap(a, b, swap);
    } else if constexpr (std::is_integral_v<T> && sizeof(T) <= 8) {
        // Independent min/max blend: both cmovs depend only on the compare
        // flag, not on each other, so they issue in parallel.
        const bool swap = comp(proj(b), proj(a));
        const T lo = swap ? b : a;
        const T hi = swap ? a : b;
        a = lo;
        b = hi;
    } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
        // 8-byte trivially-copyable aggregate with a custom comparator (e.g.
        // sorted by a key field, carrying the whole struct): the ternary can be
        // lowered to a branch, the XOR-mask swap cannot.
        const bool swap = comp(proj(b), proj(a));
        word_swap(a, b, swap);
    } else {
        const bool swap = comp(proj(b), proj(a));
        T tmp = a;
        a = swap ? b : a;
        b = swap ? tmp : b;
    }
}

using net_pair = std::pair<int, int>;

template <class It, class Cmp, class Proj, std::size_t S, std::size_t... Is>
PARTITIONS_ALWAYS_INLINE void apply_network(
    It first, const std::array<net_pair, S>& net,
    std::index_sequence<Is...>, Cmp& comp, Proj& proj) {
    (cswap(first[net[Is].first], first[net[Is].second], comp, proj), ...);
}

namespace nets {
using P = net_pair;
inline constexpr std::array<P, 1> h2 = {{{0,1}}};
inline constexpr std::array<P, 2> h3 = {{{0,2},{0,1}}};
inline constexpr std::array<P, 4> h4 = {{{1,3},{0,1},{2,3},{1,2}}};
inline constexpr std::array<P, 6> h5 = {{{0,3},{0,2},{1,3},{0,1},{2,4},{1,2}}};
inline constexpr std::array<P, 8> h6 = {{{0,5},{1,3},{2,4},{1,2},{3,4},{0,3},{2,5},{2,3}}};
inline constexpr std::array<P, 11> h7 = {{{0,6},{2,4},{1,3},{3,6},{4,5},{2,3},{0,1},{0,5},{3,4},{1,2},{2,3}}};
inline constexpr std::array<P, 14> h8 = {{{0,7},{1,4},{2,5},{3,6},{0,2},{1,3},{4,6},{5,7},{0,6},{1,7},{2,4},{3,5},{2,5},{3,4}}};
inline constexpr std::array<P, 17> h9 = {{{2,5},{4,8},{0,7},{2,4},{3,8},{5,6},{0,2},{1,3},{4,5},{7,8},{1,4},{3,6},{5,7},{3,5},{2,3},{4,5},{3,4}}};
inline constexpr std::array<P, 20> h10 = {{{0,8},{1,9},{2,7},{0,2},{4,5},{1,4},{3,6},{5,8},{7,9},{2,4},{5,7},{8,9},{0,1},{1,5},{2,3},{4,8},{6,7},{3,5},{4,6},{4,5}}};
inline constexpr std::array<P, 23> h11 = {{{0,9},{1,6},{0,1},{3,5},{4,10},{2,8},{1,10},{6,9},{1,3},{4,7},{8,10},{0,4},{1,2},{3,7},{5,9},{4,5},{7,8},{2,4},{3,6},{6,7},{3,4},{5,6},{4,5}}};
inline constexpr std::array<P, 27> h12 = {{{0,8},{1,7},{2,6},{5,9},{0,1},{2,5},{3,4},{6,9},{10,11},{0,2},{4,10},{2,11},{1,6},{9,11},{7,8},{3,6},{5,7},{1,4},{3,5},{6,8},{7,10},{2,5},{6,9},{4,6},{5,7},{5,6},{4,7}}};
inline constexpr std::array<P, 30> h13 = {{{0,12},{1,10},{2,9},{3,7},{5,11},{6,8},{1,6},{2,3},{4,11},{7,9},{8,10},{3,6},{7,8},{9,10},{5,9},{8,11},{1,2},{3,8},{4,7},{8,12},{5,8},{2,5},{6,9},{5,7},{0,4},{3,4},{6,8},{4,5},{6,7},{5,6}}};
inline constexpr std::array<P, 34> h14 = {{{0,1},{8,9},{2,3},{4,5},{6,7},{10,11},{6,10},{12,13},{0,2},{4,8},{5,9},{10,12},{3,7},{11,13},{2,8},{7,9},{0,6},{1,5},{7,13},{8,12},{2,10},{3,11},{4,6},{1,3},{5,11},{6,7},{3,10},{3,6},{5,8},{7,10},{5,6},{7,8},{6,7},{1,12}}};
inline constexpr std::array<P, 39> h15 = {{{1,2},{4,14},{5,8},{6,13},{9,11},{1,5},{2,8},{3,7},{6,9},{10,12},{11,13},{0,7},{1,6},{2,9},{4,10},{5,11},{8,13},{12,14},{0,6},{2,4},{3,5},{7,14},{7,11},{8,10},{9,12},{1,14},{0,3},{4,7},{5,9},{6,8},{4,6},{7,9},{3,5},{10,12},{8,10},{5,6},{2,11},{7,8},{6,7}}};
inline constexpr std::array<P, 44> h16 = {{{2,15},{3,14},{4,8},{7,11},{9,10},{0,5},{1,7},{2,9},{3,4},{6,13},{8,14},{10,15},{0,1},{2,3},{10,11},{14,15},{0,2},{7,9},{1,3},{4,10},{5,11},{6,7},{8,9},{12,14},{3,13},{13,15},{1,2},{3,12},{10,12},{4,6},{5,7},{8,10},{9,11},{13,14},{2,6},{5,8},{7,10},{9,13},{3,6},{9,12},{8,9},{6,8},{7,9},{7,8}}};
inline constexpr std::array<P, 48> h17 = {{{0,11},{1,15},{2,10},{7,11},{6,14},{4,5},{4,6},{8,12},{3,5},{9,16},{13,14},{0,6},{1,8},{2,8},{5,15},{3,7},{4,9},{6,16},{10,11},{0,9},{12,14},{0,2},{1,4},{5,6},{7,13},{8,9},{10,12},{11,14},{15,16},{2,5},{6,11},{9,13},{12,15},{3,4},{5,10},{7,8},{3,7},{4,8},{6,12},{6,9},{2,7},{4,5},{10,12},{5,7},{8,10},{6,8},{7,9},{7,8}}};
inline constexpr std::array<P, 52> h18 = {{{12,14},{0,1},{6,7},{16,17},{1,2},{15,16},{12,13},{10,11},{14,17},{4,5},{8,9},{9,11},{6,10},{3,5},{1,4},{13,16},{12,15},{7,16},{2,5},{0,3},{4,17},{0,13},{4,13},{10,13},{1,14},{2,14},{4,9},{3,10},{5,17},{8,15},{7,9},{0,12},{10,15},{4,12},{1,6},{6,12},{9,15},{3,8},{8,10},{5,11},{5,13},{8,12},{2,10},{5,16},{5,7},{2,5},{9,14},{5,9},{7,10},{7,12},{5,12},{7,9}}};
inline constexpr std::array<P, 56> h19 = {{{12,14},{10,11},{15,16},{17,18},{12,13},{15,17},{12,15},{16,18},{4,5},{8,9},{14,17},{0,1},{9,11},{2,3},{3,5},{13,16},{1,5},{0,3},{2,4},{6,7},{4,17},{0,13},{6,10},{2,14},{3,18},{4,13},{5,11},{7,16},{10,13},{4,9},{3,10},{13,16},{8,15},{5,18},{7,9},{1,14},{5,17},{10,15},{0,4},{3,8},{2,4},{8,10},{1,7},{5,13},{4,12},{8,12},{9,14},{1,6},{5,7},{7,10},{6,12},{9,15},{5,7},{5,9},{7,12},{7,9}}};
inline constexpr std::array<P, 60> h20 = {{{13,15},{7,8},{16,17},{11,12},{18,19},{1,2},{16,18},{17,19},{15,18},{0,2},{5,6},{3,4},{9,10},{10,12},{13,14},{1,5},{4,6},{3,5},{14,17},{2,6},{1,16},{0,4},{7,11},{5,18},{0,14},{12,18},{3,15},{4,19},{5,14},{11,14},{13,16},{1,7},{4,11},{5,10},{2,17},{6,19},{9,16},{8,10},{12,14},{6,12},{11,16},{2,15},{4,9},{3,13},{0,5},{9,11},{6,17},{2,11},{10,15},{5,13},{6,8},{2,6},{10,16},{9,13},{7,13},{6,10},{8,11},{8,13},{6,13},{8,10}}};
inline constexpr std::array<P, 65> h21 = {{{6,7},{2,3},{4,5},{16,20},{0,1},{5,7},{8,9},{10,11},{17,18},{9,11},{14,15},{0,2},{12,13},{4,6},{8,10},{16,17},{0,8},{1,3},{12,14},{4,12},{17,19},{13,15},{1,9},{7,15},{5,13},{6,14},{3,11},{3,7},{8,12},{2,10},{3,12},{10,14},{7,19},{3,17},{7,20},{7,17},{9,13},{13,17},{12,20},{2,6},{6,13},{6,9},{3,16},{7,10},{2,16},{9,16},{4,8},{10,18},{0,8},{11,13},{1,5},{5,10},{5,6},{10,18},{6,12},{11,15},{11,14},{1,7},{6,8},{7,8},{10,16},{11,12},{10,11},{9,10},{8,10}}};
inline constexpr std::array<P, 68> h22 = {{{8,9},{0,1},{20,21},{2,3},{16,17},{12,13},{18,19},{19,21},{18,20},{0,2},{4,5},{1,3},{6,7},{10,11},{14,15},{4,6},{5,7},{15,17},{14,16},{2,6},{15,19},{1,5},{16,20},{8,12},{5,17},{3,7},{11,13},{6,20},{1,15},{5,21},{7,20},{6,15},{12,15},{0,4},{2,18},{6,11},{14,18},{4,16},{2,6},{6,14},{5,12},{0,1},{1,14},{7,17},{3,19},{13,21},{10,18},{9,11},{7,15},{12,18},{3,16},{5,10},{7,13},{3,12},{10,12},{7,19},{7,9},{3,7},{4,8},{11,16},{11,18},{8,14},{7,11},{9,12},{10,14},{9,14},{7,14},{9,11}}};
inline constexpr std::array<P, 74> h23 = {{{0,1},{8,9},{10,11},{14,15},{12,13},{7,21},{12,14},{2,3},{0,2},{4,5},{9,11},{16,17},{18,19},{4,6},{17,19},{20,22},{1,3},{21,22},{8,10},{7,20},{2,6},{16,18},{1,5},{17,21},{10,14},{13,15},{8,12},{0,4},{3,15},{9,13},{6,14},{1,7},{12,17},{18,20},{3,11},{3,20},{5,13},{13,15},{5,18},{2,10},{9,16},{19,22},{7,16},{3,4},{0,8},{6,19},{13,19},{10,18},{3,12},{10,21},{4,17},{1,9},{5,8},{4,12},{6,8},{11,21},{14,17},{2,9},{6,9},{9,10},{10,12},{4,9},{13,22},{12,16},{8,12},{18,20},{8,9},{13,14},{11,13},{11,18},{11,12},{7,10},{10,11},{9,11}}};
inline constexpr std::array<P, 80> h24 = {{{4,6},{20,22},{14,15},{2,6},{12,13},{16,17},{1,3},{12,14},{7,21},{18,19},{10,11},{13,15},{8,9},{1,5},{17,19},{21,22},{9,11},{8,10},{0,23},{17,21},{2,3},{10,14},{16,18},{7,20},{2,10},{3,15},{18,20},{8,12},{4,5},{3,11},{9,13},{19,22},{12,17},{5,13},{3,20},{6,14},{9,16},{5,18},{18,21},{1,7},{14,20},{5,8},{6,19},{6,8},{3,4},{14,23},{4,17},{6,9},{1,2},{0,4},{10,18},{11,22},{3,12},{7,16},{4,12},{0,7},{11,17},{13,19},{4,7},{2,9},{7,8},{9,12},{12,16},{15,21},{11,13},{11,14},{17,18},{16,23},{8,12},{12,17},{11,12},{9,10},{7,10},{8,10},{13,16},{13,15},{10,11},{11,12},{13,14},{11,13}}};
}  // namespace nets

// Apply the halver for compile-time N; reorders [first, first+N) so the N/2
// smallest occupy [first, first+N/2).  Returns the split point first + N/2.
// Each branch calls `apply_network` DIRECTLY rather than through a local
// lambda: a `[&]` lambda here is not always_inline and GCC has been seen to
// outline it (losing vectorisation) for some element types.
#define PARTITIONS_HALVE_APPLY(net) \
    apply_network(first, net, std::make_index_sequence<net.size()>{}, comp, proj)
template <std::size_t N, class It, class Cmp = std::less<>, class Proj = std::identity>
PARTITIONS_ALWAYS_INLINE It halve_n(It first, Cmp comp = {}, Proj proj = {}) {
    if constexpr (N == 2) PARTITIONS_HALVE_APPLY(nets::h2);
    else if constexpr (N == 3) PARTITIONS_HALVE_APPLY(nets::h3);
    else if constexpr (N == 4) PARTITIONS_HALVE_APPLY(nets::h4);
    else if constexpr (N == 5) PARTITIONS_HALVE_APPLY(nets::h5);
    else if constexpr (N == 6) PARTITIONS_HALVE_APPLY(nets::h6);
    else if constexpr (N == 7) PARTITIONS_HALVE_APPLY(nets::h7);
    else if constexpr (N == 8) PARTITIONS_HALVE_APPLY(nets::h8);
    else if constexpr (N == 9) PARTITIONS_HALVE_APPLY(nets::h9);
    else if constexpr (N == 10) PARTITIONS_HALVE_APPLY(nets::h10);
    else if constexpr (N == 11) PARTITIONS_HALVE_APPLY(nets::h11);
    else if constexpr (N == 12) PARTITIONS_HALVE_APPLY(nets::h12);
    else if constexpr (N == 13) PARTITIONS_HALVE_APPLY(nets::h13);
    else if constexpr (N == 14) PARTITIONS_HALVE_APPLY(nets::h14);
    else if constexpr (N == 15) PARTITIONS_HALVE_APPLY(nets::h15);
    else if constexpr (N == 16) PARTITIONS_HALVE_APPLY(nets::h16);
    else if constexpr (N == 17) PARTITIONS_HALVE_APPLY(nets::h17);
    else if constexpr (N == 18) PARTITIONS_HALVE_APPLY(nets::h18);
    else if constexpr (N == 19) PARTITIONS_HALVE_APPLY(nets::h19);
    else if constexpr (N == 20) PARTITIONS_HALVE_APPLY(nets::h20);
    else if constexpr (N == 21) PARTITIONS_HALVE_APPLY(nets::h21);
    else if constexpr (N == 22) PARTITIONS_HALVE_APPLY(nets::h22);
    else if constexpr (N == 23) PARTITIONS_HALVE_APPLY(nets::h23);
    else if constexpr (N == 24) PARTITIONS_HALVE_APPLY(nets::h24);
    return first + static_cast<std::iter_difference_t<It>>(N / 2);
}
#undef PARTITIONS_HALVE_APPLY

// Runtime-size dispatch: returns the split point first + n/2.  Intended for
// n <= 24 (the bounded network set); larger n falls back to a full sort so the
// postcondition still holds, but the dispatch tiers never take it there.
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline It halve(It first, It last, Cmp comp = {}, Proj proj = {}) {
    const auto n = static_cast<std::size_t>(last - first);
    switch (n) {
        case 0: case 1: return first + static_cast<std::iter_difference_t<It>>(n / 2);
        case 2:  return halve_n<2>(first, comp, proj);
        case 3:  return halve_n<3>(first, comp, proj);
        case 4:  return halve_n<4>(first, comp, proj);
        case 5:  return halve_n<5>(first, comp, proj);
        case 6:  return halve_n<6>(first, comp, proj);
        case 7:  return halve_n<7>(first, comp, proj);
        case 8:  return halve_n<8>(first, comp, proj);
        case 9:  return halve_n<9>(first, comp, proj);
        case 10: return halve_n<10>(first, comp, proj);
        case 11: return halve_n<11>(first, comp, proj);
        case 12: return halve_n<12>(first, comp, proj);
        case 13: return halve_n<13>(first, comp, proj);
        case 14: return halve_n<14>(first, comp, proj);
        case 15: return halve_n<15>(first, comp, proj);
        case 16: return halve_n<16>(first, comp, proj);
        case 17: return halve_n<17>(first, comp, proj);
        case 18: return halve_n<18>(first, comp, proj);
        case 19: return halve_n<19>(first, comp, proj);
        case 20: return halve_n<20>(first, comp, proj);
        case 21: return halve_n<21>(first, comp, proj);
        case 22: return halve_n<22>(first, comp, proj);
        case 23: return halve_n<23>(first, comp, proj);
        case 24: return halve_n<24>(first, comp, proj);
        default:
            std::sort(first, last, [&](const auto& a, const auto& b) {
                return static_cast<bool>(std::invoke(
                    comp, std::invoke(proj, a), std::invoke(proj, b)));
            });
            return first + static_cast<std::iter_difference_t<It>>(n / 2);
    }
}

// Sort a small block (n <= 24) by recursive balanced halving -- the leaf the
// source repo's pure quicksort uses below its halve cutoff.
template <class It, class Cmp, class Proj>
inline void halve_sort(It first, It last, Cmp comp, Proj proj) {
    const auto n = last - first;
    if (n <= 1) return;
    It mid = halve(first, last, comp, proj);
    halve_sort(first, mid, comp, proj);
    halve_sort(mid, last, comp, proj);
}

// ===========================================================================
// Pair-lex comparators for PARTITION SCANS (element vs one fixed key), and
// the substitution the public entry points perform.
//
// The comparator's optimal form is CLASS-dependent, and the two classes have
// opposite answers (both measured; do not port a fix across the boundary):
//
//   * element-vs-element compare-EXCHANGES (the halver/sorter networks): the
//     fully branchless `(b0<a0)|(b0==a0 & b1<a1)` + XOR-mask swap wins
//     2.4-4.5x -- that is cswap's fast path above, unchanged.
//   * element-vs-FIXED-KEY scans (every partition kernel in this file): the
//     source repo's algorithms.hpp keeps the branchless form as a NEGATIVE
//     result (~30-47% slower at large n): the good lowering of the
//     short-circuit is `cmp first; je <second-key>` -- the branch fires only
//     on first-key TIES, rare on high-cardinality data, so it predicts
//     perfectly and the second-key load is SKIPPED.  The branchless form
//     pays the second key on every element for a predictor win that does
//     not exist here.
//
// PROBLEM: which lowering `pair::operator<` actually gets is decided by GCC
// per inlining context; some contexts get a ~50/50 mispredicting branch on
// the first-key ORDER instead (measured 1.9-3.6 ns/elem where the good
// lowering runs ~0.6 -- the round-2/3 forensics in cyclic_partitions.txt).
// The two comparators below remove the lottery:
//
//   * lex_less_packed: a 4+4-byte INTEGER pair loads as one u64
//     (little-endian: .first in the low half); rotl(32) + a sign-bias XOR
//     maps lex order onto ONE unsigned compare.  Exactly operator<, no
//     branch, no setcc chain, nothing to mis-lower; strictly the fastest
//     known form for these types.
//   * lex_less_semibranch: the documented-optimal short-circuit shape,
//     PINNED at the source level -- one compare of .first serves both the
//     [[unlikely]] tie branch and the ordering result, so every context
//     compiles to the algorithms.hpp fast form (cmov/setcc main path +
//     rarely-taken je).  Matches operator< exactly on +/-0.0 ties; NaN is
//     outside the strict-weak-ordering contract as everywhere else.
//
// The 3-setcc lex_less_branchless is kept for reference measurement and as
// the big-endian fallback; scans never pick it by default anymore.
// ===========================================================================

struct lex_less_branchless {
    template <class P>
    PARTITIONS_ALWAYS_INLINE bool operator()(const P& a, const P& b) const {
        const auto af = a.first;
        const auto as = a.second;
        const auto bf = b.first;
        const auto bs = b.second;
        const unsigned lt0 = static_cast<unsigned>(af < bf);
        const unsigned eq0 = static_cast<unsigned>(af == bf);
        const unsigned lt1 = static_cast<unsigned>(as < bs);
        return (lt0 | (eq0 & lt1)) != 0;
    }
};

struct lex_less_semibranch {
    template <class P>
    PARTITIONS_ALWAYS_INLINE bool operator()(const P& a, const P& b) const {
        if (a.first == b.first) [[unlikely]]
            return a.second < b.second;
        return a.first < b.first;
    }
};

template <class T>
concept int_pair_packable =
    lex_pair_like<T> && sizeof(T) == 8 &&
    std::is_integral_v<std::remove_reference_t<decltype(std::declval<T&>().first)>> &&
    std::is_integral_v<std::remove_reference_t<decltype(std::declval<T&>().second)>> &&
    sizeof(std::declval<T&>().first) == 4 &&
    sizeof(std::declval<T&>().second) == 4 &&
    std::endian::native == std::endian::little;

struct lex_less_packed {
    template <class P>
    PARTITIONS_ALWAYS_INLINE static std::uint64_t pack(const P& p) {
        std::uint64_t v;
        std::memcpy(&v, &p, 8);  // little-endian: .first is the LOW half
        v = std::rotl(v, 32);    // .first -> high half
        constexpr std::uint64_t bias =
            (std::is_signed_v<std::remove_reference_t<decltype(p.first)>>
                 ? 0x8000000000000000ull
                 : 0ull) |
            (std::is_signed_v<std::remove_reference_t<decltype(p.second)>>
                 ? 0x80000000ull
                 : 0ull);
        return v ^ bias;
    }
    template <class P>
    PARTITIONS_ALWAYS_INLINE bool operator()(const P& a, const P& b) const {
        return pack(a) < pack(b);
    }
};

// Substitute at a public entry point only when the comparison the kernels
// would otherwise run is std::pair's operator< with the pair as the key:
// the key IS the element type, the comparator is the default less, and the
// projection is identity.
template <class It, class K, class Comp, class Proj>
inline constexpr bool use_lex_subst =
    lex_pair_like<std::remove_cvref_t<K>> &&
    std::is_same_v<std::remove_cvref_t<K>, std::iter_value_t<It>> &&
    (std::is_same_v<Comp, std::less<>> ||
     std::is_same_v<Comp, std::less<std::iter_value_t<It>>>) &&
    std::is_same_v<Proj, std::identity>;

// The comparator the substitution installs for a given element type.
template <class T>
using subst_lex_comp_t =
    std::conditional_t<int_pair_packable<T>, lex_less_packed,
                       lex_less_semibranch>;

}  // namespace partitions::detail::net

// ===========================================================================
// detail::scalar -- portable partition kernels.
// ===========================================================================
namespace partitions::detail::scalar {

// ---------------------------------------------------------------------------
// Hoare -- two pointers converging from the ends.  The scalar small-n fallback
// of the block partition: no offset-buffer setup, ~half of Lomuto's swaps, and
// it splits all-equal input down the middle.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I hoare(I first, S last, K pivot, Comp comp, Proj proj) {
    auto below = [&](I it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };
    I lo = first;
    I hi = last;
    while (true) {
        while (lo != hi && below(lo)) ++lo;
        do {
            if (lo == hi) return lo;
            --hi;
        } while (!below(hi));
        std::iter_swap(lo, hi);
        ++lo;
    }
}

// ---------------------------------------------------------------------------
// Branchless Lomuto (orlp.net "gap" method) -- the small-block fast path.
//
// A single element is lifted out to open a "gap"; each iteration performs two
// moves into and out of the gap and advances the store position by the *value*
// of the predicate (0 or 1) instead of branching on it.  The compiler lowers
// `i += below(...)` to a conditional add, eliminating the mispredicted branch
// that dominates plain Lomuto on random data.
//
// Reference: Orson Peters, "Branchless Lomuto partitioning", orlp.net.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I lomuto_branchless(I first, S last, K pivot, Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    const D n = last - first;
    if (n == 0) return first;
    auto below = [&](D idx) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, first[idx]), pivot));
    };

    auto v = first;  // index with v[k]
    std::iter_value_t<I> tmp = std::move(v[0]);
    D i = 0;
    for (D j = 0; j < n - 1; ++j) {
        v[j] = std::move(v[i]);
        // gap is now at j; pull the next element into the gap at i.
        v[i] = std::move(v[j + 1]);
        i += static_cast<D>(below(i));
    }
    v[n - 1] = std::move(v[i]);
    v[i] = std::move(tmp);
    i += static_cast<D>(below(i));
    return first + i;
}

// ---------------------------------------------------------------------------
// Halver-based small partition ("sort_mid"): sort the tiny block by recursive
// balanced halving (pure branchless compare-exchanges, no data-dependent
// branch anywhere), then locate the boundary with a branchless count.  The
// sorted output is a valid partition around ANY key.  Requires `comp` to be a
// strict weak ordering (it compares elements to each other, not only to the
// key).  The dispatch gate (which sizes/types route here) is measured -- see
// tuning::halve_mid_max and the file header.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I halve_mid(I first, S last, K pivot, Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    I end = first + (last - first);
    net::halve_sort(first, end, comp, proj);
    // Branchless boundary: count of below-key elements (block is sorted,
    // but a straight setcc-add count over <= 24 elements vectorises and
    // never mispredicts, unlike a binary search).
    D m = 0;
    for (I it = first; it != end; ++it)
        m += static_cast<D>(static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot)));
    return first + m;
}

// ---------------------------------------------------------------------------
// Boost / pdqsort branchless block partition -- the large-block workhorse.
//
// Orson Peters' heavily micro-optimised refinement of BlockQuicksort (as
// shipped in boost::sort::pdqsort and the reference pdqsort), adapted to the
// pivot+comparator+projection primitive.  The three tricks that make it the
// fastest scalar partition known:
//
//   * CACHE: the two offset buffers are over-allocated and cacheline-aligned,
//     so each sits on its own 64-byte line -- the dense sequential writes never
//     straddle a line boundary or false-share with the other buffer.
//
//   * BRANCH PREDICTION: the offset-fill loops are fully branchless.  Each step
//     does `offsets[num] = i; num += predicate;` -- the misplaced index is
//     written unconditionally and the counter advances by the predicate's value
//     (0/1), which the compiler lowers to `setcc`/conditional-add with no
//     data-dependent jump.  The loops are unrolled 8x (block_size is a multiple
//     of 8) to amortise the loop counter.
//
//   * REGISTER PRESSURE / MOVES: when both buffers carry the same count the
//     swaps are a pure rotation (`swap_offsets` with use_swaps=false) -- one
//     temporary and 2N moves instead of 3N for N independent swaps.  The
//     `use_swaps=true` path is kept for the case the counts differ, which keeps
//     the descending distribution O(n).
//
// Measured 1.6-5.8x faster than scalar Hoare on random data across n=24..2^22
// for both 8- and 16-byte elements (see the source repo's docs/).
// ---------------------------------------------------------------------------

template <class T>
PARTITIONS_ALWAYS_INLINE T* align_cacheline(T* p) {
    auto ip = reinterpret_cast<std::uintptr_t>(p);
    ip = (ip + (tuning::cacheline_bytes - 1)) &
         ~(static_cast<std::uintptr_t>(tuning::cacheline_bytes) - 1);
    return reinterpret_cast<T*>(ip);
}

// Apply the at-most-`num` cross swaps recorded in the offset buffers.  With
// equal counts (use_swaps=false) this is a cyclic rotation: one temporary,
// two moves per element.  Otherwise fall back to independent iter_swaps.
template <class Iter>
PARTITIONS_ALWAYS_INLINE void swap_offsets(Iter first, Iter last,
                                           const unsigned char* offsets_l,
                                           const unsigned char* offsets_r,
                                           std::size_t num, bool use_swaps) {
    using T = typename std::iterator_traits<Iter>::value_type;
    if (use_swaps) {
        for (std::size_t i = 0; i < num; ++i)
            std::iter_swap(first + offsets_l[i], last - offsets_r[i]);
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
    }
}

// Partition [begin, end) around the value `pivot`.  Elements for which
// `comp(proj(x), pivot)` holds go left, the rest (incl. equal) go right;
// returns the boundary.  `Guarded` keeps the `first < last` bound on the
// initial left scan (value pivot, possibly absent); pass false only when an
// in-block element is guaranteed to stop it (position pivot sentinel).
template <bool Guarded, class Iter, class K, class Compare, class Proj>
inline Iter branchless_partition(Iter begin, Iter end, K pivot,
                                 Compare comp, Proj proj) {
    auto below = [&](Iter it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };

    Iter first = begin;
    Iter last = end;

    // Find the first element >= pivot.
    if constexpr (Guarded) {
        while (first < last && below(first)) ++first;
    } else {
        while (below(first)) ++first;  // in-block pivot is the sentinel
    }

    // Find the first element strictly below the pivot from the right.  Guard
    // the search if there was no element before *first.
    if (first == begin) {
        while (begin < last && !below(--last)) {
        }
    } else {
        while (!below(--last)) {
        }
    }

    // If the first pair that should be swapped is the same element, the input
    // was already partitioned.
    if (first < last) {
        std::iter_swap(first, last);
        ++first;

        alignas(tuning::cacheline_bytes)
            unsigned char offsets_l_storage[tuning::block_size + tuning::cacheline_bytes];
        alignas(tuning::cacheline_bytes)
            unsigned char offsets_r_storage[tuning::block_size + tuning::cacheline_bytes];
        unsigned char* offsets_l = align_cacheline(offsets_l_storage);
        unsigned char* offsets_r = align_cacheline(offsets_r_storage);

        Iter offsets_l_base = first;
        Iter offsets_r_base = last;
        std::size_t num_l = 0, num_r = 0, start_l = 0, start_r = 0;

        while (first < last) {
            // How many elements to scan into each offset block this round.
            std::size_t num_unknown = static_cast<std::size_t>(last - first);
            std::size_t left_split =
                num_l == 0 ? (num_r == 0 ? num_unknown / 2 : num_unknown) : 0;
            std::size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

            // Fill the left offset block with elements that belong on the right.
            if (left_split >= tuning::block_size) {
                for (std::size_t i = 0; i < tuning::block_size;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                }
            } else {
                for (std::size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = static_cast<unsigned char>(i++); num_l += !below(first); ++first;
                }
            }

            // Fill the right offset block with elements that belong on the left.
            if (right_split >= tuning::block_size) {
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
            } else {
                for (std::size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--last);
                }
            }

            // Swap the identified out-of-place pairs and advance the boundaries.
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

        // Drain any leftover offsets against the opposite end.
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
    return first;  // [begin,first) < pivot; [first,end) >= pivot
}

// Value pivot (possibly absent): guarded scan, scalar fallback for small n.
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I boost_block(I first, S last, K pivot, Comp comp, Proj proj) {
    I end = first + (last - first);
    if (end - first < tuning::block_scalar_cutoff)
        return hoare(first, end, pivot, comp, proj);
    return branchless_partition<true>(first, end, pivot, comp, proj);
}

// Position pivot (a real in-block element): unguarded scan -- the pivot is
// its own sentinel, so the per-element bound check is compiled out.
template <std::random_access_iterator I, std::sentinel_for<I> S,
          class Comp, class Proj>
inline I boost_block_at(I first, S last_s, I pivot, Comp comp, Proj proj) {
    I last = first + (last_s - first);
    auto key = std::invoke(proj, *pivot);  // stable copy
    if (last - first < tuning::block_scalar_cutoff)
        return hoare(first, last, key, comp, proj);
    return branchless_partition<false>(first, last, key, comp, proj);
}

// ---------------------------------------------------------------------------
// Size-dispatching scalar partitioner -- the portable tier.  Routes to the
// sub-partitioner that wins at the given block size: a cheap branchless Lomuto
// for small blocks (tuning::lomuto_cutoff), the pdqsort branchless block
// partition for large ones.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I sized(I first, S last, K pivot, Comp comp, Proj proj) {
    I end = first + (last - first);
    if (end - first <= tuning::lomuto_cutoff<std::iter_value_t<I>>)
        return lomuto_branchless(first, end, pivot, comp, proj);
    return boost_block(first, end, pivot, comp, proj);
}

// Position-aware fast path: the pivot is an iterator into the block, so the
// large-n branch can elide the left scan's bound check (sentinel).  Only
// sound for position pivots -- a key may be absent from the block.
template <std::random_access_iterator I, std::sentinel_for<I> S,
          class Comp, class Proj>
inline I sized_at(I first, S last_s, I pivot, Comp comp, Proj proj) {
    I last = first + (last_s - first);
    if (last - first <= tuning::lomuto_cutoff<std::iter_value_t<I>>) {
        auto key = std::invoke(proj, *pivot);  // stable copy (no sentinel path)
        return lomuto_branchless(first, last, key, comp, proj);
    }
    return boost_block_at(first, last, pivot, comp, proj);
}

}  // namespace partitions::detail::scalar

// ===========================================================================
// detail::avx2 -- AVX2 partition kernels for cheap-compare narrow integer
// keys (i32/i64, identity projection, `<`, contiguous storage).  Compiled
// only under -mavx2/-march=native.
// ===========================================================================
#if defined(__AVX2__)
namespace partitions::detail::avx2 {

// ---- 16-entry pack LUTs for 4-lane (i64) AVX2 fills -------------------------
// For a 4-bit lane mask, `pos[m]` packs the set lanes' local indices (0..3) in
// INCREASING order (matching the scalar left fill's scan order); `roff[m]`
// packs the right-fill offsets (4 - lane) in INCREASING offset-value order,
// matching the scalar right fill.  Order matters: the leftover-drain phase is
// only collision-free when offsets are monotonic, exactly as pdqsort produces
// them scalar-side.  The unused high bytes of each store are overwritten by
// the next group, exactly like the scalar loop's unconditional byte write.
struct Lut4 {
    std::uint32_t pos[16];
    std::uint32_t roff[16];
    std::uint8_t cnt[16];
};
inline constexpr Lut4 make_lut4() {
    Lut4 t{};
    for (int m = 0; m < 16; ++m) {
        std::uint8_t bp[4] = {0, 0, 0, 0};
        std::uint8_t br[4] = {0, 0, 0, 0};
        int k = 0;
        for (int b = 0; b < 4; ++b)
            if (m & (1 << b)) bp[k++] = static_cast<std::uint8_t>(b);
        int j = 0;
        for (int b = 3; b >= 0; --b)  // descending lane -> ascending offset 1..4
            if (m & (1 << b)) br[j++] = static_cast<std::uint8_t>(4 - b);
        t.cnt[m] = static_cast<std::uint8_t>(k);
        t.pos[m] = std::uint32_t(bp[0]) | (std::uint32_t(bp[1]) << 8) |
                   (std::uint32_t(bp[2]) << 16) | (std::uint32_t(bp[3]) << 24);
        t.roff[m] = std::uint32_t(br[0]) | (std::uint32_t(br[1]) << 8) |
                    (std::uint32_t(br[2]) << 16) | (std::uint32_t(br[3]) << 24);
    }
    return t;
}
inline constexpr Lut4 kLut4 = make_lut4();

// 256-entry equivalent for 8-lane (i32) fills.  pos/roff pack up to 8 bytes.
struct Lut8 {
    std::uint64_t pos[256];
    std::uint64_t roff[256];
    std::uint8_t cnt[256];
};
inline constexpr Lut8 make_lut8() {
    Lut8 t{};
    for (int m = 0; m < 256; ++m) {
        std::uint64_t pos = 0, roff = 0;
        int k = 0;
        for (int b = 0; b < 8; ++b)
            if (m & (1 << b)) pos |= std::uint64_t(b) << (8 * k++);
        int j = 0;
        for (int b = 7; b >= 0; --b)  // descending lane -> ascending offset 1..8
            if (m & (1 << b)) roff |= std::uint64_t(8 - b) << (8 * j++);
        t.cnt[m] = static_cast<std::uint8_t>(k);
        t.pos[m] = pos;
        t.roff[m] = roff;
    }
    return t;
}
inline constexpr Lut8 kLut8 = make_lut8();

// Is the (type, comparator, projection) a SIMD fast path?  Then
// below(x) == (x < pivot) on a contiguous block of a 4- or 8-byte signed int.
template <class T, class Comp, class Proj>
inline constexpr bool simd_eligible =
    std::is_same_v<Proj, std::identity> &&
    (std::is_same_v<Comp, std::less<>> || std::is_same_v<Comp, std::less<T>>) &&
    std::is_integral_v<T> && std::is_signed_v<T> &&
    (sizeof(T) == 8 || sizeof(T) == 4);

// Append, to offsets_l[num_l..], the local indices in [0,BS) of elements that
// are NOT below the pivot (>= pivot -> belong right).  base points at the block.
// 4-wide for 8-byte keys (vpcmpgtq + movmskpd, 16-entry LUT, 4-byte store);
// 8-wide for 4-byte keys (vpcmpgtd + movmskps, 256-entry LUT, 8-byte store).
template <class T>
PARTITIONS_ALWAYS_INLINE void vfill_left(const T* base, T pivot,
                                         unsigned char* offl,
                                         std::size_t& num_l,
                                         std::size_t BS) {
    if constexpr (sizeof(T) == 8) {
        const __m256i vp = _mm256_set1_epi64x(static_cast<long long>(pivot));
        for (std::size_t i = 0; i < BS; i += 4) {
            __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + i));
            __m256i lt = _mm256_cmpgt_epi64(vp, x);  // x < pivot
            unsigned bm = static_cast<unsigned>(
                _mm256_movemask_pd(_mm256_castsi256_pd(lt)));
            unsigned ge = (~bm) & 0xF;  // !below
            std::uint32_t packed =
                kLut4.pos[ge] + static_cast<std::uint32_t>(i) * 0x01010101u;
            std::memcpy(offl + num_l, &packed, 4);
            num_l += kLut4.cnt[ge];
        }
    } else {  // sizeof(T) == 4
        const __m256i vp = _mm256_set1_epi32(static_cast<int>(pivot));
        for (std::size_t i = 0; i < BS; i += 8) {
            __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + i));
            __m256i lt = _mm256_cmpgt_epi32(vp, x);  // x < pivot
            unsigned bm = static_cast<unsigned>(
                _mm256_movemask_ps(_mm256_castsi256_ps(lt)));
            unsigned ge = (~bm) & 0xFF;
            std::uint64_t packed =
                kLut8.pos[ge] + static_cast<std::uint64_t>(i) * 0x0101010101010101ull;
            std::memcpy(offl + num_l, &packed, 8);
            num_l += kLut8.cnt[ge];
        }
    }
}

// Append, to offsets_r[num_r..], the 1-based offsets-from-`rlast` (1..BS) of the
// elements that ARE below the pivot (belong left), scanning the block
// descending -- mirrors the scalar `offsets_r[num_r]=++i; num_r += below(--last)`.
template <class T>
PARTITIONS_ALWAYS_INLINE void vfill_right(const T* rlast, T pivot,
                                          unsigned char* offr,
                                          std::size_t& num_r,
                                          std::size_t BS) {
    if constexpr (sizeof(T) == 8) {
        const __m256i vp = _mm256_set1_epi64x(static_cast<long long>(pivot));
        for (std::size_t g = 0; g < BS; g += 4) {
            const T* p = rlast - g - 4;  // lane b -> offset g+(4-b)
            __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            __m256i lt = _mm256_cmpgt_epi64(vp, x);
            unsigned bm = static_cast<unsigned>(
                _mm256_movemask_pd(_mm256_castsi256_pd(lt)));
            std::uint32_t packed =
                kLut4.roff[bm] + static_cast<std::uint32_t>(g) * 0x01010101u;
            std::memcpy(offr + num_r, &packed, 4);
            num_r += kLut4.cnt[bm];
        }
    } else {  // sizeof(T) == 4
        const __m256i vp = _mm256_set1_epi32(static_cast<int>(pivot));
        for (std::size_t g = 0; g < BS; g += 8) {
            const T* p = rlast - g - 8;  // lane b -> offset g+(8-b)
            __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            __m256i lt = _mm256_cmpgt_epi32(vp, x);
            unsigned bm = static_cast<unsigned>(
                _mm256_movemask_ps(_mm256_castsi256_ps(lt)));
            std::uint64_t packed =
                kLut8.roff[bm] + static_cast<std::uint64_t>(g) * 0x0101010101010101ull;
            std::memcpy(offr + num_r, &packed, 8);
            num_r += kLut8.cnt[bm];
        }
    }
}

// ---- compaction (single-pass, swap-free) partition --------------------------
// vpermd compaction control LUTs.  For a lane mask, output lane `dst` reads
// input lane `src`: the set ("< pivot") lanes are gathered to the front, the
// clear (">= pivot") lanes to the back.  i32: 256 entries on 8 lanes.  i64: 16
// entries on 4 lanes, expressed as int32 pairs so one vpermd permutes 64-bit
// lanes.
struct PermLut32 { alignas(32) std::int32_t v[256][8]; };
struct PermLut64 { alignas(32) std::int32_t v[16][8]; };
inline constexpr PermLut32 make_perm32() {
    PermLut32 t{};
    for (int m = 0; m < 256; ++m) {
        int nl = 0;
        for (int i = 0; i < 8; ++i) nl += (m >> i) & 1;
        int l = 0, r = nl;
        for (int i = 0; i < 8; ++i) {
            int dst = ((m >> i) & 1) ? l++ : r++;
            t.v[m][dst] = i;
        }
    }
    return t;
}
inline constexpr PermLut64 make_perm64() {
    PermLut64 t{};
    for (int m = 0; m < 16; ++m) {
        int nl = 0;
        for (int i = 0; i < 4; ++i) nl += (m >> i) & 1;
        int l = 0, r = nl;
        for (int i = 0; i < 4; ++i) {
            int dst = ((m >> i) & 1) ? l++ : r++;
            t.v[m][2 * dst] = 2 * i;
            t.v[m][2 * dst + 1] = 2 * i + 1;
        }
    }
    return t;
}
inline constexpr PermLut32 kPerm32 = make_perm32();
inline constexpr PermLut64 kPerm64 = make_perm64();

// Partition [a, a+n) around `pivot` (a[0..m) < pivot <= a[m..n)) in ONE pass:
// read full vectors from both ends inward, compact each (< lanes to the front),
// and DOUBLE-STORE the compacted vector at the left write cursor and the right
// write cursor -- advancing left by popcount and right by the complement, so
// the overwritten halves are reclaimed as the cursors converge (the vectorised
// analogue of the fulcrum double write, with no offset buffer and no swap
// pass).  The two end vectors are preloaded so their slots are free; the < W
// middle remainder + the two buffered vectors are finished by a tiny scalar
// tail.  Reads are kept ahead of writes by always refilling the side with less
// slack.  Requires n >= 2*W.
template <class T>
inline std::size_t compress_partition(T* a, std::size_t n, T pivot) {
    constexpr std::size_t W = 32 / sizeof(T);
    const __m256i vp = sizeof(T) == 8
                           ? _mm256_set1_epi64x(static_cast<long long>(pivot))
                           : _mm256_set1_epi32(static_cast<int>(pivot));
    std::size_t i = W, j = n - W, sl = 0, sr = n;
    __m256i vL = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vR = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + n - W));

    auto cstore = [&](__m256i x) {
        unsigned mask;
        if constexpr (sizeof(T) == 8) {
            mask = static_cast<unsigned>(_mm256_movemask_pd(
                _mm256_castsi256_pd(_mm256_cmpgt_epi64(vp, x))));
            x = _mm256_permutevar8x32_epi32(
                x, _mm256_load_si256(reinterpret_cast<const __m256i*>(kPerm64.v[mask])));
        } else {
            mask = static_cast<unsigned>(_mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(vp, x))));
            x = _mm256_permutevar8x32_epi32(
                x, _mm256_load_si256(reinterpret_cast<const __m256i*>(kPerm32.v[mask])));
        }
        int nl = std::popcount(mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + sl), x);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + sr - W), x);
        sl += static_cast<std::size_t>(nl);
        sr -= (W - static_cast<std::size_t>(nl));
    };

    while (j - i >= W) {
        __m256i v;
        if ((i - sl) <= (sr - j)) {
            v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            i += W;
        } else {
            j -= W;
            v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
        }
        cstore(v);
    }

    // scalar tail: the two preloaded vectors + the < W middle remainder.
    T tmp[3 * W];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vL);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp + W), vR);
    std::size_t cnt = 2 * W;
    for (std::size_t k = i; k < j; ++k) tmp[cnt++] = a[k];
    for (std::size_t k = 0; k < cnt; ++k) {
        T x = tmp[k];
        if (x < pivot)
            a[sl++] = x;
        else
            a[--sr] = x;
    }
    return sl;  // sl == sr == m
}

// Mirror of scalar::branchless_partition, with the two FULL-block fills
// optionally vectorised.  `UseSimd` is the compile-time opt-in; the actual
// SIMD path is taken only when the type/comp/proj/iterator also qualify, else
// the identical scalar fill runs.  BS is the offset-block size (multiple of 8).
template <bool Guarded, bool UseSimd, std::size_t BS, class Iter, class K,
          class Compare, class Proj>
inline Iter branchless_partition_amd(Iter begin, Iter end, K pivot, Compare comp,
                                     Proj proj) {
    using T = std::iter_value_t<Iter>;
    auto below = [&](Iter it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };
    constexpr bool kSimd =
        UseSimd && std::contiguous_iterator<Iter> && simd_eligible<T, Compare, Proj>;

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
                    vfill_left<T>(&*first, pivot, offsets_l, num_l, BS);
                    first += BS;
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
                    vfill_right<T>(&*last, pivot, offsets_r, num_r, BS);
                    last -= BS;
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
            scalar::swap_offsets(offsets_l_base, offsets_r_base, offsets_l + start_l,
                                 offsets_r + start_r, num, num_l == num_r);
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;
            if (num_l == 0) { start_l = 0; offsets_l_base = first; }
            if (num_r == 0) { start_r = 0; offsets_r_base = last; }
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
    return first;
}

// Block partition with AVX2-vectorised offset fill for the i32/i64 fast path.
// Bandwidth-friendly (few writes): the winner once the block spills L2.
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I block_simd(I first, S last, K pivot, Comp comp, Proj proj) {
    I end = first + (last - first);
    if (end - first < tuning::block_scalar_cutoff)
        return scalar::hoare(first, end, pivot, comp, proj);
    return branchless_partition_amd<true, true, tuning::simd_block_size>(
        first, end, pivot, comp, proj);
}

// Position form: unguarded scan (the pivot element is its own sentinel).
template <std::random_access_iterator I, std::sentinel_for<I> S,
          class Comp, class Proj>
inline I block_simd_at(I first, S last_s, I pivot, Comp comp, Proj proj) {
    I last = first + (last_s - first);
    auto key = std::invoke(proj, *pivot);
    if (last - first < tuning::block_scalar_cutoff)
        return scalar::hoare(first, last, key, comp, proj);
    return branchless_partition_amd<false, true, tuning::simd_block_size>(
        first, last, key, comp, proj);
}

// Single-pass AVX2 compaction partition (no offset buffer, no swap pass) for
// the i32/i64 fast path; everything else falls back to boost_block.  The
// vectorised "swap" is the compaction itself: each loaded vector is compacted
// and committed to both ends in one pass.  Double-stores every element, so it
// wins only while the block is L2-resident.
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I block_compress(I first, S last, K pivot, Comp comp, Proj proj) {
    I end = first + (last - first);
    const auto n = end - first;
    using T = std::iter_value_t<I>;
    if constexpr (std::contiguous_iterator<I> && simd_eligible<T, Comp, Proj>) {
        if (n >= tuning::compress_min)
            return first + static_cast<std::iter_difference_t<I>>(
                               compress_partition<T>(&*first,
                                                     static_cast<std::size_t>(n),
                                                     static_cast<T>(pivot)));
    }
    return scalar::boost_block(first, end, pivot, comp, proj);
}

}  // namespace partitions::detail::avx2
#endif  // __AVX2__

// ===========================================================================
// sized_partition -- the general-use keyed forward partition: one entry point
// that dispatches to the fastest kernel for the (array size, element size,
// key / comparator cost, ISA) at hand.  See the file header for the tier
// table.  sized_partition_at is the position form (sentinel elision).
// ===========================================================================
namespace partitions {

namespace detail {
#if defined(__AVX2__)
// A contiguous block of a 4/8-byte signed integer key compared with the
// default `<` and an identity projection, on an AVX2 machine.
template <class I, class Comp, class Proj>
inline constexpr bool simd_fast_path =
    std::contiguous_iterator<I> &&
    avx2::simd_eligible<std::iter_value_t<I>, Comp, Proj>;
#else
template <class I, class Comp, class Proj>
inline constexpr bool simd_fast_path = false;
#endif
}  // namespace detail

template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp = std::less<>, class Proj = std::identity>
I sized_partition(I first, S last, K key, Comp comp = {}, Proj proj = {}) {
    // Pair-lex under the default `<`: swap in the pinned scan comparator
    // (lex_less_packed for 4+4-byte integer pairs, lex_less_semibranch
    // otherwise -- see detail::net) and re-dispatch.  Same order, immune to
    // the context-dependent lowering of pair::operator< (measured up to
    // ~3x; cyclic_partitions.txt rounds 2-3).  Skipped when the halver tier
    // is enabled: that tier really sorts, and the networks' own comparator
    // fast path (cswap) must keep seeing the default less.
    if constexpr (detail::net::use_lex_subst<I, K, Comp, Proj> &&
                  tuning::halve_mid_max == 0) {
        return sized_partition(
            first, last, key,
            detail::net::subst_lex_comp_t<std::iter_value_t<I>>{}, proj);
    } else {
    I end = first + (last - first);
    const auto n = end - first;
    if constexpr (tuning::halve_mid_max > 0) {
        if (n <= tuning::halve_mid_max)
            return detail::scalar::halve_mid(first, end, key, comp, proj);
    }
    if constexpr (detail::simd_fast_path<I, Comp, Proj>) {
#if defined(__AVX2__)
        if (n < tuning::simd_lomuto_cutoff)
            return detail::scalar::lomuto_branchless(first, end, key, comp, proj);
        if (n <= tuning::compress_max<std::iter_value_t<I>>)
            return detail::avx2::block_compress(first, end, key, comp, proj);
        return detail::avx2::block_simd(first, end, key, comp, proj);
#endif
    } else {
        return detail::scalar::sized(first, end, key, comp, proj);
    }
    }  // use_lex_subst
}

// Position form.  For the wide/expensive path keep `sized`'s sentinel fast
// path; for the halver/SIMD paths the kernels are value-based, so read the
// key and route by size.
template <std::random_access_iterator I, std::sentinel_for<I> S,
          class Comp = std::less<>, class Proj = std::identity>
I sized_partition_at(I first, S last_s, I pivot, Comp comp = {}, Proj proj = {}) {
    // Same pinned-comparator substitution as sized_partition (the key here
    // is proj(*pivot) = the element type itself).
    if constexpr (detail::net::use_lex_subst<I, std::iter_value_t<I>, Comp,
                                             Proj> &&
                  tuning::halve_mid_max == 0) {
        return sized_partition_at(
            first, last_s, pivot,
            detail::net::subst_lex_comp_t<std::iter_value_t<I>>{}, proj);
    } else {
    I last = first + (last_s - first);
    const auto n = last - first;
    if constexpr (tuning::halve_mid_max > 0) {
        if (n <= tuning::halve_mid_max) {
            auto key = std::invoke(proj, *pivot);
            return detail::scalar::halve_mid(first, last, key, comp, proj);
        }
    }
    if constexpr (detail::simd_fast_path<I, Comp, Proj>) {
#if defined(__AVX2__)
        auto key = std::invoke(proj, *pivot);
        if (n < tuning::simd_lomuto_cutoff)
            return detail::scalar::lomuto_branchless(first, last, key, comp, proj);
        if (n <= tuning::compress_max<std::iter_value_t<I>>)
            return detail::avx2::block_compress(first, last, key, comp, proj);
        return detail::avx2::block_simd_at(first, last, pivot, comp, proj);
#endif
    } else {
        return detail::scalar::sized_at(first, last, pivot, comp, proj);
    }
    }  // use_lex_subst
}

}  // namespace partitions

// ===========================================================================
// detail::pivot + median_partition -- partition around an ESTIMATED MEDIAN;
// no key is taken.
//
// This is where the halver networks earn their keep (the keyed dispatcher
// above deliberately leaves them off): with no key given, a keyed partition
// would first have to SELECT a pivot, while a halver splits by rank directly
// -- deterministic n/2 balance, no pivot-selection step, and ~20-33% fewer
// compare-exchanges than a full sorting network.  The size tiers are the
// measured settings of the source repo's pure quicksort (its
// docs/pure_quicksort.md and docs/quicksort_pivot_tiers.md):
//
//   n <= 24      one halver network application         -> exact n/2 split
//   n <= 65536   ninther pivot (9 spread samples)       -> keyed partition
//   n >  65536   median-of-5-medians-of-5 (25 samples)  -> keyed partition
//
// Two deliberate choices inherited from that quicksort:
//   * ninther, not median_of_3, on the mid-size tier: m3 is the textbook
//     median-of-3 killer -- on sorted/reverse input the partitioner's
//     deterministic output feeds it near-EXTREME pivots, collapsing a
//     quicksort built on this split to O(n^2); ninther's 9 spread samples
//     resist that.  m5m5's 25 samples buy extra balance on huge blocks where
//     the sampling cost fully amortises over the O(n) partition.
//   * the keyed step runs through sized_partition (the repo's quicksort uses
//     its scalar `sized` there), so AVX2-eligible types keep their SIMD
//     kernels.
//
// MEASURED (Zen 3, GCC -O3 -march=native, batched independent random blocks,
// min ns/elem, single standalone split): vs a median_of_3 + keyed-partition
// alternative, the halver tier wins on i64 (n=8: 0.61 vs 1.35; n=24: 0.82 vs
// 0.83) and, with the branchless member-wise compare-exchange (see
// pair_like_swappable), wins or ties on 16-byte pairs too (n=8: 2.11 vs
// 3.10; n=16: 3.00 vs 3.60; n=24: 3.64 vs 3.51) -- while giving an EXACT n/2
// split where m3's is a noisy 3-sample estimate.  (Before that fix, std::pair
// elements fell onto a branch-per-compare-exchange path and the halver lost
// 1.3-2.4x on pairs.)  The tiers follow the source quicksort's settings.
//
// CONTRACT.  Reorders [first, last) and returns a split point m such that no
// element of the right side is comp-less than any element of the left side:
//
//     for all i in [first, m), j in [m, last):  !comp(proj(*j), proj(*i))
//
//   * n <= 24:  m = first + n/2 EXACTLY (a rank split: both sides remain
//               unordered and no element is at its final sorted position).
//   * n >  24:  *m is the sampled pseudo-median placed at its final sorted
//               rank: [first, m) is strictly < proj(*m) and [m+1, last) is
//               >= proj(*m).  A quicksort/quickselect built on this should
//               recurse on [first, m) and [m+1, last) -- excluding m
//               guarantees progress even on heavy-duplicate input, where
//               m == first is possible (nothing is strictly below the median
//               value).  m < last always holds for n >= 1.
//
// `comp` must be a strict weak ordering: both tiers compare elements with
// each other, not merely against a fixed key.
// ===========================================================================
namespace partitions::detail::pivot {

// Median of the values referenced by three iterators (returns the iterator to
// the median value).  `less(x, y)` compares the values at iterators x and y.
template <class I, class Less>
inline I median3(I a, I b, I c, Less less) {
    if (less(a, b)) {
        if (less(b, c)) return b;
        return less(a, c) ? c : a;
    }
    if (less(a, c)) return a;
    return less(b, c) ? c : b;
}

// Median value among the iterators in [lo, hi); chosen by sorting a copy of
// the iterators (cheap for the small groups used here), input untouched.
template <class I, class Less>
inline I median_of_range(I* lo, I* hi, Less less) {
    std::sort(lo, hi, [&](I x, I y) { return less(x, y); });
    return lo[(hi - lo) / 2];
}

// Build the "less over iterators" comparator from comp + proj.
template <class Comp, class Proj>
inline auto make_less(Comp comp, Proj proj) {
    return [comp = std::move(comp), proj = std::move(proj)](auto x, auto y) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *x), std::invoke(proj, *y)));
    };
}

template <class I, class S, class Comp, class Proj>
inline I median_of_3(I first, S last, Comp comp, Proj proj) {
    const auto n = last - first;
    if (n < 3) return first + n / 2;
    auto less = make_less(comp, proj);
    return median3(first, first + n / 2, first + (n - 1), less);
}

template <class I, class S, class Comp, class Proj>
inline I median_of_5(I first, S last, Comp comp, Proj proj) {
    const auto n = last - first;
    if (n < 5) return median_of_3(first, last, comp, proj);
    auto less = make_less(comp, proj);
    I s[5] = {first, first + n / 4, first + n / 2, first + (3 * n) / 4,
              first + (n - 1)};
    return median_of_range(s, s + 5, less);
}

// Tukey's ninther: median of three medians-of-three taken over nine evenly
// spaced elements.  A cheap, high-quality pivot for large blocks.
template <class I, class S, class Comp, class Proj>
inline I ninther(I first, S last, Comp comp, Proj proj) {
    const auto n = last - first;
    if (n < 9) return median_of_3(first, last, comp, proj);
    auto less = make_less(comp, proj);
    const auto e = n / 8;  // spacing
    I lo = median3(first, first + e, first + 2 * e, less);
    I mid = median3(first + n / 2 - e, first + n / 2, first + n / 2 + e, less);
    I hi = median3(first + (n - 1) - 2 * e, first + (n - 1) - e,
                   first + (n - 1), less);
    return median3(lo, mid, hi, less);
}

// Median of five medians-of-five: the 5x5 analogue of the ninther.  Samples 25
// evenly-spaced elements, takes the median of each consecutive group of five,
// then the median of those five medians.  A constant-cost pseudo-median that
// samples more widely than the ninther (25 vs 9).
template <class I, class S, class Comp, class Proj>
inline I median_of_5_medians_of_5(I first, S last, Comp comp, Proj proj) {
    const auto n = last - first;
    if (n < 25) return median_of_5(first, last, comp, proj);
    auto less = make_less(comp, proj);
    const auto step = n - 1;
    I samp[25];
    for (int k = 0; k < 25; ++k)
        samp[k] = first + (static_cast<decltype(n)>(k) * step) / 24;
    I med[5];
    for (int g = 0; g < 5; ++g)
        med[g] = median_of_range(samp + g * 5, samp + g * 5 + 5, less);
    return median_of_range(med, med + 5, less);
}

}  // namespace partitions::detail::pivot

namespace partitions {

template <std::random_access_iterator It, class Comp = std::less<>,
          class Proj = std::identity>
It median_partition(It first, It last, Comp comp = {}, Proj proj = {}) {
    const auto n = last - first;
    if (n <= tuning::median_halve_cutoff)  // exact n/2 rank split, no pivot step
        return detail::net::halve(first, last, comp, proj);
    It p = n > tuning::median_m5m5_cutoff
               ? detail::pivot::median_of_5_medians_of_5(first, last, comp, proj)
               : detail::pivot::ninther(first, last, comp, proj);
    std::iter_swap(first, p);              // pivot to front, out of the block
    auto key = std::invoke(proj, *first);  // read the key AFTER the swap
    It m = sized_partition(first + 1, last, key, comp, proj);
    std::iter_swap(first, m - 1);          // pivot to its final sorted rank
    return m - 1;
}

}  // namespace partitions

// ===========================================================================
// detail::offset + offset_partition -- forward partition with a known
// all->= prefix.
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
// ===========================================================================
namespace partitions::detail::offset {

// ---------------------------------------------------------------------------
// Branchless Lomuto gap method started at `offset` -- the narrow-type /
// prefix-dominated route.  scalar::lomuto_branchless with the warm-up replaced
// by the precondition: lifting v[offset] opens the gap exactly in the state
// the standard loop would reach after `offset` iterations that found nothing
// below (i = 0, [0, offset) not-below).  Cost is p-flat: 1 compare and 2
// sequential moves per suffix element regardless of the below fraction.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I gap_off(I first, S last_s, std::iter_difference_t<I> offset, K key,
                 Comp comp, Proj proj) {
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

// ---------------------------------------------------------------------------
// Prefix-fill -- the offset-aware algorithm proper.
//
// INSIGHT (from measurement): in a forward block partition the LEFT cursor
// only advances as far as the boundary c, so when c < offset a fused
// prefix-aware block partition ends up COMPARING the prefix from the right
// side anyway.  The prefix must be treated as what it is: a supply of swap
// TARGETS, not scan input.
//
// Phase 1: scan the suffix from the right in branchless blocks (the pdqsort
// offset fill: unconditional byte write + setcc add, 8x unrolled); each found
// below element is swapped DIRECTLY into the next front slot.  The left side
// of the pairing is consecutive (lo, lo+1, ...), so the same
// 2-moves-per-element cyclic rotation the block partition uses applies -- no
// compare ever touches the prefix, and each below element is moved exactly
// once.  The consumed right block becomes all->= and is settled.
//
// Phase 2 (slots nearly exhausted -- c approaching offset -- or suffix
// exhausted): partition the remaining suffix, then ONE swap_ranges bridges the
// leftover [lo, pfx_end) gap.  Loop granularity guarantees phase 1 only runs
// whole blocks, so no element is ever rescanned.  The phase-2 partition goes
// through sized_partition, so an AVX2-eligible integer suffix gets the
// vectorised kernels here too (the source repo used its scalar `sized`; the
// dispatch is contract-identical and strictly faster on the eligible types).
//
// Net cost: exactly (n - offset) compares; min(c, offset)-ish elements moved
// once (rotation), the rest as an optimal partition + one block swap.
// ---------------------------------------------------------------------------

// Move the `num` below elements recorded in the right offset buffer into the
// consecutive slots [lo, lo + num), displacing the slot contents to the
// vacated right positions.  This is swap_offsets' use_swaps=false cyclic
// rotation (one temporary, two moves per element) specialised to a
// CONSECUTIVE left side: indexing lo[i] directly instead of through an
// identity offset table saves a byte load and an address dependency per
// element (GCC does not fold a constant identity table).
template <class Iter>
PARTITIONS_ALWAYS_INLINE void fill_slots(Iter lo, Iter r_base,
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

template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I prefix_fill(I first, S last_s, std::iter_difference_t<I> offset, K key,
                     Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    constexpr D B = static_cast<D>(tuning::block_size);
    I last = first + (last_s - first);
    I lo = first;             // next front slot to fill (always < pfx_end)
    I pfx_end = first + offset;
    auto below = [&](I it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), key));
    };

    alignas(tuning::cacheline_bytes)
        unsigned char offsets_r_storage[tuning::block_size +
                                        tuning::cacheline_bytes];
    unsigned char* offsets_r = scalar::align_cacheline(offsets_r_storage);

    // Phase 1: whole right blocks while both slots and suffix last.
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
        // Pair the found belows with consecutive front slots: a
        // 2-moves-per-element cyclic rotation (see fill_slots).
        fill_slots(lo, last + B, offsets_r, num_r);
        lo += static_cast<D>(num_r);
    }

    // Phase 2: partition what is left of the suffix, then bridge the
    // remaining [lo, pfx_end) gap with one block swap.  After the
    // partition the layout is
    //     [ <key : lo-first ][ >=key gap ][ <key : c_rem ][ >=key ]
    // and c_rem <= slots swaps the below run with the FIRST c_rem gap
    // slots; c_rem > slots swaps the whole gap with the LAST slots
    // elements of the below run (disjoint since c_rem > slots).  Both
    // arms are two sequential streams -- branch-free and vectorised.
    I m = sized_partition(pfx_end, last, key, comp, proj);
    const D c_rem = m - pfx_end;                  // [pfx_end, m) < key
    const D slots = pfx_end - lo;
    if (c_rem <= slots)
        std::swap_ranges(lo, lo + c_rem, pfx_end);
    else
        std::swap_ranges(lo, pfx_end, m - slots);
    return lo + c_rem;
}

// ---------------------------------------------------------------------------
// Size-dispatching offset partitioner -- THE general-use offset partition,
// mirror of the forward dispatcher.  Thresholds from the source repo's
// bench_offset_partition sweep:
//
//   * tiny suffix (<= tuning::lomuto_cutoff<T>: 512 narrow / 24 wide):
//     gap_off -- no setup, branch-free, over the COMPARED length (the suffix);
//   * narrow T (<= 8 bytes) with offset >= suffix: gap_off -- its flat
//     2-moves-per-element stream beats prefix_fill's per-below scattered
//     rotation ON AVERAGE OVER the below fraction p when the prefix dominates
//     (p is unknowable at call time, so this is an expected-cost choice);
//   * everything else: prefix_fill (for 16-byte elements gap_off's double
//     wide moves lose 1.4-1.5x at small offset fractions and 2-3x in batched
//     small blocks, and only ever reach ~5% parity when the prefix dominates).
//
// At offset == 0 both routed algorithms degenerate to a raw forward partition
// by construction -- gap_off(0) IS lomuto_branchless (identical loop body),
// and prefix_fill(0) fails its phase-1 guard immediately and partitions the
// full range (empty bridge) -- with measured overhead indistinguishable from
// the code-placement noise floor.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
          class Comp, class Proj>
inline I sized_off(I first, S last_s, std::iter_difference_t<I> offset, K key,
                   Comp comp, Proj proj) {
    I last = first + (last_s - first);
    const auto suffix = (last - first) - offset;
    if (suffix <= tuning::lomuto_cutoff<std::iter_value_t<I>>)
        return gap_off(first, last, offset, key, comp, proj);
    if constexpr (sizeof(std::iter_value_t<I>) <= 8) {
        if (offset >= suffix)
            return gap_off(first, last, offset, key, comp, proj);
    }
    return prefix_fill(first, last, offset, key, comp, proj);
}

}  // namespace partitions::detail::offset

namespace partitions {

// Offset partition entry point: returns the COUNT of below-key elements (now
// at the front).  Precondition: [first, first+offset) all >= key, i.e.
// !comp(proj(x), key) for every element of the prefix.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::ptrdiff_t offset_partition(It first, It last,
                                std::iter_difference_t<It> offset, K key,
                                Comp comp = {}, Proj proj = {}) {
    // Pinned pair-lex scan comparator, exactly as in sized_partition.
    if constexpr (detail::net::use_lex_subst<It, K, Comp, Proj>) {
        return offset_partition(
            first, last, offset, key,
            detail::net::subst_lex_comp_t<std::iter_value_t<It>>{}, proj);
    } else {
        return detail::offset::sized_off(first, last, offset, key, comp,
                                         proj) -
               first;
    }
}

}  // namespace partitions

// ===========================================================================
// detail::cyclic + cyclic_partition / cyclic_offset_partition -- the keyed
// forward partition and the offset partition over a possibly-WRAPPED range of
// a cyclic buffer [buf_begin, buf_end).
//
// RANGE CONVENTION.  The logical range is given by (first, last) inside the
// buffer:
//     first <= last : the ordinary contiguous [first, last)
//     first >  last : the WRAPPED range  S1 ++ S2,
//                     S1 = [first, buf_end),  S2 = [buf_begin, last)
// first == last means EMPTY (the standard ring convention -- a full wrapped
// buffer is not expressible; pass the flat [buf_begin, buf_end) instead).
// Elements outside the logical range are never read or written.
//
// CONTRACT (logical view, identical to the flat entry points): on return,
// logical positions [0, c) satisfy comp(proj(x), key) and [c, n) do not,
// where c is the below-key count.  cyclic_partition returns the PHYSICAL
// boundary iterator normalized into [buf_begin, buf_end) (logical index c;
// when first + c lands exactly on buf_end it wraps to buf_begin);
// cyclic_partition_count and cyclic_offset_partition return c itself.
//
// DESIGN (full idea log + measurements: cyclic_partitions.txt).  A wrapped
// range is two contiguous segments, and partitions are not stable, so no
// per-element wrap arithmetic is ever needed:
//
//   * gap_cyclic -- small-n tier.  The branchless Lomuto gap method run over
//     the logical range: the gap cursor j is CHUNKED (flat over S1, one
//     explicit wrap step, flat over S2 -- zero per-step wrap cost), and the
//     data-dependent boundary cursor i, which cannot be chunked, advances by
//     `pi += below; pi = pi == buf_end ? buf_begin : pi` -- a cmp+cmov, never
//     a branch.  While j is still inside S1, i <= j keeps pi in S1 too, so
//     chunk A carries NO wrap check at all.
//
//   * seg_bridge -- large-n tier.  Flat-partition S1 and S2 independently
//     (each through sized_partition, so the AVX2 kernels are reused
//     unchanged), then join with ONE swap_ranges "bridge": with layouts
//     S1 = [b1|g1], S2 = [b2|g2], swapping min(|b2|, |g1|) elements makes the
//     below groups logically contiguous -- if |b2| <= |g1| pull all of b2
//     into the front of g1, else swap all of g1 with the LAST |g1| elements
//     of b2 (disjoint).  Exactly n compares; the bridge is a vectorized
//     optimal exchange bounded by min(len1, len2, c, n-c).
//
//   * cross_fill2 / seg_fill -- the offset-partition machinery generalized
//     across the wrap.  prefix_fill's slot region and scan region never had
//     to be contiguous with each other: after flat-partitioning S1, its >=
//     tail g1 is a supply of all->= swap slots, and S2 is scanned from the
//     RIGHT in branchless offset blocks, each found below rotated directly
//     into the next slot (2 moves, moved exactly once -- no bridge
//     double-handling).  cross_fill2 takes the slot supply as up to TWO
//     contiguous runs so the cyclic offset partition's spilled prefix
//     (S1 entirely >= plus a leading piece of S2) fits the same kernel:
//     fill_slots is semantically a pairwise slot[i] <-> src[i] exchange, so
//     splitting one block's rotation across the run boundary into two calls
//     is legal.  On AVX2 the block scan uses vfill_right for eligible
//     integer types.
//
// The pivot key is BY VALUE for the same aliasing reason as everywhere else
// in this file.
// ===========================================================================
namespace partitions::detail::cyclic {

// ---------------------------------------------------------------------------
// wrap_hoare -- correctness reference ONLY (never dispatched): Hoare over
// logical indices with a wrap check on every access.  This is the naive cost
// model the real kernels are measured against.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> wrap_hoare(I first, I last, I buf_begin,
                                            I buf_end, K key, Comp comp,
                                            Proj proj) {
    using D = std::iter_difference_t<I>;
    const D len1 = buf_end - first;
    const D n = len1 + (last - buf_begin);
    auto at = [&](D k) { return k < len1 ? first + k : buf_begin + (k - len1); };
    auto below = [&](D k) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *at(k)), key));
    };
    D lo = 0, hi = n;
    while (true) {
        while (lo != hi && below(lo)) ++lo;
        do {
            if (lo == hi) return lo;
            --hi;
        } while (!below(hi));
        std::iter_swap(at(lo), at(hi));
        ++lo;
    }
}

// ---------------------------------------------------------------------------
// gap_cyclic -- branchless Lomuto gap method over a WRAPPED range, with the
// gap opened at logical `offset` (offset = 0 gives the plain partition; a
// positive offset is the offset-partition form, exactly like gap_off).
//
// PRECONDITIONS: the range wraps (len1 >= 1, len2 >= 1) and offset < n.
// Cost: (n - offset) compares + 2(n - offset) moves, p-flat; the only wrap
// tax is one cmp+cmov per iteration on the boundary cursor -- and none at
// all while the gap cursor is still in S1 (i <= j keeps pi in S1 there).
// Returns the below-key count.
//
// LOOP-BODY SHAPE (load once, compare the REGISTER, then store).  The naive
// transcription `*pj = move(*pi); *pi = move(pj[1]); i += below(pi);` makes
// the predicate read back through *pi.  Whether GCC then keeps the value in
// a register is a register-allocation lottery: in several instantiation
// contexts (measured: the public dispatcher, and any standalone
// instantiation, for pair<float,int>/pair<int,int>/pair<double,long>) the
// setcc scratch register was allocated on top of a value RELOADED from *pi,
// and the partial-register merge (`setnp al` into a register whose upper
// bytes came from that load) put a store-to-load forward onto the
// loop-carried dependency -- ~12 cycles/iteration instead of ~4 (2.6x,
// p-independent, no extra branches; full forensics in cyclic_partitions.txt
// round 2).  Naming the moved element `x` and comparing x BEFORE storing it
// removes the memory round-trip from the predicate entirely, which pins the
// fast shape in every context.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> gap_cyclic(I first, I last, I buf_begin,
                                            I buf_end,
                                            std::iter_difference_t<I> offset,
                                            K key, Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    using T = std::iter_value_t<I>;
    const D len1 = buf_end - first;
    auto below_v = [&](const T& v) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, v), key));
    };

    I gap = offset < len1 ? first + offset : buf_begin + (offset - len1);
    T tmp = std::move(*gap);
    I pi = first;  // physical cursor of the boundary index i (starts at 0)

    if (offset < len1) {
        // Chunk A: j in [offset, len1-1) -- the gap and its successor both in
        // S1, and i <= j < len1-1 keeps pi in S1: no wrap handling at all.
        for (I pj = gap; pj != buf_end - 1; ++pj) {
            *pj = std::move(*pi);
            T x = std::move(pj[1]);
            const bool b = below_v(x);
            *pi = std::move(x);
            pi += static_cast<D>(b);
        }
        // j = len1-1: the gap's successor is across the wrap (S2[0]); pi may
        // step onto buf_end exactly here, normalize once.
        *(buf_end - 1) = std::move(*pi);
        {
            T x = std::move(*buf_begin);
            const bool b = below_v(x);
            *pi = std::move(x);
            pi += static_cast<D>(b);
        }
        pi = pi == buf_end ? buf_begin : pi;
    }
    // Chunk B: j in [max(offset, len1), n-1) -- flat over S2.  pi crosses the
    // wrap AT MOST ONCE, so the crossing test must not ride the pointer
    // dependency chain: while pi is still in S1 it is a predicted-not-taken
    // BRANCH (a cmov here measured ~1.6 cycles/elem slower -- it serialises
    // the pi chain; see cyclic_partitions.txt R1), and once pi has crossed
    // the remaining iterations run with no test at all (pi <= j keeps it
    // strictly inside S2 from then on).
    I pj = offset < len1 ? buf_begin : gap;
    if (pi >= first) {  // pi still in S1
        for (; pj != last - 1; ++pj) {
            *pj = std::move(*pi);
            T x = std::move(pj[1]);
            const bool b = below_v(x);
            *pi = std::move(x);
            pi += static_cast<D>(b);
            if (pi == buf_end) {  // one-shot wrap
                pi = buf_begin;
                ++pj;
                break;
            }
        }
    }
    for (; pj != last - 1; ++pj) {
        *pj = std::move(*pi);
        T x = std::move(pj[1]);
        const bool b = below_v(x);
        *pi = std::move(x);
        pi += static_cast<D>(b);
    }
    // j = n-1: the lifted element comes home through the final gap.
    *(last - 1) = std::move(*pi);
    const bool btmp = below_v(tmp);
    *pi = std::move(tmp);
    const D i = pi >= first ? pi - first : len1 + (pi - buf_begin);
    return i + static_cast<D>(btmp);
}

// ---------------------------------------------------------------------------
// bridge -- join two independently partitioned segments with ONE swap_ranges.
// PRECONDITION: [m1, s1_end) is all >= key and [s2_begin, m2) is all < key,
// with segment 1 logically preceding segment 2.  Swaps k = min(m2 - s2_begin,
// s1_end - m1) elements; both arms degenerate to empty swaps, so no special
// cases.  (Proof of both arms: cyclic_partitions.txt section 3.)
// ---------------------------------------------------------------------------
template <class I>
PARTITIONS_ALWAYS_INLINE void bridge(I m1, I s1_end, I s2_begin, I m2) {
    const auto c2 = m2 - s2_begin;
    const auto g1 = s1_end - m1;
    if (c2 <= g1)
        std::swap_ranges(s2_begin, m2, m1);   // all of b2 into the front of g1
    else
        std::swap_ranges(m1, s1_end, m2 - g1);  // all of g1 with b2's tail
}

// ---------------------------------------------------------------------------
// seg_bridge -- flat-partition both segments (every measured flat kernel,
// including the AVX2 compress/block_simd tiers, reused unchanged), then one
// bridge.  Exactly n compares.  Returns the below-key count.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> seg_bridge(I first, I last, I buf_begin,
                                            I buf_end, K key, Comp comp,
                                            Proj proj) {
    I m1 = sized_partition(first, buf_end, key, comp, proj);
    I m2 = sized_partition(buf_begin, last, key, comp, proj);
    bridge(m1, buf_end, buf_begin, m2);
    return (m1 - first) + (m2 - buf_begin);
}

// ---------------------------------------------------------------------------
// cross_fill2 -- prefix_fill generalized across the wrap: move every
// below-key element of the contiguous suffix [sfirst, slast) into the leading
// SLOTS, where the slot supply is up to TWO contiguous all->= runs
// [r1lo, r1end) then [r2lo, r2end), logically ordered runs-then-suffix.
// Returns the number of below elements found in the suffix.
//
// Phase 1 scans the suffix from the RIGHT in branchless offset blocks
// (unconditional byte write + setcc add, 8x unrolled; vfill_right on the AVX2
// integer fast path) and rotates each block's belows into the next slots
// (fill_slots: 2 moves per element, moved exactly once).  A block's belows
// may straddle the run boundary: fill_slots is a pairwise slot[i] <-> src[i]
// exchange, so it is split into two calls.  Phase 2 flat-partitions the
// remaining suffix and bridges the below-run into the remaining slot runs
// (<= 2 swap_ranges per arm; the min() split keeps filled slots a LEADING
// logical run in both arms).  With no slots at all this degenerates to a
// plain flat partition of the suffix.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> cross_fill2(I r1lo, I r1end, I r2lo, I r2end,
                                             I sfirst, I slast, K key,
                                             Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    constexpr D B = static_cast<D>(tuning::block_size);
    auto below = [&](I it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), key));
    };
#if defined(__AVX2__)
    using T = std::iter_value_t<I>;
    constexpr bool kSimd =
        std::contiguous_iterator<I> && avx2::simd_eligible<T, Comp, Proj>;
#else
    constexpr bool kSimd = false;
#endif

    alignas(tuning::cacheline_bytes)
        unsigned char offsets_r_storage[tuning::block_size +
                                        tuning::cacheline_bytes];
    unsigned char* offsets_r = scalar::align_cacheline(offsets_r_storage);

    I lo = r1lo, lo_end = r1end;
    if (lo == lo_end) {  // run 1 empty: consume run 2 as the current run
        lo = r2lo;
        lo_end = r2end;
        r2lo = r2end;
    }
    D slots_rem = (lo_end - lo) + (r2end - r2lo);
    D found = 0;

    // Phase 1: whole right blocks while a full block's belows always fit.
    while (slots_rem >= B && slast - sfirst >= B) {
        std::size_t num_r = 0;
        if constexpr (kSimd) {
#if defined(__AVX2__)
            avx2::vfill_right<T>(std::to_address(slast), key, offsets_r, num_r,
                                 tuning::block_size);
            slast -= B;
#endif
        } else {
            for (std::size_t i = 0; i < tuning::block_size;) {
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
                offsets_r[num_r] = static_cast<unsigned char>(++i); num_r += below(--slast);
            }
        }
        const D rem = static_cast<D>(num_r);
        const D run1_room = lo_end - lo;
        const D k1 = rem < run1_room ? rem : run1_room;
        offset::fill_slots(lo, slast + B, offsets_r,
                           static_cast<std::size_t>(k1));
        lo += k1;
        if (lo == lo_end) {  // run exhausted exactly: switch to run 2
            lo = r2lo;
            lo_end = r2end;
            r2lo = r2end;
        }
        if (k1 < rem) {  // block straddled the run boundary: finish in run 2
            offset::fill_slots(lo, slast + B, offsets_r + k1,
                               static_cast<std::size_t>(rem - k1));
            lo += rem - k1;
        }
        slots_rem -= rem;
        found += rem;
    }

    // Phase 2: flat-partition the remaining suffix, bridge into the <= 2
    // remaining slot runs.
    I m = sized_partition(sfirst, slast, key, comp, proj);
    const D c_rem = m - sfirst;
    const D s1r = lo_end - lo;
    const D s2r = r2end - r2lo;
    if (c_rem <= s1r + s2r) {
        // The whole below-run fits the slots: fill them LEADING-first.
        const D k1 = c_rem < s1r ? c_rem : s1r;
        std::swap_ranges(sfirst, sfirst + k1, lo);
        std::swap_ranges(sfirst + k1, m, r2lo);
    } else {
        // Fill ALL remaining slots from the TAIL of the below-run (disjoint:
        // the slot runs never overlap the suffix).
        I t = m - (s1r + s2r);
        std::swap_ranges(lo, lo_end, t);
        std::swap_ranges(r2lo, r2end, t + s1r);
    }
    return found + c_rem;
}

// ---------------------------------------------------------------------------
// seg_fill -- the fused plain cyclic partition: flat-partition S1, then treat
// its >= tail as the slot run and cross-fill S2's belows straight into it.
// Each S2 below is moved once (no bridge double-handling).
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> seg_fill(I first, I last, I buf_begin,
                                          I buf_end, K key, Comp comp,
                                          Proj proj) {
    I m1 = sized_partition(first, buf_end, key, comp, proj);
    return (m1 - first) + cross_fill2(m1, buf_end, buf_end, buf_end, buf_begin,
                                      last, key, comp, proj);
}

// ---------------------------------------------------------------------------
// Wrapped offset-partition kernels.  W1: the >= prefix ends inside S1 --
// flat-offset-partition S1 (its own prefix logic), then S2's belows go into
// S1's >= tail.  W2: the prefix spills into S2 (S1 is ENTIRELY prefix, pure
// swap targets, never compared) -- the slot supply is S1 plus S2's leading
// prefix piece, i.e. cross_fill2's two runs.  Each has a seg (partition +
// bridge) and a fused (cross_fill2) variant; the dispatcher picks the
// measured winner.
// ---------------------------------------------------------------------------
template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> off_w1_fill(I first, I last, I buf_begin,
                                             I buf_end,
                                             std::iter_difference_t<I> offset,
                                             K key, Comp comp, Proj proj) {
    I m1 = offset::sized_off(first, buf_end, offset, key, comp, proj);
    return (m1 - first) + cross_fill2(m1, buf_end, buf_end, buf_end, buf_begin,
                                      last, key, comp, proj);
}

template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> off_w1_seg(I first, I last, I buf_begin,
                                            I buf_end,
                                            std::iter_difference_t<I> offset,
                                            K key, Comp comp, Proj proj) {
    I m1 = offset::sized_off(first, buf_end, offset, key, comp, proj);
    I m2 = sized_partition(buf_begin, last, key, comp, proj);
    bridge(m1, buf_end, buf_begin, m2);
    return (m1 - first) + (m2 - buf_begin);
}

template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> off_w2_fill(I first, I last, I buf_begin,
                                             I buf_end,
                                             std::iter_difference_t<I> offset,
                                             K key, Comp comp, Proj proj) {
    I p = buf_begin + (offset - (buf_end - first));
    return cross_fill2(first, buf_end, buf_begin, p, p, last, key, comp, proj);
}

template <std::random_access_iterator I, class K, class Comp, class Proj>
inline std::iter_difference_t<I> off_w2_seg(I first, I last, I buf_begin,
                                            I buf_end,
                                            std::iter_difference_t<I> offset,
                                            K key, Comp comp, Proj proj) {
    I m2 = offset::sized_off(buf_begin, last, offset - (buf_end - first), key,
                             comp, proj);
    bridge(first, buf_end, buf_begin, m2);  // "g1" := all of S1 (pure prefix)
    return m2 - buf_begin;
}

// The comparator class the cyclic gates key on: a pair-like element compared
// with the default `<` is a short-circuit LEX compare (a data-dependent
// branch per element in every scan); everything else on the focus list is a
// branchless single compare.
template <class T, class Comp, class Proj>
inline constexpr bool cyclic_lexish =
    net::lex_pair_like<T> && net::is_default_less<Comp, Proj>;

// The pair-lex scan comparators and the substitution machinery live in
// detail::net (they now also serve the FLAT entry points, which are defined
// before this section); re-exported here for the cyclic dispatchers and for
// existing external references.
using net::lex_less_branchless;
using net::lex_less_packed;
using net::lex_less_semibranch;
using net::subst_lex_comp_t;
template <class It, class K, class Comp, class Proj>
inline constexpr bool use_branchless_lex = net::use_lex_subst<It, K, Comp, Proj>;

}  // namespace partitions::detail::cyclic

namespace partitions {

// Below-key count form of the cyclic partition (see the section header for
// the range convention and contract).  The count is what the kernels compute
// natively; the iterator form is derived from it.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::iter_difference_t<It> cyclic_partition_count(It first, It last,
                                                  It buf_begin, It buf_end,
                                                  K key, Comp comp = {},
                                                  Proj proj = {}) {
    using D = std::iter_difference_t<It>;
    // Pair-lex under the default `<`: swap in the branchless lex comparator
    // (identical order, no data-dependent branch) and re-dispatch -- the
    // substituted type then flows through every internal kernel.
    if constexpr (detail::cyclic::use_branchless_lex<It, K, Comp, Proj>) {
        return cyclic_partition_count(
            first, last, buf_begin, buf_end, key,
            detail::cyclic::subst_lex_comp_t<std::iter_value_t<It>>{}, proj);
    } else {
    if (first <= last)  // flat (includes empty)
        return sized_partition(first, last, key, comp, proj) - first;
    if (last == buf_begin)  // wrapped notation, but S2 is empty
        return sized_partition(first, buf_end, key, comp, proj) - first;
    const D n = (buf_end - first) + (last - buf_begin);
    if constexpr (detail::cyclic::cyclic_lexish<std::iter_value_t<It>, Comp,
                                                Proj>) {
        if (n <= tuning::cyclic_gap_max_lex)
            return detail::cyclic::gap_cyclic(first, last, buf_begin, buf_end,
                                              D{0}, key, comp, proj);
        // Branchy-compare pairs: the vectorised bridge beats the fused
        // fill's scattered rotation at every measured mid wrap.
        return detail::cyclic::seg_bridge(first, last, buf_begin, buf_end, key,
                                          comp, proj);
    } else if constexpr (std::is_same_v<Comp,
                                        detail::cyclic::lex_less_branchless>) {
        // Multi-setcc lex predicate: tiny gap tier (its loop codegen is
        // context-fragile -- see tuning::cyclic_gap_max_mlex), bridge through
        // mid sizes, fused fill above.
        if (n <= tuning::cyclic_gap_max_mlex)
            return detail::cyclic::gap_cyclic(first, last, buf_begin, buf_end,
                                              D{0}, key, comp, proj);
        if (n <= tuning::cyclic_bridge_max_mlex)
            return detail::cyclic::seg_bridge(first, last, buf_begin, buf_end,
                                              key, comp, proj);
        return detail::cyclic::seg_fill(first, last, buf_begin, buf_end, key,
                                        comp, proj);
    } else {
        constexpr D gap_max = detail::simd_fast_path<It, Comp, Proj>
                                  ? tuning::cyclic_gap_max_simd
                                  : tuning::cyclic_gap_max;
        if (n <= gap_max)
            return detail::cyclic::gap_cyclic(first, last, buf_begin, buf_end,
                                              D{0}, key, comp, proj);
        // Cheap-compare elements (incl. the AVX2 integer fast path): the
        // fused fill moves each S2 below exactly once.
        return detail::cyclic::seg_fill(first, last, buf_begin, buf_end, key,
                                        comp, proj);
    }
    }  // use_branchless_lex
}

// Physical boundary iterator, normalized into [buf_begin, buf_end).
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
It cyclic_partition(It first, It last, It buf_begin, It buf_end, K key,
                    Comp comp = {}, Proj proj = {}) {
    const auto c =
        cyclic_partition_count(first, last, buf_begin, buf_end, key, comp, proj);
    if (first <= last) return first + c;
    const auto len1 = buf_end - first;
    return c < len1 ? first + c : buf_begin + (c - len1);
}

// Cyclic offset partition: PRECONDITION that the logical prefix [0, offset)
// is all >= key (never compared, only swapped into).  Returns the below-key
// count c; on return logical [0, c) is all < key, [c, n) all >= key.
template <std::random_access_iterator It, class K, class Comp = std::less<>,
          class Proj = std::identity>
std::iter_difference_t<It> cyclic_offset_partition(
    It first, It last, It buf_begin, It buf_end,
    std::iter_difference_t<It> offset, K key, Comp comp = {}, Proj proj = {}) {
    using D = std::iter_difference_t<It>;
    // Pair-lex under the default `<`: branchless-comparator substitution,
    // exactly as in cyclic_partition_count.
    if constexpr (detail::cyclic::use_branchless_lex<It, K, Comp, Proj>) {
        return cyclic_offset_partition(
            first, last, buf_begin, buf_end, offset, key,
            detail::cyclic::subst_lex_comp_t<std::iter_value_t<It>>{}, proj);
    } else {
    if (first <= last)
        return detail::offset::sized_off(first, last, offset, key, comp, proj) -
               first;
    if (last == buf_begin)
        return detail::offset::sized_off(first, buf_end, offset, key, comp,
                                         proj) -
               first;
    const D len1 = buf_end - first;
    const D n = len1 + (last - buf_begin);
    const D suffix = n - offset;
    if (suffix <= 0) return 0;  // empty suffix: no below elements exist
    // NOTE: the flat dispatcher's "narrow T && offset >= suffix -> gap" rule
    // does NOT carry over: the cyclic fill scans with vfill_right on the AVX2
    // fast path and beat the gap walk 2-3x in exactly that regime (measured;
    // cyclic_partitions.txt).  The only gap tier left is a small SUFFIX.
    if constexpr (detail::cyclic::cyclic_lexish<std::iter_value_t<It>, Comp,
                                                Proj>) {
        if (suffix <= tuning::cyclic_off_gap_max_lex)
            return detail::cyclic::gap_cyclic(first, last, buf_begin, buf_end,
                                              offset, key, comp, proj);
        // Branchy-compare pairs: partition + bridge wins while the prefix
        // ends inside S1; once the prefix spills into S2 the two-run fused
        // fill wins (W1 -> seg, W2 -> fill: regret 0.5% vs 9% for either
        // kernel alone).
        if (offset <= len1)
            return detail::cyclic::off_w1_seg(first, last, buf_begin, buf_end,
                                              offset, key, comp, proj);
        return detail::cyclic::off_w2_fill(first, last, buf_begin, buf_end,
                                           offset, key, comp, proj);
    } else {
        // The lex comparators gate at 256 (measured: the 16-byte gap walk's
        // two wide moves per suffix element lose to the fill's once-moved
        // rotation earlier than for first-projection keys).
        constexpr D gap_max =
            std::is_same_v<Comp, detail::cyclic::lex_less_branchless> ||
                    std::is_same_v<Comp, detail::net::lex_less_semibranch>
                ? tuning::cyclic_off_gap_max_mlex
                : detail::simd_fast_path<It, Comp, Proj>
                      ? tuning::cyclic_off_gap_max_simd
                      : tuning::cyclic_off_gap_max;
        if (suffix <= gap_max)
            return detail::cyclic::gap_cyclic(first, last, buf_begin, buf_end,
                                              offset, key, comp, proj);
        if (offset <= len1)
            return detail::cyclic::off_w1_fill(first, last, buf_begin, buf_end,
                                               offset, key, comp, proj);
        return detail::cyclic::off_w2_fill(first, last, buf_begin, buf_end,
                                           offset, key, comp, proj);
    }
    }  // use_branchless_lex
}

}  // namespace partitions

// ===========================================================================
// detail::minsel + find_min / cyclic_find_min -- find a minimal element of a
// flat or possibly-WRAPPED range (range convention as in cyclic_partition).
//
// Adapted from the source repo's measured find-minimum study
// (quicksort_lr.hpp detail::find_min + docs/quicksort_lr.md); every design
// choice below is that study's, re-verified here:
//
//   * The best kernel is PROJECTED-KEY-WIDTH dependent.  Narrow keys
//     (register-width scalars) update the running minimum with the BRANCHY
//     source form `if (comp(k, best)) { best = k; pos = it; }` -- GCC lowers
//     it to ONE cmp + two cmovs (value + position); the double-ternary
//     "branchless" form wastes a second compare.  Wide keys (16-byte lex
//     pairs) keep the same source form as a REAL branch: a running-minimum
//     update fires with probability ~1/i, the branch predicts almost
//     perfectly, and the 16-byte copy runs only on genuine updates --
//     forcing a per-element blend is 2-3x slower (kept negative result).
//   * TWO independent accumulators (the even/odd "double scan") break the
//     cmp -> cmov(best) -> cmp carried chain and win from n ~ 16 UP; below
//     that the single accumulator's zero setup wins (and >= 4 accumulators
//     or tournament networks always lose -- kept negative results).
//   * Lex pairs re-use the round-3 comparator playbook: 4+4-byte integer
//     pairs are scanned as PACKED u64 keys (pack once per element -- the key
//     becomes narrow-class and the whole scan is branchless); other lex
//     pairs run the wide kernel under lex_less_semibranch (the pinned
//     short-circuit the wide kernel's rationale depends on).
//
// CYCLIC DESIGN.  A minimum is order-independent, so a wrapped range costs
// NOTHING extra by construction: the running (best, pos) state simply
// continues through two back-to-back flat segment loops -- no wrap check in
// any loop body, no second initialisation, no combine step (M1).  The
// two-accumulator large tier (M2) carries both lanes across the wrap the
// same way (lane parity is not global; each segment drains its odd tail
// into lane 0).  TIE RULE: some minimal element is returned (the source
// study's contract, not std::min_element's first-occurrence -- the lanes of
// the double scan already forfeit that); on tie-free data flat and wrapped
// agree.  find_min requires n >= 1; cyclic_find_min returns `first` for the
// empty (first == last) range.
// ===========================================================================
namespace partitions::detail::minsel {

// dst = c ? src : dst, forced through general-purpose registers where a
// naive conditional copy would leave the value in XMM and become a branch
// (float/double keys), or through a two-word XOR blend for 16-byte keys.
template <class K>
PARTITIONS_ALWAYS_INLINE void cond_assign(K& dst, const K& src, bool c) {
    if constexpr (std::is_trivially_copyable_v<K> && sizeof(K) == 16) {
        struct two_words { std::uint64_t a, b; };
        auto d = std::bit_cast<two_words>(dst);
        const auto s = std::bit_cast<two_words>(src);
        const std::uint64_t mask = 0ull - static_cast<std::uint64_t>(c);
        d.a ^= (d.a ^ s.a) & mask;
        d.b ^= (d.b ^ s.b) & mask;
        dst = std::bit_cast<K>(d);
    } else if constexpr (std::is_trivially_copyable_v<K> && sizeof(K) == 8 &&
                         !std::is_integral_v<K>) {
        auto d = std::bit_cast<std::uint64_t>(dst);
        const auto s = std::bit_cast<std::uint64_t>(src);
        const std::uint64_t mask = 0ull - static_cast<std::uint64_t>(c);
        d ^= (d ^ s) & mask;
        dst = std::bit_cast<K>(d);
    } else if constexpr (std::is_trivially_copyable_v<K> && sizeof(K) == 4 &&
                         !std::is_integral_v<K>) {
        auto d = std::bit_cast<std::uint32_t>(dst);
        const auto s = std::bit_cast<std::uint32_t>(src);
        const std::uint32_t mask = 0u - static_cast<std::uint32_t>(c);
        d ^= (d ^ s) & mask;
        dst = std::bit_cast<K>(d);
    } else {
        dst = c ? src : dst;  // CMOV for register-width integrals
    }
}

// Continue a running minimum (best, pos) through the flat segment
// [first, last).  One source form serves both key classes: narrow keys
// if-convert to cmp + 2 cmov, wide keys keep the rarely-taken branch.
template <class It, class K, class Comp, class Proj>
PARTITIONS_ALWAYS_INLINE void scan1(It first, It last, K& best, It& pos,
                                    Comp& comp, Proj& proj) {
    for (It it = first; it != last; ++it) {
        K k = std::invoke(proj, *it);
        if (static_cast<bool>(std::invoke(comp, k, best))) {
            best = k;
            pos = it;
        }
    }
}

// Continue TWO lanes through [first, last); the odd tail drains into lane 0.
template <class It, class K, class Comp, class Proj>
PARTITIONS_ALWAYS_INLINE void scan2(It first, It last, K& b0, It& p0, K& b1,
                                    It& p1, Comp& comp, Proj& proj) {
    using D = std::iter_difference_t<It>;
    It pair_end = first + ((last - first) & ~D{1});
    It it = first;
    for (; it != pair_end; it += 2) {
        K k0 = std::invoke(proj, it[0]);
        K k1 = std::invoke(proj, it[1]);
        if (static_cast<bool>(std::invoke(comp, k0, b0))) {
            b0 = k0;
            p0 = it;
        }
        if (static_cast<bool>(std::invoke(comp, k1, b1))) {
            b1 = k1;
            p1 = it + 1;
        }
    }
    if (it != last) {
        K k = std::invoke(proj, *it);
        if (static_cast<bool>(std::invoke(comp, k, b0))) {
            b0 = k;
            p0 = it;
        }
    }
}

template <class K>
inline constexpr bool narrow_key = std::is_arithmetic_v<K> && sizeof(K) <= 8;

// Projection composing the packed-u64 lex image (turns a 4+4-byte integer
// pair scan into a narrow-class u64 scan).
struct pack_proj {
    template <class P>
    PARTITIONS_ALWAYS_INLINE std::uint64_t operator()(const P& p) const {
        return net::lex_less_packed::pack(p);
    }
};

// Below this length the single accumulator's zero setup beats the double
// scan (the source study's L ~ 16 crossover; re-measured in
// cyclic_partitions.txt round 4).
inline constexpr std::ptrdiff_t two_acc_min = 16;

// WIDE keys also profit from two lanes -- with the BRANCHY update in each
// lane (two independent rarely-taken branches pipeline fine; it is the
// branchless per-element blend that the source study rejected).  The
// crossover is member-dependent (measured, round 4): float-first pairs pay
// ~1.9 ns/elem in one lane and cross at ~24; integer 16-byte pairs cross
// only at ~256.
template <class K>
inline constexpr std::ptrdiff_t wide_two_acc_min = [] {
    if constexpr (net::lex_pair_like<K>) {
        if constexpr (std::is_floating_point_v<std::remove_reference_t<
                          decltype(std::declval<K&>().first)>>)
            return 24;
        else
            return 256;
    } else {
        return 256;
    }
}();

// Flat kernel, class-dispatched.  Requires n >= 1.
template <class It, class Comp, class Proj>
PARTITIONS_ALWAYS_INLINE It find_min_flat(It first, It last, Comp comp,
                                          Proj proj) {
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    constexpr std::ptrdiff_t two_min =
        narrow_key<K> ? two_acc_min : wide_two_acc_min<K>;
    if (last - first >= two_min) {
        K b0 = std::invoke(proj, first[0]);
        K b1 = std::invoke(proj, first[1]);
        It p0 = first, p1 = first + 1;
        scan2(first + 2, last, b0, p0, b1, p1, comp, proj);
        return static_cast<bool>(std::invoke(comp, b1, b0)) ? p1 : p0;
    }
    K best = std::invoke(proj, *first);
    It pos = first;
    scan1(first + 1, last, best, pos, comp, proj);
    return pos;
}

// Wrapped kernel: three tiers, all measured (cyclic_partitions.txt round 4).
//   n <  seg_ilp_max : PER-SEGMENT scans + one combine (M3).  Two
//                      independent (best, pos) chains let the two segment
//                      scans overlap in the out-of-order window -- measured
//                      10-25% faster than the single carried chain below
//                      n ~ 14 (a segment-level "double scan").
//   narrow, larger   : unified TWO-accumulator lanes carried across the
//                      wrap (M2).
//   wide, larger     : unified single accumulator, branchy update (M1).
// Requires a genuinely wrapped range (len1 >= 1, len2 >= 1).
inline constexpr std::ptrdiff_t seg_ilp_max = 14;

template <class It, class Comp, class Proj>
PARTITIONS_ALWAYS_INLINE It find_min_cyc(It first, It last, It buf_begin,
                                         It buf_end, Comp comp, Proj proj) {
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
    const auto n = (buf_end - first) + (last - buf_begin);
    if (n < seg_ilp_max) {
        It m1 = find_min_flat(first, buf_end, comp, proj);
        It m2 = find_min_flat(buf_begin, last, comp, proj);
        return static_cast<bool>(std::invoke(comp, std::invoke(proj, *m2),
                                             std::invoke(proj, *m1)))
                   ? m2
                   : m1;
    }
    constexpr std::ptrdiff_t two_min =
        narrow_key<K> ? two_acc_min : wide_two_acc_min<K>;
    if (n >= two_min) {
        K b0 = std::invoke(proj, *first);
        K b1 = b0;
        It p0 = first, p1 = first;
        scan2(first + 1, buf_end, b0, p0, b1, p1, comp, proj);
        scan2(buf_begin, last, b0, p0, b1, p1, comp, proj);
        return static_cast<bool>(std::invoke(comp, b1, b0)) ? p1 : p0;
    }
    K best = std::invoke(proj, *first);
    It pos = first;
    scan1(first + 1, buf_end, best, pos, comp, proj);
    scan1(buf_begin, last, best, pos, comp, proj);
    return pos;
}

// COMPARATOR POLICY for min scans (P-C: yet another compare class).  A
// running-minimum compare is element-vs-RUNNING-BEST and fires ~1/i of the
// time, so the raw short-circuiting operator< is already optimal here (the
// source study's wide-kernel rationale) -- measured in round 4:
// substituting lex_less_semibranch made float-pair scans ~2-3x slower per
// call while std::min_element with the raw compare stayed fast, so unlike
// the partition entries NO semibranch substitution happens.  The ONLY
// rewrite kept is the packed-u64 image for 4+4-byte integer pairs, which
// turns the whole scan narrow-class (branchless cmp+cmov) and wins from
// n ~ 16 while tying below.
template <class It, class Comp, class Proj>
inline constexpr bool use_packed_min =
    net::int_pair_packable<std::iter_value_t<It>> &&
    (std::is_same_v<Comp, std::less<>> ||
     std::is_same_v<Comp, std::less<std::iter_value_t<It>>>) &&
    std::is_same_v<Proj, std::identity>;

}  // namespace partitions::detail::minsel

namespace partitions {

// Flat find-minimum: an iterator to A minimal element of [first, last)
// under (comp, proj).  Requires last - first >= 1.
template <std::random_access_iterator It, class Comp = std::less<>,
          class Proj = std::identity>
It find_min(It first, It last, Comp comp = {}, Proj proj = {}) {
    if constexpr (detail::minsel::use_packed_min<It, Comp, Proj>) {
        return detail::minsel::find_min_flat(first, last, std::less<>{},
                                             detail::minsel::pack_proj{});
    } else {
        return detail::minsel::find_min_flat(first, last, comp, proj);
    }
}

// Cyclic find-minimum (range convention as cyclic_partition: first > last
// wraps as [first, buf_end) ++ [buf_begin, last); first == last is empty and
// returns first).  Returns a PHYSICAL iterator to A minimal element.
template <std::random_access_iterator It, class Comp = std::less<>,
          class Proj = std::identity>
It cyclic_find_min(It first, It last, It buf_begin, It buf_end,
                   Comp comp = {}, Proj proj = {}) {
    if (first <= last) {  // flat (includes empty)
        if (first == last) return first;
        return find_min(first, last, comp, proj);
    }
    if (last == buf_begin)  // wrapped notation, S2 empty
        return find_min(first, buf_end, comp, proj);
    if constexpr (detail::minsel::use_packed_min<It, Comp, Proj>) {
        return detail::minsel::find_min_cyc(first, last, buf_begin, buf_end,
                                            std::less<>{},
                                            detail::minsel::pack_proj{});
    } else {
        return detail::minsel::find_min_cyc(first, last, buf_begin, buf_end,
                                            comp, proj);
    }
}

}  // namespace partitions

#undef PARTITIONS_ALWAYS_INLINE

#endif  // REFACTORED_PARTITIONS_HPP
