#ifndef PARTITIONS_SMALL_SORT_HPP
#define PARTITIONS_SMALL_SORT_HPP

// Fast scalar small-array sort (N <= 24) with custom comparator + projection.
//
// Design rationale (see CLAUDE.md for the bigger picture):
//
//   * Every comparator in every network is a branchless ternary-blend
//     compare-exchange (`cswap`).  GCC/Clang lower it to a single CMOV chain
//     per output on int64_t, and to two CMOVs per output (high/low) on
//     pair64.  This is the same pattern as libc++'s `__cond_swap` and the
//     AlphaDev sort3/4/5 (Mankowitz et al., Nature 2023) -- except libc++
//     gates the branchless path on `is_arithmetic && sizeof<=sizeof(void*)`,
//     so its win is invisible to pair64.  We do not gate.
//
//   * For N in [2..24] we hand-list the best-known SIZE-optimal networks from
//     Bert Dobbelaere's database (bertdobbelaere.github.io/sorting_networks.html,
//     aggregating SorterHunter results and the classical Floyd/Knuth optima).
//     Size-optimal -- not depth-optimal -- because this module targets a
//     throughput-bound regime (a hot loop over many fixed-size blocks), where
//     total compare-exchange count, not critical-path depth, sets the cost.
//     Every network is proved correct here by exhaustive 0/1 enumeration in
//     `tools/verify_small_sort.cpp` (2^N inputs, complete up to N=24 by Knuth's
//     zero-one principle), so a transcription slip cannot pass silently.
//
//   * Fewer compare-exchanges helps most exactly where each compare-exchange
//     is expensive: pair64, whose swap is two 64-bit conditional moves.  At
//     N=24 the best-known network (120 CE) runs ~8% faster than the
//     generically-synthesised Batcher odd-even network (`sort_network_oems`,
//     kept for the benchmark) and ~1.6-2x faster than std::sort / Boost /
//     cpp-sort's own network sorter.  See docs/small_sort_findings.md.
//
//   * SIMD does not help pair64: AVX2 lanes are 32 bytes (only 2 pair64
//     elements at a time), the lex compare needs a 128-bit predicate, and
//     packed CMOV doesn't exist below AVX-512 mask intrinsics.  The whole
//     module is scalar; what we exploit is wide ILP through cswap chains
//     with shallow data dependencies.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "types.hpp"  // pair64

namespace partitions::small_sort {

// ---------------------------------------------------------------------------
// Branchless compare-exchange.
//
// Two issues need handling separately on x86-64 for 16-byte aggregates:
//
//   (1) The COMPARE: pair<i64,i64>::operator< (from default <=>) is a
//       short-circuit lex compare and GCC emits actual branches (cmp/jne/jl)
//       for it.  Branch-predictor misses cost ~15-20 cycles per cswap on
//       random data, which dominates the per-comparator budget.
//
//   (2) The BLEND: `pair64 = bool ? pair64 : pair64` is lowered by GCC as a
//       branch + xmm copy, NOT as two cmovs on the halves, because there is
//       no 128-bit CMOV in x86-64.  Even with the bool already computed, the
//       conditional 16-byte copy reintroduces a branch.
//
// Fix: when T is a 16-byte trivial aggregate AND the user supplied no
// custom comparator (std::less + std::identity), specialise to
//   (a) compute the lex compare branchlessly: (b0<a0) | ((b0==a0)&(b1<a1))
//   (b) decompose the swap into 4 explicit cmovs over the two 64-bit halves.
// For everything else, the canonical "ternary on T" pattern is used; GCC
// lowers it to a CMOV for register-sized arithmetic types.
// ---------------------------------------------------------------------------

namespace detail_cs {

// Detect "16-byte trivially-copyable aggregate" -- the case where naive
// ternary-on-T produces a branch and we want explicit half-decomposition.
template <class T>
constexpr bool is_16byte_trivial =
    sizeof(T) == 16 && std::is_trivially_copyable_v<T>;

// Detect "default" comparator and projection -- the case where we know how
// to compute the lex predicate branchlessly ourselves.
template <class Cmp, class Proj>
constexpr bool is_default_less =
    std::is_same_v<std::decay_t<Cmp>, std::less<>> &&
    std::is_same_v<std::decay_t<Proj>, std::identity>;

// Branchless 16-byte swap via integer XOR-mask (mask = 0 or all-ones).
//
// IMPORTANT: this must be XOR-mask, not a `swap ? hi : lo` cmov form.  When the
// element contains a floating field (e.g. pair_di {double,int}) GCC keeps it in
// XMM registers; it has no cheap branchless conditional move of an XMM pair, so
// a cmov-style swap is lowered to a *branch per compare-exchange* -- which then
// mispredicts on random data and dominates the cost.  Reinterpreting the 16
// bytes as two integers and blending with an arithmetic mask forces the swap
// into GP registers with no branch and no XMM dependency.
template <class T>
[[gnu::always_inline]] inline void half_swap(T& a, T& b, bool swap) {
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

// XOR-mask swap for any trivially-copyable type of size <= 8 -- guaranteed
// branchless (no per-half cmov that GCC might turn back into a branch).
template <class T>
[[gnu::always_inline]] inline void word_swap(T& a, T& b, bool swap) {
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

// Detect "looks like a lexicographic pair": a trivially-copyable class with
// public `.first` and `.second`, ordered first-then-second (which is exactly
// what a defaulted operator<=> produces).  Covers pair64, pair_fi, pair_di and
// std::pair.  We only take this path under the default comparator/projection,
// where we know the order is lexicographic and can compute the predicate
// ourselves without the short-circuit branch the struct's operator< emits.
template <class T>
concept lex_pair_like = std::is_class_v<T> && std::is_trivially_copyable_v<T> &&
                        requires(T t) {
                            t.first;
                            t.second;
                        };

}  // namespace detail_cs

template <class T, class Cmp, class Proj>
[[gnu::always_inline]] inline void cswap(T& a, T& b, Cmp& comp, Proj& proj) {
    if constexpr (detail_cs::lex_pair_like<T> &&
                  detail_cs::is_default_less<Cmp, Proj>) {
        // Generalised branchless lexicographic compare-exchange for any
        // first/second aggregate under the natural order -- pair64 {long,long},
        // pair_fi {float,int}, pair_di {double,int}, std::pair, ...  The
        // predicate is computed without the short-circuit branch the defaulted
        // operator< emits, and the swap is a size-appropriate branchless blend.
        //
        // NOTE: we deliberately do NOT special-case pair64 with an explicit
        // field-level XOR-mask any more.  Letting GCC see the whole-half blend
        // (via word_swap/half_swap) lets it emit CMOV (4 per CE) instead of the
        // ~6 explicit XORs the hand-written version produced -- ~30% fewer
        // instructions and measurably faster (pair64 N=24: 190 -> ~160 ns).
        //
        // Crucially `lt0` and `eq0` are computed from the SAME operand pair
        // (b0, a0): the compiler issues a SINGLE compare (one comisd for a
        // floating first key) and reads both `<` and `==` off its flags
        // (seta + sete).  Writing the equality as `a0 < b0` instead would be a
        // second, operand-swapped compare -- doubling the first-key compare
        // cost, which on pair_di is the whole budget.  Under default IEEE the
        // floating `==` adds a NaN/parity sequence; `-ffast-math`
        // (-ffinite-math-only, no NaNs assumed) drops it, so the pair collapses
        // to one comisd + two set-from-flags.
        const auto a0 = a.first, a1 = a.second;
        const auto b0 = b.first, b1 = b.second;
        const unsigned lt0 = static_cast<unsigned>(b0 < a0);
        const unsigned eq0 = static_cast<unsigned>(b0 == a0);
        const unsigned lt1 = static_cast<unsigned>(b1 < a1);
        const bool swap = (lt0 | (eq0 & lt1)) != 0;
        if constexpr (sizeof(T) == 16) {
            detail_cs::half_swap(a, b, swap);
        } else {
            detail_cs::word_swap(a, b, swap);
        }
    } else if constexpr (detail_cs::is_16byte_trivial<T>) {
        // Generic 16-byte type with user comparator.
        const bool swap = comp(proj(b), proj(a));
        detail_cs::half_swap(a, b, swap);
    } else if constexpr (std::is_integral_v<T> && sizeof(T) <= 8) {
        // Independent min/max blend for register-sized integrals.  Both cmovs
        // depend only on the compare flag, not on each other, so the OOO core
        // issues them in parallel.  cpp-sort's `min; y ^= dx^x` form uses one
        // fewer register but is *serial* (the two xors wait on the min result);
        // measured back-to-back the parallel form is faster for us at every N
        // (e.g. i64 N=24: 27 vs 32 ns), because these networks are throughput-
        // bound and latency per compare-exchange is what matters.
        const bool swap = comp(proj(b), proj(a));
        const T lo = swap ? b : a;
        const T hi = swap ? a : b;
        a = lo;
        b = hi;
    } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
        // 8-byte trivially-copyable aggregate with a custom comparator/projection
        // -- e.g. pair_fi {float,int} sorted *by .first only* (compare the key,
        // carry the whole pair).  The plain `swap ? b : a` ternary on such a
        // struct can be lowered to a branch (the value may sit in an XMM reg);
        // the XOR-mask word_swap forces a branchless GP-register swap, matching
        // the 16-byte half_swap path one size down.
        const bool swap = comp(proj(b), proj(a));
        detail_cs::word_swap(a, b, swap);
    } else {
        // Generic fallback.
        const bool swap = comp(proj(b), proj(a));
        T tmp = a;
        a = swap ? b : a;
        b = swap ? tmp : b;
    }
}

// ===========================================================================
// Compile-time Batcher's odd-even merge sort (OEMS), pad-to-pow2 variant.
// ===========================================================================
//
// Batcher's OEMS (1968) is cleanly defined for power-of-2 N.  For arbitrary
// N we run the pow2-N' network where N' = next_pow2(N), and treat virtual
// positions N..N'-1 as +infinity sentinels: every comparator that touches a
// virtual index is a no-op (the real value is always <= +inf) so we simply
// skip emitting it.  Correctness follows from the standard OEMS proof.
//
// Comparator counts vs proven optimum:
//
//   N     2 3 4 5 6 7 8  9 10 11 12 13 14 15 16
//   OEMS  1 3 5 9 12 16 19 26 31 37 41 48 53 58 63
//   opt   1 3 5 9 12 16 19 25 29 35 39 45 51 56 60
//   delta 0 0 0 0  0  0  0  1  2  2  2  3  2  2  3
//
// Within ~5% of optimum; the runtime difference vs hand-coded best-known
// networks is small (a single extra CE is 2-3 cycles on a modern OOO core
// with deep ILP -- it overlaps with surrounding work).

namespace detail {

constexpr int next_pow2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

// Emit cswap only if both indices are within the real array.
template <int I, int J, int Real, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void maybe_cswap(It first, Cmp& c, Proj& p) {
    if constexpr (I < Real && J < Real) {
        cswap(first[I], first[J], c, p);
    }
}

// emit (I, I+R), (I+S, I+S+R), ... while I+R <= Limit
template <int I, int Limit, int R, int Step, int Real, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void oe_emit(It first, Cmp& c, Proj& p) {
    if constexpr (I + R <= Limit) {
        maybe_cswap<I, I + R, Real>(first, c, p);
        oe_emit<I + Step, Limit, R, Step, Real>(first, c, p);
    }
}

template <int Lo, int N, int R, int Real, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void oe_merge(It first, Cmp& c, Proj& p) {
    constexpr int step = R * 2;
    if constexpr (step < N) {
        oe_merge<Lo, N, step, Real>(first, c, p);
        oe_merge<Lo + R, N - R, step, Real>(first, c, p);
        oe_emit<Lo + R, Lo + N - R, R, step, Real>(first, c, p);
    } else {
        maybe_cswap<Lo, Lo + R, Real>(first, c, p);
    }
}

template <int Lo, int N, int Real, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void oe_sort_pow2(It first, Cmp& c, Proj& p) {
    if constexpr (N > 1) {
        constexpr int M = N / 2;
        oe_sort_pow2<Lo, M, Real>(first, c, p);
        oe_sort_pow2<Lo + M, M, Real>(first, c, p);
        oe_merge<Lo, N, 1, Real>(first, c, p);
    }
}

template <int N, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void oems_sort(It first, Cmp& c, Proj& p) {
    oe_sort_pow2<0, next_pow2(N), N>(first, c, p);
}

}  // namespace detail

// ===========================================================================
// Hand-coded best-known networks for N=2..8 (proved size-optimal).
// All verified by 0/1 enumeration.  These are small enough to inline without
// any concern about icache pressure.
// ===========================================================================

namespace nets {

using P = std::pair<int, int>;

inline constexpr std::array<P, 1> n2 = {{ {0,1} }};
inline constexpr std::array<P, 3> n3 = {{ {0,2},{0,1},{1,2} }};
inline constexpr std::array<P, 5> n4 = {{ {0,2},{1,3},{0,1},{2,3},{1,2} }};
inline constexpr std::array<P, 9> n5 = {{
    {0,3},{1,4},{0,2},{1,3},{0,1},{2,4},{1,2},{3,4},{2,3}
}};
inline constexpr std::array<P, 12> n6 = {{
    {0,5},{1,3},{2,4},{1,2},{3,4},{0,3},
    {2,5},{0,1},{2,3},{4,5},{1,2},{3,4}
}};
// Knuth/best-known N=7, 16 CE, depth 6.  Trace: verified for several 0/1
// inputs including the canonical (0,1,1,0,0,0,0) which broke an earlier
// transcription.
inline constexpr std::array<P, 16> n7 = {{
    {0,4},{1,5},{2,6},{0,2},{1,3},{4,6},{2,4},{3,5},
    {0,1},{2,3},{4,5},{1,4},{3,6},{1,2},{3,4},{5,6}
}};
inline constexpr std::array<P, 19> n8 = {{
    {0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7},{0,1},{2,3},
    {4,5},{6,7},{2,4},{3,5},{1,4},{3,6},{1,2},{3,4},{5,6}
}};

// -----------------------------------------------------------------------
// Best-known SIZE-optimal networks for N=9..24, transcribed from Bert
// Dobbelaere's database (https://bertdobbelaere.github.io/sorting_networks.html),
// which aggregates the SorterHunter results and the classical optima.
// Every one of these is proved correct here by full 0/1 enumeration in
// tools/verify_small_sort.cpp (2^N inputs, exhaustive up to N=24), so a
// transcription slip cannot pass silently.  CE counts in the array sizes
// equal the best-known S(N); see the table in the OEMS comment above for
// how much this saves over the synthesised odd-even network.
// -----------------------------------------------------------------------
inline constexpr std::array<P, 25> n9 = {{
    {0,3},{1,7},{2,5},{4,8},{0,7},{2,4},{3,8},{5,6},{0,2},{1,3},
    {4,5},{7,8},{1,4},{3,6},{5,7},{0,1},{2,4},{3,5},{6,8},{2,3},
    {4,5},{6,7},{1,2},{3,4},{5,6}
}};
inline constexpr std::array<P, 29> n10 = {{
    {0,8},{1,9},{2,7},{3,5},{4,6},{0,2},{1,4},{5,8},{7,9},{0,3},
    {2,4},{5,7},{6,9},{0,1},{3,6},{8,9},{1,5},{2,3},{4,8},{6,7},
    {1,2},{3,5},{4,6},{7,8},{2,3},{4,5},{6,7},{3,4},{5,6}
}};
inline constexpr std::array<P, 35> n11 = {{
    {0,9},{1,6},{2,4},{3,7},{5,8},{0,1},{3,5},{4,10},{6,9},{7,8},
    {1,3},{2,5},{4,7},{8,10},{0,4},{1,2},{3,7},{5,9},{6,8},{0,1},
    {2,6},{4,5},{7,8},{9,10},{2,4},{3,6},{5,7},{8,9},{1,2},{3,4},
    {5,6},{7,8},{2,3},{4,5},{6,7}
}};
inline constexpr std::array<P, 39> n12 = {{
    {0,8},{1,7},{2,6},{3,11},{4,10},{5,9},{0,1},{2,5},{3,4},{6,9},
    {7,8},{10,11},{0,2},{1,6},{5,10},{9,11},{0,3},{1,2},{4,6},{5,7},
    {8,11},{9,10},{1,4},{3,5},{6,8},{7,10},{1,3},{2,5},{6,9},{8,10},
    {2,3},{4,5},{6,7},{8,9},{4,6},{5,7},{3,4},{5,6},{7,8}
}};
inline constexpr std::array<P, 45> n13 = {{
    {0,12},{1,10},{2,9},{3,7},{5,11},{6,8},{1,6},{2,3},{4,11},{7,9},
    {8,10},{0,4},{1,2},{3,6},{7,8},{9,10},{11,12},{4,6},{5,9},{8,11},
    {10,12},{0,5},{3,8},{4,7},{6,11},{9,10},{0,1},{2,5},{6,9},{7,8},
    {10,11},{1,3},{2,4},{5,6},{9,10},{1,2},{3,4},{5,7},{6,8},{2,3},
    {4,5},{6,7},{8,9},{3,4},{5,6}
}};
inline constexpr std::array<P, 51> n14 = {{
    {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{0,2},{1,3},{4,8},
    {5,9},{10,12},{11,13},{0,4},{1,2},{3,7},{5,8},{6,10},{9,13},{11,12},
    {0,6},{1,5},{3,9},{4,10},{7,13},{8,12},{2,10},{3,11},{4,6},{7,9},
    {1,3},{2,8},{5,11},{6,7},{10,12},{1,4},{2,6},{3,5},{7,11},{8,10},
    {9,12},{2,4},{3,6},{5,8},{7,10},{9,11},{3,4},{5,6},{7,8},{9,10},
    {6,7}
}};
inline constexpr std::array<P, 56> n15 = {{
    {1,2},{3,10},{4,14},{5,8},{6,13},{7,12},{9,11},{0,14},{1,5},{2,8},
    {3,7},{6,9},{10,12},{11,13},{0,7},{1,6},{2,9},{4,10},{5,11},{8,13},
    {12,14},{0,6},{2,4},{3,5},{7,11},{8,10},{9,12},{13,14},{0,3},{1,2},
    {4,7},{5,9},{6,8},{10,11},{12,13},{0,1},{2,3},{4,6},{7,9},{10,12},
    {11,13},{1,2},{3,5},{8,10},{11,12},{3,4},{5,6},{7,8},{9,10},{2,3},
    {4,5},{6,7},{8,9},{10,11},{5,6},{7,8}
}};
inline constexpr std::array<P, 60> n16 = {{
    {0,13},{1,12},{2,15},{3,14},{4,8},{5,6},{7,11},{9,10},{0,5},{1,7},
    {2,9},{3,4},{6,13},{8,14},{10,15},{11,12},{0,1},{2,3},{4,5},{6,8},
    {7,9},{10,11},{12,13},{14,15},{0,2},{1,3},{4,10},{5,11},{6,7},{8,9},
    {12,14},{13,15},{1,2},{3,12},{4,6},{5,7},{8,10},{9,11},{13,14},{1,4},
    {2,6},{5,8},{7,10},{9,13},{11,14},{2,4},{3,6},{9,12},{11,13},{3,5},
    {6,8},{7,9},{10,12},{3,4},{5,6},{7,8},{9,10},{11,12},{6,7},{8,9}
}};
inline constexpr std::array<P, 71> n17 = {{
    {0,11},{1,15},{2,10},{3,5},{4,6},{8,12},{9,16},{13,14},{0,6},{1,13},
    {2,8},{4,14},{5,15},{7,11},{0,8},{3,7},{4,9},{6,16},{10,11},{12,14},
    {0,2},{1,4},{5,6},{7,13},{8,9},{10,12},{11,14},{15,16},{0,3},{2,5},
    {6,11},{7,10},{9,13},{12,15},{14,16},{0,1},{3,4},{5,10},{6,9},{7,8},
    {11,15},{13,14},{1,2},{3,7},{4,8},{6,12},{11,13},{14,15},{1,3},{2,7},
    {4,5},{9,11},{10,12},{13,14},{2,3},{4,6},{5,7},{8,10},{3,4},{6,8},
    {7,9},{10,12},{5,6},{7,8},{9,10},{11,12},{4,5},{6,7},{8,9},{10,11},
    {12,13}
}};
inline constexpr std::array<P, 77> n18 = {{
    {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{0,2},
    {1,3},{4,12},{5,13},{6,8},{9,11},{14,16},{15,17},{0,14},{1,16},{2,15},
    {3,17},{0,6},{1,10},{2,9},{7,16},{8,15},{11,17},{1,4},{3,9},{5,7},
    {8,14},{10,12},{13,16},{0,1},{2,5},{3,13},{4,14},{7,9},{8,10},{12,15},
    {16,17},{1,2},{3,5},{4,6},{11,13},{12,14},{15,16},{4,8},{5,12},{6,10},
    {7,11},{9,13},{1,4},{2,8},{3,6},{5,7},{9,15},{10,12},{11,14},{13,16},
    {2,4},{5,8},{6,10},{7,11},{9,12},{13,15},{3,5},{6,8},{7,10},{9,11},
    {12,14},{3,4},{5,6},{7,8},{9,10},{11,12},{13,14}
}};
inline constexpr std::array<P, 85> n19 = {{
    {0,12},{1,4},{2,8},{3,5},{6,17},{7,11},{9,14},{10,13},{15,16},{0,2},
    {1,7},{3,6},{4,11},{5,17},{8,12},{10,15},{13,16},{14,18},{3,10},{4,14},
    {5,15},{6,13},{7,9},{11,17},{16,18},{0,7},{1,10},{4,6},{9,15},{11,16},
    {12,17},{13,14},{0,3},{2,6},{5,7},{8,11},{12,16},{1,8},{2,9},{3,4},
    {6,15},{7,13},{10,11},{12,18},{1,3},{2,5},{6,9},{7,12},{8,10},{11,14},
    {17,18},{0,1},{2,3},{4,8},{6,10},{9,12},{14,15},{16,17},{1,2},{5,8},
    {6,7},{9,11},{10,13},{14,16},{15,17},{3,6},{4,5},{7,9},{8,10},{11,12},
    {13,14},{15,16},{3,4},{5,6},{7,8},{9,10},{11,13},{12,14},{2,3},{4,5},
    {6,7},{8,9},{10,11},{12,13},{14,15}
}};
inline constexpr std::array<P, 91> n20 = {{
    {0,3},{1,7},{2,5},{4,8},{6,9},{10,13},{11,15},{12,18},{14,17},{16,19},
    {0,14},{1,11},{2,16},{3,17},{4,12},{5,19},{6,10},{7,15},{8,18},{9,13},
    {0,4},{1,2},{3,8},{5,7},{11,16},{12,14},{15,19},{17,18},{1,6},{2,12},
    {3,5},{4,11},{7,17},{8,15},{13,18},{14,16},{0,1},{2,6},{7,10},{9,12},
    {13,17},{18,19},{1,6},{5,9},{7,11},{8,12},{10,14},{13,18},{3,5},{4,7},
    {8,10},{9,11},{12,15},{14,16},{1,3},{2,4},{5,7},{6,10},{9,13},{12,14},
    {15,17},{16,18},{1,2},{3,4},{6,7},{8,9},{10,11},{12,13},{15,16},{17,18},
    {2,3},{4,6},{5,8},{7,9},{10,12},{11,14},{13,15},{16,17},{4,5},{6,8},
    {7,10},{9,12},{11,13},{14,15},{3,4},{5,6},{7,8},{9,10},{11,12},{13,14},
    {15,16}
}};
inline constexpr std::array<P, 99> n21 = {{
    {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},
    {0,2},{1,3},{4,6},{5,7},{8,10},{9,11},{12,14},{13,15},{16,18},{17,19},
    {0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15},{0,4},{1,5},
    {3,7},{6,20},{8,12},{9,13},{10,14},{15,19},{2,6},{3,18},{7,20},{2,16},
    {3,6},{5,18},{7,17},{11,20},{0,2},{3,8},{6,12},{7,10},{9,16},{11,15},
    {13,17},{14,18},{19,20},{1,7},{2,3},{4,9},{10,11},{13,16},{15,18},{17,19},
    {1,4},{5,10},{6,13},{7,8},{11,14},{12,16},{15,17},{18,19},{1,2},{3,4},
    {5,6},{10,12},{11,13},{14,16},{17,18},{2,3},{4,5},{6,9},{10,11},{12,13},
    {14,15},{16,17},{6,7},{8,9},{15,16},{4,6},{7,8},{9,12},{13,15},{3,4},
    {5,7},{8,10},{9,11},{12,14},{5,6},{7,8},{9,10},{11,12},{13,14}
}};
inline constexpr std::array<P, 106> n22 = {{
    {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},
    {20,21},{0,2},{1,3},{4,6},{5,7},{8,12},{9,13},{14,16},{15,17},{18,20},
    {19,21},{0,4},{1,5},{2,6},{3,7},{8,10},{9,12},{11,13},{14,18},{15,19},
    {16,20},{17,21},{0,14},{1,15},{2,18},{3,19},{4,16},{5,17},{6,20},{7,21},
    {9,11},{10,12},{2,8},{3,11},{6,9},{10,18},{12,15},{13,19},{0,2},{1,10},
    {3,16},{5,18},{6,14},{7,15},{8,12},{9,13},{11,20},{19,21},{2,6},{3,10},
    {4,8},{5,12},{9,16},{11,18},{13,17},{15,19},{1,4},{7,13},{8,14},{9,12},
    {17,20},{1,2},{3,8},{4,6},{7,11},{10,14},{13,18},{15,17},{19,20},{2,4},
    {5,10},{7,9},{11,16},{12,14},{17,19},{5,6},{7,8},{9,11},{10,12},{13,14},
    {15,16},{3,5},{6,7},{8,10},{9,12},{11,13},{14,15},{16,18},{3,4},{5,6},
    {7,8},{9,10},{11,12},{13,14},{15,16},{17,18}
}};
inline constexpr std::array<P, 114> n23 = {{
    {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},
    {20,21},{0,2},{1,3},{4,6},{5,7},{8,10},{9,11},{12,14},{13,15},{16,18},
    {17,19},{21,22},{0,4},{1,5},{2,6},{3,7},{8,12},{9,13},{10,14},{11,15},
    {17,21},{18,20},{19,22},{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},
    {7,15},{1,2},{5,18},{7,19},{9,16},{10,21},{12,20},{15,22},{5,9},{6,7},
    {10,18},{11,21},{12,17},{13,20},{14,15},{3,17},{6,16},{7,14},{8,12},{15,19},
    {20,21},{3,4},{5,8},{6,10},{9,12},{13,16},{14,15},{17,18},{19,21},{0,5},
    {1,8},{2,12},{3,9},{4,10},{7,13},{11,17},{14,16},{18,20},{2,6},{3,5},
    {4,8},{7,11},{10,12},{13,18},{14,17},{15,20},{1,3},{2,5},{6,9},{7,10},
    {11,13},{12,14},{15,18},{16,17},{19,20},{2,3},{4,6},{8,9},{11,12},{13,14},
    {15,16},{17,19},{3,4},{5,6},{7,8},{9,10},{12,13},{14,15},{17,18},{4,5},
    {6,7},{8,9},{10,11},{16,17}
}};
inline constexpr std::array<P, 120> n24 = {{
    {0,20},{1,12},{2,16},{3,23},{4,6},{5,10},{7,21},{8,14},{9,15},{11,22},
    {13,18},{17,19},{0,3},{1,11},{2,7},{4,17},{5,13},{6,19},{8,9},{10,18},
    {12,22},{14,15},{16,21},{20,23},{0,1},{2,4},{3,12},{5,8},{6,9},{7,10},
    {11,20},{13,16},{14,17},{15,18},{19,21},{22,23},{2,5},{4,8},{6,11},{7,14},
    {9,16},{12,17},{15,19},{18,21},{1,8},{3,14},{4,7},{9,20},{10,12},{11,13},
    {15,22},{16,19},{0,7},{1,5},{3,4},{6,11},{8,15},{9,14},{10,13},{12,17},
    {16,23},{18,22},{19,20},{0,2},{1,6},{4,7},{5,9},{8,10},{13,15},{14,18},
    {16,19},{17,22},{21,23},{2,3},{4,5},{6,8},{7,9},{10,11},{12,13},{14,16},
    {15,17},{18,19},{20,21},{1,2},{3,6},{4,10},{7,8},{9,11},{12,14},{13,19},
    {15,16},{17,20},{21,22},{2,3},{5,10},{6,7},{8,9},{13,18},{14,15},{16,17},
    {20,21},{3,4},{5,7},{10,12},{11,13},{16,18},{19,20},{4,6},{8,10},{9,12},
    {11,14},{13,15},{17,19},{5,6},{7,8},{9,10},{11,12},{13,14},{15,16},{17,18}
}};

}  // namespace nets

template <class It, class Cmp, class Proj, std::size_t S, std::size_t... Is>
[[gnu::always_inline]] inline void apply_network(
    It first, const std::array<nets::P, S>& net,
    std::index_sequence<Is...>, Cmp& comp, Proj& proj) {
    (cswap(first[net[Is].first], first[net[Is].second], comp, proj), ...);
}

// ---------------------------------------------------------------------------
// Sort a known compile-time-size block.  Below 9 we use the optimal
// hand-coded networks; from 9 upward we synthesise via Batcher OEMS.
// ---------------------------------------------------------------------------
template <std::size_t N, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void sort_network(It first, Cmp comp, Proj proj) {
    if constexpr (N <= 1) {
        return;
    } else if constexpr (N == 2) {
        apply_network(first, nets::n2, std::make_index_sequence<nets::n2.size()>{}, comp, proj);
    } else if constexpr (N == 3) {
        apply_network(first, nets::n3, std::make_index_sequence<nets::n3.size()>{}, comp, proj);
    } else if constexpr (N == 4) {
        apply_network(first, nets::n4, std::make_index_sequence<nets::n4.size()>{}, comp, proj);
    } else if constexpr (N == 5) {
        apply_network(first, nets::n5, std::make_index_sequence<nets::n5.size()>{}, comp, proj);
    } else if constexpr (N == 6) {
        apply_network(first, nets::n6, std::make_index_sequence<nets::n6.size()>{}, comp, proj);
    } else if constexpr (N == 7) {
        apply_network(first, nets::n7, std::make_index_sequence<nets::n7.size()>{}, comp, proj);
    } else if constexpr (N == 8) {
        apply_network(first, nets::n8, std::make_index_sequence<nets::n8.size()>{}, comp, proj);
    } else if constexpr (N == 9) {
        apply_network(first, nets::n9, std::make_index_sequence<nets::n9.size()>{}, comp, proj);
    } else if constexpr (N == 10) {
        apply_network(first, nets::n10, std::make_index_sequence<nets::n10.size()>{}, comp, proj);
    } else if constexpr (N == 11) {
        apply_network(first, nets::n11, std::make_index_sequence<nets::n11.size()>{}, comp, proj);
    } else if constexpr (N == 12) {
        apply_network(first, nets::n12, std::make_index_sequence<nets::n12.size()>{}, comp, proj);
    } else if constexpr (N == 13) {
        apply_network(first, nets::n13, std::make_index_sequence<nets::n13.size()>{}, comp, proj);
    } else if constexpr (N == 14) {
        apply_network(first, nets::n14, std::make_index_sequence<nets::n14.size()>{}, comp, proj);
    } else if constexpr (N == 15) {
        apply_network(first, nets::n15, std::make_index_sequence<nets::n15.size()>{}, comp, proj);
    } else if constexpr (N == 16) {
        apply_network(first, nets::n16, std::make_index_sequence<nets::n16.size()>{}, comp, proj);
    } else if constexpr (N == 17) {
        apply_network(first, nets::n17, std::make_index_sequence<nets::n17.size()>{}, comp, proj);
    } else if constexpr (N == 18) {
        apply_network(first, nets::n18, std::make_index_sequence<nets::n18.size()>{}, comp, proj);
    } else if constexpr (N == 19) {
        apply_network(first, nets::n19, std::make_index_sequence<nets::n19.size()>{}, comp, proj);
    } else if constexpr (N == 20) {
        apply_network(first, nets::n20, std::make_index_sequence<nets::n20.size()>{}, comp, proj);
    } else if constexpr (N == 21) {
        apply_network(first, nets::n21, std::make_index_sequence<nets::n21.size()>{}, comp, proj);
    } else if constexpr (N == 22) {
        apply_network(first, nets::n22, std::make_index_sequence<nets::n22.size()>{}, comp, proj);
    } else if constexpr (N == 23) {
        apply_network(first, nets::n23, std::make_index_sequence<nets::n23.size()>{}, comp, proj);
    } else if constexpr (N == 24) {
        apply_network(first, nets::n24, std::make_index_sequence<nets::n24.size()>{}, comp, proj);
    } else {
        detail::oems_sort<(int)N>(first, comp, proj);
    }
}

// OEMS-synthesised network entry point.  Identical interface to
// `sort_network` but always uses Batcher's odd-even merge sort, even for
// N<=24 where a hand-listed best-known network exists.  Kept so the
// benchmark can quantify exactly what the best-known networks buy over the
// generic synthesised one.
template <std::size_t N, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void sort_network_oems(It first, Cmp comp, Proj proj) {
    if constexpr (N > 1) detail::oems_sort<(int)N>(first, comp, proj);
}

// ---------------------------------------------------------------------------
// Pure branchless insertion sort.  Kept as a benchmark baseline: same cswap
// primitive, no network at all.
// ---------------------------------------------------------------------------
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void insertion_sort(It first, It last, Cmp comp = {}, Proj proj = {}) {
    const auto n = static_cast<std::size_t>(last - first);
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = i; j > 0; --j) {
            cswap(first[j - 1], first[j], comp, proj);
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level runtime dispatch for N=0..24.  The switch becomes a jump table
// of fully-inlined, fully-unrolled networks.
// ---------------------------------------------------------------------------
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void sort(It first, It last, Cmp comp = {}, Proj proj = {}) {
    const auto n = static_cast<std::size_t>(last - first);
    switch (n) {
        case 0: case 1: return;
        case 2:  sort_network<2>(first, comp, proj);  return;
        case 3:  sort_network<3>(first, comp, proj);  return;
        case 4:  sort_network<4>(first, comp, proj);  return;
        case 5:  sort_network<5>(first, comp, proj);  return;
        case 6:  sort_network<6>(first, comp, proj);  return;
        case 7:  sort_network<7>(first, comp, proj);  return;
        case 8:  sort_network<8>(first, comp, proj);  return;
        case 9:  sort_network<9>(first, comp, proj);  return;
        case 10: sort_network<10>(first, comp, proj); return;
        case 11: sort_network<11>(first, comp, proj); return;
        case 12: sort_network<12>(first, comp, proj); return;
        case 13: sort_network<13>(first, comp, proj); return;
        case 14: sort_network<14>(first, comp, proj); return;
        case 15: sort_network<15>(first, comp, proj); return;
        case 16: sort_network<16>(first, comp, proj); return;
        case 17: sort_network<17>(first, comp, proj); return;
        case 18: sort_network<18>(first, comp, proj); return;
        case 19: sort_network<19>(first, comp, proj); return;
        case 20: sort_network<20>(first, comp, proj); return;
        case 21: sort_network<21>(first, comp, proj); return;
        case 22: sort_network<22>(first, comp, proj); return;
        case 23: sort_network<23>(first, comp, proj); return;
        case 24: sort_network<24>(first, comp, proj); return;
        default:
            // Beyond N=24 fall back to the caller's std::sort -- this header
            // is intentionally bounded.  Static_assert at use sites for
            // compile-time-known N.
            for (std::size_t i = 1; i < n; ++i)
                for (std::size_t j = i; j > 0; --j)
                    cswap(first[j - 1], first[j], comp, proj);
            return;
    }
}

// Compile-time-N entry point — collapses the dispatch when N is known.
template <std::size_t N, class It, class Cmp = std::less<>, class Proj = std::identity>
[[gnu::always_inline]] inline void sort_n(It first, Cmp comp = {}, Proj proj = {}) {
    static_assert(N <= 24, "sort_n<N> bounded at 24");
    sort_network<N>(first, comp, proj);
}

// ===========================================================================
// Register-blocked runtime-size dispatch (the array length is a *run-time*
// value, not a template parameter).
// ===========================================================================
//
// libc++'s and AlphaDev's `varsortN` routines get part of their speed from
// REGISTER BLOCKING: load the whole short run into named locals, run the
// network with no intermediate memory traffic, then store back.  We generalise
// that to every best-known network up to N=24: copy the block into a local
// array `buf[N]` of compile-time size, which the compiler scalar-replaces into
// registers (after the copy it provably cannot alias the caller's storage), run
// the network on `buf`, copy back.  Versus sorting in place through the
// iterator this removes the per-compare-exchange store/reload the optimiser is
// otherwise forced to keep when it cannot prove non-aliasing -- the effect is
// largest on pair64, whose every CE writes 16 bytes.

namespace detail {
template <std::size_t N, class T, class It>
[[gnu::always_inline]] inline void load_block(T (&buf)[N], It first) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((buf[Is] = first[Is]), ...);
    }(std::make_index_sequence<N>{});
}
template <std::size_t N, class T, class It>
[[gnu::always_inline]] inline void store_block(const T (&buf)[N], It first) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((first[Is] = buf[Is]), ...);
    }(std::make_index_sequence<N>{});
}
}  // namespace detail

// Sort one compile-time-size block with register blocking.
template <std::size_t N, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void sort_block_reg(It first, Cmp comp, Proj proj) {
    using T = typename std::iterator_traits<It>::value_type;
    T buf[N];
    detail::load_block<N>(buf, first);
    sort_network<N>(static_cast<T*>(buf), comp, proj);
    detail::store_block<N>(buf, first);
}

// Runtime-size entry point: register-blocked best-known networks, N in [0,24].
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void sort_reg(It first, It last, Cmp comp = {}, Proj proj = {}) {
    const auto n = static_cast<std::size_t>(last - first);
    switch (n) {
        case 0: case 1: return;
        case 2:  sort_block_reg<2>(first, comp, proj);  return;
        case 3:  sort_block_reg<3>(first, comp, proj);  return;
        case 4:  sort_block_reg<4>(first, comp, proj);  return;
        case 5:  sort_block_reg<5>(first, comp, proj);  return;
        case 6:  sort_block_reg<6>(first, comp, proj);  return;
        case 7:  sort_block_reg<7>(first, comp, proj);  return;
        case 8:  sort_block_reg<8>(first, comp, proj);  return;
        case 9:  sort_block_reg<9>(first, comp, proj);  return;
        case 10: sort_block_reg<10>(first, comp, proj); return;
        case 11: sort_block_reg<11>(first, comp, proj); return;
        case 12: sort_block_reg<12>(first, comp, proj); return;
        case 13: sort_block_reg<13>(first, comp, proj); return;
        case 14: sort_block_reg<14>(first, comp, proj); return;
        case 15: sort_block_reg<15>(first, comp, proj); return;
        case 16: sort_block_reg<16>(first, comp, proj); return;
        case 17: sort_block_reg<17>(first, comp, proj); return;
        case 18: sort_block_reg<18>(first, comp, proj); return;
        case 19: sort_block_reg<19>(first, comp, proj); return;
        case 20: sort_block_reg<20>(first, comp, proj); return;
        case 21: sort_block_reg<21>(first, comp, proj); return;
        case 22: sort_block_reg<22>(first, comp, proj); return;
        case 23: sort_block_reg<23>(first, comp, proj); return;
        case 24: sort_block_reg<24>(first, comp, proj); return;
        default:
            for (std::size_t i = 1; i < n; ++i)
                for (std::size_t j = i; j > 0; --j)
                    cswap(first[j - 1], first[j], comp, proj);
            return;
    }
}

// ===========================================================================
// `varsort`: faithful port of the libc++/AlphaDev-style routines supplied by
// the user, for direct head-to-head comparison.  The distinguishing pieces are
//   (a) `cas`  -- a *generic* ternary-blend compare-exchange (min/max into two
//                 locals).  Unlike our `cswap` it is NOT specialised for the
//                 16-byte pair64 lex order, so on pair64 it tests whether the
//                 plain ternary blend is as good as our hand-tuned path.
//   (b) `partially_sorted_swap` -- libc++'s 3-input partial sort used by the
//                 size-3/4/5 routines (fewer ops than a 2-CE decomposition when
//                 a partial order already holds).
// Sizes 2..5 use the user's exact routines; 6..24 fall back to our
// register-blocked best-known networks (also register-blocked, so the only
// variable at 2..5 is the CE primitive itself).
// ===========================================================================

template <class T, class Comp, class Proj>
[[gnu::always_inline]] inline void cas(T& x, T& y, Comp comp, Proj proj) {
    bool test = std::invoke(comp, std::invoke(proj, x), std::invoke(proj, y));
    T min_val = test ? x : y;
    T max_val = test ? y : x;
    x = min_val;
    y = max_val;
}

template <class T, class Comp, class Proj>
[[gnu::always_inline]] inline void partially_sorted_swap(T& x, T& y, T& z,
                                                         Comp comp, Proj proj) {
    // Exact libc++ __partially_sorted_swap, adapted for projection.
    // Assumes (after projection) that y <= z already.
    bool r1 = std::invoke(comp, std::invoke(proj, z), std::invoke(proj, x));
    T tmp   = r1 ? z : x;
    z       = r1 ? x : z;
    bool r2 = std::invoke(comp, std::invoke(proj, tmp), std::invoke(proj, y));
    x       = r2 ? x : y;
    y       = r2 ? y : tmp;
}

template <std::size_t N, class It, class Cmp, class Proj>
[[gnu::always_inline]] inline void varsort_block(It first, Cmp comp, Proj proj) {
    using T = typename std::iterator_traits<It>::value_type;
    if constexpr (N == 2) {
        T x0 = first[0], x1 = first[1];
        cas(x0, x1, comp, proj);
        first[0] = x0; first[1] = x1;
    } else if constexpr (N == 3) {
        T x0 = first[0], x1 = first[1], x2 = first[2];
        cas(x0, x1, comp, proj);
        cas(x1, x2, comp, proj);
        cas(x0, x1, comp, proj);
        first[0] = x0; first[1] = x1; first[2] = x2;
    } else if constexpr (N == 4) {
        T x0 = first[0], x1 = first[1], x2 = first[2], x3 = first[3];
        cas(x0, x1, comp, proj);
        cas(x1, x2, comp, proj);
        cas(x0, x1, comp, proj);
        cas(x2, x3, comp, proj);
        cas(x1, x2, comp, proj);
        cas(x0, x1, comp, proj);
        first[0] = x0; first[1] = x1; first[2] = x2; first[3] = x3;
    } else if constexpr (N == 5) {
        T x0 = first[0], x1 = first[1], x2 = first[2], x3 = first[3], x4 = first[4];
        cas(x0, x1, comp, proj);
        cas(x3, x4, comp, proj);
        partially_sorted_swap(x2, x3, x4, comp, proj);
        cas(x1, x4, comp, proj);
        partially_sorted_swap(x0, x2, x3, comp, proj);
        partially_sorted_swap(x1, x2, x3, comp, proj);
        first[0] = x0; first[1] = x1; first[2] = x2; first[3] = x3; first[4] = x4;
    } else {
        sort_block_reg<N>(first, comp, proj);
    }
}

template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void varsort(It first, It last, Cmp comp = {}, Proj proj = {}) {
    const auto n = static_cast<std::size_t>(last - first);
    switch (n) {
        case 0: case 1: return;
        case 2:  varsort_block<2>(first, comp, proj);  return;
        case 3:  varsort_block<3>(first, comp, proj);  return;
        case 4:  varsort_block<4>(first, comp, proj);  return;
        case 5:  varsort_block<5>(first, comp, proj);  return;
        case 6:  sort_block_reg<6>(first, comp, proj);  return;
        case 7:  sort_block_reg<7>(first, comp, proj);  return;
        case 8:  sort_block_reg<8>(first, comp, proj);  return;
        case 9:  sort_block_reg<9>(first, comp, proj);  return;
        case 10: sort_block_reg<10>(first, comp, proj); return;
        case 11: sort_block_reg<11>(first, comp, proj); return;
        case 12: sort_block_reg<12>(first, comp, proj); return;
        case 13: sort_block_reg<13>(first, comp, proj); return;
        case 14: sort_block_reg<14>(first, comp, proj); return;
        case 15: sort_block_reg<15>(first, comp, proj); return;
        case 16: sort_block_reg<16>(first, comp, proj); return;
        case 17: sort_block_reg<17>(first, comp, proj); return;
        case 18: sort_block_reg<18>(first, comp, proj); return;
        case 19: sort_block_reg<19>(first, comp, proj); return;
        case 20: sort_block_reg<20>(first, comp, proj); return;
        case 21: sort_block_reg<21>(first, comp, proj); return;
        case 22: sort_block_reg<22>(first, comp, proj); return;
        case 23: sort_block_reg<23>(first, comp, proj); return;
        case 24: sort_block_reg<24>(first, comp, proj); return;
        default:
            for (std::size_t i = 1; i < n; ++i)
                for (std::size_t j = i; j > 0; --j)
                    cswap(first[j - 1], first[j], comp, proj);
            return;
    }
}

}  // namespace partitions::small_sort

#endif  // PARTITIONS_SMALL_SORT_HPP
