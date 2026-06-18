#ifndef PARTITIONS_SMALL_HALVE_HPP
#define PARTITIONS_SMALL_HALVE_HPP

// Branchless small-array HALVERS (n <= 24).
//
// A *halver* is a comparator network that, unlike a sorter, only splits the
// block by RANK: after `halve_n<N>` the N/2 smallest elements occupy the bottom
// half [first, first+N/2) and the N/2 largest the top half (each half unordered).
// That is a perfectly balanced partition *by median rank* -- exactly what a
// quicksort node wants -- and it needs strictly fewer compare-exchanges than a
// full sort:
//
//     N        2  3  4  5  6  7  8   12   16   20   24
//     sort     1  3  5  9 12 16 19   39   60   91  120   (optimal sorting net)
//     halver   1  2  4  6  8 12 15   29   47   63   88   (this file)  ~20-33% fewer
//
// The 0-1 principle does NOT make a halver a sorter: a halver only guarantees
// `max(bottom) <= min(top)`, not full grouping, so it is genuinely cheaper.
// (Networks were found by randomized-restart greedy pruning of the best-known
// sorting networks and proved correct by exhaustive 0/1 enumeration in
// tools/verify_small_halve.cpp -- the halver property holds for all 2^N inputs.)
//
// Every comparator reuses `small_sort::cswap` (the same branchless compare-
// exchange, including the pair64 half/word-swap specialisations), so halvers
// inherit all of small_sort's low-level codegen.  `halve_n<N>` returns the split
// point `first + N/2`.

#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <utility>

#include "small_sort.hpp"  // cswap, apply_network, load_block/store_block

namespace partitions::small_halve {

namespace nets {
using P = std::pair<int, int>;
inline constexpr std::array<P, 1> h2 = {{{0,1}}};
inline constexpr std::array<P, 2> h3 = {{{0,2},{0,1}}};
inline constexpr std::array<P, 4> h4 = {{{1,3},{0,1},{2,3},{1,2}}};
inline constexpr std::array<P, 6> h5 = {{{0,3},{0,2},{1,3},{0,1},{2,4},{1,2}}};
inline constexpr std::array<P, 8> h6 = {{{0,5},{1,3},{2,4},{1,2},{3,4},{0,3},{2,5},{2,3}}};
inline constexpr std::array<P, 12> h7 = {{{0,4},{1,5},{2,6},{0,2},{1,3},{4,6},{2,4},{3,5},{0,1},{2,3},{4,5},{1,4}}};
// 11-comparator halver for n=7 at split@3 (3 smallest in [0,3)), matching the
// boundary halve_n<7> returns (first+3).  Obtained by reflecting the wires of the
// split@4 11-comparator net (i -> 6-i): reflection maps split@k <-> split@(n-k),
// and 7-4 = 3.  One fewer comparator than h7 (12); verified by 0/1 enumeration.
inline constexpr std::array<P, 11> h7_alt = {{{0, 6}, {2, 4}, {1, 3}, {3, 6}, {4, 5}, {2, 3}, {0, 1}, {0, 5}, {3, 4}, {1, 2}, {2, 3}}};
inline constexpr std::array<P, 15> h8 = {{{0,2},{1,3},{5,7},{0,4},{1,5},{3,7},{0,1},{2,3},{4,5},{6,7},{2,4},{3,5},{1,4},{3,6},{3,4}}};
// 14-comparator, depth-4 halver for n=8 (split@4); one fewer comparator and two
// shallower than h8.  Verified by 0/1 enumeration.  Faster than h8 in real
// (scalar, one-block-per-call) use for all key types.
inline constexpr std::array<P, 14> h8_new = {{{0,7},{1,4},{2,5},{3,6},{0,2},{1,3},{4,6},{5,7},{0,6},{1,7},{2,4},{3,5},{2,5},{3,4}}};
inline constexpr std::array<P, 17> h9 = {{{2,5},{4,8},{0,7},{2,4},{3,8},{5,6},{0,2},{1,3},{4,5},{7,8},{1,4},{3,6},{5,7},{3,5},{2,3},{4,5},{3,4}}};
// 20-comparator, depth-6 halver for n=10 (split@5): one fewer comparator than
// the previous 21-CE net and measurably faster for all three element types --
// see Plan10.md.  Found by iterated-local-search over the 1024-input 0/1
// validity check, chosen by direct benchmark from the pool of 20-CE minima
// (tools/search_halve.cpp).  Verified by exhaustive 0/1 enumeration.
inline constexpr std::array<P, 20> h10 = {{{0,8},{1,9},{2,7},{0,2},{4,5},{1,4},{3,6},{5,8},{7,9},{2,4},{5,7},{8,9},{0,1},{1,5},{2,3},{4,8},{6,7},{3,5},{4,6},{4,5}}};
// 23-comparator halver for n=11 (split@5): two fewer comparators than the
// previous 25-CE net and measurably faster for all three element types -- see
// Plan11.md.  Found by iterated-local-search over the 2048-input 0/1 validity
// check, chosen by direct benchmark from the pool of 23-CE minima
// (tools/search_halve.cpp).  Verified by exhaustive 0/1 enumeration.
inline constexpr std::array<P, 23> h11 = {{{0,9},{1,6},{0,1},{3,5},{4,10},{2,8},{1,10},{6,9},{1,3},{4,7},{8,10},{0,4},{1,2},{3,7},{5,9},{4,5},{7,8},{2,4},{3,6},{6,7},{3,4},{5,6},{4,5}}};
// 27-comparator, depth-8 halver for n=12 (split@6): two fewer comparators and
// one shallower than the previous 29-CE/depth-9 net, and measurably faster for
// all three element types (i64, pair64 lex, pair64 first-key) -- see Plan12.md.
// Found by iterated-local-search (greedy CE deletion + perturb-reprune) over the
// 4096-input 0/1 validity check, then chosen by direct benchmark from the pool
// of distinct 27-CE minima (tools/search_halve12.cpp).  Verified by exhaustive
// 0/1 enumeration (tools/verify_small_halve{,_rev}.cpp).
inline constexpr std::array<P, 27> h12 = {{{0,8},{1,7},{2,6},{5,9},{0,1},{2,5},{3,4},{6,9},{10,11},{0,2},{4,10},{2,11},{1,6},{9,11},{7,8},{3,6},{5,7},{1,4},{3,5},{6,8},{7,10},{2,5},{6,9},{4,6},{5,7},{5,6},{4,7}}};
// 30-comparator halver for n=13 (split@6): three fewer comparators than the
// previous 33-CE net and measurably faster for all three element types -- see
// Plan13.md.  Found by iterated-local-search over the 8192-input 0/1 validity
// check, chosen by direct benchmark from the pool of 30-CE minima
// (tools/search_halve.cpp).  Verified by exhaustive 0/1 enumeration.
inline constexpr std::array<P, 30> h13 = {{{0,12},{1,10},{2,9},{3,7},{5,11},{6,8},{1,6},{2,3},{4,11},{7,9},{8,10},{3,6},{7,8},{9,10},{5,9},{8,11},{1,2},{3,8},{4,7},{8,12},{5,8},{2,5},{6,9},{5,7},{0,4},{3,4},{6,8},{4,5},{6,7},{5,6}}};
// 34-comparator, depth-8 halver for n=14 (split@7): four fewer comparators and
// two shallower than the previous 38-CE net, and measurably faster for all three
// element types -- see Plan14.md.  Found by iterated-local-search over the
// 16384-input 0/1 validity check, chosen by direct benchmark from the pool of
// 34-CE minima (tools/search_halve.cpp).  Verified by exhaustive 0/1 enumeration.
inline constexpr std::array<P, 34> h14 = {{{0,1},{8,9},{2,3},{4,5},{6,7},{10,11},{6,10},{12,13},{0,2},{4,8},{5,9},{10,12},{3,7},{11,13},{2,8},{7,9},{0,6},{1,5},{7,13},{8,12},{2,10},{3,11},{4,6},{1,3},{5,11},{6,7},{3,10},{3,6},{5,8},{7,10},{5,6},{7,8},{6,7},{1,12}}};
// 39-comparator halver for n=15 (split@7): four fewer comparators than the
// previous 43-CE net and measurably faster for all three element types -- see
// Plan15.md.  Found by iterated-local-search over the 32768-input 0/1 validity
// check, chosen by direct benchmark from the pool of 39-CE minima.  Verified by
// exhaustive 0/1 enumeration.
inline constexpr std::array<P, 39> h15 = {{{1,2},{4,14},{5,8},{6,13},{9,11},{1,5},{2,8},{3,7},{6,9},{10,12},{11,13},{0,7},{1,6},{2,9},{4,10},{5,11},{8,13},{12,14},{0,6},{2,4},{3,5},{7,14},{7,11},{8,10},{9,12},{1,14},{0,3},{4,7},{5,9},{6,8},{4,6},{7,9},{3,5},{10,12},{8,10},{5,6},{2,11},{7,8},{6,7}}};
// 44-comparator halver for n=16 (split@8): three fewer comparators than the
// previous 47-CE net and measurably faster for all three element types -- see
// Plan16.md.  Found by iterated-local-search over the 65536-input 0/1 validity
// check, chosen by direct benchmark from the pool of 44-CE minima.  Verified by
// exhaustive 0/1 enumeration.
inline constexpr std::array<P, 44> h16 = {{{2,15},{3,14},{4,8},{7,11},{9,10},{0,5},{1,7},{2,9},{3,4},{6,13},{8,14},{10,15},{0,1},{2,3},{10,11},{14,15},{0,2},{7,9},{1,3},{4,10},{5,11},{6,7},{8,9},{12,14},{3,13},{13,15},{1,2},{3,12},{10,12},{4,6},{5,7},{8,10},{9,11},{13,14},{2,6},{5,8},{7,10},{9,13},{3,6},{9,12},{8,9},{6,8},{7,9},{7,8}}};
// 48-comparator halver for n=17 (split@8): EIGHT fewer comparators than the
// previous 56-CE net and substantially faster for all three element types -- see
// Plan17.md.  Found by iterated-local-search over the 131072-input 0/1 validity
// check (two waves, 24-way parallel, the second reseeded from 48-CE minima),
// chosen by direct benchmark from the pool of 48-CE minima (tools/search_halve.cpp).
// Verified by exhaustive 0/1 enumeration.
inline constexpr std::array<P, 48> h17 = {{{0,11},{1,15},{2,10},{7,11},{6,14},{4,5},{4,6},{8,12},{3,5},{9,16},{13,14},{0,6},{1,8},{2,8},{5,15},{3,7},{4,9},{6,16},{10,11},{0,9},{12,14},{0,2},{1,4},{5,6},{7,13},{8,9},{10,12},{11,14},{15,16},{2,5},{6,11},{9,13},{12,15},{3,4},{5,10},{7,8},{3,7},{4,8},{6,12},{6,9},{2,7},{4,5},{10,12},{5,7},{8,10},{6,8},{7,9},{7,8}}};
inline constexpr std::array<P, 52> h18 = {{{12,14},{0,1},{6,7},{16,17},{1,2},{15,16},{12,13},{10,11},{14,17},{4,5},{8,9},{9,11},{6,10},{3,5},{1,4},{13,16},{12,15},{7,16},{2,5},{0,3},{4,17},{0,13},{4,13},{10,13},{1,14},{2,14},{4,9},{3,10},{5,17},{8,15},{7,9},{0,12},{10,15},{4,12},{1,6},{6,12},{9,15},{3,8},{8,10},{5,11},{5,13},{8,12},{2,10},{5,16},{5,7},{2,5},{9,14},{5,9},{7,10},{7,12},{5,12},{7,9}}};
inline constexpr std::array<P, 56> h19 = {{{12,14},{10,11},{15,16},{17,18},{12,13},{15,17},{12,15},{16,18},{4,5},{8,9},{14,17},{0,1},{9,11},{2,3},{3,5},{13,16},{1,5},{0,3},{2,4},{6,7},{4,17},{0,13},{6,10},{2,14},{3,18},{4,13},{5,11},{7,16},{10,13},{4,9},{3,10},{13,16},{8,15},{5,18},{7,9},{1,14},{5,17},{10,15},{0,4},{3,8},{2,4},{8,10},{1,7},{5,13},{4,12},{8,12},{9,14},{1,6},{5,7},{7,10},{6,12},{9,15},{5,7},{5,9},{7,12},{7,9}}};
inline constexpr std::array<P, 60> h20 = {{{13,15},{7,8},{16,17},{11,12},{18,19},{1,2},{16,18},{17,19},{15,18},{0,2},{5,6},{3,4},{9,10},{10,12},{13,14},{1,5},{4,6},{3,5},{14,17},{2,6},{1,16},{0,4},{7,11},{5,18},{0,14},{12,18},{3,15},{4,19},{5,14},{11,14},{13,16},{1,7},{4,11},{5,10},{2,17},{6,19},{9,16},{8,10},{12,14},{6,12},{11,16},{2,15},{4,9},{3,13},{0,5},{9,11},{6,17},{2,11},{10,15},{5,13},{6,8},{2,6},{10,16},{9,13},{7,13},{6,10},{8,11},{8,13},{6,13},{8,10}}};
// 65-comparator halver for n=21 (split@10): FOUR fewer comparators than the
// previous 69-CE net (Batcher-derived, never CEGAR-searched -- it was larger than
// the new n=22 at 68!) and substantially faster for all three element types --
// see Plan21.md.  Found by plateau-walking iterated-local-search with a CEGAR +
// monotonicity ("k-zero inputs only", C(21,11)=352,716 instead of 2^21) validity
// check, 6-way parallel across four reseeded waves (69->66->65; 64 not found
// across 1181 distinct 65-CE minima), then chosen by direct benchmark (median of
// 9) from that pool (optimizers/search_halve_cegar2.cpp).  Verified by exhaustive
// 0/1 enumeration (tools/verify_small_halve{,_rev}).
inline constexpr std::array<P, 65> h21 = {{{6,7},{2,3},{4,5},{16,20},{0,1},{5,7},{8,9},{10,11},{17,18},{9,11},{14,15},{0,2},{12,13},{4,6},{8,10},{16,17},{0,8},{1,3},{12,14},{4,12},{17,19},{13,15},{1,9},{7,15},{5,13},{6,14},{3,11},{3,7},{8,12},{2,10},{3,12},{10,14},{7,19},{3,17},{7,20},{7,17},{9,13},{13,17},{12,20},{2,6},{6,13},{6,9},{3,16},{7,10},{2,16},{9,16},{4,8},{10,18},{0,8},{11,13},{1,5},{5,10},{5,6},{10,18},{6,12},{11,15},{11,14},{1,7},{6,8},{7,8},{10,16},{11,12},{10,11},{9,10},{8,10}}};
// 68-comparator halver for n=22 (split@11): SIX fewer comparators than the
// previous 74-CE net (Batcher-derived, never CEGAR-searched -- it had more CE
// than n=21's 69!) and substantially faster for all three element types -- see
// Plan22.md.  Found by plateau-walking iterated-local-search with a CEGAR +
// monotonicity ("k-zero inputs only", C(22,11)=705,432 instead of 2^22) validity
// check, 6-way parallel across three reseeded waves (74->68; 67 not found across
// 1192 distinct 68-CE minima), then chosen by direct benchmark (median of 9) from
// that pool (optimizers/search_halve_cegar2.cpp).  Verified by exhaustive 0/1
// enumeration (tools/verify_small_halve{,_rev}).
inline constexpr std::array<P, 68> h22 = {{{8,9},{0,1},{20,21},{2,3},{16,17},{12,13},{18,19},{19,21},{18,20},{0,2},{4,5},{1,3},{6,7},{10,11},{14,15},{4,6},{5,7},{15,17},{14,16},{2,6},{15,19},{1,5},{16,20},{8,12},{5,17},{3,7},{11,13},{6,20},{1,15},{5,21},{7,20},{6,15},{12,15},{0,4},{2,18},{6,11},{14,18},{4,16},{2,6},{6,14},{5,12},{0,1},{1,14},{7,17},{3,19},{13,21},{10,18},{9,11},{7,15},{12,18},{3,16},{5,10},{7,13},{3,12},{10,12},{7,19},{7,9},{3,7},{4,8},{11,16},{11,18},{8,14},{7,11},{9,12},{10,14},{9,14},{7,14},{9,11}}};
// 74-comparator halver for n=23 (split@11): TEN fewer comparators than the
// original 84-CE Batcher net, and FOUR below the 78-CE the earlier CEGAR search
// got stuck at -- substantially faster for all three element types -- see
// Plan23.md (PLAN B).  KEY INSIGHT: the 78 floor was a Batcher-seed-bias local
// optimum; seeding the search from a STRUCTURALLY DIFFERENT basin -- the n=24
// halver with a wire forced to +inf, its (.,23) comparators dropped (a valid
// split@12 23-halver), reversed to split@11 -- let the plateau-walk descend
// 78->75->74.  A violation-annealing engine (optimizers/anneal_halve.cpp) then
// showed size 73 is (almost certainly) infeasible: the best 73-net still
// mis-sorts 260/1,352,078 k-zero inputs across ~78 strong restarts, so 74 is the
// practical floor.  Chosen by direct benchmark (median of 9) from the 550-net
// pool of 74-CE minima.  Verified by exhaustive 0/1 enumeration
// (tools/verify_small_halve{,_rev}).
inline constexpr std::array<P, 74> h23 = {{{0,1},{8,9},{10,11},{14,15},{12,13},{7,21},{12,14},{2,3},{0,2},{4,5},{9,11},{16,17},{18,19},{4,6},{17,19},{20,22},{1,3},{21,22},{8,10},{7,20},{2,6},{16,18},{1,5},{17,21},{10,14},{13,15},{8,12},{0,4},{3,15},{9,13},{6,14},{1,7},{12,17},{18,20},{3,11},{3,20},{5,13},{13,15},{5,18},{2,10},{9,16},{19,22},{7,16},{3,4},{0,8},{6,19},{13,19},{10,18},{3,12},{10,21},{4,17},{1,9},{5,8},{4,12},{6,8},{11,21},{14,17},{2,9},{6,9},{9,10},{10,12},{4,9},{13,22},{12,16},{8,12},{18,20},{8,9},{13,14},{11,13},{11,18},{11,12},{7,10},{10,11},{9,11}}};
// 80-comparator halver for n=24 (split@12): TWO below the 82-CE the original
// CEGAR search got stuck at (which was Batcher-seed-bias, like n=23's 78) -- see
// Plan24.md (PLAN B).  KEY INSIGHT (mirroring n=23): seed the search from a
// STRUCTURALLY DIFFERENT basin -- the new n=23 halver (74) extended to a split@12
// 24-halver (h23 on wires 0..22, then min of {11..22,23} sifted to position 11;
// 86 CE, validated) -- a non-Batcher seed.  The plateau-walk then descended
// 82->81->80; four deep reseeded waves found 1091 distinct 80-CE nets but no 79.
// Chosen by direct benchmark (median of 9) from that pool: faster overall (big
// pair64 win) at a slight i64 cost.  Verified by exhaustive 0/1 enumeration
// (tools/verify_small_halve{,_rev}).  [79 hunt + annealer-verification ongoing.]
inline constexpr std::array<P, 80> h24 = {{{4,6},{20,22},{14,15},{2,6},{12,13},{16,17},{1,3},{12,14},{7,21},{18,19},{10,11},{13,15},{8,9},{1,5},{17,19},{21,22},{9,11},{8,10},{0,23},{17,21},{2,3},{10,14},{16,18},{7,20},{2,10},{3,15},{18,20},{8,12},{4,5},{3,11},{9,13},{19,22},{12,17},{5,13},{3,20},{6,14},{9,16},{5,18},{18,21},{1,7},{14,20},{5,8},{6,19},{6,8},{3,4},{14,23},{4,17},{6,9},{1,2},{0,4},{10,18},{11,22},{3,12},{7,16},{4,12},{0,7},{11,17},{13,19},{4,7},{2,9},{7,8},{9,12},{12,16},{15,21},{11,13},{11,14},{17,18},{16,23},{8,12},{12,17},{11,12},{9,10},{7,10},{8,10},{13,16},{13,15},{10,11},{11,12},{13,14},{11,13}}};
}  // namespace nets

// Apply the halver for compile-time N; reorders [first, first+N) so the N/2
// smallest occupy [first, first+N/2).  Returns the split point first + N/2.
// NB: each branch calls `apply_network` DIRECTLY rather than through a local
// lambda.  A `[&]` lambda here is not `always_inline`, and GCC outlined it into
// a `.isra` clone *and called it* for some element types (pair64) while inlining
// it for others (i64) -- which both lost vectorisation and made microbenchmarks
// wildly context-dependent.  Direct calls guarantee the network inlines.
#define PARTITIONS_HALVE_APPLY(net) \
    small_sort::apply_network(first, net, std::make_index_sequence<net.size()>{}, comp, proj)
template <std::size_t N, class It, class Cmp = std::less<>, class Proj = std::identity>
[[gnu::always_inline]] inline It halve_n(It first, Cmp comp = {}, Proj proj = {}) {
    if constexpr (N == 2) PARTITIONS_HALVE_APPLY(nets::h2);
    else if constexpr (N == 3) PARTITIONS_HALVE_APPLY(nets::h3);
    else if constexpr (N == 4) PARTITIONS_HALVE_APPLY(nets::h4);
    else if constexpr (N == 5) PARTITIONS_HALVE_APPLY(nets::h5);
    else if constexpr (N == 6) PARTITIONS_HALVE_APPLY(nets::h6);
    else if constexpr (N == 7) PARTITIONS_HALVE_APPLY(nets::h7_alt);
    else if constexpr (N == 8) PARTITIONS_HALVE_APPLY(nets::h8_new);
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

// Register-blocked halver: copy the block into a local array (scalar-replaced
// into registers), run the network with no intermediate memory traffic, store
// back.  Same idea as small_sort::sort_block_reg -- helps most for pair64.
template <std::size_t N, class It, class Cmp = std::less<>, class Proj = std::identity>
[[gnu::always_inline]] inline It halve_reg(It first, Cmp comp = {}, Proj proj = {}) {
    using T = typename std::iterator_traits<It>::value_type;
    T buf[N];
    small_sort::detail::load_block<N>(buf, first);
    halve_n<N>(static_cast<T*>(buf), comp, proj);
    small_sort::detail::store_block<N>(buf, first);
    return first + static_cast<std::iter_difference_t<It>>(N / 2);
}

// Runtime-size dispatch: returns the split point first + n/2.  For n > 24 falls
// back to a full sort (this header is intentionally bounded at 24).
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
            small_sort::sort(first, last, comp, proj);
            return first + static_cast<std::iter_difference_t<It>>(n / 2);
    }
}

}  // namespace partitions::small_halve

#endif  // PARTITIONS_SMALL_HALVE_HPP
