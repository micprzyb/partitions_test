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
inline constexpr std::array<P, 15> h8 = {{{0,2},{1,3},{5,7},{0,4},{1,5},{3,7},{0,1},{2,3},{4,5},{6,7},{2,4},{3,5},{1,4},{3,6},{3,4}}};
inline constexpr std::array<P, 17> h9 = {{{2,5},{4,8},{0,7},{2,4},{3,8},{5,6},{0,2},{1,3},{4,5},{7,8},{1,4},{3,6},{5,7},{3,5},{2,3},{4,5},{3,4}}};
inline constexpr std::array<P, 21> h10 = {{{0,8},{1,9},{2,7},{3,5},{4,6},{0,2},{1,4},{5,8},{7,9},{3,6},{2,4},{5,7},{0,1},{8,9},{1,5},{2,3},{4,8},{6,7},{3,5},{4,6},{4,5}}};
inline constexpr std::array<P, 25> h11 = {{{0,9},{2,4},{3,7},{5,8},{0,1},{3,5},{4,10},{6,9},{7,8},{1,3},{2,5},{4,7},{8,10},{0,4},{1,2},{3,7},{5,9},{4,5},{7,8},{3,6},{2,4},{5,7},{3,4},{5,6},{4,5}}};
inline constexpr std::array<P, 29> h12 = {{{1,7},{2,6},{3,11},{4,10},{5,9},{0,1},{2,5},{3,4},{6,9},{7,8},{10,11},{0,2},{1,6},{5,10},{9,11},{1,2},{4,6},{5,7},{9,10},{1,4},{3,5},{6,8},{7,10},{2,5},{6,9},{6,7},{4,6},{5,7},{5,6}}};
inline constexpr std::array<P, 33> h13 = {{{0,12},{1,10},{2,9},{3,7},{5,11},{2,3},{4,11},{7,9},{0,4},{1,2},{3,6},{7,8},{11,12},{5,9},{4,6},{8,11},{10,12},{0,1},{2,5},{3,8},{4,7},{6,11},{9,10},{6,9},{7,8},{1,3},{2,4},{3,4},{5,7},{6,8},{4,5},{6,7},{5,6}}};
inline constexpr std::array<P, 38> h14 = {{{0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{0,2},{1,3},{4,8},{5,9},{11,13},{6,10},{0,4},{1,2},{3,7},{5,8},{9,13},{11,12},{1,5},{3,9},{4,10},{7,13},{8,12},{2,10},{3,11},{4,6},{7,9},{1,3},{2,8},{5,11},{10,12},{2,6},{3,5},{7,11},{7,10},{5,6},{7,8},{6,7}}};
inline constexpr std::array<P, 43> h15 = {{{1,2},{3,10},{4,14},{5,8},{6,13},{9,11},{0,14},{1,5},{2,8},{3,7},{6,9},{10,12},{11,13},{0,7},{1,6},{2,9},{4,10},{5,11},{8,13},{12,14},{0,6},{2,4},{3,5},{7,11},{8,10},{9,12},{13,14},{0,3},{1,2},{4,7},{5,9},{6,8},{10,11},{12,13},{2,3},{4,6},{7,9},{10,12},{3,5},{8,10},{5,6},{7,8},{6,7}}};
inline constexpr std::array<P, 47> h16 = {{{0,13},{1,12},{2,15},{4,8},{5,6},{7,11},{9,10},{0,5},{1,7},{2,9},{3,4},{6,13},{8,14},{10,15},{11,12},{0,1},{2,3},{4,5},{6,8},{7,9},{10,11},{12,13},{14,15},{0,2},{1,3},{4,10},{5,11},{6,7},{8,9},{12,14},{13,15},{1,2},{3,12},{4,6},{5,7},{8,10},{9,11},{13,14},{2,6},{5,8},{7,10},{9,13},{3,6},{9,12},{6,8},{7,9},{7,8}}};
inline constexpr std::array<P, 56> h17 = {{{0,11},{1,15},{2,10},{3,5},{4,6},{8,12},{9,16},{13,14},{1,13},{2,8},{4,14},{5,15},{7,11},{6,16},{0,8},{3,7},{4,9},{10,11},{12,14},{5,6},{15,16},{0,2},{1,4},{7,13},{8,9},{10,12},{11,14},{0,3},{2,5},{6,11},{7,10},{9,13},{12,15},{14,16},{0,1},{3,4},{5,10},{6,9},{7,8},{11,15},{13,14},{1,2},{3,7},{4,8},{6,12},{11,13},{2,7},{4,5},{9,11},{10,12},{4,6},{5,7},{8,10},{6,8},{7,9},{7,8}}};
inline constexpr std::array<P, 53> h18 = {{{0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{0,2},{1,3},{4,12},{5,13},{6,8},{9,11},{14,16},{15,17},{0,14},{1,16},{2,15},{3,17},{0,6},{1,10},{2,9},{7,16},{8,15},{11,17},{1,4},{3,9},{5,7},{8,14},{10,12},{13,16},{2,5},{3,13},{4,14},{7,9},{8,10},{12,15},{3,5},{4,6},{11,13},{12,14},{5,12},{6,10},{7,11},{3,6},{5,7},{10,12},{11,14},{6,10},{7,11},{7,10}}};
inline constexpr std::array<P, 59> h19 = {{{0,12},{1,4},{2,8},{3,5},{6,17},{7,11},{9,14},{15,16},{0,2},{1,7},{3,6},{4,11},{5,17},{8,12},{10,15},{13,16},{14,18},{3,10},{4,14},{5,15},{6,13},{7,9},{11,17},{16,18},{0,7},{1,10},{4,6},{9,15},{11,16},{12,17},{13,14},{0,3},{2,6},{5,7},{8,11},{12,16},{1,8},{2,9},{3,4},{6,15},{7,13},{10,11},{12,18},{2,5},{6,9},{7,12},{8,10},{11,14},{4,8},{6,10},{9,12},{5,8},{6,7},{9,11},{10,13},{7,9},{7,8},{9,10},{8,9}}};
inline constexpr std::array<P, 63> h20 = {{{0,3},{1,7},{4,8},{6,9},{10,13},{11,15},{12,18},{14,17},{16,19},{0,14},{1,11},{2,16},{3,17},{4,12},{5,19},{6,10},{7,15},{8,18},{9,13},{0,4},{1,2},{3,8},{5,7},{11,16},{12,14},{15,19},{17,18},{2,12},{3,5},{4,11},{7,17},{8,15},{14,16},{0,1},{18,19},{2,6},{7,10},{9,12},{13,17},{1,6},{5,9},{7,11},{8,12},{10,14},{13,18},{3,5},{4,7},{8,10},{9,11},{12,15},{14,16},{5,7},{6,10},{9,13},{12,14},{6,7},{8,9},{10,11},{12,13},{7,9},{7,10},{9,12},{9,10}}};
inline constexpr std::array<P, 69> h21 = {{{0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},{0,2},{1,3},{4,6},{5,7},{8,10},{9,11},{12,14},{13,15},{16,18},{17,19},{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15},{0,4},{1,5},{3,7},{8,12},{9,13},{10,14},{15,19},{2,6},{3,18},{7,20},{2,16},{3,6},{5,18},{7,17},{11,20},{9,16},{3,8},{6,12},{7,10},{11,15},{13,17},{14,18},{4,9},{1,7},{10,11},{13,16},{5,10},{6,13},{7,8},{11,14},{12,16},{5,6},{10,12},{11,13},{8,9},{10,11},{6,7},{7,8},{9,11},{8,10},{9,10}}};
inline constexpr std::array<P, 74> h22 = {{{0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},{20,21},{0,2},{1,3},{4,6},{5,7},{8,12},{9,13},{14,16},{15,17},{18,20},{19,21},{0,4},{1,5},{2,6},{3,7},{8,10},{9,12},{11,13},{14,18},{15,19},{16,20},{17,21},{0,14},{1,15},{2,18},{3,19},{4,16},{5,17},{6,20},{7,21},{9,11},{10,12},{2,8},{3,11},{6,9},{10,18},{12,15},{13,19},{1,10},{3,16},{5,18},{6,14},{7,15},{8,12},{9,13},{11,20},{3,10},{4,8},{5,12},{9,16},{11,18},{13,17},{7,13},{8,14},{9,12},{7,11},{10,14},{5,10},{7,9},{11,16},{12,14},{9,11},{10,12},{9,12}}};
inline constexpr std::array<P, 84> h23 = {{{0,1},{2,3},{4,5},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},{20,21},{0,2},{4,6},{5,7},{8,10},{9,11},{12,14},{13,15},{16,18},{17,19},{21,22},{0,4},{1,5},{2,6},{3,7},{8,12},{9,13},{10,14},{11,15},{17,21},{18,20},{19,22},{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,19},{15,22},{1,2},{5,18},{9,16},{10,21},{12,20},{6,7},{15,19},{5,9},{10,18},{11,21},{12,17},{13,20},{6,16},{14,15},{3,17},{8,12},{20,21},{13,16},{3,4},{5,8},{9,12},{17,18},{7,13},{14,16},{1,8},{2,12},{3,9},{4,10},{11,17},{18,20},{2,6},{4,8},{7,11},{10,12},{13,18},{14,17},{6,9},{7,10},{11,13},{12,14},{8,9},{11,12},{9,10},{10,11}}};
inline constexpr std::array<P, 88> h24 = {{{0,20},{1,12},{2,16},{3,23},{4,6},{5,10},{8,14},{9,15},{11,22},{13,18},{17,19},{0,3},{1,11},{4,17},{5,13},{6,19},{8,9},{10,18},{12,22},{14,15},{16,21},{20,23},{0,1},{2,4},{3,12},{5,8},{6,9},{7,10},{11,20},{13,16},{14,17},{15,18},{19,21},{22,23},{2,5},{4,8},{6,11},{7,14},{9,16},{12,17},{15,19},{18,21},{1,8},{3,14},{4,7},{9,20},{10,12},{11,13},{15,22},{16,19},{0,7},{3,4},{6,11},{8,15},{9,14},{10,13},{12,17},{16,23},{18,22},{19,20},{1,6},{4,7},{5,9},{8,10},{14,18},{16,19},{17,22},{12,13},{4,5},{6,8},{7,9},{10,11},{14,16},{15,17},{13,19},{7,8},{9,11},{12,14},{15,16},{5,10},{13,18},{8,9},{14,15},{10,12},{11,13},{9,12},{11,14},{11,12}}};
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
