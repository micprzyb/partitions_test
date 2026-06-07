#ifndef PARTITIONS_SMALL_HALVE_REV_HPP
#define PARTITIONS_SMALL_HALVE_REV_HPP

// REVERSED (descending) small-array halvers -- the mirror of small_halve.hpp.
//
// A forward halver `halve_n<N>` splits the block by rank so the N/2 SMALLEST
// occupy the bottom half [first, first+N/2).  Recursively applied (halve_sort)
// it sorts ASCENDING.  Here we want the opposite: the N/2 LARGEST in the bottom
// half, so recursive halving sorts DESCENDING -- the leaf primitive of the
// reversed quicksort.
//
// HOW (and why NOT a negated comparator).  A comparator network's correctness
// rests on its compare-exchanges being a strict-weak order (0/1 principle).
// Feeding the network the negated comparator `ge(a,b)=!(a<b)` is unsound: `ge`
// is reflexive, so the proofs and equal-element handling break.  Instead we keep
// the *strict* `<` comparator but REVERSE THE COMPARE-EXCHANGE DIRECTION -- each
// `cswap_rev` puts the LARGER element at the lower index.  By comparator-network
// duality (reversing every comparator == complementing the order: run on -x and
// negate) the SAME network arrays from `small_halve::nets` become valid
// descending halvers, and equal elements are never swapped (strict, irreflexive)
// -- which is exactly correct for a sort.  Proven by 0/1 enumeration in
// tools/verify_small_halve_rev.cpp (min(bottom) >= max(top) for all 2^N inputs).
//
// `cswap_rev` mirrors every `small_sort::cswap` specialization one-for-one,
// reusing the same branchless swap primitives (detail_cs::half_swap/word_swap)
// so the descending halver inherits identical low-level codegen to the forward
// one -- only the compare operand order / blend selection flips.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "small_halve.hpp"  // nets::hN, the shared network arrays
#include "small_sort.hpp"   // detail_cs::*, detail::load_block/store_block

namespace partitions::small_halve_rev {

namespace dcs = small_sort::detail_cs;

// Descending compare-exchange: leave the LARGER (by comp) element at `a` (the
// lower index) and the smaller at `b`.  Exact mirror of small_sort::cswap.
template <class T, class Cmp, class Proj>
[[gnu::always_inline]] inline void cswap_rev(T& a, T& b, Cmp& comp, Proj& proj) {
    if constexpr (dcs::lex_pair_like<T> && dcs::is_default_less<Cmp, Proj>) {
        // Branchless lexicographic compare-exchange, descending.  Forward swaps
        // when b<a (put min low); here we swap when a<b (put max low).  Same
        // single compare on one operand pair, read `<` and `==` off its flags.
        const auto a0 = a.first, a1 = a.second;
        const auto b0 = b.first, b1 = b.second;
        const unsigned lt0 = static_cast<unsigned>(a0 < b0);
        const unsigned eq0 = static_cast<unsigned>(a0 == b0);
        const unsigned lt1 = static_cast<unsigned>(a1 < b1);
        const bool swap = (lt0 | (eq0 & lt1)) != 0;
        if constexpr (sizeof(T) == 16) {
            dcs::half_swap(a, b, swap);
        } else {
            dcs::word_swap(a, b, swap);
        }
    } else if constexpr (dcs::is_16byte_trivial<T>) {
        const bool swap = comp(proj(a), proj(b));  // a < b  -> put max (b) low
        dcs::half_swap(a, b, swap);
    } else if constexpr (std::is_integral_v<T> && sizeof(T) <= 8) {
        const bool swap = comp(proj(a), proj(b));  // a < b
        const T hi = swap ? b : a;                 // larger -> lower index
        const T lo = swap ? a : b;                 // smaller -> higher index
        a = hi;
        b = lo;
    } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
        const bool swap = comp(proj(a), proj(b));
        dcs::word_swap(a, b, swap);
    } else {
        const bool swap = comp(proj(a), proj(b));
        T tmp = a;
        a = swap ? b : a;
        b = swap ? tmp : b;
    }
}

template <class It, class Cmp, class Proj, std::size_t S, std::size_t... Is>
[[gnu::always_inline]] inline void apply_network_rev(
    It first, const std::array<small_sort::nets::P, S>& net,
    std::index_sequence<Is...>, Cmp& comp, Proj& proj) {
    (cswap_rev(first[net[Is].first], first[net[Is].second], comp, proj), ...);
}

// Apply the descending halver for compile-time N; reorders [first, first+N) so
// the N/2 LARGEST occupy [first, first+N/2).  Returns the split point
// first + N/2.  Reuses the forward network arrays from small_halve::nets.
#define PARTITIONS_HALVE_REV_APPLY(net)                                     \
    apply_network_rev(first, net, std::make_index_sequence<net.size()>{},   \
                      comp, proj)
template <std::size_t N, class It, class Cmp = std::less<>, class Proj = std::identity>
[[gnu::always_inline]] inline It halve_rev_n(It first, Cmp comp = {}, Proj proj = {}) {
    namespace nets = small_halve::nets;
    if constexpr (N == 2) PARTITIONS_HALVE_REV_APPLY(nets::h2);
    else if constexpr (N == 3) PARTITIONS_HALVE_REV_APPLY(nets::h3);
    else if constexpr (N == 4) PARTITIONS_HALVE_REV_APPLY(nets::h4);
    else if constexpr (N == 5) PARTITIONS_HALVE_REV_APPLY(nets::h5);
    else if constexpr (N == 6) PARTITIONS_HALVE_REV_APPLY(nets::h6);
    else if constexpr (N == 7) PARTITIONS_HALVE_REV_APPLY(nets::h7_alt);
    else if constexpr (N == 8) PARTITIONS_HALVE_REV_APPLY(nets::h8_new);
    else if constexpr (N == 9) PARTITIONS_HALVE_REV_APPLY(nets::h9);
    else if constexpr (N == 10) PARTITIONS_HALVE_REV_APPLY(nets::h10);
    else if constexpr (N == 11) PARTITIONS_HALVE_REV_APPLY(nets::h11);
    else if constexpr (N == 12) PARTITIONS_HALVE_REV_APPLY(nets::h12);
    else if constexpr (N == 13) PARTITIONS_HALVE_REV_APPLY(nets::h13);
    else if constexpr (N == 14) PARTITIONS_HALVE_REV_APPLY(nets::h14);
    else if constexpr (N == 15) PARTITIONS_HALVE_REV_APPLY(nets::h15);
    else if constexpr (N == 16) PARTITIONS_HALVE_REV_APPLY(nets::h16);
    else if constexpr (N == 17) PARTITIONS_HALVE_REV_APPLY(nets::h17);
    else if constexpr (N == 18) PARTITIONS_HALVE_REV_APPLY(nets::h18);
    else if constexpr (N == 19) PARTITIONS_HALVE_REV_APPLY(nets::h19);
    else if constexpr (N == 20) PARTITIONS_HALVE_REV_APPLY(nets::h20);
    else if constexpr (N == 21) PARTITIONS_HALVE_REV_APPLY(nets::h21);
    else if constexpr (N == 22) PARTITIONS_HALVE_REV_APPLY(nets::h22);
    else if constexpr (N == 23) PARTITIONS_HALVE_REV_APPLY(nets::h23);
    else if constexpr (N == 24) PARTITIONS_HALVE_REV_APPLY(nets::h24);
    return first + static_cast<std::iter_difference_t<It>>(N / 2);
}
#undef PARTITIONS_HALVE_REV_APPLY

// Descending insertion sort -- the n>24 fallback (mirror of the small_sort::sort
// fallback used by small_halve::halve).  Only reached for blocks larger than the
// hand-built networks; quicksort_rev's leaf cutoff is 24, so this is rarely hit.
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline void sort_rev(It first, It last, Cmp comp = {}, Proj proj = {}) {
    const auto n = static_cast<std::size_t>(last - first);
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = i; j > 0; --j)
            cswap_rev(first[j - 1], first[j], comp, proj);
}

// Runtime-size dispatch: descending halve, returns first + n/2.
template <class It, class Cmp = std::less<>, class Proj = std::identity>
inline It halve_rev(It first, It last, Cmp comp = {}, Proj proj = {}) {
    const auto n = static_cast<std::size_t>(last - first);
    switch (n) {
        case 0: case 1: return first + static_cast<std::iter_difference_t<It>>(n / 2);
        case 2:  return halve_rev_n<2>(first, comp, proj);
        case 3:  return halve_rev_n<3>(first, comp, proj);
        case 4:  return halve_rev_n<4>(first, comp, proj);
        case 5:  return halve_rev_n<5>(first, comp, proj);
        case 6:  return halve_rev_n<6>(first, comp, proj);
        case 7:  return halve_rev_n<7>(first, comp, proj);
        case 8:  return halve_rev_n<8>(first, comp, proj);
        case 9:  return halve_rev_n<9>(first, comp, proj);
        case 10: return halve_rev_n<10>(first, comp, proj);
        case 11: return halve_rev_n<11>(first, comp, proj);
        case 12: return halve_rev_n<12>(first, comp, proj);
        case 13: return halve_rev_n<13>(first, comp, proj);
        case 14: return halve_rev_n<14>(first, comp, proj);
        case 15: return halve_rev_n<15>(first, comp, proj);
        case 16: return halve_rev_n<16>(first, comp, proj);
        case 17: return halve_rev_n<17>(first, comp, proj);
        case 18: return halve_rev_n<18>(first, comp, proj);
        case 19: return halve_rev_n<19>(first, comp, proj);
        case 20: return halve_rev_n<20>(first, comp, proj);
        case 21: return halve_rev_n<21>(first, comp, proj);
        case 22: return halve_rev_n<22>(first, comp, proj);
        case 23: return halve_rev_n<23>(first, comp, proj);
        case 24: return halve_rev_n<24>(first, comp, proj);
        default:
            sort_rev(first, last, comp, proj);
            return first + static_cast<std::iter_difference_t<It>>(n / 2);
    }
}

}  // namespace partitions::small_halve_rev

#endif  // PARTITIONS_SMALL_HALVE_REV_HPP
