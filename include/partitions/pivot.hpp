#ifndef PARTITIONS_PIVOT_HPP
#define PARTITIONS_PIVOT_HPP

// Pivot-selection strategies.
//
// A strategy chooses a pivot for [first, last).  It may return the pivot in
// either of two conventions:
//
//   * a *position* -- an iterator into the block (the pivot is *that element).
//   * a *value*    -- a `value_pivot<K>{key}`, a key that need NOT occur in the
//                     block (e.g. (min+max)/2).  Use this when the strategy does
//                     not track the element's location, or computes a synthetic
//                     pivot.
//
//     I              operator()(I first, S last, Comp comp, Proj proj) const;
//     value_pivot<K> operator()(I first, S last, Comp comp, Proj proj) const;
//
// Downstream code is agnostic to the convention: `pivot::pivot_key_of` extracts
// the key from either, and `partition_with_pivot` (partition_with_pivot.hpp)
// drives the partition by position or by key accordingly.
//
// A strategy MAY reorder [first, last) as a side effect -- the typical
// median-of-medians implementation moves group medians together and runs a
// quickselect, leaving the block partly arranged "to help" the subsequent
// partition.  Such strategies are marked with `static constexpr bool
// reorders = true;` (the default is false).  The canonical usage is therefore
//
//     auto p   = strategy(first, last, comp, proj);   // may reorder the block
//     auto key = proj(*p);                            // read the pivot key now
//     auto mid = partition_by_position(alg, first, last, p, comp, proj);
//
// and every harness reads the pivot key *after* selection and never assumes the
// block is unchanged by it.  (Balance is permutation-invariant, so it can still
// be measured on the reordered block.)
//
// To add a strategy (e.g. your own median scheme), define a function object
// with a `name` and the operator() above -- set `reorders = true` if it mutates
// the block -- then add it to `default_pivots`.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "types.hpp"

namespace partitions::pivot {

// Trait: does a strategy reorder the block?  Strategies opt in with
// `static constexpr bool reorders = true;`; the default is false.
template <class P, class = void>
struct reorders : std::false_type {};
template <class P>
struct reorders<P, std::void_t<decltype(P::reorders)>>
    : std::bool_constant<P::reorders> {};
template <class P>
inline constexpr bool reorders_v = reorders<P>::value;

// A pivot expressed as a key value that need not appear in the block.
template <class K>
struct value_pivot {
    K key;
};
template <class K>
value_pivot(K) -> value_pivot<K>;

template <class T>
struct is_value_pivot : std::false_type {};
template <class K>
struct is_value_pivot<value_pivot<K>> : std::true_type {};
template <class T>
inline constexpr bool is_value_pivot_v =
    is_value_pivot<std::remove_cvref_t<T>>::value;

// Extract the pivot key from whatever a strategy returned: the value itself for
// a value_pivot, or proj(*it) for a position.  Always returns by value, so it
// is safe even after the referenced element is moved.
template <class R, class Proj = std::identity>
auto pivot_key_of(const R& r, Proj proj = {}) {
    if constexpr (is_value_pivot_v<R>)
        return r.key;
    else
        return std::invoke(proj, *r);
}

namespace detail {

// (a + b) / 2 for the supported key types, avoiding overflow for integrals.
template <class K>
K key_midpoint(const K& a, const K& b) {
    if constexpr (std::is_arithmetic_v<K>) {
        return std::midpoint(a, b);
    } else if constexpr (std::is_same_v<K, pair64>) {
        return pair64{std::midpoint(a.first, b.first),
                      std::midpoint(a.second, b.second)};
    } else {
        return a;  // no meaningful midpoint; fall back to an endpoint
    }
}


// Median of the values referenced by three iterators (returns the iterator to
// the median value).  `less(x, y)` compares the values at iterators x and y.
template <class I, class Less>
I median3(I a, I b, I c, Less less) {
    if (less(a, b)) {
        if (less(b, c)) return b;
        return less(a, c) ? c : a;
    }
    if (less(a, c)) return a;
    return less(b, c) ? c : b;
}

// Median value among the iterators in [lo, hi); chosen by sorting a copy of
// the iterators (cheap for the small groups we use) so the input is untouched.
template <class I, class Less>
I median_of_range(I* lo, I* hi, Less less) {
    std::sort(lo, hi, [&](I x, I y) { return less(x, y); });
    return lo[(hi - lo) / 2];
}

// In-place insertion sort of a small range by comp(proj(.), proj(.)).
template <class I, class Comp, class Proj>
void small_sort(I first, I last, Comp comp, Proj proj) {
    for (I i = first; i < last; ++i)
        for (I j = i; j > first &&
                      std::invoke(comp, std::invoke(proj, *j),
                                  std::invoke(proj, *(j - 1)));
             --j)
            std::iter_swap(j - 1, j);
}

// In-place quickselect with a median-of-medians-of-5 pivot (BFPRT).  Reorders
// [first, last) so that the returned iterator holds the element of sorted rank
// k, with everything before it <= and everything after >= it.  This is the
// element-moving core of a "typical median-of-medians" pivot routine.
template <class I, class Comp, class Proj>
I select_nth(I first, I last, std::iter_difference_t<I> k, Comp comp, Proj proj) {
    using D = std::iter_difference_t<I>;
    while (true) {
        const D n = last - first;
        if (n <= 1) return first;
        if (n <= 16) {
            small_sort(first, last, comp, proj);
            return first + k;
        }
        // Sort each group of five and gather the group medians at the front.
        I store = first;
        for (I g = first; g < last; g += 5) {
            I ge = (last - g >= 5) ? g + 5 : last;
            small_sort(g, ge, comp, proj);
            std::iter_swap(store, g + (ge - g) / 2);
            ++store;
        }
        // Pivot key = median of the gathered medians (recurse on the front).
        I mom = select_nth(first, store, (store - first) / 2, comp, proj);
        auto pivot_key = std::invoke(proj, *mom);  // stable copy before moving
        // Three-way (Dutch-flag) partition of [first, last) around pivot_key.
        I lt = first, i = first, gt = last;
        while (i < gt) {
            const auto& ck = std::invoke(proj, *i);
            if (std::invoke(comp, ck, pivot_key)) {
                std::iter_swap(lt, i);
                ++lt;
                ++i;
            } else if (std::invoke(comp, pivot_key, ck)) {
                --gt;
                std::iter_swap(i, gt);
            } else {
                ++i;
            }
        }
        // [first,lt) < pivot, [lt,gt) == pivot, [gt,last) > pivot.
        const D lt_off = lt - first;
        const D gt_off = gt - first;
        if (k < lt_off) {
            last = lt;
        } else if (k < gt_off) {
            return first + k;
        } else {
            first = gt;
            k -= gt_off;
        }
    }
}

}  // namespace detail

// Build the "less over iterators" comparator from comp + proj.
template <class Comp, class Proj>
auto make_less(Comp comp, Proj proj) {
    return [comp = std::move(comp), proj = std::move(proj)](auto x, auto y) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *x), std::invoke(proj, *y)));
    };
}

struct first_element {
    static constexpr const char* name = "first";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S, Comp = {}, Proj = {}) const {
        return first;
    }
};

struct middle_element {
    static constexpr const char* name = "middle";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp = {}, Proj = {}) const {
        return first + (last - first) / 2;
    }
};

struct last_element {
    static constexpr const char* name = "last";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp = {}, Proj = {}) const {
        return first + ((last - first) - 1);
    }
};

struct median_of_3 {
    static constexpr const char* name = "median_of_3";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 3) return first + n / 2;
        auto less = make_less(comp, proj);
        return detail::median3(first, first + n / 2, first + (n - 1), less);
    }
};

struct median_of_5 {
    static constexpr const char* name = "median_of_5";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 5) return median_of_3{}(first, last, comp, proj);
        auto less = make_less(comp, proj);
        I s[5] = {first, first + n / 4, first + n / 2, first + (3 * n) / 4,
                  first + (n - 1)};
        return detail::median_of_range(s, s + 5, less);
    }
};

// Tukey's ninther: median of three medians-of-three taken over nine evenly
// spaced elements.  A cheap, high-quality pivot for large blocks.
struct ninther {
    static constexpr const char* name = "ninther";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 9) return median_of_3{}(first, last, comp, proj);
        auto less = make_less(comp, proj);
        const auto e = n / 8;  // spacing
        I lo = detail::median3(first, first + e, first + 2 * e, less);
        I mid = detail::median3(first + n / 2 - e, first + n / 2, first + n / 2 + e, less);
        I hi = detail::median3(first + (n - 1) - 2 * e, first + (n - 1) - e,
                               first + (n - 1), less);
        return detail::median3(lo, mid, hi, less);
    }
};

// Median of five medians-of-five: the 5x5 analogue of the ninther.  Samples 25
// evenly-spaced elements, takes the median of each consecutive group of five,
// then the median of those five medians.  A constant-cost (O(1)) pseudo-median
// that samples more widely than the ninther (25 vs 9) -- a position-returning
// counterpart to the 15-sample value pivot `pseudo15`.
struct median_of_5_medians_of_5 {
    static constexpr const char* name = "median_of_5_medians_of_5";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 25) return median_of_5{}(first, last, comp, proj);
        auto less = make_less(comp, proj);
        const auto step = n - 1;
        I samp[25];
        for (int k = 0; k < 25; ++k)
            samp[k] = first + (static_cast<decltype(n)>(k) * step) / 24;
        I med[5];
        for (int g = 0; g < 5; ++g)
            med[g] = detail::median_of_range(samp + g * 5, samp + g * 5 + 5, less);
        return detail::median_of_range(med, med + 5, less);
    }
};

// Median of 5 medians-of-5 with CLUSTERED sampling: 8 elements at the start,
// 9 in the middle (centred on n/2), and 8 at the end (8+9+8 = 25).  These 25
// reads land in only 3 cache lines on i64 (1 per cluster) vs 25 separate
// cache lines for the evenly-spaced variant -- a huge cache-locality win on
// large blocks (n ~ 2^20+).  Groups of 5 are taken in array order, so two of
// the five groups straddle a cluster boundary.
struct median_of_5_medians_of_5_898 {
    static constexpr const char* name = "median_of_5_medians_of_5_898";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 25) return median_of_5{}(first, last, comp, proj);
        auto less = make_less(comp, proj);
        I samp[25];
        for (int k = 0; k < 8; ++k) samp[k]      = first + k;                // 0..7
        const I mid  = first + (n / 2 - 4);
        for (int k = 0; k < 9; ++k) samp[8 + k]  = mid + k;                  // 8..16
        const I tail = first + (n - 8);
        for (int k = 0; k < 8; ++k) samp[17 + k] = tail + k;                 // 17..24

        I med[5];
        for (int g = 0; g < 5; ++g)
            med[g] = detail::median_of_range(samp + g * 5, samp + g * 5 + 5, less);
        return detail::median_of_range(med, med + 5, less);
    }
};

// Median of five medians-of-three: samples 15 evenly-spaced elements as 5
// groups of 3, takes each group's median, then the median of those five
// medians.  Same sample count as `pseudo15`, but uses median-of-3 sub-trees
// instead of a full 15-input sorting network -- cheaper comparison count, at
// the cost of slightly weaker robustness vs adversarial inputs.
struct median_of_5_medians_of_3 {
    static constexpr const char* name = "median_of_5_medians_of_3";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 15) return median_of_5{}(first, last, comp, proj);
        auto less = make_less(comp, proj);
        const auto step = n - 1;
        I samp[15];
        for (int k = 0; k < 15; ++k)
            samp[k] = first + (static_cast<decltype(n)>(k) * step) / 14;
        I med[5];
        for (int g = 0; g < 5; ++g)
            med[g] = detail::median3(samp[g * 3], samp[g * 3 + 1],
                                     samp[g * 3 + 2], less);
        return detail::median_of_range(med, med + 5, less);
    }
};

// Median of three medians-of-five: samples 15 evenly-spaced elements as 3
// groups of 5, takes each group's median, then the median of those three
// medians.  Same sample count as `pseudo15` / `median_of_5_medians_of_3`; the
// 3-groups-of-5 shape is the other obvious factoring of 15 and is informative
// to compare against the 5-groups-of-3 shape.
struct median_of_3_medians_of_5 {
    static constexpr const char* name = "median_of_3_medians_of_5";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 15) return median_of_5{}(first, last, comp, proj);
        auto less = make_less(comp, proj);
        const auto step = n - 1;
        I samp[15];
        for (int k = 0; k < 15; ++k)
            samp[k] = first + (static_cast<decltype(n)>(k) * step) / 14;
        I med[3];
        for (int g = 0; g < 3; ++g)
            med[g] = detail::median_of_range(samp + g * 5, samp + g * 5 + 5, less);
        return detail::median3(med[0], med[1], med[2], less);
    }
};

// Median-of-medians (BFPRT pivot): split into groups of five, take each
// group's median, then the *exact* median of those medians.  This guarantees
// the pivot lies between the 30th and 70th percentile of the block.  (Taking a
// median-of-medians-of-medians recursion instead would be only a heuristic and
// can fall outside that band, so we select the true median of the one medians
// level.)  Operates on a temporary array of iterators, leaving input untouched.
struct median_of_medians_5 {
    static constexpr const char* name = "median_of_medians_5";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n <= 0) return first;
        auto less = make_less(comp, proj);

        std::vector<I> medians;
        medians.reserve(static_cast<std::size_t>(n) / 5 + 1);
        auto k = decltype(n){0};
        for (; k + 5 <= n; k += 5) {
            I g[5] = {first + k, first + k + 1, first + k + 2, first + k + 3,
                      first + k + 4};
            medians.push_back(detail::median_of_range(g, g + 5, less));
        }
        if (k < n) {  // leftover group of 1..4
            std::vector<I> g;
            for (; k < n; ++k) g.push_back(first + k);
            medians.push_back(detail::median_of_range(g.data(), g.data() + g.size(), less));
        }

        const auto mid = static_cast<std::ptrdiff_t>(medians.size() / 2);
        std::nth_element(medians.begin(), medians.begin() + mid, medians.end(),
                         [&](I x, I y) { return less(x, y); });
        return medians[static_cast<std::size_t>(mid)];
    }
};

// Reproducible random pivot.  Carries RNG state (declared mutable so the call
// operator can stay const like the others).
struct random_pivot {
    static constexpr const char* name = "random";
    mutable std::mt19937_64 rng{0x9E3779B97F4A7C15ull};

    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp = {}, Proj = {}) const {
        const auto n = last - first;
        if (n <= 1) return first;
        std::uniform_int_distribution<std::ptrdiff_t> dist(0, n - 1);
        return first + dist(rng);
    }
};

// ---- Strategies that return a VALUE (no position) -------------------------

// Pivot = (min + max) / 2 of the block.  A synthetic key that is typically
// NOT present in the block; returned as a value_pivot.  Well centred for
// uniform data, arbitrary otherwise.  O(n).
struct midpoint_min_max {
    static constexpr const char* name = "midpoint_min_max";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
        if (first == last) return value_pivot<K>{K{}};
        K lo = std::invoke(proj, *first), hi = lo;
        for (I it = first; it != last; ++it) {
            K k = std::invoke(proj, *it);
            if (comp(k, lo)) lo = k;
            if (comp(hi, k)) hi = k;
        }
        return value_pivot<K>{detail::key_midpoint(lo, hi)};
    }
};

// Pivot = (first + last) / 2.  O(1) synthetic key (the midpoint of the two end
// elements); also returned as a value, since it need not be an element.
struct midpoint_first_last {
    static constexpr const char* name = "midpoint_first_last";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I first, S last, Comp = {}, Proj proj = {}) const {
        using K = std::remove_cvref_t<decltype(std::invoke(proj, *first))>;
        const auto n = last - first;
        if (n <= 0) return value_pivot<K>{K{}};
        K a = std::invoke(proj, *first);
        K b = std::invoke(proj, *(first + (n - 1)));
        return value_pivot<K>{detail::key_midpoint(a, b)};
    }
};

// ---- Strategies that REORDER the block ------------------------------------

// Median-of-3 that places the median at the middle position (and orders the
// three sampled elements), like the median-of-3 used inside many quicksorts.
// Returns the middle iterator, which now holds the median of {first,mid,last}.
struct median_of_3_inplace {
    static constexpr const char* name = "median_of_3_inplace";
    static constexpr bool reorders = true;
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n < 3) return first + n / 2;
        auto less = make_less(comp, proj);
        I lo = first, mid = first + n / 2, hi = first + (n - 1);
        if (less(mid, lo)) std::iter_swap(mid, lo);
        if (less(hi, lo)) std::iter_swap(hi, lo);
        if (less(hi, mid)) std::iter_swap(hi, mid);
        return mid;  // lo <= mid <= hi
    }
};

// Median-of-medians-of-5 implemented in place via quickselect (BFPRT).  This
// is the "typical" element-moving implementation: it sorts groups of five,
// gathers their medians, and selects the exact median of the block, leaving the
// block partly arranged.  Returns the iterator to the median element.
struct median_of_medians_5_inplace {
    static constexpr const char* name = "median_of_medians_5_inplace";
    static constexpr bool reorders = true;
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    I operator()(I first, S last, Comp comp = {}, Proj proj = {}) const {
        const auto n = last - first;
        if (n <= 0) return first;
        return detail::select_nth(first, first + n, n / 2, comp, proj);
    }
};


// Pseudomedian of 15 evenly-spaced samples via a custom selection network
// with 27 unique compare-and-swaps (13 produce both outputs, 14 produce
// only one) and critical depth 9 -- much shallower and tighter than any
// AxMED catalog variant with the same +/-1 error tolerance (s_15_5 is
// depth 14, 41 CAS).
//
// Codegen: each comparator pair is materialised as `bool b = comp(a,b);`
// followed by one or two ternary assignments.  gcc/clang at -O3 lower this
// to cmp + cmov (single output) or cmp + 2x cmov (both outputs).  The
// pattern guarantees a single comparison per pair -- which a naive
// std::min(a,b) + std::max(a,b) pair does not, because the compiler can't
// always prove the comparator is pure across templated calls.
//
// Variable naming: m_X_Y = min(vX,vY), M_X_Y = max(vX,vY); intermediates
// (v21, v25, ...) follow the original numeric labels of the source network.
struct pseudo15 {
    static constexpr const char* name = "pseudo15";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 15;
        auto v0  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v9  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v10 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v11 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v12 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v13 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v14 = std::invoke(proj, *begin);

        // Operations are interleaved (not strictly layered) so that L1 outputs
        // are consumed by L2/L3 as early as possible.  This shrinks peak live
        // set from ~15 (strict-layered) to ~10-11, which lets gcc avoid two of
        // the callee-saved register saves we'd otherwise pay for.

        // Block A: build v21, v25, M_21 keepalives, kill four L1 outputs fast.
        bool c0 = comp(v1,  v11); auto m_1_11  = c0 ? v1  : v11; auto M_1_11  = c0 ? v11 : v1;
        bool c1 = comp(v13, v5);  auto m_13_5  = c1 ? v13 : v5;  auto M_13_5  = c1 ? v5  : v13;
        auto v21 = comp(m_1_11, m_13_5) ? m_13_5 : m_1_11;     // m_1_11, m_13_5 dead
        auto v25 = comp(M_13_5, M_1_11) ? M_13_5 : M_1_11;     // M_1_11, M_13_5 dead

        // Block B: same shape for the (v4,v10),(v12,v6) group.
        bool c2 = comp(v4,  v10); auto m_4_10  = c2 ? v4  : v10; auto M_4_10  = c2 ? v10 : v4;
        bool c3 = comp(v12, v6);  auto m_12_6  = c3 ? v12 : v6;  auto M_12_6  = c3 ? v6  : v12;
        auto v28 = comp(m_4_10, m_12_6) ? m_12_6 : m_4_10;     // m_4_10, m_12_6 dead
        auto v29 = comp(M_4_10, M_12_6) ? M_4_10 : M_12_6;     // M_4_10, M_12_6 dead

        // Block C: collapse v21,v25,v28,v29 immediately into the L3 (BOTH) pairs.
        bool c11 = comp(v21, v28); auto m_21_28 = c11 ? v21 : v28; auto M_21_28 = c11 ? v28 : v21;
        bool c12 = comp(v25, v29); auto m_25_29 = c12 ? v25 : v29; auto M_25_29 = c12 ? v29 : v25;

        // Block D: remaining L1 pairs.
        bool c4 = comp(v14, v2);  auto m_14_2  = c4 ? v14 : v2;  auto v38     = c4 ? v2  : v14;
        bool c5 = comp(v7,  v3);  auto m_7_3   = c5 ? v7  : v3;  auto M_7_3   = c5 ? v3  : v7;
        bool c6 = comp(v8,  v0);  auto m_8_0   = c6 ? v8  : v0;  auto M_8_0   = c6 ? v0  : v8;
        bool c7 = comp(v9, v38);  auto m_9_38  = c7 ? v9  : v38; auto M_9_38  = c7 ? v38 : v9;  // v38 dead

        // Block E: inner sub-DAGs for v74/v66, collapsing four more outputs.
        auto x_inner_max = comp(m_9_38, m_8_0) ? m_8_0  : m_9_38;   // m_9_38, m_8_0 dead
        auto x_inner_min = comp(M_8_0,  M_9_38) ? M_8_0  : M_9_38;  // M_8_0, M_9_38 dead

        // Block F: assemble v50, v51, v66, v74.
        auto v50 = comp(m_21_28, m_7_3)  ? m_7_3  : m_21_28;        // m_21_28, m_7_3 dead
        auto v51 = comp(m_25_29, m_14_2) ? m_14_2 : m_25_29;        // m_25_29, m_14_2 dead
        auto v66 = comp(M_25_29, x_inner_min) ? M_25_29 : x_inner_min;  // M_25_29, x_inner_min dead
        auto v74 = comp(x_inner_max, M_7_3) ? x_inner_max : M_7_3;  // x_inner_max, M_7_3 dead

        // L5: (v50,v51) and (v74,v61) pairs.
        bool c14 = comp(v50, v51); auto v62 = c14 ? v50 : v51; auto v61 = c14 ? v51 : v50;
        bool c15 = comp(v74, v61); auto m_74_61 = c15 ? v74 : v61; auto M_74_61 = c15 ? v61 : v74;

        // L6: v86 and v102.
        auto v86  = comp(m_74_61, v62)     ? v62      : m_74_61;    // max(m_74_61, v62)
        auto v102 = comp(M_74_61, M_21_28) ? M_74_61  : M_21_28;    // min

        // L7: (v66,v86) pair.
        bool c17 = comp(v66, v86); auto v103 = c17 ? v66 : v86; auto v104 = c17 ? v86 : v66;

        // L8 & final.
        auto v112 = comp(v104, v102) ? v104 : v102;                 // min
        auto out  = comp(v103, v112) ? v112 : v103;                 // max -- the pseudomedian

        using K = std::remove_cvref_t<decltype(out)>;
        return value_pivot<K>{out};
    }
};

// Pseudomedian of 15 via the AxMED s_15_5 selection network (41 CAS, depth 14,
// d_L=d_H=1, Q(M)=0.009).  Same n/15 stride as pseudo15.  All CASes are
// emitted using the proven bool+ternary pattern -- gcc/clang at -O3 lower
// them to cmp + cmov (single output) or cmp + cmov + cmov (both outputs)
// with no branches.  Source: https://github.com/ehw-fit/approximate-medians
// netlist/s_15_5.cha.
struct pseudo15_s5 {
    static constexpr const char* name = "pseudo15_s5";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 15;
        auto v0  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v9  = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v10 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v11 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v12 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v13 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v14 = std::invoke(proj, *begin);
        // L1
        bool b0  = comp(v13, v11); auto v15 = b0  ? v13 : v11; auto v16 = b0  ? v11 : v13;
        bool b1  = comp(v7, v9);   auto v17 = b1  ? v9  : v7;  auto v18 = b1  ? v7  : v9;
        bool b2  = comp(v5, v14);  auto v19 = b2  ? v14 : v5;  auto v20 = b2  ? v5  : v14;
        bool b4  = comp(v0, v12);  auto v23 = b4  ? v0  : v12; auto v24 = b4  ? v12 : v0;
        bool b5  = comp(v6, v8);   auto v25 = b5  ? v8  : v6;  auto v26 = b5  ? v6  : v8;
        bool b7  = comp(v3, v4);   auto v29 = b7  ? v4  : v3;  auto v30 = b7  ? v3  : v4;
        bool b11 = comp(v10, v2);  auto v37 = b11 ? v2  : v10; auto v38 = b11 ? v10 : v2;
        // L2
        bool b3  = comp(v17, v1);  auto v21 = b3  ? v17 : v1;  auto v22 = b3  ? v1  : v17;
        bool b8  = comp(v26, v30); auto v31 = b8  ? v26 : v30; auto v32 = b8  ? v30 : v26;
        bool b9  = comp(v18, v15); auto v33 = b9  ? v15 : v18; auto v34 = b9  ? v18 : v15;
        bool b12 = comp(v16, v19); auto v39 = b12 ? v19 : v16; auto v40 = b12 ? v16 : v19;
        bool b14 = comp(v20, v23); auto v43 = b14 ? v23 : v20; auto v44 = b14 ? v20 : v23;
        // L3
        bool b13 = comp(v37, v33); auto v41 = b13 ? v37 : v33; auto v42 = b13 ? v33 : v37;
        auto v46 = comp(v34, v31) ? v31 : v34;
        bool b16 = comp(v21, v32); auto v47 = b16 ? v32 : v21; auto v48 = b16 ? v21 : v32;
        bool b23 = comp(v39, v24); auto v61 = b23 ? v24 : v39; auto v62 = b23 ? v39 : v24;
        // L4
        bool b19 = comp(v47, v40); auto v53 = b19 ? v40 : v47; auto v54 = b19 ? v47 : v40;
        bool b20 = comp(v38, v41); auto v55 = b20 ? v41 : v38; auto v56 = b20 ? v38 : v41;
        bool b24 = comp(v62, v25); auto v63 = b24 ? v25 : v62; auto v64 = b24 ? v62 : v25;
        bool b30 = comp(v43, v46); auto v75 = b30 ? v46 : v43; auto v76 = b30 ? v43 : v46;
        // L5
        bool b21 = comp(v54, v55); auto v57 = b21 ? v54 : v55; auto v58 = b21 ? v55 : v54;
        auto v69 = comp(v61, v63) ? v61 : v63;
        auto v71 = comp(v53, v42) ? v53 : v42;
        bool b33 = comp(v56, v75); auto v81 = b33 ? v75 : v56; auto v82 = b33 ? v56 : v75;
        auto v98 = comp(v76, v48) ? v48 : v76;
        // L6
        auto v78 = comp(v44, v57) ? v57 : v44;
        bool b32 = comp(v29, v58); auto v79 = b32 ? v29 : v58; auto v80 = b32 ? v58 : v29;
        auto v103 = comp(v98, v82) ? v82 : v98;
        // L7
        auto v83 = comp(v22, v80) ? v22 : v80;
        bool b40 = comp(v79, v81); auto v95 = b40 ? v79 : v81; auto v96 = b40 ? v81 : v79;
        bool b47 = comp(v103, v78); auto v109 = b47 ? v78 : v103; auto v110 = b47 ? v103 : v78;
        // L8
        bool b36 = comp(v71, v83); auto v87 = b36 ? v83 : v71; auto v88 = b36 ? v71 : v83;
        auto v102 = comp(v69, v96) ? v69 : v96;
        // L9
        bool b37 = comp(v88, v64); auto v89 = b37 ? v64 : v88; auto v90 = b37 ? v88 : v64;
        // L10
        auto v91 = comp(v87, v89) ? v87 : v89;
        auto v100 = comp(v95, v90) ? v90 : v95;
        // L11
        auto v106 = comp(v102, v91) ? v102 : v91;
        // L12
        bool b46 = comp(v106, v100); auto v107 = b46 ? v100 : v106; auto v108 = b46 ? v106 : v100;
        // L13
        auto v112 = comp(v107, v109) ? v107 : v109;
        auto v113 = comp(v108, v110) ? v110 : v108;
        // L14
        auto v115 = comp(v112, v113) ? v113 : v112;
        using K = std::remove_cvref_t<decltype(v115)>;
        return value_pivot<K>{v115};
    }
};

// Pseudomedian of 9 evenly-spaced samples via the s_9_6 selection network
// (Knuth Vol 3).  After dead-code elimination of the 5 unused wires in the
// original 19-CAS netlist, the network reduces to:
//
//   * 14 compare-and-swaps  (6 produce BOTH outputs; 8 produce only ONE)
//   * critical path depth   = 6  (vs sorting_network_9's depth 7)
//   * sample positions      = same n/9 stride as sorting_network_9
//
// A pseudomedian may differ from the true median by at most one sorted rank.
//
// Codegen strategy: each CAS is encoded as `bool b = comp(a,b);` followed by
// a single ternary (when only one output is live) or by two ternaries sharing
// the bool (when both outputs are live).  gcc/clang at -O3 lower these to
// `cmp + cmov` and `cmp + cmov + cmov` respectively, with no branches.  Dead
// CASes are not written at all, so register pressure stays low and the
// compiler does not need to spill callee-saved registers.
struct pseudo9 {
    static constexpr const char* name = "pseudo9";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 9;
        auto v0 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8 = std::invoke(proj, *begin);

        // Layer 1 (4 CAS, every output is consumed downstream -> both kept).
        bool b0 = comp(v6, v1);  auto v9  = b0 ? v6 : v1;  auto v10 = b0 ? v1 : v6;
        bool b1 = comp(v2, v0);  auto v11 = b1 ? v0 : v2;  auto v12 = b1 ? v2 : v0;
        bool b2 = comp(v5, v8);  auto v13 = b2 ? v8 : v5;  auto v14 = b2 ? v5 : v8;
        bool b3 = comp(v7, v3);  auto v15 = b3 ? v3 : v7;  auto v16 = b3 ? v7 : v3;

        // Layer 2 (3 CAS keep one output, 1 CAS keeps both).
        auto v17 = comp(v11, v13) ? v11 : v13;          // CAS 4: min only
        auto v19 = comp(v14, v12) ? v12 : v14;          // CAS 5: max only
        bool b6 = comp(v10, v16);
        auto v21 = b6 ? v16 : v10;                       // CAS 6: max
        auto v22 = b6 ? v10 : v16;                       // CAS 6: min
        auto v24 = comp(v9,  v4)  ? v4  : v9;            // CAS 7: max only

        // Layer 3 (3 CAS, single output each).
        auto v28 = comp(v15, v24) ? v15 : v24;          // CAS 9:  min only
        auto v32 = comp(v22, v19) ? v19 : v22;          // CAS 11: max only
        auto v37 = comp(v17, v21) ? v17 : v21;          // CAS 14: min only

        // Layer 4 (1 CAS, both outputs).
        bool b16 = comp(v32, v37);
        auto v41 = b16 ? v32 : v37;                      // CAS 16: min
        auto v42 = b16 ? v37 : v32;                      // CAS 16: max

        // Layer 5 (1 CAS, max only).
        auto v43 = comp(v41, v28) ? v28 : v41;          // CAS 17: max only

        // Layer 6 (1 CAS, the pseudomedian).
        auto v45 = comp(v42, v43) ? v42 : v43;          // CAS 18: min

        using K = std::remove_cvref_t<decltype(v45)>;
        return value_pivot<K>{v45};
    }
};

// ---------------------------------------------------------------------------
// AxMED catalogue: pseudo9_s1 ... pseudo9_s5
// ---------------------------------------------------------------------------
//
// Approximate-median networks for 9 inputs from Mrazek & Vasicek, "AxMED:
// Formal Analysis and Automated Design of Approximate Median Filters using
// BDDs" (ISCAS 2025); netlists from https://github.com/ehw-fit/approximate-
// medians .  Catalog parameters for the 9-input variants with at most one
// rank of error (d_L = d_H <= 1):
//
//   name        catalog  d_L,d_H  Q(M)    ops  delay   notes
//   pseudo9_s1  s_9_1    0,0      0.000   19   8       exact median of 9
//   pseudo9_s2  s_9_2    1,1      0.015   18   8
//   pseudo9_s3  s_9_3    1,1      0.055   17   8
//   pseudo9_s4  s_9_4    1,1      0.103   16   8
//   pseudo9_s5  s_9_5    1,1      0.126   15   7
//   pseudo9     s_9_6    1,1      0.285   14   6       <- current default
//
// Q(M) is roughly the fraction of inputs producing a non-exact median.
// All networks sample at the same n/9 stride and are generated from their
// .cha netlists by dead-code-eliminating wires that do not reach the output
// and scheduling each compare-and-swap into its earliest layer.  Each CAS is
// emitted as `bool b = comp(a,b);` plus one or two ternary assignments (gcc
// /clang at -O3 lower the pattern to cmp + cmov, with no branches).
//
// Generated by /tmp/gen_pseudo9.py (one-shot helper, not committed) - see
// the chat log for the parser.

// pseudo9_s1: AxMED s_9_1 (19 CAS, depth 8).  This is the EXACT median of 9.
struct pseudo9_s1 {
    static constexpr const char* name = "pseudo9_s1";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 9;
        auto v0 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8 = std::invoke(proj, *begin);
        // L1
        bool b0 = comp(v5, v0);  auto v9  = b0 ? v5 : v0;  auto v10 = b0 ? v0 : v5;
        bool b1 = comp(v2, v1);  auto v11 = b1 ? v1 : v2;  auto v12 = b1 ? v2 : v1;
        bool b2 = comp(v7, v4);  auto v13 = b2 ? v7 : v4;  auto v14 = b2 ? v4 : v7;
        // L2
        bool b3 = comp(v6, v13); auto v15 = b3 ? v13: v6;  auto v16 = b3 ? v6 : v13;
        bool b4 = comp(v3, v12); auto v17 = b4 ? v12: v3;  auto v18 = b4 ? v3 : v12;
        bool b5 = comp(v9, v8);  auto v19 = b5 ? v9 : v8;  auto v20 = b5 ? v8 : v9;
        // L3
        bool b6 = comp(v10, v20); auto v21 = b6 ? v10: v20; auto v22 = b6 ? v20: v10;
        bool b7 = comp(v17, v11); auto v23 = b7 ? v17: v11; auto v24 = b7 ? v11: v17;
        bool b8 = comp(v15, v14); auto v25 = b8 ? v14: v15; auto v26 = b8 ? v15: v14;
        auto v27 = comp(v19, v16) ? v16 : v19;
        // L4
        auto v29 = comp(v22, v24) ? v22 : v24;
        bool b11 = comp(v26, v23); auto v31 = b11? v26: v23; auto v32 = b11? v23: v26;
        auto v33 = comp(v18, v27) ? v27 : v18;
        // L5
        auto v35 = comp(v31, v21) ? v21 : v31;
        auto v38 = comp(v25, v29) ? v25 : v29;
        // L6
        auto v40 = comp(v32, v35) ? v32 : v35;
        bool b16 = comp(v33, v38); auto v41 = b16? v38: v33; auto v42 = b16? v33: v38;
        // L7
        auto v44 = comp(v40, v41) ? v40 : v41;
        // L8
        auto v45 = comp(v44, v42) ? v42 : v44;
        using K = std::remove_cvref_t<decltype(v45)>;
        return value_pivot<K>{v45};
    }
};

// pseudo9_s2: AxMED s_9_2 (18 CAS, depth 8). Q(M)=0.015, d_L=d_H=1.
struct pseudo9_s2 {
    static constexpr const char* name = "pseudo9_s2";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 9;
        auto v0 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8 = std::invoke(proj, *begin);
        // L1
        bool b0 = comp(v4, v5);  auto v9  = b0 ? v5 : v4;  auto v10 = b0 ? v4 : v5;
        bool b1 = comp(v8, v0);  auto v11 = b1 ? v0 : v8;  auto v12 = b1 ? v8 : v0;
        bool b3 = comp(v1, v3);  auto v15 = b3 ? v1 : v3;  auto v16 = b3 ? v3 : v1;
        // L2
        bool b2 = comp(v12, v7); auto v13 = b2 ? v7 : v12; auto v14 = b2 ? v12: v7;
        bool b5 = comp(v11, v2); auto v19 = b5 ? v2 : v11; auto v20 = b5 ? v11: v2;
        // L3
        auto v18 = comp(v10, v14) ? v14 : v10;
        auto v24 = comp(v19, v16) ? v19 : v16;
        bool b9  = comp(v15, v20); auto v27 = b9 ? v15: v20; auto v28 = b9 ? v20: v15;
        bool b10 = comp(v9, v13);  auto v29 = b10? v13: v9;  auto v30 = b10? v9 : v13;
        // L4
        bool b6  = comp(v6, v18);  auto v21 = b6 ? v18: v6;  auto v22 = b6 ? v6 : v18;
        bool b11 = comp(v28, v30); auto v31 = b11? v30: v28; auto v32 = b11? v28: v30;
        auto v34 = comp(v24, v29) ? v24 : v29;
        // L5
        auto v35 = comp(v21, v31) ? v21 : v31;
        auto v37 = comp(v27, v32) ? v32 : v27;
        // L6
        bool b15 = comp(v34, v35); auto v39 = b15? v34: v35; auto v40 = b15? v35: v34;
        auto v42 = comp(v37, v22) ? v22 : v37;
        // L7
        auto v44 = comp(v39, v42) ? v42 : v39;
        // L8
        auto v45 = comp(v44, v40) ? v44 : v40;
        using K = std::remove_cvref_t<decltype(v45)>;
        return value_pivot<K>{v45};
    }
};

// pseudo9_s3: AxMED s_9_3 (17 CAS, depth 8). Q(M)=0.055, d_L=d_H=1.
struct pseudo9_s3 {
    static constexpr const char* name = "pseudo9_s3";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 9;
        auto v0 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8 = std::invoke(proj, *begin);
        // L1
        bool b0 = comp(v3, v7);  auto v9  = b0 ? v3 : v7;  auto v10 = b0 ? v7 : v3;
        bool b1 = comp(v1, v8);  auto v11 = b1 ? v1 : v8;  auto v12 = b1 ? v8 : v1;
        bool b3 = comp(v0, v4);  auto v15 = b3 ? v0 : v4;  auto v16 = b3 ? v4 : v0;
        // L2
        bool b2 = comp(v9, v6);  auto v13 = b2 ? v9 : v6;  auto v14 = b2 ? v6 : v9;
        auto v18 = comp(v16, v10) ? v16 : v10;
        bool b5 = comp(v2, v15); auto v19 = b5 ? v2 : v15; auto v20 = b5 ? v15: v2;
        // L3
        bool b7  = comp(v12, v14); auto v23 = b7 ? v14: v12; auto v24 = b7 ? v12: v14;
        auto v25 = comp(v13, v11) ? v11 : v13;
        bool b10 = comp(v5, v18);  auto v29 = b10? v5 : v18; auto v30 = b10? v18: v5;
        // L4
        bool b11 = comp(v24, v20); auto v31 = b11? v24: v20; auto v32 = b11? v20: v24;
        auto v34 = comp(v23, v30) ? v23 : v30;
        auto v38 = comp(v19, v25) ? v25 : v19;
        // L5
        auto v36 = comp(v31, v29) ? v29 : v31;
        auto v40 = comp(v34, v32) ? v34 : v32;
        // L6
        bool b16 = comp(v40, v36); auto v41 = b16? v40: v36; auto v42 = b16? v36: v40;
        // L7
        auto v43 = comp(v42, v38) ? v42 : v38;
        // L8
        auto v45 = comp(v43, v41) ? v41 : v43;
        using K = std::remove_cvref_t<decltype(v45)>;
        return value_pivot<K>{v45};
    }
};

// pseudo9_s4: AxMED s_9_4 (16 CAS, depth 8). Q(M)=0.103, d_L=d_H=1.
struct pseudo9_s4 {
    static constexpr const char* name = "pseudo9_s4";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 9;
        auto v0 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8 = std::invoke(proj, *begin);
        // L1
        bool b0 = comp(v0, v5);  auto v9  = b0 ? v5 : v0;  auto v10 = b0 ? v0 : v5;
        bool b1 = comp(v3, v4);  auto v11 = b1 ? v3 : v4;  auto v12 = b1 ? v4 : v3;
        bool b2 = comp(v6, v1);  auto v13 = b2 ? v6 : v1;  auto v14 = b2 ? v1 : v6;
        bool b4 = comp(v8, v7);  auto v17 = b4 ? v7 : v8;  auto v18 = b4 ? v8 : v7;
        // L2
        bool b3 = comp(v2, v9);  auto v15 = b3 ? v9 : v2;  auto v16 = b3 ? v2 : v9;
        auto v24 = comp(v10, v13) ? v13 : v10;
        auto v25 = comp(v17, v14) ? v17 : v14;
        // L3
        bool b5 = comp(v16, v18); auto v19 = b5 ? v18: v16; auto v20 = b5 ? v16: v18;
        auto v31 = comp(v25, v15) ? v25 : v15;
        // L4
        bool b9 = comp(v24, v19); auto v27 = b9 ? v24: v19; auto v28 = b9 ? v19: v24;
        auto v34 = comp(v20, v11) ? v11 : v20;
        // L5
        auto v38 = comp(v27, v34) ? v34 : v27;
        auto v39 = comp(v28, v12) ? v28 : v12;
        // L6
        bool b16 = comp(v38, v39); auto v41 = b16? v38: v39; auto v42 = b16? v39: v38;
        // L7
        auto v44 = comp(v31, v41) ? v41 : v31;
        // L8
        auto v46 = comp(v42, v44) ? v42 : v44;
        using K = std::remove_cvref_t<decltype(v46)>;
        return value_pivot<K>{v46};
    }
};

// pseudo9_s5: AxMED s_9_5 (15 CAS, depth 7). Q(M)=0.126, d_L=d_H=1.
struct pseudo9_s5 {
    static constexpr const char* name = "pseudo9_s5";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 9;
        auto v0 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v1 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v2 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v3 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v4 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v5 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v6 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v7 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto v8 = std::invoke(proj, *begin);
        // L1
        bool b0 = comp(v4, v3);  auto v9  = b0 ? v4 : v3;  auto v10 = b0 ? v3 : v4;
        bool b1 = comp(v6, v5);  auto v11 = b1 ? v5 : v6;  auto v12 = b1 ? v6 : v5;
        bool b2 = comp(v8, v2);  auto v13 = b2 ? v8 : v2;  auto v14 = b2 ? v2 : v8;
        bool b3 = comp(v7, v0);  auto v15 = b3 ? v7 : v0;  auto v16 = b3 ? v0 : v7;
        // L2
        auto v17 = comp(v15, v13) ? v13 : v15;
        auto v22 = comp(v11, v10) ? v11 : v10;
        auto v26 = comp(v9, v12)  ? v12 : v9;
        auto v28 = comp(v14, v16) ? v14 : v16;
        // L3
        bool b5  = comp(v17, v1);  auto v19 = b5 ? v17: v1;  auto v20 = b5 ? v1 : v17;
        bool b11 = comp(v26, v22); auto v31 = b11? v26: v22; auto v32 = b11? v22: v26;
        // L4
        auto v35 = comp(v32, v20) ? v32 : v20;
        auto v40 = comp(v31, v19) ? v19 : v31;
        // L5
        bool b16 = comp(v28, v35); auto v41 = b16? v35: v28; auto v42 = b16? v28: v35;
        // L6
        auto v44 = comp(v40, v41) ? v40 : v41;
        // L7
        auto v46 = comp(v42, v44) ? v44 : v42;
        using K = std::remove_cvref_t<decltype(v46)>;
        return value_pivot<K>{v46};
    }
};

// Exact median of 9 evenly-spaced samples via the size-and-depth-optimal
// 9-input sorting network (25 comparators, 7 layers; Floyd 1964 / Codish-
// Cruz-Filipe-Frank-Schneider-Kamp 2016 size-optimality, Parberry 1989
// depth-optimality).  Returns wire 4, which after a full sort holds the
// median.
//
// Branch prediction: every comparator is written as `bool lt = comp(a,b);
// lo = lt?a:b; hi = lt?b:a;` -- the canonical pattern gcc/clang lower to
// CMP + two CMOVs, so the network has *no* data-dependent branches.  Cost
// on the critical path is the network depth (7 layers of cmp+cmov ≈ 7
// dependent cycles on modern OoO cores); the remaining 18 comparators
// overlap via ILP within their respective layers.
//
// Source: Bert Dobbelaere's optimal sorting network catalogue
// (https://bertdobbelaere.github.io/sorting_networks.html), entry "9
// inputs, size 25, depth 7".  Uses the same n/9 stride as `pseudo9`.
struct sorting_network_9 {
    static constexpr const char* name = "sorting_network_9";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin, S end, Comp comp = {}, Proj proj = {}) const {
        auto stride = std::ranges::distance(begin, end) / 9;

        auto w0 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w1 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w2 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w3 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w4 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w5 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w6 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w7 = std::invoke(proj, *begin); std::advance(begin, stride);
        auto w8 = std::invoke(proj, *begin);

        // Branchless compare-and-swap on two wires.  Lowered to cmp + 2 cmov.
        auto cas = [&comp](auto& a, auto& b) {
            bool lt = comp(a, b);
            auto lo = lt ? a : b;
            auto hi = lt ? b : a;
            a = lo;
            b = hi;
        };

        // Layer 1 (4 CAS, independent)
        cas(w0, w3); cas(w1, w7); cas(w2, w5); cas(w4, w8);
        // Layer 2 (4 CAS, independent)
        cas(w0, w7); cas(w2, w4); cas(w3, w8); cas(w5, w6);
        // Layer 3 (4 CAS, independent)
        cas(w0, w2); cas(w1, w3); cas(w4, w5); cas(w7, w8);
        // Layer 4 (3 CAS, independent)
        cas(w1, w4); cas(w3, w6); cas(w5, w7);
        // Layer 5 (4 CAS, independent)
        cas(w0, w1); cas(w2, w4); cas(w3, w5); cas(w6, w8);
        // Layer 6 (3 CAS, independent)
        cas(w2, w3); cas(w4, w5); cas(w6, w7);
        // Layer 7 (3 CAS, independent)
        cas(w1, w2); cas(w3, w4); cas(w5, w6);

        using K = std::remove_cvref_t<decltype(w4)>;
        return value_pivot<K>{w4};
    }
};
/*
template <typename T, typename Compare = std::less<T>, typename Projection = std::identity>
class s_9_6_network {
private:
    Compare comp;
    Projection proj;

    // Force inline for all comparison-swap operations
    template <typename U>
    [[gnu::always_inline]] inline void cas(U& a, U& b) noexcept {
        // Use branchless comparison when possible to avoid branch misprediction
        if (comp(proj(b), proj(a))) {
            std::swap(a, b);
        }
    }

    // Branchless conditional swap for primitive types
    template <typename U>
    [[gnu::always_inline]] inline typename std::enable_if<std::is_arithmetic<U>::value>::type
    cas_branchless(U& a, U& b) noexcept {
        bool condition = comp(proj(b), proj(a));
        U temp = a;
        a = condition ? b : a;
        b = condition ? temp : b;
    }

public:
    explicit s_9_6_network(Compare c = Compare(), Projection p = Projection())
        : comp(std::move(c)), proj(std::move(p)) {}

    // Main sorting function - optimized for cache locality
    void operator()(T* arr) noexcept {
        // Stage 1: Initial comparisons (depth 1)
        cas(arr[0], arr[1]);
        cas(arr[2], arr[3]);
        cas(arr[4], arr[5]);
        cas(arr[6], arr[7]);

        // Stage 2: Cross comparisons (depth 2)
        cas(arr[0], arr[2]);
        cas(arr[1], arr[3]);
        cas(arr[4], arr[6]);
        cas(arr[5], arr[7]);

        // Stage 3: Deeper cross comparisons (depth 3)
        cas(arr[0], arr[4]);
        cas(arr[1], arr[5]);
        cas(arr[2], arr[6]);
        cas(arr[3], arr[7]);

        // Stage 4: Final ordering (depth 4)
        cas(arr[1], arr[2]);
        cas(arr[3], arr[4]);
        cas(arr[5], arr[6]);

        // Stage 5: Last adjustments (depth 5)
        cas(arr[2], arr[4]);
        cas(arr[3], arr[5]);

        // Stage 6: Final pass (depth 6)
        cas(arr[1], arr[2]);
        cas(arr[3], arr[4]);
        cas(arr[5], arr[6]);

        // Handle the 9th element (arr[8]) - insert into sorted array
        insert_ninth(arr);
    }

private:
    // Optimized insertion of the 9th element into the sorted 8-element array
    [[gnu::noinline]] void insert_ninth(T* arr) noexcept {
        // Binary search insertion point using projection
        auto val_proj = proj(arr[8]);

        // Use linear search for small arrays (better branch prediction)
        // Start from the end since the 9th element is likely to be larger
        if (!comp(val_proj, proj(arr[7]))) {
            // Element is largest, already in position
            return;
        }

        if (comp(val_proj, proj(arr[0]))) {
            // Element is smallest, rotate right
            T temp = std::move(arr[8]);
            for (int i = 8; i > 0; --i) {
                arr[i] = std::move(arr[i-1]);
            }
            arr[0] = std::move(temp);
            return;
        }

        // Find insertion point with linear search from the right
        int pos = 7;
        while (pos > 0 && comp(val_proj, proj(arr[pos]))) {
            --pos;
        }
        ++pos; // Insert after the smaller element

        // Rotate elements to make space
        T temp = std::move(arr[8]);
        for (int i = 8; i > pos; --i) {
            arr[i] = std::move(arr[i-1]);
        }
        arr[pos] = std::move(temp);
    }

public:
    // Convenience function for array
    void sort(T (&arr)[9]) noexcept {
        operator()(arr);
    }

    // STL-compatible interface
    template <typename Iterator>
    void sort(Iterator first, Iterator last) noexcept {
        static_assert(std::is_same<typename std::iterator_traits<Iterator>::iterator_category,
                                  std::random_access_iterator_tag>::value,
                     "Requires random access iterator");

        T temp[9];
        std::move(first, last, temp);
        operator()(temp);
        std::move(temp, temp + 9, first);
    }
};

// Convenience function for automatic type deduction
template <typename T, typename Compare = std::less<T>, typename Projection = std::identity>
inline void s_9_6_sort(T (&arr)[9], Compare comp = Compare(), Projection proj = Projection()) noexcept {
    s_9_6_network<T, Compare, Projection> sorter(comp, proj);
    sorter(arr);
}
*/
}  // namespace partitions::pivot

#endif  // PARTITIONS_PIVOT_HPP
