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


struct pseudo15 {
    static constexpr const char* name = "pseudo15";
    template <class I, class S, class Comp = std::less<>, class Proj = std::identity>
    auto operator()(I begin,  S end, Comp comp = {}, Proj proj = {}) const
{
        auto stride = std::ranges::distance(begin, end)/15;
    auto v0  = std::invoke(proj, *begin);
        std::advance(begin, stride);
    auto v1  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v2  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v3  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v4  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v5  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v6  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v7  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v8  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v9  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v10  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v11  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v12  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v13  = std::invoke(proj, *begin);
        std::advance(begin, stride);
        auto v14  = std::invoke(proj, *begin);


    auto v21 = std::max(std::min(v1, v11, comp), std::min(v13, v5, comp), comp);
    auto v28 = std::max(std::min(v4, v10, comp), std::min(v12, v6, comp), comp);

    auto v25 = std::min(std::max(v13, v5, comp), std::max(v1, v11, comp), comp);
    auto v29 = std::min(std::max(v4, v10, comp), std::max(v12, v6, comp), comp);

    auto v38 = std::max(v14, v2, comp);

    // v50 and v51 are branch poTs (both outputs of the min/max pair are live later)
    auto v50 = std::max(std::min(v21, v28, comp), std::min(v7, v3, comp), comp);
    auto v51 = std::max(std::min(v25, v29, comp), std::min(v14, v2, comp), comp);

    auto v61 = std::max(v50, v51, comp);
    auto v62 = std::min(v50, v51, comp);

    // v74 is a heavy branch poT (used in two independent min/max trees)
    auto v74 = std::min(std::max(std::min(v9, v38, comp), std::min(v8, v0, comp), comp), std::max(v7, v3, comp), comp);

    // v66 is a branch poT (both min and max of this sub-DAG are live)
    auto v66 = std::min(std::max(v25, v29, comp), std::min(std::max(v8, v0, comp), std::max(v9, v38, comp), comp), comp);

    // v86 is a branch poT (both outputs live in the final stage)
    auto v86 = std::max(std::min(v74, v61, comp), v62, comp);

    // Final stage - only the required min/max are kept
    auto v102 = std::min(std::max(v74, v61, comp), std::max(v21, v28, comp), comp);

    auto v103 = std::min(v66, v86, comp);
    auto v104 = std::max(v66, v86, comp);
    auto v112 = std::min(v104, v102, comp);

        using K = std::remove_cvref_t<decltype(std::invoke(proj, *begin))>;
    return value_pivot<K> {std::max(v103, v112, comp)};
}
};

}  // namespace partitions::pivot

#endif  // PARTITIONS_PIVOT_HPP
