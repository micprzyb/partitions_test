#ifndef PARTITIONS_PIVOT_HPP
#define PARTITIONS_PIVOT_HPP

// Pivot-selection strategies.
//
// A strategy chooses one element of [first, last) to act as the pivot and
// returns an iterator to it:
//
//     I operator()(I first, S last, Comp comp, Proj proj) const;
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
#include <random>
#include <vector>

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

namespace detail {

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

}  // namespace partitions::pivot

#endif  // PARTITIONS_PIVOT_HPP
