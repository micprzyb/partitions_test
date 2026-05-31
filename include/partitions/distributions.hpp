#ifndef PARTITIONS_DISTRIBUTIONS_HPP
#define PARTITIONS_DISTRIBUTIONS_HPP

// Input distributions -- the element arrangements a partition (and the pivot
// selection feeding it) must cope with.  Several are deliberately adversarial;
// see docs/difficult-cases.md for the rationale and references.
//
// Each generator produces a sequence of integer "ranks" and materialises them
// into the requested element type via make_value<T>, so a distribution is
// written once and reused for i32 / i64 / pair64 / keyed.
//
//   operator()<T>(n, seed) -> std::vector<T>
//
// Generators are reproducible: the same (n, seed) yields the same sequence.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "types.hpp"

namespace partitions::dist {

namespace detail {
template <class T>
std::vector<T> to_elements(const std::vector<i64>& ranks) {
    std::vector<T> out;
    out.reserve(ranks.size());
    for (i64 r : ranks) out.push_back(make_value<T>(r));
    return out;
}
}  // namespace detail

// Uniformly random ranks in [0, n): the baseline "easy" case.
struct random_uniform {
    static constexpr const char* name = "random_uniform";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t seed) const {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<i64> d(0, static_cast<i64>(n));
        std::vector<i64> r(n);
        for (auto& x : r) x = d(rng);
        return detail::to_elements<T>(r);
    }
};

// Many duplicates: only ~sqrt(n) distinct values.  Stresses how a scheme and a
// pivot cope with equal keys (all equal-to-pivot keys land on the >= side).
struct few_unique {
    static constexpr const char* name = "few_unique";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t seed) const {
        std::mt19937_64 rng(seed);
        i64 distinct = static_cast<i64>(1 + std::llround(std::sqrt(static_cast<double>(n))));
        std::uniform_int_distribution<i64> d(0, distinct - 1);
        std::vector<i64> r(n);
        for (auto& x : r) x = d(rng);
        return detail::to_elements<T>(r);
    }
};

// Every element equal.  Lomuto's worst case (swaps every element); a good
// scheme must still split this near the middle, and every pivot is "balanced 0%
// smaller" -- the canonical reminder that a 2-way < / >= split cannot balance
// equal keys.
struct all_equal {
    static constexpr const char* name = "all_equal";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        return std::vector<T>(n, make_value<T>(42));
    }
};

// Only two distinct values.
struct binary {
    static constexpr const char* name = "binary";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t seed) const {
        std::mt19937_64 rng(seed);
        std::bernoulli_distribution d(0.5);
        std::vector<i64> r(n);
        for (auto& x : r) x = d(rng) ? 1 : 0;
        return detail::to_elements<T>(r);
    }
};

struct sorted_ascending {
    static constexpr const char* name = "sorted_ascending";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        std::vector<i64> r(n);
        std::iota(r.begin(), r.end(), i64{0});
        return detail::to_elements<T>(r);
    }
};

struct sorted_descending {
    static constexpr const char* name = "sorted_descending";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        std::vector<i64> r(n);
        for (std::size_t i = 0; i < n; ++i) r[i] = static_cast<i64>(n - i);
        return detail::to_elements<T>(r);
    }
};

// Sorted with ~n/20 random adjacent-ish swaps: defeats naive "already sorted"
// fast paths without being fully random.
struct nearly_sorted {
    static constexpr const char* name = "nearly_sorted";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t seed) const {
        std::vector<i64> r(n);
        std::iota(r.begin(), r.end(), i64{0});
        if (n > 1) {
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<std::size_t> d(0, n - 1);
            for (std::size_t k = 0; k < n / 20 + 1; ++k)
                std::swap(r[d(rng)], r[d(rng)]);
        }
        return detail::to_elements<T>(r);
    }
};

// Ascending then descending (a "mountain").  Classic median-of-3 trap: the
// first/middle/last samples are 0, max, 0, so the chosen pivot is extreme.
struct organ_pipe {
    static constexpr const char* name = "organ_pipe";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        std::vector<i64> r(n);
        for (std::size_t i = 0; i < n; ++i)
            r[i] = static_cast<i64>(i < n / 2 ? i : n - i);
        return detail::to_elements<T>(r);
    }
};

// Descending then ascending (a "valley").
struct reverse_organ_pipe {
    static constexpr const char* name = "reverse_organ_pipe";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        std::vector<i64> r(n);
        for (std::size_t i = 0; i < n; ++i)
            r[i] = static_cast<i64>(i < n / 2 ? n / 2 - i : i - n / 2);
        return detail::to_elements<T>(r);
    }
};

// Repeating ramp of fixed period: many runs, periodic structure.
struct sawtooth {
    static constexpr const char* name = "sawtooth";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        const i64 period = 16;
        std::vector<i64> r(n);
        for (std::size_t i = 0; i < n; ++i) r[i] = static_cast<i64>(i) % period;
        return detail::to_elements<T>(r);
    }
};

// Musser's median-of-three killer permutation.  Constructed so that a
// median-of-3 pivot is always near-extreme, driving median-of-3 quicksort to
// Theta(n^2).  Defined for even n; odd n appends the maximum.
// Reference: D. Musser, "Introspective Sorting and Selection Algorithms" (1997).
struct median_of_3_killer {
    static constexpr const char* name = "median_of_3_killer";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        std::vector<i64> r(n);
        const std::size_t even_n = n - (n & 1u);
        const std::size_t m = even_n / 2;
        for (std::size_t i = 0; i < m; ++i) {
            if (i % 2 == 0)
                r[i] = static_cast<i64>(i + 1);
            else
                r[i] = static_cast<i64>(m + i + (m % 2 != 0 ? 1 : 0));
        }
        for (std::size_t i = m; i < even_n; ++i)
            r[i] = static_cast<i64>((i - m + 1) * 2);
        if (n & 1u) r[n - 1] = static_cast<i64>(n);  // pad odd n with the max
        return detail::to_elements<T>(r);
    }
};

// Sorted ascending except for a single maximal element at the front: probes
// fast paths and the "one element out of place" boundary case.
struct single_outlier {
    static constexpr const char* name = "single_outlier";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t) const {
        std::vector<i64> r(n);
        std::iota(r.begin(), r.end(), i64{0});
        if (n > 0) r[0] = static_cast<i64>(n) + 1000;
        return detail::to_elements<T>(r);
    }
};

// Sorted runs of length ~sqrt(n) whose order among runs is shuffled.
struct shuffled_blocks {
    static constexpr const char* name = "shuffled_blocks";
    template <class T>
    std::vector<T> operator()(std::size_t n, std::uint64_t seed) const {
        std::vector<i64> r(n);
        std::iota(r.begin(), r.end(), i64{0});
        std::size_t blk = static_cast<std::size_t>(1 + std::llround(std::sqrt(static_cast<double>(n))));
        std::mt19937_64 rng(seed);
        std::vector<std::size_t> starts;
        for (std::size_t s = 0; s < n; s += blk) starts.push_back(s);
        std::shuffle(starts.begin(), starts.end(), rng);
        std::vector<i64> out;
        out.reserve(n);
        for (std::size_t s : starts)
            for (std::size_t i = s; i < std::min(s + blk, n); ++i) out.push_back(r[i]);
        return detail::to_elements<T>(out);
    }
};

}  // namespace partitions::dist

#endif  // PARTITIONS_DISTRIBUTIONS_HPP
