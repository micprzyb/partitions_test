#ifndef PARTITIONS_STATISTICS_HPP
#define PARTITIONS_STATISTICS_HPP

// Pivot-quality statistics.
//
// The quality of a pivot for a 2-way (< / >=) partition is captured by the
// fraction of elements strictly smaller than the pivot key -- this is exactly
// where the partition point lands, so it is the size of the left side relative
// to the whole.  0.5 is perfectly balanced; near 0 or near 1 is a degenerate
// split that drives a quicksort built on it toward O(n^2).
//
// Equal keys are reported separately: in a < / >= split they all fall on the
// right, so a block dominated by the pivot value is unavoidably unbalanced no
// matter how the pivot is chosen.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <vector>

namespace partitions::stat {

struct balance {
    std::size_t n = 0;
    std::size_t smaller = 0;  // keys strictly < pivot
    std::size_t equal = 0;    // keys == pivot (neither < nor >)
    std::size_t greater = 0;  // keys strictly > pivot

    // Fraction of the block that ends up on the left of a forward partition.
    double smaller_fraction() const {
        return n ? static_cast<double>(smaller) / static_cast<double>(n) : 0.0;
    }
    // Distance from a perfectly balanced split (0 = ideal, 0.5 = worst).
    double imbalance() const { return std::abs(smaller_fraction() - 0.5); }
};

// Count how the block splits around `pivot_key` under comp/proj.
template <std::input_iterator I, std::sentinel_for<I> S, class Key,
          class Comp = std::less<>, class Proj = std::identity>
balance measure_balance(I first, S last, const Key& pivot_key, Comp comp = {},
                        Proj proj = {}) {
    balance b;
    for (; first != last; ++first) {
        ++b.n;
        const auto& key = std::invoke(proj, *first);
        if (std::invoke(comp, key, pivot_key))
            ++b.smaller;
        else if (std::invoke(comp, pivot_key, key))
            ++b.greater;
        else
            ++b.equal;
    }
    return b;
}

// Aggregate of the smaller-fraction across many trials of a pivot strategy.
struct summary {
    std::size_t trials = 0;
    double min_fraction = 1.0;
    double max_fraction = 0.0;
    double mean_fraction = 0.0;
    double stddev_fraction = 0.0;
    double mean_equal_fraction = 0.0;
    // Worst-case left-side share = max(left, right) over trials; the quantity a
    // recursive sort actually pays for.  Closer to 0.5 is better.
    double worst_side = 0.0;

    static summary from(const std::vector<balance>& samples) {
        summary s;
        s.trials = samples.size();
        if (samples.empty()) return s;
        double sum = 0.0, sumsq = 0.0, eq = 0.0;
        for (const auto& b : samples) {
            const double f = b.smaller_fraction();
            s.min_fraction = std::min(s.min_fraction, f);
            s.max_fraction = std::max(s.max_fraction, f);
            s.worst_side = std::max(s.worst_side, std::max(f, 1.0 - f));
            sum += f;
            sumsq += f * f;
            eq += b.n ? static_cast<double>(b.equal) / static_cast<double>(b.n) : 0.0;
        }
        const double t = static_cast<double>(s.trials);
        s.mean_fraction = sum / t;
        s.mean_equal_fraction = eq / t;
        s.stddev_fraction = std::sqrt(std::max(0.0, sumsq / t - s.mean_fraction * s.mean_fraction));
        return s;
    }
};

}  // namespace partitions::stat

#endif  // PARTITIONS_STATISTICS_HPP
