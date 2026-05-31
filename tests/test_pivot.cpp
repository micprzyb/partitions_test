// Pivot-strategy tests: validity (in-range, no out-of-bounds on tiny blocks),
// correctness of median-of-3, and the median-of-medians percentile guarantee.

#include <string>

#include "framework.hpp"
#include "partitions/partitions.hpp"

using namespace partitions;

namespace {

template <class T>
void check_in_range() {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 16u, 24u, 1000u}) {
            auto v = d.template operator()<T>(n, 7u + n);
            for_each(default_pivots(), [&](auto pv) {
                auto it = pv(v.begin(), v.end(), comp, proj);
                CHECK_MESSAGE(it >= v.begin() && it < v.end(),
                              std::string(pv.name) + " out of range, n=" +
                                  std::to_string(n));
            });
        }
    });
}

}  // namespace

TEST_CASE("pivot strategies stay in range (all types, incl. tiny blocks)") {
    check_in_range<i32>();
    check_in_range<i64>();
    check_in_range<pair64>();
    check_in_range<keyed>();
}

TEST_CASE("median_of_3 returns the true median of its three samples") {
    auto comp = std::less<>{};
    auto proj = std::identity{};
    // n=9 so samples are positions 0, 4, 8.
    std::vector<i64> v = {5, 0, 0, 0, 9, 0, 0, 0, 7};
    auto it = pivot::median_of_3{}(v.begin(), v.end(), comp, proj);
    CHECK(*it == 7);  // median of {5, 9, 7}
}

TEST_CASE("median_of_medians_5 lies within the 30-70 percentile band") {
    // On distinct, uniformly random data the BFPRT pivot is guaranteed to sit
    // between the 30th and 70th percentile.  We use a generous [0.2, 0.8] band.
    auto comp = std::less<>{};
    auto proj = std::identity{};
    dist::sorted_ascending gen;  // distinct keys 0..n-1
    for (std::size_t n : {25u, 125u, 1000u, 5000u}) {
        auto v = gen.operator()<i64>(n, 1);
        auto it = pivot::median_of_medians_5{}(v.begin(), v.end(), comp, proj);
        auto b = stat::measure_balance(v.begin(), v.end(), *it, comp, proj);
        const double f = b.smaller_fraction();
        CHECK_MESSAGE(f > 0.2 && f < 0.8,
                      "mom5 fraction=" + std::to_string(f) +
                          " n=" + std::to_string(n));
    }
}

TEST_CASE("random pivot is reproducible for a fixed seed") {
    auto comp = std::less<>{};
    auto proj = std::identity{};
    dist::sorted_ascending gen;
    auto v = gen.operator()<i64>(1000, 1);
    pivot::random_pivot a{}, b{};
    // Same seed -> same first few draws.
    for (int i = 0; i < 5; ++i) {
        auto ia = a(v.begin(), v.end(), comp, proj);
        auto ib = b(v.begin(), v.end(), comp, proj);
        CHECK(ia == ib);
    }
}
