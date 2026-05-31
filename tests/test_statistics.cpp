// Balance-statistics tests.

#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"

using namespace partitions;

TEST_CASE("measure_balance counts smaller/equal/greater correctly") {
    std::vector<i64> v = {0, 1, 2, 3, 4, 5, 5, 5};  // pivot 5
    auto b = stat::measure_balance(v.begin(), v.end(), i64{5}, std::less<>{}, std::identity{});
    CHECK(b.n == 8);
    CHECK(b.smaller == 5);  // 0..4
    CHECK(b.equal == 3);    // three 5s
    CHECK(b.greater == 0);
    CHECK(b.smaller_fraction() == 5.0 / 8.0);
}

TEST_CASE("balance on sorted data matches pivot position") {
    auto v = dist::sorted_ascending{}.operator()<i64>(1000, 0);
    // first element -> 0% smaller, last -> ~100%, middle -> ~50%.
    auto bf = stat::measure_balance(v.begin(), v.end(), v.front(), std::less<>{}, std::identity{});
    auto bl = stat::measure_balance(v.begin(), v.end(), v.back(), std::less<>{}, std::identity{});
    auto bm = stat::measure_balance(v.begin(), v.end(), v[500], std::less<>{}, std::identity{});
    CHECK(bf.smaller_fraction() == 0.0);
    CHECK(bl.smaller_fraction() == 999.0 / 1000.0);
    CHECK(bm.smaller == 500);
}

TEST_CASE("all_equal: zero smaller, all equal, worst possible balance") {
    auto v = dist::all_equal{}.operator()<i64>(1000, 0);
    auto b = stat::measure_balance(v.begin(), v.end(), v.front(), std::less<>{}, std::identity{});
    CHECK(b.smaller == 0);
    CHECK(b.equal == 1000);
    CHECK(b.imbalance() == 0.5);  // maximally unbalanced
}

TEST_CASE("summary aggregates min/max/mean over trials") {
    std::vector<stat::balance> samples;
    samples.push_back({10, 2, 0, 8});   // f = 0.2
    samples.push_back({10, 5, 0, 5});   // f = 0.5
    samples.push_back({10, 8, 0, 2});   // f = 0.8
    auto s = stat::summary::from(samples);
    CHECK(s.trials == 3);
    CHECK(s.min_fraction == 0.2);
    CHECK(s.max_fraction == 0.8);
    CHECK(std::abs(s.mean_fraction - 0.5) < 1e-12);
    CHECK(std::abs(s.worst_side - 0.8) < 1e-12);
}
