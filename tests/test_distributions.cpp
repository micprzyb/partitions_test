// Distribution-generator tests: correct sizes, reproducibility, and the
// defining properties of the named adversarial inputs.

#include <algorithm>
#include <set>
#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"

using namespace partitions;

TEST_CASE("every distribution produces the requested size and is reproducible") {
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : {0u, 1u, 4u, 23u, 1000u}) {
            auto a = d.template operator()<i64>(n, 99);
            auto b = d.template operator()<i64>(n, 99);
            CHECK_MESSAGE(a.size() == n, std::string(d.name) + " size");
            CHECK_MESSAGE(a == b, std::string(d.name) + " reproducible");
        }
    });
}

TEST_CASE("median_of_3_killer matches Musser's reference sequence (n=20)") {
    auto v = dist::median_of_3_killer{}.operator()<i64>(20, 0);
    std::vector<i64> expected = {1, 11, 3, 13, 5, 15, 7, 17, 9, 19,
                                 2, 4,  6, 8,  10, 12, 14, 16, 18, 20};
    CHECK(v == expected);
    // It is a permutation of 1..20 (distinct, no duplicates).
    std::set<i64> s(v.begin(), v.end());
    CHECK(s.size() == 20);
    CHECK(*s.begin() == 1 && *s.rbegin() == 20);
}

TEST_CASE("all_equal contains a single distinct value") {
    auto v = dist::all_equal{}.operator()<i64>(500, 0);
    CHECK(std::set<i64>(v.begin(), v.end()).size() == 1);
}

TEST_CASE("binary contains at most two distinct values") {
    auto v = dist::binary{}.operator()<i64>(500, 3);
    CHECK(std::set<i64>(v.begin(), v.end()).size() <= 2);
}

TEST_CASE("sorted_ascending / sorted_descending are monotonic") {
    auto a = dist::sorted_ascending{}.operator()<i64>(500, 0);
    auto d = dist::sorted_descending{}.operator()<i64>(500, 0);
    CHECK(std::is_sorted(a.begin(), a.end()));
    CHECK(std::is_sorted(d.rbegin(), d.rend()));
}

TEST_CASE("few_unique really has few distinct values") {
    auto v = dist::few_unique{}.operator()<i64>(10000, 5);
    CHECK(std::set<i64>(v.begin(), v.end()).size() < 200);  // ~sqrt(n)=100
}
