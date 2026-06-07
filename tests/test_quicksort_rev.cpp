// Correctness of the reversed (descending) pure-partition quicksort: the result
// must be DESCENDING-sorted by (comp, proj) and a permutation of the input
// (multiset preserved).  Mirror of test_quicksort.cpp; exercises the edge sizes
// around the halver cutoff (24) and the large-node m5m5 pivot (>65536).

#include <functional>
#include <string>
#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"
#include "partitions/quicksort_rev.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,  1,  2,  3,   4,    8,    15,   16,
                                  17, 24, 25, 33,  100,  300,  1000, 3000, 4096,
                                  70000};

template <class T>
void run_matrix(const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xC0FFEEu + n);
            auto v = base;
            quicksort_rev(v.begin(), v.end(), comp, proj);
            const std::string ctx = tname + "/" + d.name + "/n=" + std::to_string(n);
            // Descending: no element may be strictly LESS than... i.e. each
            // element must be <= its predecessor, so v[i] > v[i-1] is a failure.
            bool ok = true;
            for (std::size_t i = 1; i < v.size(); ++i)
                if (comp(std::invoke(proj, v[i - 1]), std::invoke(proj, v[i]))) ok = false;
            CHECK_MESSAGE(ok, ctx + " descending-sorted");
            CHECK_MESSAGE(ptv::same_multiset(v, base), ctx + " multiset preserved");
        }
    });
}

}  // namespace

TEST_CASE("quicksort_rev: i32") { run_matrix<i32>("i32"); }
TEST_CASE("quicksort_rev: i64") { run_matrix<i64>("i64"); }
TEST_CASE("quicksort_rev: pair64") { run_matrix<pair64>("pair64"); }
TEST_CASE("quicksort_rev: keyed (non-identity projection)") { run_matrix<keyed>("keyed"); }
