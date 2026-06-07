// Correctness of the pure-partition adaptive quicksort across types, sizes and
// distributions: the result must be sorted by (comp, proj) and a permutation of
// the input (multiset preserved).  Exercises the edge sizes around the halver
// cutoff (16), the ninther cutoff is large so only the >2048 sizes hit it.

#include <string>
#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"
#include "partitions/quicksort.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,  1,  2,  3,   4,    8,    15,   16,
                                  17, 24, 25, 33,  100,  300,  1000, 3000, 4096,
                                  70000};  // >65536 exercises the m5m5 large-node pivot

template <class T>
void run_matrix(const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xC0FFEEu + n);
            auto v = base;
            quicksort(v.begin(), v.end(), comp, proj);
            const std::string ctx = tname + "/" + d.name + "/n=" + std::to_string(n);
            bool ok = true;
            for (std::size_t i = 1; i < v.size(); ++i)
                if (comp(std::invoke(proj, v[i]), std::invoke(proj, v[i - 1]))) ok = false;
            CHECK_MESSAGE(ok, ctx + " sorted");
            CHECK_MESSAGE(ptv::same_multiset(v, base), ctx + " multiset preserved");
        }
    });
}

}  // namespace

TEST_CASE("quicksort: i32") { run_matrix<i32>("i32"); }
TEST_CASE("quicksort: i64") { run_matrix<i64>("i64"); }
TEST_CASE("quicksort: pair64") { run_matrix<pair64>("pair64"); }
TEST_CASE("quicksort: keyed (non-identity projection)") { run_matrix<keyed>("keyed"); }
