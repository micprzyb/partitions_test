// Correctness of the left-to-right, non-recursive, selection-leaf quicksort
// (partitions::quicksort_lr) across types, sizes and distributions: the result
// must be sorted by (comp, proj) and a permutation of the input (multiset
// preserved).  Sizes straddle the selection-sort threshold and the m5m5 huge-
// node pivot cutoff; several thresholds (incl. degenerate small ones that force
// the selection leaf / deep left spines) are exercised to stress every path.

#include <string>
#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"
#include "partitions/quicksort_lr.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,  1,  2,   3,    4,    8,    15,   16,
                                  17, 24, 25,  33,   100,  300,  1000, 3000,
                                  4096, 70000};  // >65536 exercises the m5m5 path

template <std::ptrdiff_t T, class Type, class Comp, class Proj>
void check_one(const std::vector<Type>& base, Comp comp, Proj proj,
               const std::string& ctx) {
    auto v = base;
    quicksort_lr<T>(v.begin(), v.end(), comp, proj);
    bool ok = true;
    for (std::size_t i = 1; i < v.size(); ++i)
        if (comp(std::invoke(proj, v[i]), std::invoke(proj, v[i - 1]))) ok = false;
    CHECK_MESSAGE(ok, ctx + " sorted");
    CHECK_MESSAGE(ptv::same_multiset(v, base), ctx + " multiset preserved");
}

template <class T>
void run_matrix(const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xC0FFEEu + n);
            const std::string ctx = tname + "/" + d.name + "/n=" + std::to_string(n);
            // Default threshold, plus a tiny one (deep selection leaf / spine)
            // and a large one (selection sort doing most of the work).
            check_one<16>(base, comp, proj, ctx + "/T16");
            check_one<4>(base, comp, proj, ctx + "/T4");
            check_one<64>(base, comp, proj, ctx + "/T64");
        }
    });
}

}  // namespace

TEST_CASE("quicksort_lr: i32") { run_matrix<i32>("i32"); }
TEST_CASE("quicksort_lr: i64") { run_matrix<i64>("i64"); }
TEST_CASE("quicksort_lr: pair64") { run_matrix<pair64>("pair64"); }
TEST_CASE("quicksort_lr: keyed (non-identity projection)") {
    run_matrix<keyed>("keyed");
}
