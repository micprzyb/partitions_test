// Correctness of the REVERSED partition family (algo_rev::*): every form must
// produce [begin, m) >= pivot and [m, end) strictly < pivot (the ptv::reverse_ok
// contract) and preserve the multiset, across types, sizes and distributions.
//
// Also cross-checks that the dedicated reversed partitioners agree (boundary +
// multiset) with the generic reversal route reverse_partition_by_key(forward,
// negate_comp) -- i.e. the rewrite is semantically the same transform, just with
// the predicate hard-wired.

#include <functional>
#include <string>
#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"
#include "partitions/reverse_partition.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,  1,  2,  3,   4,    7,   8,   15,  16,
                                  17, 24, 25, 33,  64,   100, 256, 300, 1000,
                                  4096};

// A median-of-3 pivot key, read once before partitioning (value form).
template <class T, class Comp, class Proj>
auto pick_pivot_key(const std::vector<T>& v, Comp comp, Proj proj) {
    std::vector<T> tmp = v;
    auto it = pivot::median_of_3{}(tmp.begin(), tmp.end(), comp, proj);
    return std::invoke(proj, *it);
}

template <class Alg, class T>
void run_alg(const char* aname, const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    Alg alg{};
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xC0FFEEu + n);
            const std::string ctx =
                tname + "/" + aname + "/" + d.name + "/n=" + std::to_string(n);
            if (n == 0) {
                auto v = base;
                auto m = alg(v.begin(), v.end(), std::invoke(proj, T{}), comp, proj);
                CHECK_MESSAGE(m == v.begin(), ctx + " empty boundary");
                continue;
            }
            const auto key = pick_pivot_key(base, comp, proj);

            auto v = base;
            auto m = alg(v.begin(), v.end(), key, comp, proj);
            CHECK_MESSAGE(ptv::reverse_ok(v.begin(), m, v.end(), key, comp, proj),
                          ctx + " reverse_ok");
            CHECK_MESSAGE(ptv::same_multiset(v, base), ctx + " multiset");

            // Cross-check boundary vs the generic negate_comp reversal route.
            auto v2 = base;
            auto m2 = reverse_partition_by_key(algo::sized{}, v2.begin(),
                                               v2.end(), key, comp, proj);
            CHECK_MESSAGE((m - v.begin()) == (m2 - v2.begin()),
                          ctx + " boundary matches negate_comp route");
        }
    });
}

template <class T>
void run_type(const std::string& tname) {
    run_alg<algo_rev::hoare_rev, T>("hoare_rev", tname);
    run_alg<algo_rev::lomuto_branchless_rev, T>("lomuto_branchless_rev", tname);
    run_alg<algo_rev::boost_block_rev, T>("boost_block_rev", tname);
    run_alg<algo_rev::sized_rev, T>("sized_rev", tname);
}

}  // namespace

TEST_CASE("reverse_partition: i32") { run_type<i32>("i32"); }
TEST_CASE("reverse_partition: i64") { run_type<i64>("i64"); }
TEST_CASE("reverse_partition: pair64") { run_type<pair64>("pair64"); }
TEST_CASE("reverse_partition: keyed") { run_type<keyed>("keyed"); }
