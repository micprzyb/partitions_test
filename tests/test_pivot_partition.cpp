// Tests for the (pivot strategy -> partition) pipeline, with emphasis on
//   (1) strategies that REORDER the block (median_of_3_inplace,
//       median_of_medians_5_inplace), and
//   (2) the exact "middle" return contract, including pivot keys absent from
//       the block.
//
// For every strategy x algorithm x type x distribution x size we run the
// canonical flow -- select (possibly reordering), read the pivot key, then
// partition by that position -- and check the postcondition, multiset
// preservation (which catches a reordering strategy that loses an element), and
// that the returned middle index equals the count of strictly-smaller elements.

#include <string>
#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {1, 2, 3, 4, 5, 8, 16, 17, 24, 64, 257, 1000};

template <class V, class Key, class Comp, class Proj>
std::ptrdiff_t count_less(const V& base, const Key& key, Comp comp, Proj proj) {
    std::ptrdiff_t c = 0;
    for (const auto& e : base)
        if (comp(std::invoke(proj, e), key)) ++c;
    return c;
}

template <class T>
void run_pipeline(const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();

    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xBEEFu + n);

            for_each(default_pivots(), [&](auto pv) {
                for_each(default_partitioners(), [&](auto alg) {
                    const std::string ctx = tname + "/" + d.name + "/" + pv.name +
                                            "/" + alg.name + "/n=" + std::to_string(n);

                    // ---- forward: select (maybe reorder) then partition ----
                    {
                        auto v = base;
                        auto pos = pv(v.begin(), v.end(), comp, proj);
                        REQUIRE(pos >= v.begin() && pos < v.end());
                        auto key = std::invoke(proj, *pos);
                        auto m = partition_by_position(alg, v.begin(), v.end(), pos, comp, proj);
                        CHECK_MESSAGE(ptv::forward_ok(v.begin(), m, v.end(), key, comp, proj),
                                      ctx + " fwd postcondition");
                        CHECK_MESSAGE(ptv::same_multiset(v, base), ctx + " fwd multiset");
                        // exact "middle" contract: m - first == #{x : x < pivot}
                        CHECK_MESSAGE((m - v.begin()) == count_less(base, key, comp, proj),
                                      ctx + " fwd middle index");
                    }
                    // ---- reverse: select (maybe reorder) then partition ----
                    {
                        auto v = base;
                        auto pos = pv(v.begin(), v.end(), comp, proj);
                        auto key = std::invoke(proj, *pos);
                        auto m = reverse_partition_by_position(alg, v.begin(), v.end(), pos, comp, proj);
                        CHECK_MESSAGE(ptv::reverse_ok(v.begin(), m, v.end(), key, comp, proj),
                                      ctx + " rev postcondition");
                        CHECK_MESSAGE(ptv::same_multiset(v, base), ctx + " rev multiset");
                        // reverse middle: m - first == #{x : x >= pivot} == n - #{< pivot}
                        const std::ptrdiff_t ge =
                            static_cast<std::ptrdiff_t>(n) - count_less(base, key, comp, proj);
                        CHECK_MESSAGE((m - v.begin()) == ge, ctx + " rev middle index");
                    }
                });
            });
        }
    });
}

// The "middle" contract under pivot keys that are ABSENT from the block:
// below-min must give an empty left (m == first), above-max an empty right
// (m == last), and in both the index must equal the strictly-smaller count.
template <class T>
void run_absent_key_middle(const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    using Key = std::decay_t<decltype(std::invoke(proj, std::declval<T>()))>;

    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : {std::size_t{1}, std::size_t{24}, std::size_t{500}}) {
            const auto base = d.template operator()<T>(n, 1234u + n);
            Key lo = std::invoke(proj, base[0]), hi = lo;
            for (const auto& e : base) {
                Key k = std::invoke(proj, e);
                if (comp(k, lo)) lo = k;
                if (comp(hi, k)) hi = k;
            }
            // Keys strictly below the min and strictly above the max -> absent.
            Key below = lo, above = hi;
            if constexpr (std::is_integral_v<Key>) {
                below = static_cast<Key>(lo - 1);
                above = static_cast<Key>(hi + 1);
            } else {  // pair64
                below = pair64{lo.first - 1, 0};
                above = pair64{hi.first + 1, 0};
            }

            for_each(default_partitioners(), [&](auto alg) {
                const std::string ctx = tname + "/" + d.name + "/" + alg.name +
                                        "/n=" + std::to_string(n);
                {  // below-min: everything is >= pivot, left side empty
                    auto v = base;
                    auto m = partition_by_key(alg, v.begin(), v.end(), below, comp, proj);
                    CHECK_MESSAGE(m == v.begin(), ctx + " absent-low middle==first");
                    CHECK_MESSAGE(ptv::forward_ok(v.begin(), m, v.end(), below, comp, proj),
                                  ctx + " absent-low postcondition");
                }
                {  // above-max: everything is < pivot, right side empty
                    auto v = base;
                    auto m = partition_by_key(alg, v.begin(), v.end(), above, comp, proj);
                    CHECK_MESSAGE(m == v.end(), ctx + " absent-high middle==last");
                    CHECK_MESSAGE(ptv::forward_ok(v.begin(), m, v.end(), above, comp, proj),
                                  ctx + " absent-high postcondition");
                }
            });
        }
    });
}

}  // namespace

TEST_CASE("pivot->partition pipeline (incl. reordering strategies): i32") {
    run_pipeline<i32>("i32");
}
TEST_CASE("pivot->partition pipeline (incl. reordering strategies): i64") {
    run_pipeline<i64>("i64");
}
TEST_CASE("pivot->partition pipeline (incl. reordering strategies): pair64") {
    run_pipeline<pair64>("pair64");
}
TEST_CASE("pivot->partition pipeline (incl. reordering strategies): keyed") {
    run_pipeline<keyed>("keyed");
}

TEST_CASE("middle contract holds for absent pivot keys") {
    run_absent_key_middle<i32>("i32");
    run_absent_key_middle<i64>("i64");
    run_absent_key_middle<pair64>("pair64");
    run_absent_key_middle<keyed>("keyed");
}
