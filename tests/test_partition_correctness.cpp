// Exhaustive correctness sweep:
//   {all algorithms} x {i32,i64,pair64,keyed} x {all distributions}
//   x {forward,reverse} x {pivot-by-key, pivot-by-position}
//   x {present pivot, absent-low pivot, absent-high pivot} x {many sizes}
//
// Each case checks both the partition postcondition and that the multiset of
// elements is preserved (nothing lost/duplicated).  Sizes span the small
// blocks of interest (0..24) and large ones that exercise the block scheme's
// 2*B element threshold.

#include <string>
#include <vector>

#include "framework.hpp"
#include "partitions/partitions.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,  1,  2,  3,  4,   5,    7,    8,
                                  15, 16, 17, 23, 24,  100,  300,  1000,
                                  4096};

template <class Key>
std::pair<Key, Key> outside_keys(Key lo, Key hi) {
    if constexpr (std::is_integral_v<Key>) {
        return {static_cast<Key>(lo - 1), static_cast<Key>(hi + 1)};
    } else {  // pair64
        return {pair64{lo.first - 1, 0}, pair64{hi.first + 1, 0}};
    }
}

template <class T>
void run_matrix(const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    auto algos = default_partitioners();

    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xC0FFEEu + n);

            // Build pivot keys: present (middle), plus absent low/high keys.
            using Key = std::decay_t<decltype(std::invoke(proj, base.front()))>;
            std::vector<std::pair<std::string, Key>> keys;
            if (n > 0) {
                Key lo = std::invoke(proj, base[0]);
                Key hi = lo;
                for (const auto& e : base) {
                    Key k = std::invoke(proj, e);
                    if (comp(k, lo)) lo = k;
                    if (comp(hi, k)) hi = k;
                }
                keys.emplace_back("present", std::invoke(proj, base[n / 2]));
                auto [low_absent, high_absent] = outside_keys<Key>(lo, hi);
                keys.emplace_back("absent_low", low_absent);
                keys.emplace_back("absent_high", high_absent);
            } else {
                keys.emplace_back("present", Key{});  // empty range, any key
            }

            for_each(algos, [&](auto alg) {
                const std::string ctx = tname + "/" + d.name + "/" + alg.name +
                                        "/n=" + std::to_string(n);

                for (auto& [kname, key] : keys) {
                    // forward by key
                    {
                        auto v = base;
                        auto p = partition_by_key(alg, v.begin(), v.end(), key, comp, proj);
                        CHECK_MESSAGE(
                            ptv::forward_ok(v.begin(), p, v.end(), key, comp, proj),
                            ctx + " fwd-key " + kname);
                        CHECK_MESSAGE(ptv::same_multiset(v, base),
                                      ctx + " fwd-key multiset " + kname);
                    }
                    // reverse by key
                    {
                        auto v = base;
                        auto p = reverse_partition_by_key(alg, v.begin(), v.end(), key, comp, proj);
                        CHECK_MESSAGE(
                            ptv::reverse_ok(v.begin(), p, v.end(), key, comp, proj),
                            ctx + " rev-key " + kname);
                        CHECK_MESSAGE(ptv::same_multiset(v, base),
                                      ctx + " rev-key multiset " + kname);
                    }
                }

                // pivot-by-position: exercise several positions (incl. ends).
                if (n > 0) {
                    for (std::size_t pos : {std::size_t{0}, n / 2, n - 1}) {
                        const auto off = static_cast<std::ptrdiff_t>(pos);
                        // forward by position
                        {
                            auto v = base;
                            Key key = std::invoke(proj, v[pos]);
                            auto p = partition_by_position(alg, v.begin(), v.end(),
                                                           v.begin() + off, comp, proj);
                            CHECK_MESSAGE(
                                ptv::forward_ok(v.begin(), p, v.end(), key, comp, proj),
                                ctx + " fwd-pos " + std::to_string(pos));
                            CHECK_MESSAGE(ptv::same_multiset(v, base),
                                          ctx + " fwd-pos multiset");
                        }
                        // reverse by position
                        {
                            auto v = base;
                            Key key = std::invoke(proj, v[pos]);
                            auto p = reverse_partition_by_position(
                                alg, v.begin(), v.end(), v.begin() + off, comp, proj);
                            CHECK_MESSAGE(
                                ptv::reverse_ok(v.begin(), p, v.end(), key, comp, proj),
                                ctx + " rev-pos " + std::to_string(pos));
                            CHECK_MESSAGE(ptv::same_multiset(v, base),
                                          ctx + " rev-pos multiset");
                        }
                    }
                }
            });
        }
    });
}

}  // namespace

TEST_CASE("partition correctness: i32") { run_matrix<i32>("i32"); }
TEST_CASE("partition correctness: i64") { run_matrix<i64>("i64"); }
TEST_CASE("partition correctness: pair64") { run_matrix<pair64>("pair64"); }
TEST_CASE("partition correctness: keyed (non-identity projection)") {
    run_matrix<keyed>("keyed");
}
