// Correctness of offset_low_half / offset_low_half_exact: given the
// precondition [begin, begin+offset) all >= key, the call must return k and
// leave the k SMALLEST elements of the whole range in the first k positions
// (any order: bottom-k property), preserving the multiset, with
//   exact   : k == floor(S/2), or S when S/2 < 16 (take-all base case);
//   sampled : 0 <= k <= S, and k > 0 whenever S > 0 (it either routes to the
//             exact path or partitions around a sampled value t < key).

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "framework.hpp"
#include "partitions/offset_low_half.hpp"
#include "partitions/partitions.hpp"
#include "partitions/reverse_partition.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,  1,   2,   3,    7,    16,   33,  64,
                                  100, 256, 511, 1000, 1024, 2048, 5000};

// bottom-k: max over the first k must be <= min over the rest (projected).
template <class T, class Comp, class Proj>
bool bottom_k_ok(const std::vector<T>& v, std::ptrdiff_t k, Comp comp,
                 Proj proj) {
    const auto n = static_cast<std::ptrdiff_t>(v.size());
    if (k < 0 || k > n) return false;
    if (k == 0 || k == n) return true;
    auto kof = [&](std::ptrdiff_t i) {
        return std::invoke(proj, v[static_cast<std::size_t>(i)]);
    };
    auto maxfront = kof(0);
    for (std::ptrdiff_t i = 1; i < k; ++i)
        if (comp(maxfront, kof(i))) maxfront = kof(i);
    for (std::ptrdiff_t i = k; i < n; ++i)
        if (comp(kof(i), maxfront)) return false;
    return true;
}

template <class T, class K, class Comp, class Proj>
std::ptrdiff_t count_below(const std::vector<T>& v, K key, Comp comp,
                           Proj proj) {
    std::ptrdiff_t S = 0;
    for (const auto& x : v)
        S += static_cast<bool>(comp(std::invoke(proj, x), key));
    return S;
}

template <class T, class K, class Comp, class Proj>
std::vector<T> make_input(std::vector<T> v, std::ptrdiff_t offset, K key,
                          Comp comp, Proj proj, std::uint64_t seed) {
    algo_rev::sized_rev{}(v.begin(), v.end(), key, comp, proj);
    std::mt19937_64 rng(seed);
    std::shuffle(v.begin() + offset, v.end(), rng);
    return v;
}

template <bool Exact, class T>
void run_type(const char* an, const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0x10D0u + n);
            if (n == 0) {
                auto v = base;
                auto k = Exact ? offset_low_half_exact(v.begin(), v.end(), 0,
                                                       std::invoke(proj, T{}),
                                                       comp, proj)
                               : offset_low_half(v.begin(), v.end(), 0,
                                                 std::invoke(proj, T{}), comp,
                                                 proj);
                CHECK_MESSAGE(k == 0, tname + "/" + an + " empty");
                continue;
            }
            std::vector<T> tmp = base;
            std::nth_element(tmp.begin(),
                             tmp.begin() + static_cast<std::ptrdiff_t>(n / 2),
                             tmp.end(), [&](const T& a, const T& b) {
                                 return comp(std::invoke(proj, a),
                                             std::invoke(proj, b));
                             });
            const auto key = std::invoke(proj, tmp[n / 2]);
            std::ptrdiff_t high = 0;
            for (const auto& x : base)
                high += !static_cast<bool>(comp(std::invoke(proj, x), key));
            for (std::ptrdiff_t offset :
                 {std::ptrdiff_t{0}, high / 2, high}) {
                const auto input =
                    make_input(base, offset, key, comp, proj, 0xFACEu + n);
                const auto S = count_below(input, key, comp, proj);
                auto v = input;
                const std::ptrdiff_t k =
                    Exact ? offset_low_half_exact(v.begin(), v.end(), offset,
                                                  key, comp, proj)
                          : offset_low_half(v.begin(), v.end(), offset, key,
                                            comp, proj);
                const std::string ctx = tname + "/" + an + "/" + d.name +
                                        "/n=" + std::to_string(n) +
                                        "/off=" + std::to_string(offset);
                if (Exact) {
                    const std::ptrdiff_t want = (S / 2 < 16) ? S : S / 2;
                    CHECK_MESSAGE(k == want, ctx + " k exact");
                } else {
                    CHECK_MESSAGE(k >= 0 && k <= S, ctx + " k range");
                    CHECK_MESSAGE(S == 0 || k > 0, ctx + " k nonzero");
                }
                CHECK_MESSAGE(bottom_k_ok(v, k, comp, proj), ctx + " bottomk");
                CHECK_MESSAGE(ptv::same_multiset(v, input), ctx + " multiset");
            }
        }
    });
}

}  // namespace

TEST_CASE("offset_low_half_exact: i64") { run_type<true, i64>("exact", "i64"); }
TEST_CASE("offset_low_half_exact: pair64") { run_type<true, pair64>("exact", "pair64"); }
TEST_CASE("offset_low_half_exact: keyed") { run_type<true, keyed>("exact", "keyed"); }
TEST_CASE("offset_low_half: i64") { run_type<false, i64>("sampled", "i64"); }
TEST_CASE("offset_low_half: pair64") { run_type<false, pair64>("sampled", "pair64"); }
TEST_CASE("offset_low_half: keyed") { run_type<false, keyed>("sampled", "keyed"); }
