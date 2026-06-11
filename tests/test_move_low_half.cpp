// Correctness of move_low_half / move_low_half_exact: after the call the LAST k
// positions must hold k elements that are <= every element of the first n-k (a
// valid bottom-k by (comp, proj)), the multiset must be preserved, and for the
// exact variant k must equal the spec's target: floor(S/2), or S when S/2 < 16
// (the take-all base case), where S = #{x : comp(proj(x), key)}.  The
// approximate variant's k is only required to be a valid bottom-k count with
// 0 <= k <= S (every moved element is strictly below the key estimate, which is
// itself below `key`).

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "framework.hpp"
#include "partitions/move_low_half.hpp"
#include "partitions/partitions.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,    1,    2,    3,    15,   16,   31,   32,
                                  33,   63,   64,   100,  256,  1000, 1024, 2048,
                                  2049, 3000, 8192, 70000};

template <class T, class K, class Comp, class Proj>
std::ptrdiff_t count_below(const std::vector<T>& v, K key, Comp comp, Proj proj) {
    std::ptrdiff_t s = 0;
    for (const auto& x : v)
        s += static_cast<bool>(std::invoke(comp, std::invoke(proj, x), key));
    return s;
}

// bottom-k: max key of the last k <= min key of the first n-k.
template <class T, class Comp, class Proj>
bool bottom_k(const std::vector<T>& v, std::ptrdiff_t k, Comp comp, Proj proj) {
    const auto n = static_cast<std::ptrdiff_t>(v.size());
    if (k < 0 || k > n) return false;
    if (k == 0 || k == n) return true;
    auto maxlast = std::invoke(proj, v[static_cast<std::size_t>(n - k)]);
    for (std::ptrdiff_t i = n - k + 1; i < n; ++i) {
        auto cur = std::invoke(proj, v[static_cast<std::size_t>(i)]);
        if (std::invoke(comp, maxlast, cur)) maxlast = cur;
    }
    for (std::ptrdiff_t i = 0; i < n - k; ++i)
        if (std::invoke(comp, std::invoke(proj, v[static_cast<std::size_t>(i)]),
                        maxlast))
            return false;
    return true;
}

template <class T>
void run_matrix(const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xFEEDu + n);
            // Keys at several percentiles of the actual data (plus extremes that
            // force S = 0 and, for arithmetic keys, S = n).
            using KeyT = std::remove_cvref_t<decltype(std::invoke(proj, base[0]))>;
            std::vector<KeyT> keys;
            if (n == 0) {
                keys.push_back(KeyT{});
            } else {
                auto sorted = base;
                std::sort(sorted.begin(), sorted.end(), [&](const T& a, const T& b) {
                    return std::invoke(comp, std::invoke(proj, a), std::invoke(proj, b));
                });
                for (double p : {0.0, 0.1, 0.5, 0.9}) {
                    auto idx = std::min<std::size_t>(
                        static_cast<std::size_t>(p * static_cast<double>(n)), n - 1);
                    keys.push_back(std::invoke(proj, sorted[idx]));
                }
                if constexpr (std::is_arithmetic_v<KeyT>)
                    keys.push_back(std::invoke(proj, sorted[n - 1]) + 1);  // S = n
            }
            for (const auto& key : keys) {
                const std::ptrdiff_t S = count_below(base, key, comp, proj);
                const std::ptrdiff_t want = (S / 2 < 16) ? S : S / 2;
                const std::string ctx = tname + "/" + d.name + "/n=" +
                                        std::to_string(n) + "/S=" + std::to_string(S);

                auto v = base;
                std::ptrdiff_t k = move_low_half_exact(v.begin(), v.end(), key,
                                                       comp, proj);
                CHECK_MESSAGE(k == want, ctx + " exact k==target");
                CHECK_MESSAGE(bottom_k(v, k, comp, proj), ctx + " exact bottom-k");
                CHECK_MESSAGE(ptv::same_multiset(v, base), ctx + " exact multiset");
                // Everything moved must be strictly below the key.
                bool below_ok = true;
                for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - k;
                     i < static_cast<std::ptrdiff_t>(n); ++i)
                    if (!std::invoke(comp,
                                     std::invoke(proj, v[static_cast<std::size_t>(i)]),
                                     key))
                        below_ok = false;
                CHECK_MESSAGE(below_ok, ctx + " exact all-below-key");

                auto w = base;
                std::ptrdiff_t ka = move_low_half(w.begin(), w.end(), key, comp, proj);
                CHECK_MESSAGE(ka >= 0 && ka <= S, ctx + " approx 0<=k<=S");
                CHECK_MESSAGE(bottom_k(w, ka, comp, proj), ctx + " approx bottom-k");
                CHECK_MESSAGE(ptv::same_multiset(w, base), ctx + " approx multiset");
            }
        }
    });
}

}  // namespace

TEST_CASE("move_low_half: i64") { run_matrix<i64>("i64"); }
TEST_CASE("move_low_half: pair64") { run_matrix<pair64>("pair64"); }
TEST_CASE("move_low_half: keyed (non-identity projection)") {
    run_matrix<keyed>("keyed");
}
