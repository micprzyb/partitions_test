// Correctness of the OFFSET partition family (algo_off::*) and the
// offset_partition entry point: given the precondition that [begin,
// begin+offset) is all >= key, every variant must produce [begin, m) < key,
// [m, end) >= key (ptv::forward_ok) and preserve the multiset, across types,
// sizes, distributions and offsets -- including offset == 0 (plain partition),
// offset == #(>= key) (maximal valid prefix), keys absent from the array, and
// suffixes with zero / all below-key elements.

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "framework.hpp"
#include "partitions/offset_partition.hpp"
#include "partitions/partitions.hpp"
#include "partitions/reverse_partition.hpp"
#include "verify.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {0,  1,  2,  3,   4,   7,   8,    15, 16,
                                  17, 24, 25, 33,  64,  100, 129,  256,
                                  300, 1000, 4096};

// Successor of a projected key -- used to form a key that is (typically)
// absent from the array.  Keys are either arithmetic or pair64-shaped.
template <class K>
K next_key(K k) {
    if constexpr (std::is_arithmetic_v<K>) {
        return static_cast<K>(k + 1);
    } else {
        k.second += 1;
        return k;
    }
}

// Run one variant on a prepared input and verify the contract.
template <class Alg, class T, class K, class Comp, class Proj>
void check_one(const std::string& ctx, const std::vector<T>& input,
               std::ptrdiff_t offset, K key, Comp comp, Proj proj) {
    auto v = input;
    auto m = Alg{}(v.begin(), v.end(), offset, key, comp, proj);
    CHECK_MESSAGE(ptv::forward_ok(v.begin(), m, v.end(), key, comp, proj),
                  ctx + " boundary");
    CHECK_MESSAGE(ptv::same_multiset(v, input), ctx + " multiset");
    // The boundary must equal the below-key count of the input.
    std::ptrdiff_t S = 0;
    for (const auto& x : input)
        S += static_cast<bool>(std::invoke(comp, std::invoke(proj, x), key));
    CHECK_MESSAGE(m - v.begin() == S, ctx + " count");
}

// Build a valid input for a given offset: reverse-partition a copy by `key`
// so the >= elements lead, take the first `offset` as the prefix, and shuffle
// the rest (deterministically) so the suffix is a genuine mix.
template <class T, class K, class Comp, class Proj>
std::vector<T> make_input(std::vector<T> v, std::ptrdiff_t offset, K key,
                          Comp comp, Proj proj, std::uint64_t seed) {
    algo_rev::sized_rev{}(v.begin(), v.end(), key, comp, proj);
    std::mt19937_64 rng(seed);
    std::shuffle(v.begin() + offset, v.end(), rng);
    return v;
}

template <class Alg, class T>
void run_alg(const char* aname, const std::string& tname) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            const auto base = d.template operator()<T>(n, 0xC0FFEEu + n);
            const std::string ctx0 =
                tname + "/" + std::string(aname) + "/" + d.name +
                "/n=" + std::to_string(n);
            if (n == 0) {
                auto v = base;
                auto m = Alg{}(v.begin(), v.end(), 0, std::invoke(proj, T{}),
                               comp, proj);
                CHECK_MESSAGE(m == v.begin(), ctx0 + " empty boundary");
                continue;
            }
            // Keys: the median element's key (present) and one just above it
            // (typically absent -- exercises the value-pivot semantics).
            auto keys_of = [&] {
                std::vector<T> tmp = base;
                std::nth_element(tmp.begin(),
                                 tmp.begin() + static_cast<std::ptrdiff_t>(n / 2),
                                 tmp.end(),
                                 [&](const T& a, const T& b) {
                                     return comp(std::invoke(proj, a),
                                                 std::invoke(proj, b));
                                 });
                auto k = std::invoke(proj, tmp[n / 2]);
                return std::pair{k, next_key(k)};
            }();
            for (auto key : {keys_of.first, keys_of.second}) {
                // Maximal valid prefix length: the number of >= key elements.
                std::ptrdiff_t high = 0;
                for (const auto& x : base)
                    high += !static_cast<bool>(
                        std::invoke(comp, std::invoke(proj, x), key));
                for (std::ptrdiff_t offset :
                     {std::ptrdiff_t{0}, std::min<std::ptrdiff_t>(1, high),
                      high / 2, high}) {
                    const auto input = make_input(base, offset, key, comp,
                                                  proj,
                                                  0xFEEDu + n +
                                                      static_cast<std::size_t>(offset));
                    check_one<Alg>(ctx0 + "/off=" + std::to_string(offset),
                                   input, offset, key, comp, proj);
                }
            }
        }
    });
}

template <class Alg>
void run_types(const char* aname) {
    run_alg<Alg, i32>(aname, "i32");
    run_alg<Alg, i64>(aname, "i64");
    run_alg<Alg, pair64>(aname, "pair64");
    run_alg<Alg, keyed>(aname, "keyed");
}

}  // namespace

TEST_CASE("offset_partition: scan_swap") { run_types<algo_off::scan_swap_off>("scan_swap"); }
TEST_CASE("offset_partition: gap_off") { run_types<algo_off::gap_off>("gap_off"); }
TEST_CASE("offset_partition: part_swap") { run_types<algo_off::part_swap_off>("part_swap"); }
TEST_CASE("offset_partition: fused_block") { run_types<algo_off::fused_block_off>("fused_block"); }
TEST_CASE("offset_partition: prefix_fill") { run_types<algo_off::prefix_fill_off>("prefix_fill"); }
TEST_CASE("offset_partition: sized_off") { run_types<algo_off::sized_off>("sized_off"); }

// Exhaustive tiny arrays: all value tuples over {0,1,2,3}, all preconditioned
// offsets, every variant.  Catches any boundary/straddle bookkeeping slip.
TEST_CASE("offset_partition: exhaustive small") {
    auto comp = std::less<>{};
    auto proj = std::identity{};
    const int key = 2;
    auto run_all = [&](const std::vector<i64>& v, std::ptrdiff_t off) {
        auto one = [&](auto alg, const char* nm) {
            auto w = v;
            auto m = alg(w.begin(), w.end(), off, i64{key}, comp, proj);
            const bool ok = ptv::forward_ok(w.begin(), m, w.end(), i64{key},
                                            comp, proj) &&
                            ptv::same_multiset(w, v);
            CHECK_MESSAGE(ok, std::string(nm) + " exhaustive n=" +
                                  std::to_string(v.size()) +
                                  " off=" + std::to_string(off));
        };
        one(algo_off::scan_swap_off{}, "scan_swap");
        one(algo_off::gap_off{}, "gap_off");
        one(algo_off::part_swap_off{}, "part_swap");
        one(algo_off::fused_block_off{}, "fused_block");
        one(algo_off::prefix_fill_off{}, "prefix_fill");
        one(algo_off::sized_off{}, "sized_off");
    };
    for (std::size_t n = 1; n <= 6; ++n) {
        std::vector<i64> v(n);
        const std::size_t total = std::size_t(1) << (2 * n);  // 4^n tuples
        for (std::size_t code = 0; code < total; ++code) {
            for (std::size_t i = 0; i < n; ++i)
                v[i] = static_cast<i64>((code >> (2 * i)) & 3);
            for (std::ptrdiff_t off = 0;
                 off <= static_cast<std::ptrdiff_t>(n); ++off) {
                bool valid = true;  // precondition: v[0..off) all >= key
                for (std::ptrdiff_t i = 0; i < off; ++i)
                    valid = valid && v[static_cast<std::size_t>(i)] >= key;
                if (!valid) break;  // longer offsets only add violations
                run_all(v, off);
            }
        }
    }
}

// The entry point returns the count and honours comp/proj defaulting.
TEST_CASE("offset_partition: entry point count") {
    std::mt19937_64 rng(42);
    for (std::size_t n : {std::size_t(1), std::size_t(17), std::size_t(300),
                          std::size_t(5000)}) {
        std::vector<i64> v(n);
        for (auto& x : v) x = static_cast<i64>(rng() % 1000);
        const i64 key = 500;
        auto m0 = algo_rev::sized_rev{}(v.begin(), v.end(), key,
                                        std::less<>{}, std::identity{});
        const std::ptrdiff_t high = m0 - v.begin();
        for (std::ptrdiff_t off : {std::ptrdiff_t{0}, high / 3, high}) {
            auto input = v;
            std::shuffle(input.begin() + off, input.end(), rng);
            std::ptrdiff_t S = 0;
            for (auto x : input) S += x < key;
            auto w = input;
            const auto c = offset_partition(w.begin(), w.end(), off, key);
            CHECK(c == S);
            CHECK(ptv::forward_ok(w.begin(), w.begin() + c, w.end(), key,
                                  std::less<>{}, std::identity{}));
        }
    }
}
