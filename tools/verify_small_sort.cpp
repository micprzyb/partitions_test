// Exhaustive correctness check for the small-sort networks.
//
// Uses the zero-one principle (Knuth Vol 3, 5.3.4 Thm Z): a comparator
// network correctly sorts every input iff it correctly sorts every 0/1 input.
// That gives us 2^N inputs for size N -- 65536 for N=16, all 2^24 = 16.7M
// for N=24 (still well under a second).  We also spot-check with std::sort
// on a few thousand random int sequences per size, to verify the dispatcher
// for pair64 / projection paths.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "partitions/small_sort.hpp"
#include "partitions/types.hpp"

using namespace partitions;

template <std::size_t N>
bool check_binary() {
    std::array<int, N> a{};
    const std::uint64_t lim = (N >= 64) ? ~0ull : (1ull << N);
    for (std::uint64_t bits = 0; bits < lim; ++bits) {
        for (std::size_t i = 0; i < N; ++i)
            a[i] = static_cast<int>((bits >> i) & 1);
        small_sort::sort_n<N>(a.begin());
        for (std::size_t i = 1; i < N; ++i) {
            if (a[i - 1] > a[i]) {
                std::printf("N=%zu FAILED 0/1 input bits=0x%llx\n",
                            (std::size_t)N, (unsigned long long)bits);
                return false;
            }
        }
    }
    return true;
}

template <std::size_t N>
bool check_random_i64(int trials = 4096) {
    std::mt19937_64 rng(0xC0FFEEull ^ N);
    std::uniform_int_distribution<i64> dist(-1'000'000, 1'000'000);
    std::array<i64, N> a{}, ref{};
    for (int t = 0; t < trials; ++t) {
        for (auto& x : a) x = dist(rng);
        ref = a;
        small_sort::sort_n<N>(a.begin());
        std::sort(ref.begin(), ref.end());
        if (a != ref) {
            std::printf("N=%zu i64 trial %d FAILED\n", (std::size_t)N, t);
            return false;
        }
    }
    return true;
}

template <class P, std::size_t N>
bool check_random_pairT(const char* name, std::uint64_t salt, int trials = 4096) {
    std::mt19937_64 rng(0xBADC0DEull ^ N ^ (salt << 24));
    std::uniform_int_distribution<i64> dist(-100, 100);  // many ties
    std::array<P, N> a{}, ref{};
    for (int t = 0; t < trials; ++t) {
        for (auto& x : a) {
            if constexpr (std::is_same_v<P, pair64>)
                x = pair64{dist(rng), dist(rng)};
            else if constexpr (std::is_same_v<P, pair_fi>)
                x = pair_fi{static_cast<float>(dist(rng)), static_cast<int>(dist(rng))};
            else
                x = pair_di{static_cast<double>(dist(rng)), static_cast<int>(dist(rng))};
        }
        ref = a;
        small_sort::sort_n<N>(a.begin());
        std::sort(ref.begin(), ref.end());
        if (a != ref) {
            std::printf("N=%zu %s trial %d FAILED\n", (std::size_t)N, name, t);
            return false;
        }
    }
    return true;
}

template <std::size_t N>
bool check_random_pair() {
    return check_random_pairT<pair64, N>("pair64", 0) &&
           check_random_pairT<pair_fi, N>("pair_fi", 1) &&
           check_random_pairT<pair_di, N>("pair_di", 2);
}

// Verify the runtime-size dispatchers (register-blocked `sort_reg` and the
// libc++/AlphaDev-style `varsort`) against std::sort on random i64 + pair64,
// across sizes passed as a *run-time* length.
template <class T, class Fn>
bool check_runtime(const char* name, std::uint64_t salt, Fn fn, std::size_t n,
                   int trials = 3000) {
    std::mt19937_64 rng(0xD15EA5Eull ^ (n << 7) ^ (salt << 20));
    std::uniform_int_distribution<i64> dist(-50, 50);  // many ties
    std::vector<T> a(n), ref(n);
    for (int t = 0; t < trials; ++t) {
        for (auto& x : a) {
            if constexpr (std::is_same_v<T, pair64>) x = pair64{dist(rng), dist(rng)};
            else x = static_cast<T>(dist(rng));
        }
        ref = a;
        fn(a.data(), a.data() + n);
        std::sort(ref.begin(), ref.end());
        if (a != ref) {
            std::printf("%s n=%zu trial %d FAILED\n", name, n, t);
            return false;
        }
    }
    return true;
}

bool check_runtime_dispatchers() {
    bool ok = true;
    for (std::size_t n = 0; n <= 24; ++n) {
        ok = check_runtime<i64>("sort_reg/i64", 1, [](i64* a, i64* b) {
                 small_sort::sort_reg(a, b);
             }, n) && ok;
        ok = check_runtime<pair64>("sort_reg/pair64", 2, [](pair64* a, pair64* b) {
                 small_sort::sort_reg(a, b);
             }, n) && ok;
        ok = check_runtime<i64>("varsort/i64", 3, [](i64* a, i64* b) {
                 small_sort::varsort(a, b);
             }, n) && ok;
        ok = check_runtime<pair64>("varsort/pair64", 4, [](pair64* a, pair64* b) {
                 small_sort::varsort(a, b);
             }, n) && ok;
    }
    return ok;
}

template <std::size_t... Ns>
bool run_all_impl(std::index_sequence<Ns...>) {
    bool ok = true;
    ((ok = check_binary<Ns + 2>() && ok), ...);
    ((ok = check_random_i64<Ns + 2>() && ok), ...);
    ((ok = check_random_pair<Ns + 2>() && ok), ...);
    ok = check_runtime_dispatchers() && ok;
    return ok;
}

int main() {
    // Test N = 2 .. 24
    bool ok = run_all_impl(std::make_index_sequence<23>{});
    if (ok) {
        std::printf("ALL networks pass 0/1 enumeration and random i64+pair64 trials.\n");
        return 0;
    }
    return 1;
}
