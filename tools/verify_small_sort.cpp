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

template <std::size_t N>
bool check_random_pair(int trials = 4096) {
    std::mt19937_64 rng(0xBADC0DEull ^ N);
    std::uniform_int_distribution<i64> dist(-100, 100);  // many ties
    std::array<pair64, N> a{}, ref{};
    for (int t = 0; t < trials; ++t) {
        for (auto& x : a) x = pair64{dist(rng), dist(rng)};
        ref = a;
        small_sort::sort_n<N>(a.begin());
        std::sort(ref.begin(), ref.end());
        if (a != ref) {
            std::printf("N=%zu pair64 trial %d FAILED\n", (std::size_t)N, t);
            return false;
        }
    }
    return true;
}

template <std::size_t... Ns>
bool run_all_impl(std::index_sequence<Ns...>) {
    bool ok = true;
    ((ok = check_binary<Ns + 2>() && ok), ...);
    ((ok = check_random_i64<Ns + 2>() && ok), ...);
    ((ok = check_random_pair<Ns + 2>() && ok), ...);
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
