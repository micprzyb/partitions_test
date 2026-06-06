// Exhaustive correctness proof for the small halver networks (small_halve.hpp).
//
// By the 0-1 principle a comparator network has the halver property for ALL
// inputs iff it has it for all 2^N 0/1 inputs.  For each N in [2,24] we run
// halve_n<N> on every 0/1 vector and check the halver postcondition:
//
//     max(bottom half [0,N/2)) <= min(top half [N/2,N))
//
// i.e. the N/2 smallest ranks land in the bottom half.  We also confirm each
// network is NOT a sorter (the whole point: a halver is cheaper) and that it
// preserves the multiset (trivially true for compare-exchange networks; checked
// via popcount).  Exit code is nonzero on any failure.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <utility>

#include "partitions/small_halve.hpp"

using namespace partitions;

namespace {

template <int N>
bool verify_one() {
    constexpr int m = N / 2;
    bool always_sorted = true;
    for (std::uint32_t mask = 0; mask < (1u << N); ++mask) {
        int a[N];
        int pop = 0;
        for (int i = 0; i < N; ++i) { a[i] = (mask >> i) & 1; pop += a[i]; }
        small_halve::halve_n<N>(static_cast<int*>(a), std::less<>{}, std::identity{});
        int maxb = 0;
        for (int i = 0; i < m; ++i) maxb = a[i] > maxb ? a[i] : maxb;
        int mint = 1;
        for (int i = m; i < N; ++i) mint = a[i] < mint ? a[i] : mint;
        if (maxb > mint) {
            std::printf("  N=%d FAIL: halver property violated for mask 0x%x\n", N, mask);
            return false;
        }
        int pop2 = 0;
        for (int i = 0; i < N; ++i) pop2 += a[i];
        if (pop2 != pop) {
            std::printf("  N=%d FAIL: multiset not preserved for mask 0x%x\n", N, mask);
            return false;
        }
        bool sorted = true;
        for (int i = 1; i < N; ++i) if (a[i] < a[i - 1]) sorted = false;
        if (!sorted) always_sorted = false;
    }
    std::printf("  N=%2d  halver OK (2^%d inputs)%s\n", N, N,
                always_sorted ? "  [note: also a sorter]" : "  [not a sorter -- cheaper]");
    return true;
}

template <int... Ns>
bool verify_all(std::integer_sequence<int, Ns...>) {
    return (verify_one<Ns + 2>() & ...);  // N = 2 .. 24
}

}  // namespace

int main() {
    std::printf("Verifying small halvers by exhaustive 0/1 enumeration:\n");
    bool ok = verify_all(std::make_integer_sequence<int, 23>{});
    std::printf(ok ? "ALL HALVERS CORRECT\n" : "FAILURES DETECTED\n");
    return ok ? 0 : 1;
}
