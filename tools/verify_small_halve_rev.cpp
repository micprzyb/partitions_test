// Exhaustive correctness proof for the REVERSED (descending) halver networks
// (small_halve_rev.hpp).  Mirror of verify_small_halve.cpp.
//
// By the 0-1 principle, a comparator network has the descending-halver property
// for ALL inputs iff it has it for all 2^N 0/1 inputs.  For each N in [2,24] we
// run halve_rev_n<N> on every 0/1 vector and check the DESCENDING postcondition:
//
//     min(bottom half [0,N/2)) >= max(top half [N/2,N))
//
// i.e. the N/2 LARGEST ranks land in the bottom half (so recursive halving sorts
// descending).  We also confirm the multiset is preserved (popcount) and report
// whether the network happens to be a descending sorter.  Nonzero exit on any
// failure.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <utility>

#include "partitions/small_halve_rev.hpp"

using namespace partitions;

namespace {

template <int N>
bool verify_one() {
    constexpr int m = N / 2;
    bool always_sorted = true;  // descending-sorted?
    for (std::uint32_t mask = 0; mask < (1u << N); ++mask) {
        int a[N];
        int pop = 0;
        for (int i = 0; i < N; ++i) { a[i] = (mask >> i) & 1; pop += a[i]; }
        small_halve_rev::halve_rev_n<N>(static_cast<int*>(a), std::less<>{},
                                        std::identity{});
        int minb = 1;
        for (int i = 0; i < m; ++i) minb = a[i] < minb ? a[i] : minb;
        int maxt = 0;
        for (int i = m; i < N; ++i) maxt = a[i] > maxt ? a[i] : maxt;
        if (minb < maxt) {
            std::printf("  N=%d FAIL: descending-halver property violated for mask 0x%x\n", N, mask);
            return false;
        }
        int pop2 = 0;
        for (int i = 0; i < N; ++i) pop2 += a[i];
        if (pop2 != pop) {
            std::printf("  N=%d FAIL: multiset not preserved for mask 0x%x\n", N, mask);
            return false;
        }
        bool sorted = true;  // descending: each <= previous
        for (int i = 1; i < N; ++i) if (a[i] > a[i - 1]) sorted = false;
        if (!sorted) always_sorted = false;
    }
    std::printf("  N=%2d  desc-halver OK (2^%d inputs)%s\n", N, N,
                always_sorted ? "  [note: also a descending sorter]"
                              : "  [not a sorter -- cheaper]");
    return true;
}

template <int... Ns>
bool verify_all(std::integer_sequence<int, Ns...>) {
    return (verify_one<Ns + 2>() & ...);  // N = 2 .. 24
}

}  // namespace

int main() {
    std::printf("Verifying REVERSED small halvers by exhaustive 0/1 enumeration:\n");
    bool ok = verify_all(std::make_integer_sequence<int, 23>{});
    std::printf(ok ? "ALL REVERSED HALVERS CORRECT\n" : "FAILURES DETECTED\n");
    return ok ? 0 : 1;
}
