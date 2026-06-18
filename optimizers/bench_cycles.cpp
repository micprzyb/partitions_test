// Cycle-accurate two-network comparison harness (reliable benchmark).
//
// WHY (vs the ns timing in bench_pool.cpp): wall-clock ns swings with turbo /
// frequency and is contaminated by the per-rep buffer copy.  The halver kernels
// are FULLY BRANCHLESS (cmov compare-exchange, zero conditional branches), so the
// retired-instruction stream and the cycle count are DATA-INDEPENDENT.  That lets
// us (a) drop the per-rep copy entirely and just hammer the kernel on one L1-
// resident batch, and (b) measure with `perf` hardware counters:
//   * cycles/element  -- frequency-invariant (true pipeline cost, not ns)
//   * IPC             -- exposes the *mechanism* (dependency stalls show as low IPC)
//   * instructions/el -- the portable "work" metric
//
// Driven by optimizers/bench_cycles.sh, which generates /tmp/cyc_nets.inc with two
// nets `nets::A` and `nets::B` (+ `#define HN`), compiles A/B x {i64,pair64,pair64f}
// in parallel, and perf-stats each on one quiet P-core (median of N reps).
//
// Build (per binary): g++ -O3 -march=native -std=c++23 -DNET=A -DVARIANT=1 \
//   -DCYC_INC='"/tmp/cyc_nets.inc"' -Iinclude optimizers/bench_cycles.cpp -o ...
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>
#include "partitions/small_sort.hpp"
#include "partitions/types.hpp"
using namespace partitions;
using P = std::pair<int, int>;
struct first_key { template <class T> auto operator()(const T& p) const { return p.first; } };
#ifndef CYC_INC
#error "define -DCYC_INC='\"/tmp/cyc_nets.inc\"' (defines nets::A, nets::B, HN)"
#endif
namespace nets {
#include CYC_INC                 // inline constexpr std::array<P,..> A,B; #define HN <n>
}
#ifndef NET
#define NET A
#endif
#ifndef VARIANT
#define VARIANT 1
#endif
#if VARIANT == 0
using T = i64; using PROJ = std::identity; static const char* VN = "i64";
#elif VARIANT == 1
using T = pair64; using PROJ = std::identity; static const char* VN = "pair64";
#else
using T = pair64; using PROJ = first_key; static const char* VN = "pair64f";
#endif

template <class U>
static U mk(std::mt19937_64& r, std::uniform_int_distribution<std::int64_t>& d) {
    if constexpr (std::is_same_v<U, pair64>) return pair64{d(r), d(r)};
    else return (U)d(r);
}
template <const auto& Net>
[[gnu::noinline, gnu::optimize("no-tree-vectorize")]]
void kern(T* a, std::size_t B) {
    std::less<> c; PROJ p;
    for (std::size_t b = 0; b < B; ++b)
        small_sort::apply_network(a + b * HN, Net, std::make_index_sequence<Net.size()>{}, c, p);
    asm volatile("" : : : "memory");
}
int main(int argc, char** argv) {
    std::size_t B = (argc > 2 ? std::atol(argv[2]) : 96);     // blocks (keep L1-resident)
    std::size_t R = (argc > 1 ? std::atol(argv[1]) : 30000);  // measured reps
    std::size_t total = B * HN;
    std::mt19937_64 r(123);
    std::uniform_int_distribution<std::int64_t> d(0, 1000000000);
    std::vector<T> v(total);
    for (auto& x : v) x = mk<T>(r, d);
    for (int w = 0; w < 300; ++w) kern<nets::NET>(v.data(), B);     // warmup
    for (std::size_t it = 0; it < R; ++it) kern<nets::NET>(v.data(), B);
    std::printf("ELEMS %zu VN %s HN %d\n", (std::size_t)B * R * HN, VN, (int)HN);
    volatile char sink = *(char*)v.data(); (void)sink;   // keep v live; return 0 (set -e)
    return 0;
}
