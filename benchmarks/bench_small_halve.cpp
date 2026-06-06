// Small-array HALVER benchmark.
//
// A halver and a full sort both produce the SAME balanced median-rank split
// (the N/2 smallest in the bottom half), so -- per the "same result => compare
// time" rule -- this is a pure raw-time comparison of producing that split:
//
//   ss::sort      small_sort::sort_n<N>   (full sort, then split at N/2)
//   halve         small_halve::halve_n<N> (the halver network, in place)
//   halve_reg     small_halve::halve_reg<N> (register-blocked halver)
//
// Element types: i64, pair64 (lexicographic), pair64f (compare .first only,
// full-entropy first key). min ns/element over batched blocks.
//
// Usage: bench_small_halve [N]   (single N, else a representative sweep)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/small_halve.hpp"
#include "partitions/small_sort.hpp"

using namespace partitions;

namespace {

struct first_key {
    template <class P>
    auto operator()(const P& p) const { return p.first; }
};

template <class T>
std::vector<T> gen(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int64_t> d(0, 1'000'000'000);
    std::vector<T> v(n);
    for (auto& x : v) {
        std::int64_t k = d(rng);
        if constexpr (std::is_same_v<T, pair64>) x = pair64{k, d(rng)};
        else x = static_cast<T>(k);
    }
    return v;
}

constexpr std::size_t kEPI = 1u << 16;

// Each method is a dedicated NON-inline batch applier.  This matters: inlining
// the network into the measure() lambda (and computing the returned boundary)
// gave the optimiser a worse register schedule and falsely inflated the halver's
// time at large n -- a benchmark artifact, not a real cost.  Isolating the batch
// loop in its own function gives clean, representative codegen.
// Each method is a dedicated NON-inline batch applier ending in a memory clobber
// (so the reordering is not dead-code-eliminated).  We time these directly with
// a min-over-reps loop rather than through a lambda passed to measure(): inlining
// the network into the measure() machinery gave wildly unstable, physically
// impossible numbers (i64 jumping 0.8 -> 1.7 between n=8 and n=10) -- a
// measurement artifact.  The direct loop matches an isolated standalone micro.
template <std::size_t N, class T, class Proj>
[[gnu::noinline]] void apply_halve(T* a, std::size_t batch, Proj proj) {
    for (std::size_t b = 0; b < batch; ++b)
        small_halve::halve_n<N>(a + b * N, std::less<>{}, proj);
    asm volatile("" : : : "memory");
}
template <std::size_t N, class T, class Proj>
[[gnu::noinline]] void apply_halve_reg(T* a, std::size_t batch, Proj proj) {
    for (std::size_t b = 0; b < batch; ++b)
        small_halve::halve_reg<N>(a + b * N, std::less<>{}, proj);
    asm volatile("" : : : "memory");
}
template <std::size_t N, class T, class Proj>
[[gnu::noinline]] void apply_sort(T* a, std::size_t batch, Proj proj) {
    for (std::size_t b = 0; b < batch; ++b)
        small_sort::sort_n<N>(a + b * N, std::less<>{}, proj);
    asm volatile("" : : : "memory");
}

template <std::size_t N, class T, class Proj, class Fn>
double best_ns(std::vector<T>& work, const std::vector<T>& master, std::size_t batch,
               std::size_t total, Proj proj, Fn fn) {
    double best = 1e18;
    for (int r = 0; r < 400; ++r) {
        work = master;
        auto t0 = std::chrono::steady_clock::now();
        fn(work.data(), batch, proj);
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                    static_cast<double>(total);
        if (ns < best) best = ns;
    }
    return best;
}

template <std::size_t N, class T, class Proj>
void run(const char* tname, Proj proj) {
    const std::size_t batch = kEPI / N;
    const std::size_t total = N * batch;
    auto master = gen<T>(total, 0xABCD + N);
    std::vector<T> work(total);
    double h = best_ns<N, T>(work, master, batch, total, proj,
                             [](T* a, std::size_t b, Proj p) { apply_halve<N>(a, b, p); });
    double r = best_ns<N, T>(work, master, batch, total, proj,
                             [](T* a, std::size_t b, Proj p) { apply_halve_reg<N>(a, b, p); });
    double s = best_ns<N, T>(work, master, batch, total, proj,
                             [](T* a, std::size_t b, Proj p) { apply_sort<N>(a, b, p); });
    std::printf("%s,%zu,%.4f,%.4f,%.4f\n", tname, N, h, r, s);
    std::fflush(stdout);
}

template <std::size_t N>
void run_all_types() {
    run<N, i64>("i64", std::identity{});
    run<N, pair64>("pair64", std::identity{});
    run<N, pair64>("pair64f", first_key{});
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("type,n,halve_ns,halve_reg_ns,sort_ns\n");
    if (argc > 1) {
        std::size_t N = std::strtoull(argv[1], nullptr, 10);
        // dispatch a single N
        auto one = [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
            ((Ns + 2 == N ? (run_all_types<Ns + 2>(), 0) : 0), ...);
        };
        one(std::make_index_sequence<23>{});
        return 0;
    }
    run_all_types<4>();
    run_all_types<6>();
    run_all_types<8>();
    run_all_types<10>();
    run_all_types<12>();
    run_all_types<16>();
    run_all_types<20>();
    run_all_types<24>();
    return 0;
}
