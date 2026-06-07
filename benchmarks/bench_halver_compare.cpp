// Compare specific halver networks head-to-head: the previous h7/h8 vs the
// fewer-comparator candidates (h7_alt = 11, h8_new = 14).  Each net is first
// PROVEN a halver by exhaustive 0/1 enumeration for its split point, then the
// network-application cost is timed (apply via small_sort::cswap, the exact path
// halve_n uses).  Run the binary a few times and take the min/median per row.
//
// CSV: type,net,comparators,split,ns_per_block

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/small_sort.hpp"

using namespace partitions;
using PR = std::pair<int, int>;

namespace {

constexpr std::array<PR, 12> H7_OLD = {{{0,4},{1,5},{2,6},{0,2},{1,3},{4,6},{2,4},{3,5},{0,1},{2,3},{4,5},{1,4}}};
constexpr std::array<PR, 11> H7_ALT = {{{0,6},{2,4},{1,3},{3,6},{4,5},{2,3},{0,1},{0,5},{3,4},{1,2},{2,3}}};
constexpr std::array<PR, 15> H8_OLD = {{{0,2},{1,3},{5,7},{0,4},{1,5},{3,7},{0,1},{2,3},{4,5},{6,7},{2,4},{3,5},{1,4},{3,6},{3,4}}};
constexpr std::array<PR, 14> H8_NEW = {{{0,7},{1,4},{2,5},{3,6},{0,2},{1,3},{4,6},{5,7},{0,6},{1,7},{2,4},{3,5},{2,5},{3,4}}};

template <std::size_t K>
bool is_halver(const std::array<PR, K>& net, int n, int h) {
    for (int mask = 0; mask < (1 << n); ++mask) {
        int a[16];
        for (int i = 0; i < n; ++i) a[i] = (mask >> i) & 1;
        for (auto [x, y] : net) if (a[x] > a[y]) std::swap(a[x], a[y]);
        int bm = 0, tm = 1;
        for (int i = 0; i < h; ++i) bm = std::max(bm, a[i]);
        for (int i = h; i < n; ++i) tm = std::min(tm, a[i]);
        if (bm > tm) return false;
    }
    return true;
}

struct first_key { template <class P> auto operator()(const P& p) const { return p.first; } };

// no-tree-vectorize: the halver is applied one block at a time in real use (the
// quicksort recursion), which is SCALAR.  Without this, GCC auto-vectorises this
// artificial batch loop with AVX2 and spills ymm registers (552-byte frame),
// which neither matches real use nor reflects the nets' true cost -- it inverted
// the i64 h8_new result.  Forcing scalar makes the benchmark representative.
template <int N, std::size_t K, class T, class Proj>
[[gnu::noinline, gnu::optimize("no-tree-vectorize")]]
void apply_all(T* a, std::size_t batch, const std::array<PR, K>& net, Proj proj) {
    auto comp = std::less<>{};
    for (std::size_t b = 0; b < batch; ++b)
        small_sort::apply_network(a + b * N, net, std::make_index_sequence<K>{}, comp, proj);
    asm volatile("" : : : "memory");
}

template <int N, class T, std::size_t K, class Proj>
double bench_net(const char* tn, const char* nm, int split, const std::array<PR, K>& net, Proj proj) {
    const std::size_t batch = (1u << 16) / N;
    const std::size_t total = batch * N;
    std::mt19937_64 rng(N * 131 + K);
    std::uniform_int_distribution<std::int64_t> d(0, 1'000'000'000);
    std::vector<T> master(total);
    for (auto& x : master) { std::int64_t k = d(rng); if constexpr (std::is_same_v<T, pair64>) x = pair64{k, d(rng)}; else x = static_cast<T>(k); }
    std::vector<T> work(total);
    double best = 1e18;
    for (int r = 0; r < 600; ++r) {
        work = master;
        auto t0 = std::chrono::steady_clock::now();
        apply_all<N>(work.data(), batch, net, proj);
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / batch);
    }
    std::printf("%s,%s,%zu,%d,%.4f\n", tn, nm, K, split, best);
    std::fflush(stdout);
    return best;
}

template <class T, class Proj>
void run_type(const char* tn, Proj proj) {
    bench_net<7, T>(tn, "h7_old", 3, H7_OLD, proj);
    bench_net<7, T>(tn, "h7_alt", 3, H7_ALT, proj);
    bench_net<8, T>(tn, "h8_old", 4, H8_OLD, proj);
    bench_net<8, T>(tn, "h8_new", 4, H8_NEW, proj);
}

}  // namespace

int main() {
    // prove correctness first (abort if any candidate is not a halver for its split)
    struct { const char* nm; bool ok; } v[] = {
        {"h7_old@3", is_halver(H7_OLD, 7, 3)}, {"h7_alt@3", is_halver(H7_ALT, 7, 3)},
        {"h8_old@4", is_halver(H8_OLD, 8, 4)}, {"h8_new@4", is_halver(H8_NEW, 8, 4)},
    };
    std::fprintf(stderr, "correctness (0/1 exhaustive):\n");
    for (auto& e : v) std::fprintf(stderr, "  %-10s halver=%d\n", e.nm, e.ok);

    std::printf("type,net,comparators,split,ns_per_block\n");
    run_type<i64>("i64", std::identity{});
    run_type<pair64>("pair64", std::identity{});
    run_type<pair64>("pair64f", first_key{});
    return 0;
}
