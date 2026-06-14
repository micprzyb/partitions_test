// Pool benchmark: time every candidate halver network in a generated include
// against the baseline, for i64 / pair64 (lex) / pair64f (first-key), and report
// the fastest per type and by total.
//
// The networks are passed as COMPILE-TIME constants (a reference NTTP), exactly
// as small_halve::halve_n uses nets::hN -- this bakes the comparator indices in.
// Passing them at runtime instead loads indices from memory and inflates the
// time ~3x (a measurement artifact), so the candidates must be #included.
//
// Build (the driver does this for you):
//   g++ -O3 -march=native -std=c++23 -DHN=<n> -DPOOL_INC='"/tmp/hpool.inc"' \
//       -I include optimizers/bench_pool.cpp -o /tmp/benchpool
// where /tmp/hpool.inc defines  cand::base  + pN candidates + HN_POOL_LIST
// (see optimizers/make_pool.py).
#ifndef HN
#error "define -DHN=<block size>"
#endif
#ifndef POOL_INC
#error "define -DPOOL_INC='\"/path/to/hpool.inc\"'"
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "partitions/small_sort.hpp"
#include "partitions/types.hpp"
using namespace partitions;
using P = std::pair<int, int>;
struct first_key { template <class T> auto operator()(const T& p) const { return p.first; } };
namespace cand {
#include POOL_INC          // defines base + p0..pK + HN_POOL_LIST
}

template <const auto& Net, class T, class Proj>
[[gnu::noinline, gnu::optimize("no-tree-vectorize")]]
void apply_cand(T* a, std::size_t batch, Proj proj) {
    constexpr std::size_t S = Net.size();
    std::less<> comp;
    for (std::size_t b = 0; b < batch; ++b)
        small_sort::apply_network(a + b * HN, Net, std::make_index_sequence<S>{}, comp, proj);
    asm volatile("" : : : "memory");
}
constexpr std::size_t kEPI = 1u << 16;
template <class T> std::vector<T> gen(std::size_t n, std::uint64_t s) {
    std::mt19937_64 r(s); std::uniform_int_distribution<std::int64_t> d(0, 1000000000);
    std::vector<T> v(n);
    for (auto& x : v) { auto k = d(r); if constexpr (std::is_same_v<T, pair64>) x = pair64{k, d(r)}; else x = (T)k; }
    return v;
}
template <const auto& Net, class T, class Proj>
double bn(std::vector<T>& w, const std::vector<T>& m, std::size_t b, std::size_t t, Proj p) {
    double best = 1e18;
    for (int r = 0; r < 300; ++r) {
        w = m; auto t0 = std::chrono::steady_clock::now();
        apply_cand<Net, T>(w.data(), b, p);
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)t;
        if (ns < best) best = ns;
    }
    return best;
}
struct Row { std::string name; double i64, p64, p64f; };
std::vector<Row> rows;
template <const auto& Net> void run_all(const char* nname) {
    const std::size_t batch = kEPI / HN, total = HN * batch;
    static auto mi = gen<i64>(total, 0xABCD + HN);
    static auto mp = gen<pair64>(total, 0xABCD + HN);
    std::vector<i64> wi(total); std::vector<pair64> wp(total);
    double a = bn<Net, i64>(wi, mi, batch, total, std::identity{});
    double b = bn<Net, pair64>(wp, mp, batch, total, std::identity{});
    double c = bn<Net, pair64>(wp, mp, batch, total, first_key{});
    rows.push_back({nname, a, b, c});
    std::printf("%-6s %.4f %.4f %.4f  sum=%.4f\n", nname, a, b, c, a + b + c);
    std::fflush(stdout);
}
int main() {
    std::printf("net    i64     pair64  pair64f   (HN=%d)\n", HN);
    run_all<cand::base>("base");
#define X(p) run_all<cand::p>(#p);
    HN_POOL_LIST
#undef X
    auto bb = [&](double Row::*f, const char* l) {
        const Row* w = &rows[0];
        for (auto& r : rows) if (r.*f < w->*f) w = &r;
        std::printf("best %-8s: %s (%.4f)\n", l, w->name.c_str(), w->*f);
    };
    bb(&Row::i64, "i64"); bb(&Row::p64, "pair64"); bb(&Row::p64f, "pair64f");
    const Row* w = &rows[0]; double bs = 1e18;
    for (auto& r : rows) { double s = r.i64 + r.p64 + r.p64f; if (s < bs) { bs = s; w = &r; } }
    std::printf("best sum     : %s (%.4f)\n", w->name.c_str(), bs);
    return 0;
}
