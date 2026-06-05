// Benchmark: completing an almost-sorted small block (n <= 24), two shapes:
//
//   case=a  EXTEND : [0,k) sorted, [k,n) unsorted.
//   case=b  MERGE  : [0,k) sorted AND [k,n) sorted.
//
// delta = n - k is the amount of "new" data; target regime is delta small (1..4).
//
// FAIRNESS / round-robin harness.  An earlier version timed each candidate in its
// own long measurement, then compared absolute numbers across candidates that ran
// seconds apart.  That is biased: the per-call `min` is stable, but the absolute
// level drifts over a program's lifetime (frequency scaling, cache state left by
// neighbouring candidates).  Proof: at delta=1 cases (a) and (b) are the IDENTICAL
// operation on IDENTICAL data, yet the old harness reported up to ~30% difference.
//
// This version measures all candidates for a given (n, delta) ROUND-ROBIN: in
// each round every candidate is timed once, so drift hits them equally and the
// per-candidate `min` is comparable.  All candidates share the same random values
// (only the prep -- which ranges are pre-sorted -- differs), so case (a) vs (b) is
// a clean apples-to-apples comparison.  The inner batch loop calls the concrete
// op directly (the std::function wraps a whole *batch*, called once per round), so
// there is no per-element type-erasure overhead.
//
// CSV: type,case,algo,n,k,delta,total_ops,samples,min_ns,p50_ns,p90_ns,mean_ns,cv_pct

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/small_merge.hpp"
#include "partitions/small_sort.hpp"
#include "partitions/types.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kTargetElems   = 1u << 16;
constexpr std::uint64_t kMinTotalOps = 5'000'000;
constexpr int kWarmupRounds          = 3;
using clk = std::chrono::steady_clock;

template <class T, class Rng, class Dist>
T make_elem(Rng& rng, Dist& dist) {
    if constexpr (std::is_same_v<T, pair64>)
        return pair64{dist(rng), dist(rng)};
    else
        return static_cast<T>(dist(rng));
}

// Shared random values: `batch` blocks of n elements, half wide / half narrow
// range (narrow = many key ties).  Prep is applied later, per candidate.
template <class T>
std::vector<T> make_values(std::size_t n, std::size_t batch, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<T> v(n * batch);
    for (std::size_t b = 0; b < batch; ++b) {
        const bool narrow = (b & 1) != 0;
        const i64 hi = narrow ? 64 : (std::numeric_limits<i64>::max() / 4);
        const i64 lo = narrow ? -64 : (std::numeric_limits<i64>::min() / 4);
        std::uniform_int_distribution<i64> d(lo, hi);
        for (std::size_t i = 0; i < n; ++i) v[b * n + i] = make_elem<T>(rng, d);
    }
    return v;
}

double pct(const std::vector<double>& s, double p) {
    if (s.empty()) return 0.0;
    const double idx = p * (s.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx), hi = std::min(lo + 1, s.size() - 1);
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo);
}

template <class T, std::size_t N>
void run_n(std::size_t delta) {
    const std::size_t k = N - delta;
    const std::size_t batch = std::max<std::size_t>(1, kTargetElems / N);
    const auto values = make_values<T>(N, batch, 0xABCDEull ^ (N << 8) ^ (k << 20));

    std::vector<std::string> label;          // "case,algo"
    std::vector<std::function<double()>> run; // measures one batch -> ns/op

    // Register a candidate: prep(block) arranges sortedness; op(block) is timed.
    auto add = [&](const char* cs, const char* algo, auto prep, auto op) {
        auto M = std::make_shared<std::vector<T>>(values);
        for (std::size_t b = 0; b < batch; ++b) prep(M->data() + b * N);
        auto W = std::make_shared<std::vector<T>>(N * batch);
        label.push_back(std::string(cs) + "," + algo);
        run.push_back([=]() -> double {
            *W = *M;  // untimed restore
            std::uintptr_t sink = 0;
            const auto t0 = clk::now();
            for (std::size_t b = 0; b < batch; ++b) {
                T* p = W->data() + b * N;
                op(p);  // concrete lambda, inlined -- no per-op type erasure
                sink ^= reinterpret_cast<std::uintptr_t>(p);
            }
            const auto t1 = clk::now();
            bench::do_not_optimize(sink);
            return std::chrono::duration<double, std::nano>(t1 - t0).count() /
                   static_cast<double>(batch);
        });
    };

    auto prep_a = [k](T* b) { std::sort(b, b + k); };                   // tail unsorted
    auto prep_b = [k](T* b) { std::sort(b, b + k); std::sort(b + k, b + N); };

    // ---- case (a): prefix sorted, tail unsorted ----
    add("a", "extend_sorted", prep_a, [k](T* a) { small_merge::extend_sorted(a, k, N); });
    add("a", "extend_scan", prep_a, [k](T* a) { small_merge::extend_sorted_scan(a, k, N); });
    add("a", "extend_bsearch", prep_a, [k](T* a) { small_merge::extend_sorted_bsearch(a, k, N); });
    add("a", "sort_network", prep_a, [](T* a) { small_sort::sort_n<N>(a); });
    add("a", "std::sort", prep_a, [](T* a) { std::sort(a, a + N); });
    // ---- case (b): both halves sorted ----
    add("b", "extend_sorted", prep_b, [k](T* a) { small_merge::extend_sorted(a, k, N); });
    add("b", "merge_sorted", prep_b, [k](T* a) { small_merge::merge_sorted(a, k, N); });
    add("b", "merge_branchless", prep_b, [k](T* a) { small_merge::merge_sorted_branchless(a, k, N); });
    add("b", "sort_network", prep_b, [](T* a) { small_sort::sort_n<N>(a); });
    add("b", "std::inplace_merge", prep_b, [k](T* a) { std::inplace_merge(a, a + k, a + N); });
    add("b", "std::merge", prep_b, [k](T* a) {
        T buf[24];
        std::merge(a, a + k, a + k, a + N, buf);
        std::copy(buf, buf + N, a);
    });

    const std::size_t C = run.size();
    const std::uint64_t R = std::max<std::uint64_t>((kMinTotalOps + batch - 1) / batch, 50);
    std::vector<std::vector<double>> samp(C);
    for (auto& s : samp) s.reserve(R);

    for (int w = 0; w < kWarmupRounds; ++w)
        for (std::size_t c = 0; c < C; ++c) (void)run[c]();
    for (std::uint64_t r = 0; r < R; ++r)            // ROUND-ROBIN over candidates
        for (std::size_t c = 0; c < C; ++c) samp[c].push_back(run[c]());

    for (std::size_t c = 0; c < C; ++c) {
        double sum = 0, sum2 = 0;
        for (double s : samp[c]) { sum += s; sum2 += s * s; }
        const double mean = sum / samp[c].size();
        const double var = sum2 / samp[c].size() - mean * mean;
        const double cv = mean > 0 ? 100.0 * std::sqrt(std::max(0.0, var)) / mean : 0.0;
        std::sort(samp[c].begin(), samp[c].end());
        std::printf("%s,%s,%zu,%zu,%zu,%llu,%zu,%.3f,%.3f,%.3f,%.3f,%.2f\n",
                    type_name<T>(), label[c].c_str(), N, k, delta,
                    static_cast<unsigned long long>(R * batch), samp[c].size(),
                    samp[c].front(), pct(samp[c], 0.50), pct(samp[c], 0.90), mean, cv);
        std::fflush(stdout);
    }
}

template <class T, std::size_t N>
void sweep_deltas() {
    for (std::size_t delta : {1ul, 2ul, 3ul, 4ul, 6ul, 8ul, 12ul}) {
        if (delta >= N) continue;
        run_n<T, N>(delta);
    }
}
template <class T, std::size_t... Ns>
void run_type_impl(std::index_sequence<Ns...>, std::size_t only_n) {
    ((only_n == 0 || only_n == Ns + 2 ? sweep_deltas<T, Ns + 2>() : void()), ...);
}
template <class T>
void run_type(std::size_t only_n) {
    run_type_impl<T>(std::make_index_sequence<23>{}, only_n);
}

// quick correctness self-check
template <class T, std::size_t N>
bool check() {
    std::mt19937_64 rng(123 + N);
    std::uniform_int_distribution<i64> d(-30, 30);
    for (int trial = 0; trial < 2000; ++trial) {
        for (std::size_t delta = 1; delta < N; ++delta) {
            const std::size_t k = N - delta;
            std::vector<T> base(N), w(N);
            for (auto& x : base) x = make_elem<T>(rng, d);
            auto chk = [&](const char* nm) {
                if (!std::is_sorted(w.begin(), w.end())) { std::printf("%s fail N=%zu k=%zu\n", nm, N, k); return false; }
                return true;
            };
            w = base; std::sort(w.begin(), w.begin() + k);
            small_merge::extend_sorted(w.data(), k, N); if (!chk("a extend")) return false;
            w = base; std::sort(w.begin(), w.begin() + k);
            small_merge::extend_sorted_scan(w.data(), k, N); if (!chk("a scan")) return false;
            w = base; std::sort(w.begin(), w.begin() + k);
            small_merge::extend_sorted_bsearch(w.data(), k, N); if (!chk("a bsearch")) return false;
            w = base; std::sort(w.begin(), w.begin() + k); std::sort(w.begin() + k, w.end());
            small_merge::merge_sorted(w.data(), k, N); if (!chk("b merge")) return false;
            w = base; std::sort(w.begin(), w.begin() + k); std::sort(w.begin() + k, w.end());
            small_merge::merge_sorted_branchless(w.data(), k, N); if (!chk("b bl")) return false;
        }
    }
    return true;
}
template <std::size_t... Ns>
bool check_all(std::index_sequence<Ns...>) {
    bool ok = true;
    ((ok = check<i64, Ns + 2>() && ok), ...);
    ((ok = check<pair64, Ns + 2>() && ok), ...);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "check") == 0) {
        const bool ok = check_all(std::make_index_sequence<23>{});
        std::printf(ok ? "merge/extend: ALL correct\n" : "FAILED\n");
        return ok ? 0 : 1;
    }
    std::size_t only_n = 0;
    if (argc > 1) only_n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,case,algo,n,k,delta,total_ops,samples,min_ns,p50_ns,p90_ns,mean_ns,cv_pct\n");
    run_type<i64>(only_n);
    run_type<pair64>(only_n);
    return 0;
}
