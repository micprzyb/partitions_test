// Small-array sort benchmark for N=2..24 of int64 and pair<int64,int64>.
//
// ------------------------- methodology (stability) -------------------------
// Each reported data point aggregates *several million* individual sorts taken
// across *multiple independent configurations*, so the number is robust to
// run-to-run jitter, frequency scaling, and any single lucky/unlucky input:
//
//   * CONFIGURATIONS.  We sweep `kConfigs` independent setups per data point.
//     They differ in (a) the RNG seed and (b) the value range: half use a wide
//     full-width range, half a narrow range that forces many ties (which, for
//     pair64, exercises the second-key tie-break of the lexicographic order).
//     A comparator-network is data-independent so this only matters for the
//     branchy library sorts -- which is exactly the point of testing it.
//
//   * VOLUME.  Per data point we run at least `kMinTotalSorts` (a few million)
//     sorts.  Sorts are issued in batches of `batch` independent blocks; the
//     batch is the timed unit (one steady_clock pair amortised over thousands
//     of sorts, so clock overhead is negligible).  We collect one ns/sort
//     sample per batch -> hundreds-to-thousands of samples per data point.
//
//   * RESTORE IS UNTIMED.  Sorting is destructive, so before each timed batch
//     we copy the pristine master back into the work buffer; that copy is
//     outside the timed region.  A few warmup batches per config prime caches
//     and the branch predictor before sampling begins.
//
//   * REPORTING.  We report the MIN (least-noisy estimate of true cost), the
//     MEDIAN (p50) and p90 ns/sort over all samples, plus the coefficient of
//     variation (cv% = stddev/mean) as an explicit stability indicator.
//
// Candidates (all driven with a COMPILE-TIME size N, std::less<> + identity):
//   std::sort, boost::pdqsort, boost::spreadsort (int64 only),
//   cppsort::sorting_network_sorter<N>, cppsort::low_moves_sorter<N>,
//   small_sort::sort_n<N> (best-known network), small_sort::sort_network_oems<N>,
//   small_sort::insertion_sort (branchless baseline).
//
// CSV columns:
//   type,algo,n,total_sorts,samples,min_ns,p50_ns,p90_ns,mean_ns,cv_pct
//
// Usage:
//   bench_small_sort                  full sweep N=2..24
//   bench_small_sort <n>              only that single N

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/small_sort.hpp"
#include "partitions/types.hpp"

// boost
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>

// cpp-sort (header-only, third_party/cpp-sort)
#include <cpp-sort/fixed/sorting_network_sorter.h>
#include <cpp-sort/fixed/low_moves_sorter.h>

using namespace partitions;

namespace {

// Tunables governing the stability/volume of every data point.
constexpr std::size_t kTargetElems   = 1u << 16;     // ~64K elements per batch
constexpr std::uint64_t kMinTotalSorts = 5'000'000;  // >= a few million / point
constexpr int kConfigs               = 8;            // independent setups / point
constexpr int kWarmupBatches         = 3;

using clk = std::chrono::steady_clock;

// Build `batch` random blocks of `n` elements, concatenated.  `narrow` shrinks
// the value range so many keys tie (exercises pair64 second-key tie-breaks and
// the branchy sorts' equal-key paths).
// Build one element of type T from a random-integer source.  For the pair
// shapes both fields draw independently so the lexicographic second-key path is
// genuinely exercised (especially in the narrow/tie-heavy config).
template <class T, class Rng, class Dist>
T make_elem(Rng& rng, Dist& dist) {
    if constexpr (std::is_same_v<T, pair64>) {
        return pair64{dist(rng), dist(rng)};
    } else if constexpr (std::is_same_v<T, pair_li>) {
        return pair_li{static_cast<long>(dist(rng)), static_cast<int>(dist(rng))};
    } else if constexpr (std::is_same_v<T, pair_fi>) {
        return pair_fi{static_cast<float>(dist(rng)), static_cast<int>(dist(rng))};
    } else if constexpr (std::is_same_v<T, pair_di>) {
        return pair_di{static_cast<double>(dist(rng)), static_cast<int>(dist(rng))};
    } else {
        return static_cast<T>(dist(rng));
    }
}

template <class T>
std::vector<T> make_master(std::size_t n, std::size_t batch, std::uint64_t seed,
                           bool narrow) {
    std::mt19937_64 rng(seed);
    const i64 hi = narrow ? 64 : (std::numeric_limits<i64>::max() / 4);
    const i64 lo = narrow ? -64 : (std::numeric_limits<i64>::min() / 4);
    std::uniform_int_distribution<i64> dist(lo, hi);
    std::vector<T> master;
    master.reserve(n * batch);
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t i = 0; i < n; ++i) master.push_back(make_elem<T>(rng, dist));
    return master;
}

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double idx = p * (sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

// One robust data point.  `sort_fn(begin, end)` sorts a single block.
template <class T, class SortFn>
void run(const char* algo, std::size_t n, SortFn&& sort_fn) {
    const std::size_t batch =
        std::max<std::size_t>(1, kTargetElems / std::max<std::size_t>(n, 1));
    const std::size_t total = n * batch;

    // Enough batches per config that all configs together exceed the floor,
    // with a minimum for decent sample counts at tiny N (large batch).
    std::uint64_t passes =
        (kMinTotalSorts / kConfigs + batch - 1) / std::max<std::size_t>(batch, 1);
    passes = std::max<std::uint64_t>(passes, 24);

    std::vector<T> work(total);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(passes) * kConfigs);
    std::uint64_t total_sorts = 0;

    auto sort_all = [&] {
        std::uintptr_t sink = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.data() + b * n;
            sort_fn(beg, beg + n);
            sink ^= reinterpret_cast<std::uintptr_t>(beg);
        }
        bench::do_not_optimize(sink);
        bench::do_not_optimize(work.data());
    };

    for (int c = 0; c < kConfigs; ++c) {
        const bool narrow = (c & 1) != 0;
        const std::uint64_t seed = 0xC0FFEEull ^ (n << 8) ^ (static_cast<std::uint64_t>(c) << 32);
        const auto master = make_master<T>(n, batch, seed, narrow);

        for (int w = 0; w < kWarmupBatches; ++w) {
            work = master;
            sort_all();
        }
        for (std::uint64_t p = 0; p < passes; ++p) {
            work = master;  // untimed restore
            const auto t0 = clk::now();
            sort_all();
            const auto t1 = clk::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            samples.push_back(ns / static_cast<double>(batch));
            total_sorts += batch;
        }
    }

    double sum = 0, sum2 = 0;
    for (double s : samples) { sum += s; sum2 += s * s; }
    const double mean = sum / samples.size();
    const double var = sum2 / samples.size() - mean * mean;
    const double cv = mean > 0 ? 100.0 * std::sqrt(std::max(0.0, var)) / mean : 0.0;
    std::sort(samples.begin(), samples.end());

    std::printf("%s,%s,%zu,%llu,%zu,%.3f,%.3f,%.3f,%.3f,%.2f\n", type_name<T>(),
                algo, n, static_cast<unsigned long long>(total_sorts),
                samples.size(), samples.front(), percentile(samples, 0.50),
                percentile(samples, 0.90), mean, cv);
    std::fflush(stdout);
}

template <class T, std::size_t N>
void run_one_size() {
    std::fprintf(stderr, "  %s N=%zu ...\n", type_name<T>(), N);
    run<T>("std::sort", N, [](T* a, T* b) { std::sort(a, b); });
    run<T>("boost::pdqsort", N, [](T* a, T* b) { boost::sort::pdqsort(a, b); });
    if constexpr (std::is_same_v<T, i64>) {
        run<T>("boost::spreadsort", N, [](T* a, T* b) {
            boost::sort::spreadsort::integer_sort(a, b);
        });
    }
    run<T>("cppsort::network", N, [](T* a, T* b) {
        cppsort::sorting_network_sorter<N>{}(a, b, std::less<>{}, std::identity{});
    });
    run<T>("cppsort::low_moves", N, [](T* a, T* b) {
        cppsort::low_moves_sorter<N>{}(a, b, std::less<>{}, std::identity{});
    });
    run<T>("small_sort::best", N, [](T* a, T* b) {
        (void)b;
        small_sort::sort_n<N>(a, std::less<>{}, std::identity{});
    });
    run<T>("small_sort::oems", N, [](T* a, T* b) {
        (void)b;
        small_sort::sort_network_oems<N>(a, std::less<>{}, std::identity{});
    });
    run<T>("small_sort::insertion", N, [](T* a, T* b) {
        small_sort::insertion_sort(a, b, std::less<>{}, std::identity{});
    });
}

template <class T, std::size_t... Ns>
void run_type_impl(std::index_sequence<Ns...>, std::size_t only_n) {
    ((only_n == 0 || only_n == Ns + 2 ? run_one_size<T, Ns + 2>() : void()), ...);
}

template <class T>
void run_type(std::size_t only_n) {
    run_type_impl<T>(std::make_index_sequence<23>{}, only_n);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t only_n = 0;
    if (argc > 1) only_n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,algo,n,total_sorts,samples,min_ns,p50_ns,p90_ns,mean_ns,cv_pct\n");
    run_type<i64>(only_n);
    run_type<pair64>(only_n);     // pair<long,long>, 16B
    run_type<pair_li>(only_n);    // pair<long,int>,  16B, narrow second key
    run_type<pair_fi>(only_n);    // pair<float,int>,  8B
    run_type<pair_di>(only_n);    // pair<double,int>, 16B, floating first key
    return 0;
}
