// Runtime-size small-array sort benchmark (N up to 24, length NOT known at
// compile time).  This is the realistic "I get a pointer and a length" case:
// every candidate must dispatch on a run-time length.
//
// Two regimes are reported (the `mode` column):
//
//   * mode=fixed : the length is a run-time value but constant across a data
//     point (the switch/jump-table target is perfectly predicted).  This
//     isolates the per-size sort cost and lets us see whether REGISTER BLOCKING
//     (load the block into locals, sort in registers, store back) beats sorting
//     in place through the iterator.  Comparable to the compile-time tables.
//
//   * mode=mix : each block's length is drawn at random from a range, so the
//     dispatch is genuinely unpredictable -- this includes the realistic cost
//     of the size dispatch itself.  Reported as one aggregate ns/sort over the
//     size mix (mix2_24 = uniform[2,24], mix2_8, mix9_24).
//
// Candidates (all take (T* first, T* last), std::less + identity, run-time len):
//   std::sort, boost::pdqsort,
//   cppsort::net(sw)   -- switch dispatch to cppsort::sorting_network_sorter<N>,
//   ss::inplace        -- small_sort::sort   (network on the iterator, in place),
//   ss::regblock       -- small_sort::sort_reg (register-blocked best-known net),
//   varsort            -- small_sort::varsort (libc++/AlphaDev cas + p.s.s. <=5).
//
// Stability: every data point aggregates >= ~5M sorts across 8 independent
// configurations (4 seeds x {wide, narrow} value ranges).  Reported: min / p50 /
// p90 / mean ns per sort and cv% (stddev/mean).
//
// CSV: type,mode,algo,len_lo,len_hi,total_sorts,samples,min_ns,p50_ns,p90_ns,mean_ns,cv_pct

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

#include <boost/sort/pdqsort/pdqsort.hpp>
#include <cpp-sort/fixed/sorting_network_sorter.h>

using namespace partitions;

namespace {

constexpr std::size_t kTargetElems    = 1u << 16;
constexpr std::uint64_t kMinTotalSorts = 5'000'000;
constexpr int kConfigs                = 8;
constexpr int kWarmupBatches          = 3;

using clk = std::chrono::steady_clock;

// Runtime switch dispatch to cpp-sort's fixed-size network sorter (a fair way
// to use a compile-time-size library when the length is only known at run time:
// a jump table, exactly like our own dispatch).
template <class T>
inline void cppsort_net_rt(T* a, T* b) {
    const std::size_t n = static_cast<std::size_t>(b - a);
    auto less = std::less<>{};
    auto id = std::identity{};
    switch (n) {
        case 0: case 1: return;
        case 2:  cppsort::sorting_network_sorter<2>{}(a, b, less, id);  return;
        case 3:  cppsort::sorting_network_sorter<3>{}(a, b, less, id);  return;
        case 4:  cppsort::sorting_network_sorter<4>{}(a, b, less, id);  return;
        case 5:  cppsort::sorting_network_sorter<5>{}(a, b, less, id);  return;
        case 6:  cppsort::sorting_network_sorter<6>{}(a, b, less, id);  return;
        case 7:  cppsort::sorting_network_sorter<7>{}(a, b, less, id);  return;
        case 8:  cppsort::sorting_network_sorter<8>{}(a, b, less, id);  return;
        case 9:  cppsort::sorting_network_sorter<9>{}(a, b, less, id);  return;
        case 10: cppsort::sorting_network_sorter<10>{}(a, b, less, id); return;
        case 11: cppsort::sorting_network_sorter<11>{}(a, b, less, id); return;
        case 12: cppsort::sorting_network_sorter<12>{}(a, b, less, id); return;
        case 13: cppsort::sorting_network_sorter<13>{}(a, b, less, id); return;
        case 14: cppsort::sorting_network_sorter<14>{}(a, b, less, id); return;
        case 15: cppsort::sorting_network_sorter<15>{}(a, b, less, id); return;
        case 16: cppsort::sorting_network_sorter<16>{}(a, b, less, id); return;
        case 17: cppsort::sorting_network_sorter<17>{}(a, b, less, id); return;
        case 18: cppsort::sorting_network_sorter<18>{}(a, b, less, id); return;
        case 19: cppsort::sorting_network_sorter<19>{}(a, b, less, id); return;
        case 20: cppsort::sorting_network_sorter<20>{}(a, b, less, id); return;
        case 21: cppsort::sorting_network_sorter<21>{}(a, b, less, id); return;
        case 22: cppsort::sorting_network_sorter<22>{}(a, b, less, id); return;
        case 23: cppsort::sorting_network_sorter<23>{}(a, b, less, id); return;
        case 24: cppsort::sorting_network_sorter<24>{}(a, b, less, id); return;
        default: std::sort(a, b); return;
    }
}

// A workload: a flat buffer plus (offset,len) block descriptors.
template <class T>
struct Workload {
    std::vector<T> master;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> blocks;  // (offset,len)
};

template <class T, class Rng, class Dist>
T make_elem(Rng& rng, Dist& dist) {
    if constexpr (std::is_same_v<T, pair64>)
        return pair64{dist(rng), dist(rng)};
    else if constexpr (std::is_same_v<T, pair_li>)
        return pair_li{static_cast<long>(dist(rng)), static_cast<int>(dist(rng))};
    else if constexpr (std::is_same_v<T, pair_fi>)
        return pair_fi{static_cast<float>(dist(rng)), static_cast<int>(dist(rng))};
    else if constexpr (std::is_same_v<T, pair_di>)
        return pair_di{static_cast<double>(dist(rng)), static_cast<int>(dist(rng))};
    else
        return static_cast<T>(dist(rng));
}

template <class T>
Workload<T> make_workload(std::size_t len_lo, std::size_t len_hi,
                          std::uint64_t seed, bool narrow) {
    std::mt19937_64 rng(seed);
    const i64 hi = narrow ? 64 : (std::numeric_limits<i64>::max() / 4);
    const i64 lo = narrow ? -64 : (std::numeric_limits<i64>::min() / 4);
    std::uniform_int_distribution<i64> vdist(lo, hi);
    std::uniform_int_distribution<std::size_t> ldist(len_lo, len_hi);

    Workload<T> w;
    w.master.reserve(kTargetElems + len_hi);
    std::size_t off = 0;
    while (off + len_hi <= kTargetElems) {
        const std::size_t len = (len_lo == len_hi) ? len_lo : ldist(rng);
        for (std::size_t i = 0; i < len; ++i)
            w.master.push_back(make_elem<T>(rng, vdist));
        w.blocks.emplace_back(static_cast<std::uint32_t>(off),
                              static_cast<std::uint32_t>(len));
        off += len;
    }
    return w;
}

double percentile(const std::vector<double>& s, double p) {
    if (s.empty()) return 0.0;
    const double idx = p * (s.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, s.size() - 1);
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo);
}

template <class T, class SortFn>
void run(const char* mode, const char* algo, std::size_t len_lo,
         std::size_t len_hi, SortFn&& sort_fn) {
    std::vector<Workload<T>> cfgs;
    cfgs.reserve(kConfigs);
    std::size_t blocks_per_cfg = 0;
    for (int c = 0; c < kConfigs; ++c) {
        const std::uint64_t seed =
            0x5A17ull ^ (len_lo << 8) ^ (len_hi << 16) ^ (static_cast<std::uint64_t>(c) << 32);
        cfgs.push_back(make_workload<T>(len_lo, len_hi, seed, (c & 1) != 0));
        blocks_per_cfg = std::max(blocks_per_cfg, cfgs.back().blocks.size());
    }

    std::uint64_t passes =
        (kMinTotalSorts / kConfigs + blocks_per_cfg - 1) / std::max<std::size_t>(blocks_per_cfg, 1);
    passes = std::max<std::uint64_t>(passes, 24);

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(passes) * kConfigs);
    std::uint64_t total_sorts = 0;
    std::vector<T> work;

    for (int c = 0; c < kConfigs; ++c) {
        const auto& wl = cfgs[c];
        work = wl.master;
        auto sort_all = [&] {
            std::uintptr_t sink = 0;
            for (auto [off, len] : wl.blocks) {
                T* beg = work.data() + off;
                sort_fn(beg, beg + len);
                sink ^= reinterpret_cast<std::uintptr_t>(beg);
            }
            bench::do_not_optimize(sink);
            bench::do_not_optimize(work.data());
        };
        for (int wcnt = 0; wcnt < kWarmupBatches; ++wcnt) {
            work = wl.master;
            sort_all();
        }
        for (std::uint64_t p = 0; p < passes; ++p) {
            work = wl.master;  // untimed restore
            const auto t0 = clk::now();
            sort_all();
            const auto t1 = clk::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            samples.push_back(ns / static_cast<double>(wl.blocks.size()));
            total_sorts += wl.blocks.size();
        }
    }

    double sum = 0, sum2 = 0;
    for (double s : samples) { sum += s; sum2 += s * s; }
    const double mean = sum / samples.size();
    const double var = sum2 / samples.size() - mean * mean;
    const double cv = mean > 0 ? 100.0 * std::sqrt(std::max(0.0, var)) / mean : 0.0;
    std::sort(samples.begin(), samples.end());
    std::printf("%s,%s,%s,%zu,%zu,%llu,%zu,%.3f,%.3f,%.3f,%.3f,%.2f\n",
                type_name<T>(), mode, algo, len_lo, len_hi,
                static_cast<unsigned long long>(total_sorts), samples.size(),
                samples.front(), percentile(samples, 0.50),
                percentile(samples, 0.90), mean, cv);
    std::fflush(stdout);
}

template <class T>
void all_candidates(const char* mode, std::size_t lo, std::size_t hi) {
    run<T>(mode, "std::sort", lo, hi, [](T* a, T* b) { std::sort(a, b); });
    run<T>(mode, "boost::pdqsort", lo, hi, [](T* a, T* b) { boost::sort::pdqsort(a, b); });
    run<T>(mode, "cppsort::net(sw)", lo, hi, [](T* a, T* b) { cppsort_net_rt(a, b); });
    run<T>(mode, "ss::inplace", lo, hi,
           [](T* a, T* b) { small_sort::sort(a, b, std::less<>{}, std::identity{}); });
    run<T>(mode, "ss::regblock", lo, hi,
           [](T* a, T* b) { small_sort::sort_reg(a, b, std::less<>{}, std::identity{}); });
    run<T>(mode, "varsort", lo, hi,
           [](T* a, T* b) { small_sort::varsort(a, b, std::less<>{}, std::identity{}); });
}

template <class T>
void run_type(std::size_t only_n) {
    if (only_n >= 2 && only_n <= 24) {
        all_candidates<T>("fixed", only_n, only_n);
        return;
    }
    for (std::size_t n = 2; n <= 24; ++n) all_candidates<T>("fixed", n, n);
    all_candidates<T>("mix2_24", 2, 24);
    all_candidates<T>("mix2_8", 2, 8);
    all_candidates<T>("mix9_24", 9, 24);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t only_n = 0;
    if (argc > 1) only_n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,mode,algo,len_lo,len_hi,total_sorts,samples,"
                "min_ns,p50_ns,p90_ns,mean_ns,cv_pct\n");
    run_type<i64>(only_n);
    run_type<pair64>(only_n);   // pair<long,long>, 16B
    run_type<pair_li>(only_n);  // pair<long,int>,  16B
    run_type<pair_fi>(only_n);  // pair<float,int>,  8B
    run_type<pair_di>(only_n);  // pair<double,int>, 16B, floating first key
    return 0;
}
