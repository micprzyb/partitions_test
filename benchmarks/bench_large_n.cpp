// Focused benchmark: pivot selection on a SINGLE LARGE block (n = 2^24).
//
// Compares three position-returning pivot strategies that sample a constant
// number of elements and combine via small median networks:
//
//   median_of_5_medians_of_3       15 samples, evenly spaced at stride n/14
//   median_of_5_medians_of_5       25 samples, evenly spaced at stride n/24
//   median_of_5_medians_of_5_898   25 samples, clustered 8+9+8 at start/middle/end
//
// At n=2^24 i64 (128 MB) the array vastly exceeds any cache level.  The
// theoretical prediction is that the clustered variant should win on cost
// (3 cache misses vs 15-25) while statistically being competitive (25
// samples > 15; same network).
//
// Benchmarking trick (per the user's suggestion): instead of regenerating the
// distribution for each trial, we generate ONE master array of size
//     N + (TRIALS - 1) * SHIFT
// and for trial k use the slice [k*SHIFT, k*SHIFT + N).  Each shift moves the
// sampled positions in the master array, so trials see "different" sample
// values without re-running the (expensive) distribution generator.  SHIFT=9
// is small enough that consecutive trials' sample positions land mostly on
// already-warmed cache lines (so warm-cache timing is meaningful) and large
// enough to give every trial a distinct random sample value for the
// random_uniform distribution.
//
// We report BOTH cold-cache cost (first call on a freshly-shifted slice) and
// warm-cache cost (per-call within a batch of 32 repeats on the same slice),
// because cold-cache cost is what matters when this pivot is invoked once per
// quicksort partition, while warm-cache cost isolates the ALU work.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kN     = 1u << 24;     // 16 777 216 elements
constexpr std::size_t kTrials = 64;
constexpr std::size_t kShift = 9;            // user-suggested shift step
constexpr std::size_t kWarmBatch = 32;

using clock_t = std::chrono::steady_clock;

template <class T, class Dist, class Pv>
void run_pivot(const std::vector<T>& master, Dist dist, Pv pv) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();

    std::vector<double> cold(kTrials), warm(kTrials), bal(kTrials);

    // Pre-touch the master array once (page-fault prepay).
    {
        volatile std::int64_t sink = 0;
        for (std::size_t i = 0; i < master.size(); i += 4096 / sizeof(T)) sink += static_cast<std::int64_t>(master[i]);
        (void)sink;
    }

    // Junk buffer for cache flushing: 64 MB is larger than any consumer L3,
    // so a full read evicts the master array from L1/L2/L3.
    static std::vector<std::int64_t> junk(8 * 1024 * 1024);

    for (std::size_t k = 0; k < kTrials; ++k) {
        auto begin = master.begin() + static_cast<std::ptrdiff_t>(k * kShift);
        auto end   = begin + static_cast<std::ptrdiff_t>(kN);

        // Flush caches: walk the 64 MB junk buffer.  This evicts ALL of the
        // master array's cache lines, giving genuinely cold-cache timing.
        {
            volatile std::int64_t s = 0;
            for (auto v : junk) s += v;
            (void)s;
        }

        // Cold: a SINGLE call after a full cache flush.  Sample reads must
        // come from DRAM (or, for the clustered strategy, page-table-walked
        // L2/L3-resident lines if the hardware prefetcher caught the
        // sequential access within each cluster).
        auto t0 = clock_t::now();
        auto r0 = pv(begin, end, comp, proj);
        auto t1 = clock_t::now();
        bench::do_not_optimize(r0);
        cold[k] = std::chrono::duration<double, std::nano>(t1 - t0).count();

        // Warm: many calls on the same slice; cache is hot after the first.
        auto t2 = clock_t::now();
        for (std::size_t j = 0; j < kWarmBatch; ++j) {
            auto r = pv(begin, end, comp, proj);
            bench::do_not_optimize(r);
        }
        auto t3 = clock_t::now();
        warm[k] = std::chrono::duration<double, std::nano>(t3 - t2).count() /
                  static_cast<double>(kWarmBatch);

        // Balance: scan the slice once to count elements strictly less.
        auto b = stat::measure_balance(begin, end,
                                       pivot::pivot_key_of(r0, proj), comp, proj);
        bal[k] = b.smaller_fraction();
    }

    auto median = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    double mean_bal = 0, worst_side = 0;
    for (double f : bal) {
        mean_bal += f;
        double s = std::max(f, 1.0 - f);
        if (s > worst_side) worst_side = s;
    }
    mean_bal /= static_cast<double>(kTrials);

    std::printf("%s,%s,%s,%.1f,%.1f,%.4f,%.4f\n",
                type_name<T>(), dist.name, pv.name,
                median(cold), median(warm), mean_bal, worst_side);
    std::fflush(stdout);
}

template <class T, class Dist>
void run_dist(Dist dist) {
    const std::size_t master_size = kN + (kTrials - 1) * kShift;
    auto master = dist.template operator()<T>(master_size, 0xA1B2C3D4u);

    run_pivot(master, dist, pivot::median_of_5_medians_of_3{});
    run_pivot(master, dist, pivot::median_of_5_medians_of_5{});
    run_pivot(master, dist, pivot::median_of_5_medians_of_5_898{});
}

}  // namespace

int main() {
    std::printf("type,distribution,strategy,cold_ns,warm_ns,mean_balance,worst_side\n");

    run_dist<i64>(dist::random_uniform{});
    run_dist<i64>(dist::sorted_ascending{});
    run_dist<i64>(dist::sorted_descending{});
    run_dist<i64>(dist::organ_pipe{});
    run_dist<i64>(dist::median_of_3_killer{});
    run_dist<i64>(dist::nearly_sorted{});
    run_dist<i64>(dist::shuffled_blocks{});

    return 0;
}
