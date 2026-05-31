#ifndef PARTITIONS_BENCH_HARNESS_HPP
#define PARTITIONS_BENCH_HARNESS_HPP

// A small steady-clock benchmark harness tailored to partitioning.
//
// Partitioning *mutates* its input, so each timed repetition must start from a
// fresh copy.  `measure` runs an untimed `setup()` before each timed `work()`
// repetition, so the restore cost is excluded from the measurement.  We report
// the minimum (least noisy estimate of the true cost) and the median.
//
// For very small blocks a single partition is far below clock resolution, so
// callers batch many blocks into one `work()` call and divide (see
// `report_row`'s ns/element, which is batch-agnostic).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

namespace bench {

// Prevent the optimiser from discarding a computed value / treating a buffer
// as dead.  Standard Google-Benchmark-style barriers.
template <class T>
inline void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}
inline void clobber() { asm volatile("" : : : "memory"); }

struct Result {
    double min_ns = 0;
    double median_ns = 0;
    std::uint64_t reps = 0;
};

using clock = std::chrono::steady_clock;

// Run `work` `reps` times, calling `setup` (untimed) before each.  A few warmup
// iterations prime caches / branch predictors before measurement begins.
template <class Setup, class Work>
Result measure(std::uint64_t reps, Setup setup, Work work, std::uint64_t warmup = 3) {
    for (std::uint64_t w = 0; w < warmup; ++w) {
        setup();
        work();
    }
    std::vector<double> samples;
    samples.reserve(reps);
    for (std::uint64_t r = 0; r < reps; ++r) {
        setup();
        const auto t0 = clock::now();
        work();
        const auto t1 = clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    Result res;
    res.reps = reps;
    res.min_ns = samples.front();
    res.median_ns = samples[samples.size() / 2];
    return res;
}

}  // namespace bench

#endif  // PARTITIONS_BENCH_HARNESS_HPP
