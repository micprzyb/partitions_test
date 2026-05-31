// Partition throughput benchmark.
//
// Sweeps {element type} x {distribution} x {block size} x {algorithm} and
// prints CSV to stdout.  A fixed median-of-3 pivot (selected once, outside the
// timed region) is used so this measures the *partition step in isolation*;
// pivot-selection cost is benchmarked separately in bench_pivot.cpp.
//
// Small blocks (4..24 elements) are far below clock resolution, so many blocks
// are batched into a single timed call and the time is normalised per element.
//
// Usage:
//   bench_partition            full sweep, sizes 4 .. 2^22
//   bench_partition quick      sizes capped at 4096 (fast smoke run)
//   bench_partition <maxsize>  cap the largest block size at <maxsize>
//
// CSV columns:
//   type,distribution,algorithm,n,batch,reps,median_ns_total,min_ns_total,ns_per_elem

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"

using namespace partitions;

namespace {

// Distributions exercised by default.  Edit freely; this is intentionally a
// representative subset rather than the full set, to keep a default run quick.
inline auto bench_distributions() {
    return std::tuple{dist::random_uniform{}, dist::few_unique{},
                      dist::all_equal{},       dist::sorted_ascending{},
                      dist::sorted_descending{}, dist::organ_pipe{},
                      dist::median_of_3_killer{}};
}

constexpr std::size_t kAllSizes[] = {4,      8,       16,      24,
                                     64,     256,     1024,    4096,
                                     1u << 16, 1u << 18, 1u << 20, 1u << 22};

// Target number of elements touched per timed iteration (sets the batch).
constexpr std::size_t kElemsPerIter = 1u << 16;

template <class T, class Dist, class Alg>
void run_one(Dist dist, Alg alg, std::size_t n) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();

    const std::size_t batch = n < kElemsPerIter ? kElemsPerIter / n : 1;
    const std::size_t total = n * batch;

    // Build `batch` independent blocks concatenated into one master buffer.
    std::vector<T> master;
    master.reserve(total);
    std::vector<std::size_t> pivot_idx(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        auto block = dist.template operator()<T>(n, 0x1234u + b + n);
        auto it = pivot::median_of_3{}(block.begin(), block.end(), comp, proj);
        pivot_idx[b] = static_cast<std::size_t>(it - block.begin());
        master.insert(master.end(), block.begin(), block.end());
    }

    std::vector<T> work(total);

    // reps: more for small blocks, fewer for the multi-megabyte ones.
    std::uint64_t reps = static_cast<std::uint64_t>((1u << 22) / total);
    reps = std::min<std::uint64_t>(std::max<std::uint64_t>(reps, 5), 200);

    auto setup = [&] { work = master; };
    auto do_work = [&] {
        std::ptrdiff_t sink = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            auto p = partition_by_position(alg, beg, beg + static_cast<std::ptrdiff_t>(n),
                                           beg + static_cast<std::ptrdiff_t>(pivot_idx[b]),
                                           comp, proj);
            sink += p - beg;
        }
        bench::do_not_optimize(sink);
    };

    auto res = bench::measure(reps, setup, do_work);
    const double ns_per_elem = res.median_ns / static_cast<double>(total);
    std::printf("%s,%s,%s,%zu,%zu,%llu,%.1f,%.1f,%.4f\n", type_name<T>(),
                dist.name, alg.name, n, batch, (unsigned long long)res.reps,
                res.median_ns, res.min_ns, ns_per_elem);
    std::fflush(stdout);
}

template <class T>
void run_type(std::size_t max_size) {
    for_each(bench_distributions(), [&](auto d) {
        for (std::size_t n : kAllSizes) {
            if (n > max_size) continue;
            for_each(default_partitioners(), [&](auto alg) { run_one<T>(d, alg, n); });
        }
    });
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 22;
    if (argc > 1) {
        if (std::strcmp(argv[1], "quick") == 0)
            max_size = 4096;
        else
            max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }

    std::printf("type,distribution,algorithm,n,batch,reps,median_ns_total,min_ns_total,ns_per_elem\n");
    run_type<i32>(max_size);
    run_type<i64>(max_size);
    run_type<pair64>(max_size);
    return 0;
}
