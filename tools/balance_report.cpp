// Pivot-balance report.
//
// For each {distribution, size, pivot strategy} it runs many trials (varying
// the seed) and aggregates the balance of the resulting partition -- the
// fraction of the block strictly smaller than the chosen pivot.  This answers
// "how well balanced a partition would this pivot give?" independently of any
// partition algorithm.
//
// Key figures per row:
//   mean_frac   average left-side share (0.5 is ideal)
//   stddev      spread across trials (0 for deterministic input+pivot)
//   worst_side  max over trials of max(left, right) share -- the split a
//               recursive sort actually pays for; closer to 0.5 is better
//   mean_equal  average share of keys equal to the pivot (these all land on
//               the >= side and cannot be balanced by a 2-way split)
//
// Usage:
//   balance_report                 default (trials=200)
//   balance_report <trials>        set trial count
//
// Emits CSV; pipe to a file or a plotter.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "partitions/partitions.hpp"

using namespace partitions;

namespace {

constexpr std::size_t kSizes[] = {16, 24, 256, 4096, 1u << 16, 1u << 20};

template <class Dist, class Pivot>
void run_one(Dist dist, Pivot pv, std::size_t n, int trials) {
    auto comp = std::less<>{};
    auto proj = std::identity{};
    std::vector<stat::balance> samples;
    samples.reserve(static_cast<std::size_t>(trials));
    for (int t = 0; t < trials; ++t) {
        auto data = dist.template operator()<i64>(n, static_cast<std::uint64_t>(t) * 2654435761u + 1);
        auto r = pv(data.begin(), data.end(), comp, proj);
        samples.push_back(stat::measure_balance(data.begin(), data.end(),
                                                pivot::pivot_key_of(r, proj), comp, proj));
    }
    auto s = stat::summary::from(samples);
    std::printf("%s,%zu,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", dist.name, n,
                pv.name, trials, s.mean_fraction, s.stddev_fraction,
                s.min_fraction, s.max_fraction, s.worst_side, s.mean_equal_fraction);
}

}  // namespace

int main(int argc, char** argv) {
    int trials = 200;
    if (argc > 1) trials = std::atoi(argv[1]);
    if (trials < 1) trials = 1;

    std::printf("distribution,n,strategy,trials,mean_frac,stddev,min_frac,max_frac,worst_side,mean_equal\n");
    for_each(default_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            for_each(default_pivots(), [&](auto pv) { run_one(d, pv, n, trials); });
        }
    });
    return 0;
}
