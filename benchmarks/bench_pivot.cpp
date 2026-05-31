// Pivot-selection benchmark.
//
// For each {type, distribution, size, strategy} it times the (non-destructive)
// pivot selection and also reports the resulting balance (fraction of the
// block strictly smaller than the chosen pivot), so cost and quality appear
// side by side.  O(1) samplers (first/median-of-3/ninther) should be flat in
// n; median-of-medians is O(n) and its ns/element is the interesting figure.
//
// Usage: same as bench_partition (`quick` or a max size).
//
// CSV columns:
//   type,distribution,strategy,n,reps,median_ns,ns_per_elem,smaller_fraction

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"

using namespace partitions;

namespace {

inline auto bench_distributions() {
    return std::tuple{dist::random_uniform{}, dist::sorted_ascending{},
                      dist::organ_pipe{}, dist::median_of_3_killer{}};
}

constexpr std::size_t kSizes[] = {8,      16,      64,      256,    1024,
                                  4096,   1u << 16, 1u << 18, 1u << 20, 1u << 22};

template <class T, class Dist, class Pivot>
void run_one(Dist dist, Pivot pv, std::size_t n) {
    auto comp = std::less<>{};
    auto proj = projection_for<T>();
    const auto master = dist.template operator()<T>(n, 0xABCDu + n);
    auto data = master;

    std::uint64_t reps = static_cast<std::uint64_t>((1u << 22) / n);
    reps = std::min<std::uint64_t>(std::max<std::uint64_t>(reps, 5), 500);

    // A reordering strategy mutates `data`, so restore it before each timed
    // run (untimed); a non-destructive one needs no restore.
    auto setup = [&] {
        if constexpr (pivot::reorders_v<Pivot>) data = master;
    };
    auto do_work = [&] {
        auto r = pv(data.begin(), data.end(), comp, proj);
        bench::do_not_optimize(pivot::pivot_key_of(r, proj));
    };
    auto res = bench::measure(reps, setup, do_work);

    // Balance is permutation-invariant; measure it on a fresh selection.
    auto bcopy = master;
    auto r = pv(bcopy.begin(), bcopy.end(), comp, proj);
    auto bal = stat::measure_balance(bcopy.begin(), bcopy.end(),
                                     pivot::pivot_key_of(r, proj), comp, proj);

    std::printf("%s,%s,%s,%zu,%llu,%.1f,%.4f,%.4f\n", type_name<T>(), dist.name,
                pv.name, n, (unsigned long long)res.reps, res.median_ns,
                res.median_ns / static_cast<double>(n), bal.smaller_fraction());
    std::fflush(stdout);
}

template <class T>
void run_type(std::size_t max_size) {
    for_each(bench_distributions(), [&](auto d) {
        for (std::size_t n : kSizes) {
            if (n > max_size) continue;
            for_each(default_pivots(), [&](auto pv) { run_one<T>(d, pv, n); });
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
    std::printf("type,distribution,strategy,n,reps,median_ns,ns_per_elem,smaller_fraction\n");
    run_type<i64>(max_size);
    run_type<pair64>(max_size);
    return 0;
}
