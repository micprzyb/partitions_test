// Reversed-partition benchmark: prove the reversed partitioner ([>=pivot | <pivot])
// is NOT slower than the forward original ([<pivot | >=pivot]), and quantify the
// three ways to obtain the reversal.
//
// SECTION 1 -- standalone partitioner (one fixed by-value pivot, batched small
// blocks; mirrors bench_partition.cpp).  Per (type, dist, n) the routes are:
//   fwd       forward algo::sized                              (baseline)
//   rev_rw    rewritten algo_rev::sized_rev (role-exchanged, strict <) (route a)
//   rev_neg   forward algo::sized + negate_comp(less)          (route b)
//   rev_view  forward algo::sized over reverse_iterators       (route c)
//
// SECTION 2 -- full quicksort (mirrors bench_quicksort.cpp):
//   qs_fwd       partitions::quicksort         (ascending baseline)
//   qs_rev_rw    partitions::quicksort_rev     (descending, route a)
//   qs_rev_view  quicksort over reverse_iters  (descending, route c)
// (route b -- quicksort + negate_comp -- is intentionally absent: negating the
//  comparator feeds the HALVER networks a reflexive `>=`, which is unsound; see
//  docs/reverse_partition_report.md.  The correctness loop here would catch it.)
//
// CSV: section,type,dist,n,route,ns_per_elem   (run several times, take medians)
//
// COST: `bench_reverse_partition quick` (<=4096) runs in seconds; the default
// (<=2^22) takes ~2-3 min/pass.  Passing maxsize >= 2^24 is EXPENSIVE -- a single
// 2^26 pass is ~50 min (the pair64 2^26 quicksort + 1 GB setup-copies dominate).
// Do not loop many 2^26 passes; settle a specific big-n cell with a targeted
// single-type/single-size micro-bench (min ns/elem) instead.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/quicksort.hpp"
#include "partitions/quicksort_rev.hpp"
#include "partitions/reverse_partition.hpp"

using namespace partitions;

namespace {

struct first_key {
    template <class P>
    auto operator()(const P& p) const { return p.first; }
};

// negate_comp, matching partition_api.hpp's reverse route.
template <class Comp>
auto negate(Comp comp) {
    return [comp](const auto& a, const auto& b) {
        return !static_cast<bool>(std::invoke(comp, a, b));
    };
}

constexpr std::size_t kSizes[] = {8,        16,       24,       64,
                                  256,      1024,     4096,     1u << 16,
                                  1u << 18, 1u << 20, 1u << 22, 1u << 24,
                                  std::size_t(1) << 26};
constexpr std::size_t kElemsPerIter = 1u << 16;

// ---- SECTION 1: standalone partitioner -----------------------------------
template <class T, class Proj, class Gen>
void run_partition(const char* tname, Proj proj, const char* dname, Gen gen,
                   std::size_t n) {
    auto comp = std::less<>{};
    const std::size_t batch = n < kElemsPerIter ? kElemsPerIter / n : 1;
    const std::size_t total = n * batch;

    std::vector<T> master;
    master.reserve(total);
    std::vector<T> keys(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        auto block = gen(n, 0x1234u + b + n);
        auto it = pivot::median_of_3{}(block.begin(), block.end(), comp, proj);
        keys[b] = *it;
        master.insert(master.end(), block.begin(), block.end());
    }
    std::vector<T> work(total);

    std::uint64_t reps = static_cast<std::uint64_t>((1u << 22) / total);
    reps = std::min<std::uint64_t>(std::max<std::uint64_t>(reps, 5), 200);
    auto setup = [&] { work = master; };

    auto emit = [&](const char* route, auto&& dowork) {
        auto res = bench::measure(reps, setup, dowork);
        std::printf("partition,%s,%s,%zu,%s,%.4f\n", tname, dname, n, route,
                    res.median_ns / static_cast<double>(total));
        std::fflush(stdout);
    };

    emit("fwd", [&] {
        std::ptrdiff_t sink = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            auto key = std::invoke(proj, keys[b]);
            auto p = algo::sized{}(beg, beg + static_cast<std::ptrdiff_t>(n), key, comp, proj);
            sink += p - beg;
        }
        bench::do_not_optimize(sink);
    });
    emit("rev_rw", [&] {
        std::ptrdiff_t sink = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            auto key = std::invoke(proj, keys[b]);
            auto p = algo_rev::sized_rev{}(beg, beg + static_cast<std::ptrdiff_t>(n), key, comp, proj);
            sink += p - beg;
        }
        bench::do_not_optimize(sink);
    });
    emit("rev_neg", [&] {
        std::ptrdiff_t sink = 0;
        auto nc = negate(comp);
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            auto key = std::invoke(proj, keys[b]);
            auto p = algo::sized{}(beg, beg + static_cast<std::ptrdiff_t>(n), key, nc, proj);
            sink += p - beg;
        }
        bench::do_not_optimize(sink);
    });
    emit("rev_view", [&] {
        std::ptrdiff_t sink = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            auto end = beg + static_cast<std::ptrdiff_t>(n);
            auto key = std::invoke(proj, keys[b]);
            auto rp = algo::sized{}(std::make_reverse_iterator(end),
                                    std::make_reverse_iterator(beg), key, comp, proj);
            sink += rp.base() - beg;
        }
        bench::do_not_optimize(sink);
    });
}

// ---- SECTION 2: full quicksort -------------------------------------------
template <class T, class Proj>
void run_quicksort(const char* tname, Proj proj, const char* dname,
                   const std::vector<T>& master, std::size_t n) {
    auto comp = std::less<>{};
    std::vector<T> work(n);
    std::uint64_t reps = n >= (1u << 23) ? 3 : std::min<std::uint64_t>(std::max<std::uint64_t>((1u << 23) / n, 5), 60);
    std::uint64_t warmup = n >= (1u << 23) ? 1 : 3;
    auto setup = [&] { work = master; };

    auto emit = [&](const char* route, auto&& dowork) {
        auto res = bench::measure(reps, setup, dowork, warmup);
        std::printf("quicksort,%s,%s,%zu,%s,%.4f\n", tname, dname, n, route,
                    res.median_ns / static_cast<double>(n));
        std::fflush(stdout);
    };

    emit("qs_fwd", [&] { quicksort(work.begin(), work.end(), comp, proj); bench::do_not_optimize(work[0]); });
    emit("qs_rev_rw", [&] { quicksort_rev(work.begin(), work.end(), comp, proj); bench::do_not_optimize(work[0]); });
    emit("qs_rev_view", [&] {
        quicksort(std::make_reverse_iterator(work.end()),
                  std::make_reverse_iterator(work.begin()), comp, proj);
        bench::do_not_optimize(work[0]);
    });
}

template <class T, class Proj>
void run_type(const char* tname, Proj proj, std::size_t max_size, bool wrap_first = false) {
    auto distros = std::tuple{dist::random_uniform{}, dist::few_unique{},
                              dist::sorted_descending{}, dist::organ_pipe{}};
    for_each(distros, [&](auto d) {
        auto gen = [&](std::size_t nn, std::uint64_t seed) -> std::vector<T> {
            if constexpr (std::is_same_v<T, pair64>) {
                if (wrap_first) {
                    auto k = d.template operator()<i64>(nn, seed);
                    std::vector<pair64> v(nn);
                    for (std::size_t i = 0; i < nn; ++i) v[i] = pair64{k[i], 0};
                    return v;
                }
            }
            return d.template operator()<T>(nn, seed);
        };
        for (std::size_t n : kSizes) {
            if (n > max_size) continue;
            run_partition<T>(tname, proj, d.name, gen, n);
        }
        // quicksort: only the larger sizes are meaningful (and not all dists)
        for (std::size_t n : {std::size_t(4096), std::size_t(1u << 20),
                              std::size_t(1u << 22), std::size_t(1u << 24),
                              std::size_t(1) << 26}) {
            if (n > max_size) continue;
            run_quicksort<T>(tname, proj, d.name, gen(n, 0x9501Du + n), n);
        }
    });
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 22;
    if (argc > 1) {
        if (std::strcmp(argv[1], "quick") == 0) max_size = 4096;
        else max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    std::printf("section,type,dist,n,route,ns_per_elem\n");
    run_type<i64>("i64", std::identity{}, max_size);
    run_type<pair64>("pair64", std::identity{}, max_size);
    run_type<pair64>("pair64f", first_key{}, max_size, /*wrap_first=*/true);
    return 0;
}
