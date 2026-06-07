// Quickselect-style partitioner benchmark.
//
// Quickselect finds the k-th element by repeatedly partitioning around a pivot
// and recursing into the ONE side that contains k.  Its total work is ~2n and is
// almost entirely PARTITIONING (no leaf-sort dominance, no two-sided recursion),
// so it is a clean way to measure a partitioner in a realistic recursive setting.
//
// The driver is templated on the partitioner; pivot selection (ninther) is the
// same for every partitioner, so the only variable is the partition step.  The
// pivot is swapped to the front and placed at the boundary afterwards, so it is
// EXCLUDED from the recursion -- guaranteeing progress even with many duplicates.
//
// We select the median (k = n/2) and report total quickselect ns / element, for
// i64, pair64 (lex), pair64f (compare .first only), across large n.  A correctness
// check against std::nth_element runs first.
//
// Usage: bench_quickselect [maxsize]
// CSV: type,n,partitioner,ns_per_elem

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/small_sort.hpp"

using namespace partitions;

namespace {

// Quickselect on [first,last) for the element of rank (nth-first), using `part`
// as the partition primitive (its value-pivot operator()).
template <class Part, class It, class Comp, class Proj>
void quickselect(It first, It last, It nth, Part part, Comp comp, Proj proj) {
    while (last - first > 24) {
        It p = pivot::ninther{}(first, last, comp, proj);  // pivot position
        std::iter_swap(first, p);                          // pivot to front
        auto key = std::invoke(proj, *first);
        It m = part(first + 1, last, key, comp, proj);     // partition the tail by value
        std::iter_swap(first, m - 1);                      // place pivot at its rank
        It pp = m - 1;
        if (nth == pp) return;
        if (nth < pp) last = pp;
        else first = pp + 1;                               // recurse one side, pivot excluded
    }
    small_sort::sort(first, last, comp, proj);             // finish the small remainder
}

struct first_key {
    template <class P>
    auto operator()(const P& p) const { return p.first; }
};

template <class T>
std::vector<T> gen(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int64_t> d(0, static_cast<std::int64_t>(n));
    std::vector<T> v(n);
    for (auto& x : v) {
        std::int64_t k = d(rng);
        if constexpr (std::is_same_v<T, pair64>) x = pair64{k, d(rng)};
        else x = static_cast<T>(k);
    }
    return v;
}

template <class T, class Proj, class Part>
void run_one(const char* tname, Proj proj, Part part, std::size_t n) {
    auto comp = std::less<>{};
    auto master = gen<T>(n, 0x51E1Eu + n);
    const std::size_t k = n / 2;
    std::vector<T> work(n);

    // correctness vs std::nth_element (median value must match a full sort)
    {
        work = master;
        quickselect(work.begin(), work.end(), work.begin() + k, part, comp, proj);
        auto ref = master;
        std::nth_element(ref.begin(), ref.begin() + k, ref.end(),
                         [&](const T& a, const T& b) {
                             return comp(std::invoke(proj, a), std::invoke(proj, b));
                         });
        if (!(std::invoke(proj, work[k]) == std::invoke(proj, ref[k]))) {
            std::fprintf(stderr, "CORRECTNESS FAIL %s/%s n=%zu\n", tname, part.name, n);
            std::abort();
        }
    }

    std::uint64_t reps = static_cast<std::uint64_t>((1u << 24) / n);
    reps = std::min<std::uint64_t>(std::max<std::uint64_t>(reps, 5), 100);
    auto setup = [&] { work = master; };
    auto do_work = [&] {
        quickselect(work.begin(), work.end(), work.begin() + k, part, comp, proj);
        bench::do_not_optimize(work[k]);
    };
    auto r = bench::measure(reps, setup, do_work);
    std::printf("%s,%zu,%s,%.4f\n", tname, n, part.name,
                r.min_ns / static_cast<double>(n));
    std::fflush(stdout);
}

constexpr std::size_t kSizes[] = {1u << 16, 1u << 18, 1u << 20, 1u << 22};

template <class T, class Proj>
void run_type(const char* tname, Proj proj, std::size_t max_size) {
    for (std::size_t n : kSizes) {
        if (n > max_size) continue;
        run_one<T>(tname, proj, algo::hoare{}, n);
        run_one<T>(tname, proj, algo::lomuto_branchless{}, n);
        run_one<T>(tname, proj, algo::block{}, n);
        run_one<T>(tname, proj, algo::boost_block{}, n);
        run_one<T>(tname, proj, algo::fulcrum{}, n);
        run_one<T>(tname, proj, algo::sized{}, n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 22;
    if (argc > 1) max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,n,partitioner,ns_per_elem\n");
    run_type<i64>("i64", std::identity{}, max_size);
    run_type<pair64>("pair64", std::identity{}, max_size);
    run_type<pair64>("pair64f", first_key{}, max_size);
    return 0;
}
