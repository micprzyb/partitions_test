// Pure-partition quicksort benchmark: size sweep (to very large n), distribution
// sweep, and the size-adaptive design "full picture", vs boost::pdqsort/std::sort.
//
// Run the binary several times and take the median per row (see the reproduce
// note) for accurate, layout/frequency-noise-robust numbers.
//
// CSV: type,dist,n,algo,ns_per_elem

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#if __has_include(<boost/sort/sort.hpp>)
#include <boost/sort/sort.hpp>
#define HAVE_BOOST 1
#endif

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/quicksort.hpp"
#include "partitions/small_halve.hpp"

using namespace partitions;

namespace {

template <class It, class Comp, class Proj>
void halver_sort(It first, It last, Comp comp, Proj proj) {
    const auto n = last - first;
    if (n <= 1) return;
    It mid = small_halve::halve(first, last, comp, proj);
    halver_sort(first, mid, comp, proj);
    halver_sort(mid, last, comp, proj);
}
struct piv_m3 {
    static constexpr const char* name = "m3";
    template <class I, class C, class P> I operator()(I f, I l, C c, P p) const { return pivot::median_of_3{}(f, l, c, p); }
};
template <std::ptrdiff_t PT>
struct piv_adapt {
    static constexpr const char* name = "adapt";
    template <class I, class C, class P> I operator()(I f, I l, C c, P p) const {
        return (l - f) > PT ? pivot::ninther{}(f, l, c, p) : pivot::median_of_3{}(f, l, c, p);
    }
};
template <int HalveT, class Piv, class Part, class It, class Comp, class Proj>
void qsort(It first, It last, Piv piv, Part part, Comp comp, Proj proj) {
    while (last - first > 1) {
        if constexpr (HalveT > 0)
            if (last - first <= HalveT) { halver_sort(first, last, comp, proj); return; }
        It p = piv(first, last, comp, proj);
        std::iter_swap(first, p);
        auto key = std::invoke(proj, *first);
        It m = part(first + 1, last, key, comp, proj);
        std::iter_swap(first, m - 1);
        It pp = m - 1;
        if (pp - first < last - (pp + 1)) { qsort<HalveT>(first, pp, piv, part, comp, proj); first = pp + 1; }
        else { qsort<HalveT>(pp + 1, last, piv, part, comp, proj); last = pp; }
    }
}

struct first_key {
    template <class P> auto operator()(const P& p) const { return p.first; }
};
template <class T, class Proj>
bool is_ok(const std::vector<T>& v, Proj proj) {
    auto comp = std::less<>{};
    return std::is_sorted(v.begin(), v.end(),
                          [&](const T& a, const T& b) { return comp(std::invoke(proj, a), std::invoke(proj, b)); });
}

template <class T, class Proj, class SortFn>
void timeit(const char* tn, const char* dn, std::size_t n, const char* algo,
            Proj proj, const std::vector<T>& master, SortFn sortfn) {
    std::vector<T> work = master;
    sortfn(work);
    if (!is_ok(work, proj)) { std::fprintf(stderr, "NOT SORTED %s/%s/%s\n", tn, dn, algo); std::abort(); }
    // fewer reps/warmup for very large n (rely on outer repeats for accuracy)
    std::uint64_t reps = n >= (1u << 23) ? 3 : std::min<std::uint64_t>(std::max<std::uint64_t>((1u << 23) / n, 5), 60);
    std::uint64_t warmup = n >= (1u << 23) ? 1 : 3;
    auto setup = [&] { work = master; };
    auto do_work = [&] { sortfn(work); bench::do_not_optimize(work[0]); };
    auto r = bench::measure(reps, setup, do_work, warmup);
    std::printf("%s,%s,%zu,%s,%.4f\n", tn, dn, n, algo, r.min_ns / static_cast<double>(n));
    std::fflush(stdout);
}

// The three "library + references" sorts for a given (type, master).
template <class T, class Proj>
void run_main(const char* tn, const char* dn, Proj proj, const std::vector<T>& master, std::size_t n) {
    auto comp = std::less<>{};
    timeit<T>(tn, dn, n, "quicksort", proj, master,
              [&](std::vector<T>& w) { quicksort(w.begin(), w.end(), comp, proj); });
#ifdef HAVE_BOOST
    timeit<T>(tn, dn, n, "boost_pdqsort", proj, master, [&](std::vector<T>& w) {
        if constexpr (std::is_same_v<Proj, std::identity>) boost::sort::pdqsort(w.begin(), w.end());
        else boost::sort::pdqsort(w.begin(), w.end(), [&](const T& a, const T& b) { return comp(std::invoke(proj, a), std::invoke(proj, b)); });
    });
#endif
    timeit<T>(tn, dn, n, "std_sort", proj, master, [&](std::vector<T>& w) {
        if constexpr (std::is_same_v<Proj, std::identity>) std::sort(w.begin(), w.end());
        else std::sort(w.begin(), w.end(), [&](const T& a, const T& b) { return comp(std::invoke(proj, a), std::invoke(proj, b)); });
    });
}

template <class T, class Proj>
void run_type(const char* tname, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    // ---- 1. full-picture incremental design sweep (2^22, random_uniform) ----
    {
        std::size_t n = 1u << 22;
        if (n <= max_size) {
            auto m = dist::random_uniform{}.template operator()<T>(n, 0x9501Du + n);
            auto QS = [&](auto piv, auto part, auto T_ic) {
                constexpr int H = decltype(T_ic)::value;
                return [&, piv, part](std::vector<T>& w) { qsort<H>(w.begin(), w.end(), piv, part, comp, proj); };
            };
            timeit<T>(tname, "random_uniform", n, "design_pure_m3_boost_T0", proj, m, QS(piv_m3{}, algo::boost_block{}, std::integral_constant<int,0>{}));
            timeit<T>(tname, "random_uniform", n, "design_halver_T24", proj, m, QS(piv_m3{}, algo::boost_block{}, std::integral_constant<int,24>{}));
            timeit<T>(tname, "random_uniform", n, "design_sized_T16", proj, m, QS(piv_m3{}, algo::sized{}, std::integral_constant<int,16>{}));
            timeit<T>(tname, "random_uniform", n, "design_adaptive_T16", proj, m, QS(piv_adapt<2048>{}, algo::sized{}, std::integral_constant<int,16>{}));
        }
    }
    // ---- 2. size sweep, random_uniform, up to very large n ----
    for (std::size_t n : {std::size_t(1u << 22), std::size_t(1u << 24), std::size_t(1u << 26)}) {
        if (n > max_size) continue;
        auto m = dist::random_uniform{}.template operator()<T>(n, 0x9501Du + n);
        run_main<T>(tname, "random_uniform", proj, m, n);
    }
    // ---- 3. distribution sweep at 2^22 (shows pattern handling) ----
    {
        std::size_t n = 1u << 22;
        if (n <= max_size) {
            auto do_dist = [&](const char* dn, auto d) { run_main<T>(tname, dn, proj, d.template operator()<T>(n, 0x9501Du + n), n); };
            do_dist("sorted_ascending", dist::sorted_ascending{});
            do_dist("sorted_descending", dist::sorted_descending{});
            do_dist("few_unique", dist::few_unique{});
            do_dist("organ_pipe", dist::organ_pipe{});
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 26;
    if (argc > 1) max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,dist,n,algo,ns_per_elem\n");
    run_type<i64>("i64", std::identity{}, max_size);
    run_type<pair64>("pair64", std::identity{}, max_size);
    run_type<pair64>("pair64f", first_key{}, max_size);
    return 0;
}
