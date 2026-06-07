// Does a SIZE-TIERED pivot (cheap for small nodes, ninther for medium, the
// expensive median-of-5-of-5 only for the few huge top nodes) beat ninther
// everywhere?  And does a bigger halver cutoff (>16) pay off?  Total sort time
// is the only criterion -- measured on random AND sorted_descending (the latter
// catches any config that reintroduces the median-of-3 killer -> O(n^2)).
//
// Partitioner fixed = algo::sized (the production choice).  CSV: type,dist,config,ns_per_elem

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
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

// pivot policies ---------------------------------------------------------------
struct P_nin {
    template <class I, class C, class P> I operator()(I f, I l, C c, P p) const {
        return pivot::ninther{}(f, l, c, p);
    }
};
// small (<=TS): m5 or m3 ; medium: ninther ; large (>TL): median_of_5_medians_of_5
template <long TS, long TL, bool SmallM5>
struct P_tier {
    template <class I, class C, class P> I operator()(I f, I l, C c, P p) const {
        auto n = l - f;
        if (n > TL) return pivot::median_of_5_medians_of_5{}(f, l, c, p);
        if (n > TS) return pivot::ninther{}(f, l, c, p);
        if constexpr (SmallM5) return pivot::median_of_5{}(f, l, c, p);
        else return pivot::median_of_3{}(f, l, c, p);
    }
};

template <int HalveT, class Piv, class It, class Comp, class Proj>
void qsort(It first, It last, Piv piv, Comp comp, Proj proj) {
    while (last - first > 1) {
        if (last - first <= HalveT) { halver_sort(first, last, comp, proj); return; }
        It pp = piv(first, last, comp, proj);
        std::iter_swap(first, pp);
        auto key = std::invoke(proj, *first);
        It m = algo::sized{}(first + 1, last, key, comp, proj);
        std::iter_swap(first, m - 1);
        It b = m - 1;
        if (b - first < last - (b + 1)) { qsort<HalveT>(first, b, piv, comp, proj); first = b + 1; }
        else { qsort<HalveT>(b + 1, last, piv, comp, proj); last = b; }
    }
}

struct first_key { template <class P> auto operator()(const P& p) const { return p.first; } };

template <class T, class Proj, class Run>
void timeit(const char* tn, const char* dn, const char* cfg, Proj proj,
            const std::vector<T>& master, Run run) {
    std::vector<T> work = master;
    run(work);
    auto comp = std::less<>{};
    if (!std::is_sorted(work.begin(), work.end(), [&](const T& a, const T& b) {
            return comp(std::invoke(proj, a), std::invoke(proj, b)); })) {
        std::fprintf(stderr, "NOT SORTED %s/%s/%s\n", tn, dn, cfg); std::abort();
    }
    const std::size_t n = master.size();
    std::uint64_t reps = std::min<std::uint64_t>(std::max<std::uint64_t>((1u << 23) / n, 4), 30);
    auto setup = [&] { work = master; };
    auto dw = [&] { run(work); bench::do_not_optimize(work[0]); };
    auto r = bench::measure(reps, setup, dw, 2);
    std::printf("%s,%s,%s,%.4f\n", tn, dn, cfg, r.min_ns / static_cast<double>(n));
    std::fflush(stdout);
}

template <class T, class Proj>
void run_type(const char* tn, Proj proj, std::size_t n) {
    auto comp = std::less<>{};
    auto mk = [&](auto dist) { return dist.template operator()<T>(n, 0x1234 + n); };
    auto rnd = mk(dist::random_uniform{});
    auto srt = mk(dist::sorted_descending{});
    auto sweep = [&](const char* dn, const std::vector<T>& m) {
        // (A) halver-cutoff sweep, pivot = ninther everywhere
        timeit<T>(tn, dn, "ninther.T8",  proj, m, [&](std::vector<T>& w){ qsort<8 >(w.begin(),w.end(),P_nin{},comp,proj); });
        timeit<T>(tn, dn, "ninther.T12", proj, m, [&](std::vector<T>& w){ qsort<12>(w.begin(),w.end(),P_nin{},comp,proj); });
        timeit<T>(tn, dn, "ninther.T16", proj, m, [&](std::vector<T>& w){ qsort<16>(w.begin(),w.end(),P_nin{},comp,proj); });
        timeit<T>(tn, dn, "ninther.T20", proj, m, [&](std::vector<T>& w){ qsort<20>(w.begin(),w.end(),P_nin{},comp,proj); });
        timeit<T>(tn, dn, "ninther.T24", proj, m, [&](std::vector<T>& w){ qsort<24>(w.begin(),w.end(),P_nin{},comp,proj); });
        // (B) tiered pivots, halver T=16, safe small tier = m5
        timeit<T>(tn, dn, "m5<=256|nin", proj, m, [&](std::vector<T>& w){ qsort<16>(w.begin(),w.end(),P_tier<256,(1L<<62),true>{},comp,proj); });
        timeit<T>(tn, dn, "nin|m5m5>4k", proj, m, [&](std::vector<T>& w){ qsort<16>(w.begin(),w.end(),P_tier<0,4096,true>{},comp,proj); });
        timeit<T>(tn, dn, "nin|m5m5>64k", proj, m, [&](std::vector<T>& w){ qsort<16>(w.begin(),w.end(),P_tier<0,65536,true>{},comp,proj); });
        timeit<T>(tn, dn, "m5<=256|nin|m5m5>64k", proj, m, [&](std::vector<T>& w){ qsort<16>(w.begin(),w.end(),P_tier<256,65536,true>{},comp,proj); });
        // (C) UNSAFE: m3 small tier (expect sorted blow-up) -- include to prove the point
        timeit<T>(tn, dn, "m3<=64|nin|m5m5>64k", proj, m, [&](std::vector<T>& w){ qsort<16>(w.begin(),w.end(),P_tier<64,65536,false>{},comp,proj); });
    };
    sweep("random_uniform", rnd);
    sweep("sorted_descending", srt);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1u << 20);
    std::printf("type,dist,config,ns_per_elem\n");
    run_type<i64>("i64", std::identity{}, n);
    run_type<pair64>("pair64", std::identity{}, n);
    run_type<pair64>("pair64f", first_key{}, n);
    return 0;
}
