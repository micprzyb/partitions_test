// Benchmark: move ~half of the below-key smallest elements to the END of the
// array.
//
// SPEC.  Given array A[0,n) and a `key`, let S = #{x : comp(proj(x), key)} (the
// elements "below" the key).  Rearrange A so the LAST k positions hold the k
// SMALLEST elements (any order) -- those k smallest are exactly the lower half
// of the below-key elements (all < key) -- with k APPROXIMATELY S/2.  Base case:
// if the target half S/2 < 16, take ALL S below-key elements instead (k = S).
// Order within the two parts is irrelevant.
//
// IDEAS BENCHMARKED (all reuse the repo's branchless `algo::sized` partition and
// the inline `detail::ninther_pos` pivot):
//   ref          : oracle -- count S, std::nth_element for rank k, swap to end.
//   count_select : count S, quickselect rank k over the WHOLE array, swap.
//   key_select   : (Approach 2, EXACT) partition by key -> S, quickselect rank k
//                  over the below-key SUBSET only, swap.  Best exact; fastest at
//                  LOW key percentile (small select).
//   key_one_part : partition by key, then ONE median(ninther)-partition of the
//                  remainder.  REJECTED: the ninther of a *partition-structured*
//                  array is a biased median -> k/S swings 0.35-0.83.
//   sample       : (the WINNER for "fast + k~=S/2") estimate the (S/2)-th value
//                  from a stride sample of the ORIGINAL (unsorted, so UNBIASED)
//                  array, then ONE partition around it.  ~n for ANY percentile,
//                  k/S within ~+-4% of 0.5.
//   part_until   : (Approach 1, the fixed example) descend-left partitioning
//                  while the pivot >= key, take the smaller part once the pivot
//                  drops below key (or the range is small) -- fast at HIGH p, but
//                  APPROXIMATE k (k/S ~ 0.6-0.74, biased high).
//
// The two winners are promoted to include/partitions/move_low_half.hpp:
//   move_low_half       == `sample`     (fast, k ~= S/2)
//   move_low_half_exact == `key_select` (exact k = floor(S/2))
//
// We verify each output (bottom-k property + multiset) and report k/S (how close
// to 0.5) alongside ns/elem.  The crucial axis is the KEY PERCENTILE p (= S/n).
//
// CSV: type,dist,p,n,algo,ns_per_elem,k_over_S

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <type_traits>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/move_low_half.hpp"  // the promoted winners (exercised below)
#include "partitions/quicksort.hpp"
#include "partitions/quicksort_lr.hpp"  // detail::ninther_pos (inline branchless pivot)
#include "partitions/small_sort.hpp"

using namespace partitions;

namespace {

struct first_key {
    template <class P>
    auto operator()(const P& p) const {
        return p.first;
    }
};

// Fair high-cardinality data for the by-first pair case (see bench_quicksort_lr).
template <class T, class Dist, class Proj>
std::vector<T> gen_data(Dist d, std::size_t n, std::uint64_t seed, Proj) {
    if constexpr (std::is_same_v<T, pair64> &&
                  !std::is_same_v<std::decay_t<Proj>, std::identity>) {
        const auto ranks = d.template operator()<i64>(n, seed);
        std::vector<pair64> out(n);
        for (std::size_t i = 0; i < n; ++i) out[i] = pair64{ranks[i], 0};
        return out;
    } else {
        return d.template operator()<T>(n, seed);
    }
}

// Quickselect: reorder [first,last) so [first,nth) <= *nth <= [nth,last).
template <class It, class Comp, class Proj>
void quickselect(It first, It last, It nth, Comp comp, Proj proj) {
    while (last - first > 24) {
        It p = detail::ninther_pos(first, last, comp, proj);
        std::iter_swap(first, p);
        auto pk = std::invoke(proj, *first);
        It m = algo::sized{}(first + 1, last, pk, comp, proj);
        std::iter_swap(first, m - 1);
        It pp = m - 1;
        if (nth == pp) return;
        if (nth < pp) last = pp; else first = pp + 1;
    }
    small_sort::sort(first, last, comp, proj);
}

// Move the front-k (a valid bottom-k) to the last k positions.  Order-agnostic,
// O(min(k, n-k)); handles k > n/2 without overlap.
template <class It>
void move_front_to_end(It a, It end, std::ptrdiff_t k) {
    const std::ptrdiff_t n = end - a;
    if (k <= 0 || k >= n) return;
    if (k <= n - k) std::swap_ranges(a, a + k, end - k);
    else std::swap_ranges(a, a + (n - k), a + k);
}

// Target count given the below-key count S: ~half, but all if the half is tiny.
inline std::ptrdiff_t target_k(std::ptrdiff_t S) { return (S / 2 < 16) ? S : S / 2; }

template <class It, class K, class Comp, class Proj>
std::ptrdiff_t count_below(It a, It end, K key, Comp comp, Proj proj) {
    std::ptrdiff_t S = 0;
    for (It it = a; it != end; ++it)
        S += static_cast<bool>(std::invoke(comp, std::invoke(proj, *it), key));
    return S;
}

// ---- ref (oracle): count + std::nth_element + move ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_ref(It a, It end, K key, Comp comp, Proj proj) {
    const auto k = target_k(count_below(a, end, key, comp, proj));
    if (k > 0 && k < end - a)
        std::nth_element(a, a + k, end, [&](const auto& x, const auto& y) {
            return static_cast<bool>(
                std::invoke(comp, std::invoke(proj, x), std::invoke(proj, y)));
        });
    move_front_to_end(a, end, k);
    return k;
}

// ---- count_select (idea A): count + quickselect over whole + move ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_count_select(It a, It end, K key, Comp comp, Proj proj) {
    const auto k = target_k(count_below(a, end, key, comp, proj));
    if (k > 0 && k < end - a) quickselect(a, end, a + k, comp, proj);
    move_front_to_end(a, end, k);
    return k;
}

// ---- key_select (Approach 2): partition by key + quickselect on below-key + move ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_key_select(It a, It end, K key, Comp comp, Proj proj) {
    It m = algo::sized{}(a, end, key, comp, proj);  // [a, m) < key
    const std::ptrdiff_t S = m - a;
    const auto k = target_k(S);
    if (k > 0 && k < S) quickselect(a, m, a + k, comp, proj);  // bottom k of below-key
    move_front_to_end(a, end, k);
    return k;
}

// ---- key_one_part (Approach 2, faithful): partition by key, then ONE median-
// partition of the below-key remainder (no full select) -> accurate k~S/2, fast ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_key_one_part(It a, It end, K key, Comp comp, Proj proj) {
    It m = algo::sized{}(a, end, key, comp, proj);  // [a, m) < key
    const std::ptrdiff_t S = m - a;
    if (S / 2 < 16) { move_front_to_end(a, end, S); return S; }  // take all (small)
    // one ninther (~median) partition of the below-key part -> bottom ~S/2 at front
    It p = detail::ninther_pos(a, m, comp, proj);
    std::iter_swap(a, p);
    auto val = std::invoke(proj, *a);
    It mid = algo::sized{}(a + 1, m, val, comp, proj);
    std::iter_swap(a, mid - 1);     // pivot to its rank; [a, mid-1) < val
    const std::ptrdiff_t k = (mid - 1) - a;
    move_front_to_end(a, end, k);
    return k;
}

// ---- sample (Floyd-Rivest flavour): estimate the (S/2)-th value from a sample
// of the ORIGINAL (unsorted -> unbiased) array, then ONE partition around it ----
template <class It, class K2, class Comp, class Proj>
std::ptrdiff_t v_sample(It a, It end, K2 key, Comp comp, Proj proj) {
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *a))>;
    const std::ptrdiff_t n = end - a;
    constexpr int M = 512;
    if (n <= 256) return v_key_select(a, end, key, comp, proj);  // small: just be exact
    const int m = static_cast<int>(std::min<std::ptrdiff_t>(n, M));
    const std::ptrdiff_t stride = n / m;
    K samp[M];
    std::ptrdiff_t below = 0;
    for (int i = 0; i < m; ++i) {
        samp[i] = std::invoke(proj, a[i * stride]);
        below += static_cast<bool>(std::invoke(comp, samp[i], key));
    }
    if (below < 16) return v_key_select(a, end, key, comp, proj);  // S small: exact handles base case
    const int r = static_cast<int>(below / 2);  // sample rank ~ the (S/2)-th overall
    std::nth_element(samp, samp + r, samp + m, [&](const K& x, const K& y) {
        return static_cast<bool>(std::invoke(comp, x, y));
    });
    const K t = samp[r];                            // estimate of the (S/2)-th value (< key)
    It mid = algo::sized{}(a, end, t, comp, proj);  // [a, mid) < t  (the bottom ~S/2)
    const std::ptrdiff_t k = mid - a;
    move_front_to_end(a, end, k);
    return k;
}

// ---- part_until (Approach 1, fixed example): descend-left, take the smaller part ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_part_until(It a, It end, K key, Comp comp, Proj proj) {
    It lo = a, hi = end;  // lo stays == a: we only ever narrow the right end.
    while (true) {
        const auto n = hi - lo;
        if (n <= 32) {
            // Base: the residual holds ALL S below-key (none were discarded, the
            // descent only drops >= key); take all of them (S is small here).
            It m = algo::sized{}(lo, hi, key, comp, proj);  // [a, m) < key
            move_front_to_end(a, end, m - a);
            return m - a;
        }
        It p = detail::ninther_pos(lo, hi, comp, proj);
        std::iter_swap(lo, p);
        auto val = std::invoke(proj, *lo);
        It m = algo::sized{}(lo + 1, hi, val, comp, proj);
        std::iter_swap(lo, m - 1);
        It mid = m - 1;  // [a, mid) < val, *mid == val, [mid+1, hi) >= val
        if (static_cast<bool>(std::invoke(comp, val, key))) {  // val < key
            // [a, mid] <= val < key are the smallest mid+1 elements; take them.
            move_front_to_end(a, end, (mid + 1) - a);
            return (mid + 1) - a;
        }
        hi = mid;  // val >= key: every below-key element is in [a, mid); descend left
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

template <class T>
bool same_multiset(std::vector<T> x, std::vector<T> y) {
    auto less = [](const T& a, const T& b) { return std::less<>{}(a, b); };
    std::sort(x.begin(), x.end(), less);
    std::sort(y.begin(), y.end(), less);
    return x == y;
}

// The key whose rank is ~p*n (so ~p of the elements are below it).
template <class T, class Comp, class Proj>
auto percentile_key(const std::vector<T>& data, double p, Comp comp, Proj proj) {
    using K = std::remove_cvref_t<decltype(std::invoke(proj, data[0]))>;
    std::vector<K> keys;
    keys.reserve(data.size());
    for (const auto& x : data) keys.push_back(std::invoke(proj, x));
    std::size_t idx = std::min<std::size_t>(
        static_cast<std::size_t>(p * static_cast<double>(data.size())), data.size() - 1);
    std::nth_element(keys.begin(), keys.begin() + idx, keys.end(), comp);
    return keys[idx];
}

template <class T, class K, class Comp, class Proj>
bool check(const std::vector<T>& after, const std::vector<T>& before, std::ptrdiff_t k,
           K key, Comp comp, Proj proj) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(after.size());
    if (!same_multiset(after, before)) return false;
    if (k < 0 || k > n) return false;
    if (k == 0 || k == n) return true;
    // bottom-k: every element in the last k must be <= every element in the
    // first n-k (by projected key).  Check max(last) <= min(first).
    auto kof = [&](const T& x) { return std::invoke(proj, x); };
    auto maxlast = kof(after[n - k]);
    for (std::ptrdiff_t i = n - k + 1; i < n; ++i)
        if (static_cast<bool>(comp(maxlast, kof(after[i])))) maxlast = kof(after[i]);
    auto minfirst = kof(after[0]);
    for (std::ptrdiff_t i = 1; i < n - k; ++i)
        if (static_cast<bool>(comp(kof(after[i]), minfirst))) minfirst = kof(after[i]);
    return !static_cast<bool>(comp(minfirst, maxlast));  // minfirst >= maxlast
}

template <class T, class K, class Proj, class Fn>
void run_one(const char* tn, const char* dn, double p, std::size_t n, const char* algo,
             K key, const std::vector<T>& master, Proj proj, std::ptrdiff_t S, Fn fn) {
    auto comp = std::less<>{};
    std::vector<T> work = master;
    std::ptrdiff_t k = fn(work);
    if (!check(work, master, k, key, comp, proj)) {
        std::fprintf(stderr, "WRONG %s/%s/p=%.2f/n=%zu/%s (k=%td)\n", tn, dn, p, n, algo, k);
        std::abort();
    }
    std::uint64_t reps = n >= (1u << 21) ? 7 : std::min<std::uint64_t>(
                             std::max<std::uint64_t>((1u << 23) / n, 7), 100);
    auto setup = [&] { work = master; };
    auto do_work = [&] { std::ptrdiff_t kk = fn(work); bench::do_not_optimize(kk); bench::do_not_optimize(work[0]); };
    auto r = bench::measure(reps, setup, do_work, 3);
    double kos = S > 0 ? static_cast<double>(k) / static_cast<double>(S) : 0.0;
    std::printf("%s,%s,%.2f,%zu,%s,%.4f,%.3f\n", tn, dn, p, n, algo,
                r.min_ns / static_cast<double>(n), kos);
    std::fflush(stdout);
}

template <class T, class Proj>
void run_type(const char* tn, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    auto do_dist = [&](const char* dn, auto dist) {
        for (double p : {0.10, 0.50, 0.90}) {
            for (std::size_t n : {std::size_t(1u << 16), std::size_t(1u << 18),
                                  std::size_t(1u << 20), std::size_t(1u << 22)}) {
                if (n > max_size) continue;
                auto master = gen_data<T>(dist, n, 0xA11CEu + n, proj);
                auto key = percentile_key(master, p, comp, proj);
                std::ptrdiff_t S = count_below(master.begin(), master.end(), key, comp, proj);
                auto R = [&](const char* nm, auto fn) {
                    run_one<T>(tn, dn, p, n, nm, key, master, proj, S,
                               [&](std::vector<T>& w) { return fn(w.begin(), w.end(), key, comp, proj); });
                };
                R("ref",          [](auto a, auto e, auto k, auto c, auto pr) { return v_ref(a, e, k, c, pr); });
                R("count_select", [](auto a, auto e, auto k, auto c, auto pr) { return v_count_select(a, e, k, c, pr); });
                // key_select and sample call the PROMOTED header functions
                // (move_low_half_exact / move_low_half) so the shipped code is
                // what's benchmarked and cannot drift from this study.
                R("key_select",   [](auto a, auto e, auto k, auto c, auto pr) { return move_low_half_exact(a, e, k, c, pr); });
                R("key_one_part", [](auto a, auto e, auto k, auto c, auto pr) { return v_key_one_part(a, e, k, c, pr); });
                R("sample",       [](auto a, auto e, auto k, auto c, auto pr) { return move_low_half(a, e, k, c, pr); });
                R("part_until",   [](auto a, auto e, auto k, auto c, auto pr) { return v_part_until(a, e, k, c, pr); });
            }
        }
    };
    do_dist("random_uniform", dist::random_uniform{});
    do_dist("few_unique", dist::few_unique{});
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t max_size = 1u << 22;
    if (argc > 1) max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,dist,p,n,algo,ns_per_elem,k_over_S\n");
    run_type<i64>("i64", std::identity{}, max_size);
    run_type<pair64>("pair64", std::identity{}, max_size);
    run_type<pair64>("pair64f", first_key{}, max_size);
    return 0;
}
