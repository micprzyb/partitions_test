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
//   fwd_key_select : (Approach 2, EXACT, forward) partition by key -> S,
//                  quickselect rank k over the below-key SUBSET only, swap.
//   key_one_part : partition by key, then ONE median(ninther)-partition of the
//                  remainder.  REJECTED: the ninther of a *partition-structured*
//                  array is a biased median -> k/S swings 0.35-0.83.
//   fwd_sample   : estimate the (S/2)-th value from a stride sample of the
//                  ORIGINAL (unsorted, so UNBIASED) array, then ONE forward
//                  partition around it + a move-to-end pass.
//   part_until   : (Approach 1, the fixed example) descend-left partitioning
//                  while the pivot >= key, take the smaller part once the pivot
//                  drops below key (or the range is small) -- fast at HIGH p, but
//                  APPROXIMATE k (k/S ~ 0.6-0.74, biased high).
//
// REVERSED round (the shipped winners).  Every forward variant above ends with
// `move_front_to_end` -- min(k, n-k) extra swaps that exist only because the
// partition puts the bottom set at the WRONG end.  Building on the reversed
// partition family (algo_rev::sized_rev, >= left | < right) the bottom set
// lands at the END as a side effect of the one partition pass, and the exact
// path's quickselect runs reversed (detail::quickselect_rev) so the answer is
// in place when the recursion stops:
//   sample         : sample + ONE reversed partition.  No move.  (shipped
//                    move_low_half; the sample rank is found with the repo
//                    quickselect, not std::nth_element)
//   key_select     : reversed key-partition -> tail holds all S below-key; the
//                    take-all base case is then ALREADY DONE, else reverse-
//                    quickselect the tail in place.  (shipped move_low_half_exact)
//   rev_count_select : count S (move-free scan), reverse-quickselect the whole
//                    array; the high-p alternative to key_select.
//   rev_part_until : part_until on the reversed partition -- the descent keeps
//                    all below-key elements in the [lo, end) suffix and the
//                    accepted tail is the answer in place (no move).  Same
//                    biased k (conditioned 9-sample pivot), pure raw speed.
//
// We verify each output (bottom-k property + multiset) and report k/S (how close
// to 0.5) alongside ns/elem.  The crucial axis is the KEY PERCENTILE p (= S/n).
//
// MODES: default/large = fixed-percentile single calls (n up to argv[1]);
// `small` = fixed-percentile batched tiny blocks; `random` = random array with
// a RANDOM key per block/run (p ~ U[0,1]), AVERAGE over many runs -- the
// expected-cost view (see the "Mode random" comment below).
//
// CSV: type,dist,p,n,algo,ns_per_elem,k_over_S,ns_per_k,ns_per_min
//   ns_per_k   = total ns / k            (cost per DELIVERED element)
//   ns_per_min = total ns / min(k, S-k)  (cost per USEFUL element: a biased cut
//                shrinks the smaller side of the below-key split and with it
//                the split's value -- the docs/partition_efficiency.md metric)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
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

// ---- rev_count_select: count S (move-free), then ONE reversed quickselect of
// the whole array puts the k smallest in the tail; take-all becomes a single
// reversed key-partition.  The reversed twin of count_select. ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_rev_count_select(It a, It end, K key, Comp comp, Proj proj) {
    const std::ptrdiff_t S = count_below(a, end, key, comp, proj);
    const auto k = target_k(S);
    if (k <= 0) return 0;
    if (k >= end - a) return k;
    if (k == S) {  // take-all: one reversed partition by key lands all S at the end
        algo_rev::sized_rev{}(a, end, key, comp, proj);
        return k;
    }
    detail::quickselect_rev(a, end, end - k, comp, proj);
    return k;
}

// ---- rev_part_until: part_until rebuilt on the reversed partition -- the
// mirror descent.  Invariant: every below-key element of the WHOLE array is in
// [lo, end) (the discarded heads were all >= some pivot >= key).  One reversed
// ninther-partition per pass: [lo, m) >= val | [m, end) < val, pivot parked at
// m-1.  If val < key, the suffix [m-1, end) (pivot included -- it is < key and
// everything left of it is >= val) is a valid bottom-k ALREADY AT THE END --
// return, no move.  Else descend into the tail (lo = m).  Base case <= 32: one
// reversed key-partition takes all S survivors exactly.  Same biased k as
// part_until (the accepted pivot is a conditioned 9-sample rank estimate), but
// the epilogue move of the forward version -- min(k, n-k) swaps, ~n/3 at large
// n -- is gone. ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_rev_part_until(It a, It end, K key, Comp comp, Proj proj) {
    It lo = a;
    while (true) {
        if (end - lo <= 32) {
            It m = algo_rev::sized_rev{}(lo, end, key, comp, proj);
            return end - m;  // all S below-key survivors, in place
        }
        It p = detail::ninther_pos(lo, end, comp, proj);
        std::iter_swap(lo, p);
        auto val = std::invoke(proj, *lo);
        It m = algo_rev::sized_rev{}(lo + 1, end, val, comp, proj);
        std::iter_swap(lo, m - 1);  // park pivot at the right edge of the >= run
        if (static_cast<bool>(std::invoke(comp, val, key)))
            return end - (m - 1);  // [m-1, end) = {val} u {< val}: bottom-k, done
        lo = m;  // val >= key: every below-key element is in [m, end)
    }
}

// ---- rev_sample_n4 (small-n experiment): the `sample` strategy scaled down to
// mid-size arrays -- m = min(n/4, 512) stride samples, repo quickselect for the
// sample rank, ONE reversed partition.  Tests whether sampling can beat the
// exact path below the shipped 2048 routing threshold (accuracy degrades as
// ~1/sqrt(p*m), so k/S is the other readout). ----
template <class It, class K2, class Comp, class Proj>
std::ptrdiff_t v_rev_sample_n4(It a, It end, K2 key, Comp comp, Proj proj) {
    using K = std::remove_cvref_t<decltype(std::invoke(proj, *a))>;
    const std::ptrdiff_t n = end - a;
    constexpr int M = 512;
    const int m = static_cast<int>(std::min<std::ptrdiff_t>(n / 4, M));
    if (m < 64) return move_low_half_exact(a, end, key, comp, proj);
    const std::ptrdiff_t stride = n / m;
    K samp[M];
    std::ptrdiff_t below = 0;
    for (int i = 0; i < m; ++i) {
        samp[i] = std::invoke(proj, a[i * stride]);
        below += static_cast<bool>(std::invoke(comp, samp[i], key));
    }
    if (below < 16) return move_low_half_exact(a, end, key, comp, proj);
    const int r = static_cast<int>(below / 2);
    detail::quickselect(samp, samp + m, samp + r, comp, std::identity{});
    It mid = algo_rev::sized_rev{}(a, end, samp[r], comp, proj);
    return end - mid;
}

// ---- sort_take (small-n): full sort, the bottom k are then the front k ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_sort_take(It a, It end, K key, Comp comp, Proj proj) {
    quicksort(a, end, comp, proj);
    std::ptrdiff_t S = 0;  // sorted -> below-key are the front run
    for (It it = a; it != end &&
                    static_cast<bool>(std::invoke(comp, std::invoke(proj, *it), key));
         ++it)
        ++S;
    const auto k = target_k(S);
    move_front_to_end(a, end, k);
    return k;
}

// ---- key_then_sort (small-n): partition by key, SORT the below-key set, take ----
template <class It, class K, class Comp, class Proj>
std::ptrdiff_t v_key_then_sort(It a, It end, K key, Comp comp, Proj proj) {
    It m = algo::sized{}(a, end, key, comp, proj);  // [a, m) < key
    const std::ptrdiff_t S = m - a;
    const auto k = target_k(S);
    if (k > 0 && k < S) quicksort(a, m, comp, proj);  // sort below-key; bottom k at front
    move_front_to_end(a, end, k);
    return k;
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
    // Normalised efficiency metrics (requester): ns per DELIVERED element (k)
    // and ns per USEFUL element min(k, S-k) -- the latter penalises a biased
    // cut the way docs/partition_efficiency.md's E does (overshooting the S/2
    // target shrinks the smaller side and with it the value of the split).
    const std::ptrdiff_t mks = std::min(k, S - k);
    double nspk = k > 0 ? r.min_ns / static_cast<double>(k) : 0.0;
    double nspm = mks > 0 ? r.min_ns / static_cast<double>(mks) : 0.0;
    std::printf("%s,%s,%.2f,%zu,%s,%.4f,%.3f,%.4f,%.4f\n", tn, dn, p, n, algo,
                r.min_ns / static_cast<double>(n), kos, nspk, nspm);
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
                // (move_low_half_exact / move_low_half, now REVERSED) so the
                // shipped code is what's benchmarked and cannot drift from this
                // study; fwd_* are the superseded forward+move formulations.
                R("fwd_key_select", [](auto a, auto e, auto k, auto c, auto pr) { return v_key_select(a, e, k, c, pr); });
                R("key_select",   [](auto a, auto e, auto k, auto c, auto pr) { return move_low_half_exact(a, e, k, c, pr); });
                R("rev_count_select", [](auto a, auto e, auto k, auto c, auto pr) { return v_rev_count_select(a, e, k, c, pr); });
                R("key_one_part", [](auto a, auto e, auto k, auto c, auto pr) { return v_key_one_part(a, e, k, c, pr); });
                R("fwd_sample",   [](auto a, auto e, auto k, auto c, auto pr) { return v_sample(a, e, k, c, pr); });
                R("sample",       [](auto a, auto e, auto k, auto c, auto pr) { return move_low_half(a, e, k, c, pr); });
                R("part_until",   [](auto a, auto e, auto k, auto c, auto pr) { return v_part_until(a, e, k, c, pr); });
                R("rev_part_until", [](auto a, auto e, auto k, auto c, auto pr) { return v_rev_part_until(a, e, k, c, pr); });
            }
        }
    };
    do_dist("random_uniform", dist::random_uniform{});
    do_dist("few_unique", dist::few_unique{});
}

// Small-n study: tiny arrays are timed BATCHED (one call is sub-clock-resolution)
// -- split a pool into size-n blocks, run the algorithm on every block, report
// ns/elem over the whole pool.  A single global key (percentile p of the pool) is
// used, so each block sees S ~= p*n below-key elements.
template <class T, class Proj>
void study_small(const char* tn, Proj proj) {
    auto comp = std::less<>{};
    const std::size_t pool = 1u << 20;
    auto master = gen_data<T>(dist::random_uniform{}, pool, 0xB00B5u, proj);
    for (std::size_t n : {std::size_t(32), std::size_t(64), std::size_t(128),
                          std::size_t(256), std::size_t(512), std::size_t(1024),
                          std::size_t(2048)}) {
        const std::size_t blocks = pool / n;
        for (double p : {0.25, 0.50, 0.90}) {
            auto key = percentile_key(master, p, comp, proj);
            auto R = [&](const char* nm, auto fn) {
                std::ptrdiff_t Ssum = 0, ksum = 0, minsum = 0;
                {  // correctness + k/S over every block
                    std::vector<T> chk = master;
                    for (std::size_t b = 0; b < blocks; ++b) {
                        auto bb = chk.begin() + static_cast<std::ptrdiff_t>(b * n);
                        std::vector<T> orig(bb, bb + n);
                        const std::ptrdiff_t Sb = count_below(bb, bb + n, key, comp, proj);
                        Ssum += Sb;
                        std::ptrdiff_t k = fn(bb, bb + n);
                        ksum += k;
                        minsum += std::min(k, Sb - k);
                        if (!check(std::vector<T>(bb, bb + n), orig, k, key, comp, proj)) {
                            std::fprintf(stderr, "WRONG small %s n=%zu p=%.2f %s blk=%zu k=%td\n",
                                         tn, n, p, nm, b, k);
                            std::abort();
                        }
                    }
                }
                std::vector<T> work = master;
                auto setup = [&] { work = master; };
                auto do_work = [&] {
                    for (std::size_t b = 0; b < blocks; ++b) {
                        auto bb = work.begin() + static_cast<std::ptrdiff_t>(b * n);
                        std::ptrdiff_t k = fn(bb, bb + n);
                        // ANTI-FUSION BARRIER (load-bearing): do_not_optimize's
                        // "memory" clobber stops GCC fusing/vectorising one block's
                        // count/partition loops into the next's, which would make
                        // batched tiny calls artificially fast.  Verified equal to
                        // a [[gnu::noinline]] per-call build (and to no-barrier,
                        // i.e. no fusion happens) within run-to-run noise.
                        bench::do_not_optimize(k);
                    }
                    bench::do_not_optimize(work[0]);
                };
                auto r = bench::measure(16, setup, do_work, 3);
                double kos = Ssum > 0 ? static_cast<double>(ksum) / static_cast<double>(Ssum) : 0.0;
                double nspk = ksum > 0 ? r.min_ns / static_cast<double>(ksum) : 0.0;
                double nspm = minsum > 0 ? r.min_ns / static_cast<double>(minsum) : 0.0;
                std::printf("%s,%zu,%.2f,%s,%.4f,%.3f,%.4f,%.4f\n", tn, n, p, nm,
                            r.min_ns / static_cast<double>(pool), kos, nspk, nspm);
                std::fflush(stdout);
            };
            R("ref",           [&](auto a, auto e) { return v_ref(a, e, key, comp, proj); });
            R("count_select",  [&](auto a, auto e) { return v_count_select(a, e, key, comp, proj); });
            R("fwd_key_select",[&](auto a, auto e) { return v_key_select(a, e, key, comp, proj); });
            R("key_select",    [&](auto a, auto e) { return move_low_half_exact(a, e, key, comp, proj); });
            R("rev_count_select", [&](auto a, auto e) { return v_rev_count_select(a, e, key, comp, proj); });
            R("rev_sample_n4", [&](auto a, auto e) { return v_rev_sample_n4(a, e, key, comp, proj); });
            R("sample",        [&](auto a, auto e) { return move_low_half(a, e, key, comp, proj); });
            R("part_until",    [&](auto a, auto e) { return v_part_until(a, e, key, comp, proj); });
            R("rev_part_until",[&](auto a, auto e) { return v_rev_part_until(a, e, key, comp, proj); });
            R("sort_take",     [&](auto a, auto e) { return v_sort_take(a, e, key, comp, proj); });
            R("key_then_sort", [&](auto a, auto e) { return v_key_then_sort(a, e, key, comp, proj); });
        }
    }
}

// ---------------------------------------------------------------------------
// Mode "random": EXPECTED cost under a uniformly random key.
//
// The fixed-percentile modes answer "how does each strategy behave at a given
// p"; this mode answers "which strategy is best if the key is arbitrary" --
// each run draws a fresh random array AND a fresh random key (the projected
// key of a uniformly random element, so the key percentile p is ~U[0,1]), and
// the AVERAGE over many runs is reported.  This is the decisive view for the
// descend-style algorithms, whose cost is itself a random function of p
// (1..4 passes), and it weights the regimes exactly as a random workload
// would: take-all (~S/2<16), low p (descent-hostile), high p (descent-
// friendly) in their natural proportions.
//
// Design: PAIRED -- per run all algorithms see the same array copy and the
// same key, so the p-randomness cancels in comparisons; reported numbers are
// aggregate ratios over all runs (sum ns / sum n, sum ns / sum k, ...), which
// is the run-weighted average and keeps ns_per_min well-defined when single
// runs land in the take-all regime (S-k == 0 contributes 0 to the
// denominator).  Small n is batched (pool split into blocks, an INDEPENDENT
// random key per block -- the per-block key loads are part of the workload),
// timed with the same anti-fusion barrier as study_small and averaged over
// reps; large n is single calls, mean over runs (key variance >> clock noise).
// Verification: bottom-k on EVERY output; full multiset check on the first
// run of each cell (the fixed-percentile modes already cover multisets
// exhaustively).
// ---------------------------------------------------------------------------

// bottom-k check only (O(n)); the multiset half of check() is O(n log n) and
// too slow to run on every of the thousands of large random runs.
template <class T, class Comp, class Proj>
bool bottom_k_ok(const std::vector<T>& v, std::ptrdiff_t k, Comp comp, Proj proj) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(v.size());
    if (k < 0 || k > n) return false;
    if (k == 0 || k == n) return true;
    auto kof = [&](const T& x) { return std::invoke(proj, x); };
    auto maxlast = kof(v[static_cast<std::size_t>(n - k)]);
    for (std::ptrdiff_t i = n - k + 1; i < n; ++i)
        if (static_cast<bool>(comp(maxlast, kof(v[static_cast<std::size_t>(i)]))))
            maxlast = kof(v[static_cast<std::size_t>(i)]);
    for (std::ptrdiff_t i = 0; i < n - k; ++i)
        if (static_cast<bool>(comp(kof(v[static_cast<std::size_t>(i)]), maxlast)))
            return false;
    return true;
}

template <class T, class Proj>
void random_small(const char* tn, Proj proj) {
    auto comp = std::less<>{};
    const std::size_t pool = 1u << 20;
    auto master = gen_data<T>(dist::random_uniform{}, pool, 0xD1CE5EEDu, proj);
    using K = std::remove_cvref_t<decltype(std::invoke(proj, master[0]))>;
    std::mt19937_64 rng(0xD1CE5EEDu);
    for (std::size_t n : {std::size_t(32), std::size_t(64), std::size_t(128),
                          std::size_t(256), std::size_t(512), std::size_t(1024),
                          std::size_t(2048)}) {
        const std::size_t blocks = pool / n;
        std::vector<K> keys(blocks);  // one uniformly random element key per block
        for (std::size_t b = 0; b < blocks; ++b)
            keys[b] = std::invoke(proj, master[b * n + rng() % n]);
        auto R = [&](const char* nm, auto fn) {
            std::ptrdiff_t Ssum = 0, ksum = 0, minsum = 0;
            {  // correctness + aggregates over every block
                std::vector<T> chk = master;
                for (std::size_t b = 0; b < blocks; ++b) {
                    auto bb = chk.begin() + static_cast<std::ptrdiff_t>(b * n);
                    std::vector<T> orig(bb, bb + n);
                    const std::ptrdiff_t Sb =
                        count_below(bb, bb + n, keys[b], comp, proj);
                    Ssum += Sb;
                    std::ptrdiff_t k = fn(bb, bb + n, keys[b]);
                    ksum += k;
                    minsum += std::min(k, Sb - k);
                    if (!check(std::vector<T>(bb, bb + n), orig, k, keys[b], comp, proj)) {
                        std::fprintf(stderr, "WRONG random-small %s n=%zu %s blk=%zu k=%td\n",
                                     tn, n, nm, b, k);
                        std::abort();
                    }
                }
            }
            std::vector<T> work = master;
            auto setup = [&] { work = master; };
            auto do_work = [&] {
                for (std::size_t b = 0; b < blocks; ++b) {
                    auto bb = work.begin() + static_cast<std::ptrdiff_t>(b * n);
                    std::ptrdiff_t k = fn(bb, bb + n, keys[b]);
                    bench::do_not_optimize(k);  // anti-fusion barrier (see study_small)
                }
                bench::do_not_optimize(work[0]);
            };
            // One pool pass already AVERAGES over `blocks` independent random
            // keys (the key randomness is inside the pass), so taking the MIN
            // over reps keeps the expected-cost semantics while rejecting
            // clock noise.  (A plain mean-of-reps was tried first and showed
            // occasional 1.5-2x single-cell outliers from a slow rep --
            // core migration / interrupts; the duplicated key_select==sample
            // cells at n<1024 exposed it.)
            auto r8 = bench::measure(8, setup, do_work, 2);
            const double ns = r8.min_ns;
            double kos = Ssum > 0 ? double(ksum) / double(Ssum) : 0.0;
            std::printf("%s,batched,%zu,%s,%zu,%.4f,%.3f,%.4f,%.4f\n", tn, n, nm,
                        blocks, ns / double(pool), kos,
                        ksum > 0 ? ns / double(ksum) : 0.0,
                        minsum > 0 ? ns / double(minsum) : 0.0);
            std::fflush(stdout);
        };
        R("fwd_key_select", [&](auto a, auto e, K k) { return v_key_select(a, e, k, comp, proj); });
        R("key_select",     [&](auto a, auto e, K k) { return move_low_half_exact(a, e, k, comp, proj); });
        R("rev_count_select", [&](auto a, auto e, K k) { return v_rev_count_select(a, e, k, comp, proj); });
        R("fwd_sample",     [&](auto a, auto e, K k) { return v_sample(a, e, k, comp, proj); });
        R("sample",         [&](auto a, auto e, K k) { return move_low_half(a, e, k, comp, proj); });
        R("part_until",     [&](auto a, auto e, K k) { return v_part_until(a, e, k, comp, proj); });
        R("rev_part_until", [&](auto a, auto e, K k) { return v_rev_part_until(a, e, k, comp, proj); });
    }
}

template <class T, class Proj>
void random_large(const char* tn, Proj proj, std::size_t max_size) {
    auto comp = std::less<>{};
    constexpr int kAlgos = 7;
    static const char* names[kAlgos] = {"fwd_key_select", "key_select",
                                        "rev_count_select", "fwd_sample",
                                        "sample", "part_until", "rev_part_until"};
    std::mt19937_64 rng(0xACE0FBA5Eu);
    for (std::size_t n : {std::size_t(1u << 16), std::size_t(1u << 18),
                          std::size_t(1u << 20), std::size_t(1u << 22)}) {
        if (n > max_size) continue;
        const int runs = n >= (1u << 22) ? 24 : n >= (1u << 20) ? 48
                         : n >= (1u << 18) ? 96 : 192;
        struct Acc { double ns = 0, k = 0, S = 0, mn = 0, elems = 0; } acc[kAlgos];
        for (int r = 0; r < runs; ++r) {
            auto master = gen_data<T>(dist::random_uniform{}, n, 0xBA5E + 977u * r, proj);
            const auto key = std::invoke(proj, master[rng() % n]);
            const std::ptrdiff_t S =
                count_below(master.begin(), master.end(), key, comp, proj);
            std::vector<T> work;
            auto one = [&](int i, auto fn) {
                work = master;
                const auto t0 = std::chrono::steady_clock::now();
                std::ptrdiff_t k = fn(work.begin(), work.end(), key, comp, proj);
                const auto t1 = std::chrono::steady_clock::now();
                bench::do_not_optimize(k);
                bench::do_not_optimize(work[0]);
                const bool ok = r == 0 ? check(work, master, k, key, comp, proj)
                                       : bottom_k_ok(work, k, comp, proj);
                if (!ok) {
                    std::fprintf(stderr, "WRONG random-large %s n=%zu %s run=%d k=%td\n",
                                 tn, n, names[i], r, k);
                    std::abort();
                }
                acc[i].ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
                acc[i].k += double(k);
                acc[i].S += double(S);
                acc[i].mn += double(std::min<std::ptrdiff_t>(k, S - k));
                acc[i].elems += double(n);
            };
            one(0, [&](auto a, auto e, auto k, auto c, auto pr) { return v_key_select(a, e, k, c, pr); });
            one(1, [&](auto a, auto e, auto k, auto c, auto pr) { return move_low_half_exact(a, e, k, c, pr); });
            one(2, [&](auto a, auto e, auto k, auto c, auto pr) { return v_rev_count_select(a, e, k, c, pr); });
            one(3, [&](auto a, auto e, auto k, auto c, auto pr) { return v_sample(a, e, k, c, pr); });
            one(4, [&](auto a, auto e, auto k, auto c, auto pr) { return move_low_half(a, e, k, c, pr); });
            one(5, [&](auto a, auto e, auto k, auto c, auto pr) { return v_part_until(a, e, k, c, pr); });
            one(6, [&](auto a, auto e, auto k, auto c, auto pr) { return v_rev_part_until(a, e, k, c, pr); });
        }
        for (int i = 0; i < kAlgos; ++i) {
            std::printf("%s,single,%zu,%s,%d,%.4f,%.3f,%.4f,%.4f\n", tn, n, names[i],
                        runs, acc[i].ns / acc[i].elems,
                        acc[i].S > 0 ? acc[i].k / acc[i].S : 0.0,
                        acc[i].k > 0 ? acc[i].ns / acc[i].k : 0.0,
                        acc[i].mn > 0 ? acc[i].ns / acc[i].mn : 0.0);
            std::fflush(stdout);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "large";
    if (mode == "small") {
        std::printf("type,n,p,algo,ns_per_elem,k_over_S,ns_per_k,ns_per_min\n");
        study_small<i64>("i64", std::identity{});
        study_small<pair64>("pair64", std::identity{});
        study_small<pair64>("pair64f", first_key{});
        return 0;
    }
    if (mode == "random") {  // random array, RANDOM key; averages over many runs
        std::size_t max_size = 1u << 22;
        if (argc > 2) max_size = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
        std::printf("type,mode,n,algo,runs,ns_per_elem,k_over_S,ns_per_k,ns_per_min\n");
        random_small<i64>("i64", std::identity{});
        random_small<pair64>("pair64", std::identity{});
        random_small<pair64>("pair64f", first_key{});
        random_large<i64>("i64", std::identity{}, max_size);
        random_large<pair64>("pair64", std::identity{}, max_size);
        random_large<pair64>("pair64f", first_key{}, max_size);
        return 0;
    }
    std::size_t max_size = 1u << 22;
    if (argc > 1) max_size = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    std::printf("type,dist,p,n,algo,ns_per_elem,k_over_S,ns_per_k,ns_per_min\n");
    run_type<i64>("i64", std::identity{}, max_size);
    run_type<pair64>("pair64", std::identity{}, max_size);
    run_type<pair64>("pair64f", first_key{}, max_size);
    return 0;
}
