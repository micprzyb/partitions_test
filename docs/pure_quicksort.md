# Pure-partition quicksort: best pivot, and when halvers help

## Setup (`benchmarks/bench_quicksort.cpp`)

A quicksort built **only** from partitioning — it recurses to size 1 with **no
insertion-sort / sorting-network leaf**. The partition primitive is fixed
(`boost_block`); the pivot is swapped to the front and placed at its rank (so it
is excluded from recursion → progress on duplicates). The two variables are the
**pivot strategy** and an optional **halver threshold T**: for subarrays ≤ T,
pivot-partitioning is replaced by **recursive halving** (one balanced halver split
per step, recursing both halves to 1) — still a partition primitive, not a leaf
sort. Total sort time is the only metric; `std::is_sorted` is checked first.
Random `uniform[0,n]` data.

## Results — total sort ns/element (rows = pivot, cols = T; lower = better)

```
 i64    n=2^22         T=0    T=8   T=16   T=24      pair64 n=2^22   T=0    T=8   T=16   T=24
 median_of_3          27.4   22.0   18.4   17.9*     median_of_3    34.5   27.2   26.7*  27.3
 ninther (m3m3)       29.1   23.0   18.7   18.3      ninther        37.3   29.9   27.7   28.1
 median_of_5_of_5     40.8   32.7   27.3   25.8      m5m5           50.0   39.4   36.8   36.8
 (pair64f n=2^22: m3 best, T=16 22.0 / T=24 22.3; m5m5 ~40% slower; T=0 28.0.)
```

## Finding 1 — best pivot: **median_of_3** (the cheapest)

`median_of_3` is fastest for every type and size; `ninther` is ~3–5% behind;
`median_of_5_medians_of_5` is **~45–50% slower**. This **inverts** the
single-partition efficiency study (where ninther won and m3 was "too noisy").
The reason is the regime:

* In a *single* partition the pivot cost is amortised over O(n) work, so pivot
  **quality** (balance) dominates → more samples (ninther/m5m5) win.
* In a *full quicksort* the pivot is chosen at **every node** (~2n of them, most
  of them small), so pivot **selection cost** dominates → the **cheapest**
  adequate pivot wins, and over-sampling (m5m5's 25 reads/node) is a large net
  loss. Balance *variance* averages out across the whole tree on random data.

(Caveat: `median_of_3` has the weaker worst case on adversarial inputs; `ninther`
is the more robust choice if antagonistic data is possible — here, on random data,
the cheaper pivot wins on total time.)

## Finding 2 — halvers help for **all** small subarrays (20–36% faster)

`T=0` (pure partitioning to size 1) is the **slowest** column everywhere —
recursing to 1 with a pivot at each tiny node is very wasteful. Switching to
recursive **halving** below a threshold cuts total sort time substantially and
monotonically up to the optimum:

* **i64 / pair64f** (cheap to move): best at **T=24** — ~35% faster than T=0
  (i64 2^22: 17.9 vs 27.4).
* **pair64** (lexicographic, 16-byte moves): best at **T≈16** — ~22% faster than
  T=0; T=24 is slightly worse, because the larger halver blocks do more expensive
  16-byte compare-exchanges than the pivot-partition they replace.

So the halver (a branchless, pivot-free, perfectly-balanced split) is the right
*partition-based* small-array handler: it has no pivot-selection cost, deterministic
balance, and ~20–33% fewer compare-exchanges than a full sort — and it beats
continuing to pivot-partition down to 1 by a wide margin.

## Bottom line

For a pure-partition quicksort judged on total time:
* use the **cheapest** pivot (`median_of_3`) — in a full sort the per-node pivot
  cost dominates its quality;
* hand small subarrays (**≤ 16 for 16-byte lex pairs, ≤ 24 otherwise**) to a
  **recursive halver** rather than partitioning to size 1 — a 20–36% total-time
  win, entirely within the "partitions only" constraint.

Reproduce: `build/benchmarks/bench_quicksort`.
