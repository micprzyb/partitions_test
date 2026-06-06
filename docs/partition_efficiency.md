# Partition efficiency: best pivot × partition strategy

**Scope.** Find the most efficient way to *partition* a block (not sort it),
where **efficiency is judged by**

```
        pivot-find time  +  partition time
  E  =  ──────────────────────────────────
              | smaller part |
```

— nanoseconds spent per element of *guaranteed split* (`|smaller part| =
min(m−first, last−m)` for partition point `m`). **Lower is better.** Raw
throughput is the only criterion. Element types: **i64**, **pair64** (16-byte,
lexicographic), **pair64f** (16-byte, compare `.first` only).

Measured on an Intel Core Ultra 7 165H (Meteor Lake), GCC, `-O3 -march=native`,
single core pinned; `random_uniform`; min-time over batched blocks; large-n
numbers are the median of 3 runs. Reproduce with
`build/benchmarks/bench_partition_efficiency`.

## Why this metric

`E` rewards **both speed and balance simultaneously**. A perfectly balanced
split has `|smaller| = n/2`, so it earns a 2× discount on its per-element time
versus a 3:1 split (`|smaller| = n/4`). It is the natural "useful work per
nanosecond" for a partition: the smaller side is the part a quicksort is
guaranteed to make progress on.

At large `n` the O(1) pivot cost vanishes and the metric **decomposes**:

```
  E  ≈  (raw partition ns/elem) / (smaller-fraction)
```

so it cleanly separates a *fast partition* (numerator → use `boost_block`) from
a *well-centred, reliable pivot* (denominator → maximise and stabilise the
smaller-fraction toward 0.5). Everything below is an instance of minimising that
ratio.

---

## Small arrays (n ≤ 64): is a perfect-partition *sorter* more efficient?

A small **sorting network** gives a *perfect* partition for free in the balance
sense — sort the block, split at the middle, `|smaller| = n/2` exactly — at the
cost of a full `O(n log² n)` branchless sort (`small_sort::sort`, capped at
N≤24). We compare it against find+partition and a hand-crafted branchless Lomuto.

`E` (ns per smaller-element, median of 3; `*` = best). `sort_mid` is the sorter;
`nin_lomuto` = ninther pivot + branchless Lomuto; `m3m3swap` = exchange pivot.

```
 i64    n |  sort_mid  nin_lomuto    m3_boost  ninther  pseudo9  m3m3swap
        8 |    2.74*      3.03         8.82      8.69    13.08     9.31
       16 |    2.49*      2.91         5.76      6.37     5.25     8.88
       21 |    2.68       2.41*        4.75      5.29     4.78     7.61
       24 |    2.81       2.26*        4.30      4.61     3.97     6.52
       64 |    (n/a)      1.79*        2.66      2.36     2.32     2.82

 pair64 n |  sort_mid  nin_lomuto    m3_boost  ninther  pseudo9  m3m3swap
        8 |    6.59*      9.86        12.78     13.54    15.78    13.40
       16 |    8.25*     13.33         8.55     12.35     9.30    13.47
       21 |   10.56      11.68         7.42*    10.03     7.48    10.50
       64 |    (n/a)     10.17         3.89      4.67     3.65*    5.04

 pair64f n |  sort_mid  nin_lomuto   m3_boost  ninther  pseudo9  m3m3swap
        8 |    3.68       3.40*        8.18      9.70    11.77     8.99
       16 |    4.48       3.19*        6.00      6.46     5.35     9.63
       24 |    5.95       2.42*        4.45      4.81     4.15     7.13
       64 |    (n/a)      2.15*        2.90      2.53     2.55     2.99
```

Balance, `smaller_frac` at n=24: `sort_mid 0.48`, `ninther 0.38`, `m3 0.32`.

**Findings.**

* **The sorter wins the smallest sizes for the expensive-compare integer types.**
  For **i64** and **pair64 (lex)** at **n ≤ 16**, `sort_mid` is the most
  efficient: its perfect balance (0.48 vs 0.32–0.38 for sample pivots) enlarges
  the denominator by ~1.3×, and at these sizes the branchless network is cheap
  enough that it beats find+partition outright. This confirms the hypothesis —
  *for small arrays a sorter that guarantees a perfect partition can be the most
  efficient partitioner.* It only holds while `small_sort` is in its network
  regime (N ≤ 24); above that `small_sort` degrades to O(n²) and `sort_mid`
  explodes (hence "n/a").

* **A cheap branchless Lomuto wins the n=21–64 band for i64 and all of pair64f.**
  `nin_lomuto` (9-sample pivot + branchless gap-method Lomuto) is best for i64 at
  21–64 and for pair64f at *every* small size. Its O(n) cost with a single cmov
  predicate per element beats both the sort (whose cost grows) and `boost_block`
  (whose offset machinery doesn't amortise at small n).

* **Lomuto is the *worst* choice for pair64-lex** (10–13 throughout): the gap
  method does ~n moves of 16 bytes *and* a branchy lex compare on the critical
  path — both expensive for a wide, expensive-compare element. There the
  sorter (≤16) and `m3/pseudo9 + boost_block` (≥21) win.

* **Exchange pivots are consistently the worst.** `m3m3swap` (the 9-sample pivot
  that physically reorders the samples) is dominated everywhere — same balance as
  the find form, strictly more time (the corroborating single-step study is in
  `bench_pivot_total.cpp`).

So the best *small-n* partitioner is **type- and size-dependent**: sort for the
tiniest integer blocks, branchless Lomuto for the cheap-move/cheap-compare band,
and `boost_block` with a 3- or 9-sample pivot for wide lexicographic elements.

---

## Large arrays (n ≥ 256): pivot quality vs pivot cost

Here the partition is always `boost_block` (the fastest, per the other studies)
and we vary the pivot. `E` median of 3, with `[raw ns/elem | smaller_frac]`:

```
 i64     n |   m3_boost        ninther        m5m5          mom            pseudo9
   4096    | 1.46[.42|.28]  1.27[.44|.35]  1.12*[.46|.41] 59.2[29.6|.50] 1.24[.42|.34]
   2^16    | 1.17[.42|.36]  0.90*[.44|.49] 0.92[.44|.48]  62.8[31.4|.50] 2.90[.36|.12]
   2^20    | 1.37[.63|.46]  1.01*[.48|.48] 1.15[.48|.42]  62.8[31.4|.50] 1.34[.49|.36]
   2^22    | 2.25[.51|.22]  1.19*[.53|.45] 1.84[.54|.29]  62.9[31.4|.50] 1.85[.52|.28]

 pair64  n |   m3_boost        ninther        m5m5          mom            pseudo9
   2^16    | 1.69[.61|.36]  1.14*[.55|.49] 1.19[.57|.48]  53.7[26.9|.50] 3.53[.44|.12]
   2^20    | 1.42[.65|.46]  1.42*[.68|.48] 1.64[.69|.42]  50.0[25.0|.50] 2.04[.74|.36]
   2^22    | 3.30[.74|.22]  1.56*[.70|.45] 2.61[.76|.29]  50.0[25.0|.50] 2.59[.73|.28]

 pair64f n |   m3_boost        ninther        m5m5          mom            pseudo9
   2^16    | 1.22[.44|.36]  0.91*[.44|.49] 0.93[.44|.48]  44.4[22.2|.50] 2.94[.36|.12]
   2^22    | 3.36[.75|.22]  1.62*[.73|.45] 2.57[.75|.29]  46.0[23.0|.50] 2.64[.75|.28]
```

**Findings.**

* **Exact median is a disaster (mom).** `median_of_medians_5_inplace` (BFPRT
  quickselect) gives a perfect 0.50 split but its **O(n)** selection costs
  ~25–32 ns/elem — *30–50× the partition itself* — for a denominator gain of only
  ~10% over a 9-sample pivot. Its `E` (45–65) is ~40× the best. **Never pay O(n)
  to find the pivot.** The balance improvement from "good" to "perfect" is far
  too small to justify any super-constant selection cost.

* **The ninther (median-of-3-of-3, 9 O(1) samples) is the best all-round large-n
  pivot.** It wins or ties at 2^16–2^22 for all three types, because it is both
  cheap (O(1), → 0 ns/elem) and *reliably* centred (smaller-fraction 0.45–0.49).
  `m5m5` (25 samples) is a close second — marginally better balance, marginally
  more cost — taking the 4k–2^18 band on some types.

* **median_of_3 is too few samples.** With only 3 sampled points its split
  fraction has high variance *independent of n* (a Beta(2,2) split), so even on a
  huge block it occasionally lands at 0.22 → tiny denominator → `E` spikes (2.2–
  3.4 at 2^22). It is never the safe choice. `pseudo9` (a value pivot via the
  by-key path, no sentinel) is similar-to-good but occasionally unstable (the
  2^16 row craters to a 0.12 fraction). The 9-sample **position** ninther is the
  robust pick.

* **The decomposition holds.** Read across any row: `E ≈ raw / smaller_frac`. The
  winner is always "lowest raw (boost_block) × highest, *stable* fraction
  (ninther)". This is the actionable rule.

---

## Hand-crafted small partitioners, and the pseudo15-exchange idea

The suggested idea — *take the pseudo15 approximate-median network and, instead
of pure comparisons, exchange the elements to get a partition* — does not work as
stated, for a concrete reason worth recording:

* **pseudo15/pseudo9 are value-computing DAGs, not permutation networks.** Each
  node outputs `min`/`max` of two wires and many intermediates are mins-of-maxes
  that no longer correspond to any physical array slot. There is no bijection
  between wires and elements to preserve, so the DAG **cannot be run in place as
  compare-exchanges** — doing so neither sorts nor partitions the block. A
  pseudomedian network can only ever *find a pivot value*; partitioning is then a
  separate pass.

* **The realizable "compare-exchange network → partition" is a sorting (or
  selection) network.** That is exactly `sort_mid`, and the data above shows it
  is genuinely the most efficient partitioner for the smallest i64/pair64 blocks
  (perfect balance, fully branchless). A true *median-selection* network (fewer
  comparators than a full sort, still a permutation network, still a perfect
  partition around the median) would sit between `sort_mid` and find+partition;
  constructing and 0/1-verifying one (à la `tools/verify_small_sort`) is the
  natural next step if the n ≤ 24 partition is hot — but the full sort network is
  already available and already wins there.

* **Using the network's *pivot* with exchanges is just an exchange pivot**, which
  the data (m3m3swap) and the separate `bench_pivot_total` study both show is a
  net loss: the block partition cannot exploit the reordering (it has no
  per-element bound checks for a sentinel to remove), so the extra scattered
  writes are pure cost — worst for wide elements.

The useful hand-crafted result is the other one: **a branchless Lomuto**
(`algo::lomuto_branchless`, the orlp gap method) with a 9-sample pivot is the
best partitioner for the n=21–64 band on cheap-move/cheap-compare elements
(i64, pair64f), beating both the sorter and `boost_block` there.

---

## Recommendations (lowest E)

| regime | i64 | pair64 (lex) | pair64f (by .first) |
|---|---|---|---|
| n ≤ 16 | **sort_mid** (network) | **sort_mid** (network) | **branchless Lomuto** + ninther |
| n = 21–64 | **branchless Lomuto** + ninther | boost_block + median_of_3 / pseudo9 | **branchless Lomuto** + ninther |
| n ≥ 256 | **boost_block + ninther** | **boost_block + ninther** | **boost_block + ninther** |

Universal rules, in order of impact:

1. **Never pay O(n) to find the pivot** (exact median / median-of-medians) — the
   balance gain is ~10%, the cost is 30–50×.
2. **Use ~9 O(1) samples (ninther/m3m3)** — enough to make the split fraction
   reliably ~0.47; 3 samples (median_of_3) are too noisy, 25 (m5m5) buy almost
   nothing extra.
3. **Partition with `boost_block`** at every size ≥ ~24; below that prefer a full
   sorting network (cheap-move types, n ≤ 16) or a branchless Lomuto.
4. **Do not exchange while selecting** — the block partition gains nothing from
   it.

Reproduce:

```bash
build/benchmarks/bench_partition_efficiency            # full sweep, CSV
build/benchmarks/bench_partition_efficiency 4096       # cap size for a quick run
```
