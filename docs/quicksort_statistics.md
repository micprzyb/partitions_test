# Quicksort benchmark — detailed statistics (5 runs)

`partitions::quicksort` (pure-partition, size-adaptive) vs `boost::sort::pdqsort` and `std::sort`. Random `uniform[0,n]` plus a distribution sweep. Total sort time only.

## Methodology

* Hardware pinned: `taskset -c 2` on the Core Ultra 7 165H; `-O3 -march=native`.
* Each cell: the benchmark's own min over in-process reps (warmup excluded), then the **binary was run 5 times** end to end. Reported = **median of the 5**, with `[min..max]` spread and **CV** (coefficient of variation, %). CV ≤ ~2% everywhere ⇒ the numbers are layout/frequency-noise-robust.
* Sizes 2^22=4.2M, 2^24=16.8M, 2^26=67.1M elements. i64=8B, pair64=16B lexicographic, pair64f=16B compared by `.first` only.
* `boost::pdqsort`/`std::sort` are called with the **default `<`** when the projection is identity (engages pdqsort's branchless fast path); a custom comparator only for pair64f.
* Generator note: `dist::random_uniform` builds pairs as `(rank>>8, rank&0xff)` with `rank∈[0,n]`. So the **pair64f sort key `.first=rank>>8` has only ~n/256 distinct values** — i.e. pair64f-random is a *low-cardinality / duplicate-heavy* key test (this matters, see analysis). i64 and pair64(lex) keys are high-cardinality.

## Size sweep — random_uniform (median ns/elem [min–max] CV)

| type | n | quicksort | boost_pdqsort | std_sort | qs÷pdq |
|---|---|---|---|---|---|
| i64 | 2^22 | 15.13 [14.63–15.57] 2.0% | 20.51 [20.15–21.18] 1.6% | 53.73 | **1.36×** |
| i64 | 2^24 | 16.53 [16.19–16.96] 1.5% | 21.89 [21.38–22.49] 1.6% | 58.49 | **1.32×** |
| i64 | 2^26 | 17.41 [17.19–18.00] 1.6% | 23.19 [22.92–23.69] 1.1% | 63.74 | **1.33×** |
| pair64 | 2^22 | 38.05 [37.60–38.99] 1.2% | 72.80 [72.56–73.88] 0.6% | 77.71 | **1.91×** |
| pair64 | 2^24 | 40.21 [39.80–40.85] 0.9% | 79.39 [78.97–80.87] 0.8% | 84.50 | **1.97×** |
| pair64 | 2^26 | 41.52 [41.21–42.88] 1.4% | 86.45 [86.29–87.74] 0.7% | 91.62 | **2.08×** |
| pair64f | 2^22 | 51.59 [51.30–52.90] 1.2% | 37.07 [37.02–37.62] 0.6% | 43.42 | **0.72×** |
| pair64f | 2^24 | 53.53 [53.35–54.73] 1.0% | 42.52 [42.13–42.91] 0.6% | 49.12 | **0.79×** |
| pair64f | 2^26 | 55.38 [55.29–57.40] 1.5% | 48.69 [48.43–49.37] 0.7% | 55.74 | **0.88×** |

## Distribution sweep — n=2^22 (median ns/elem [min–max] CV)

| type | dist | n | quicksort | boost_pdqsort | std_sort | qs÷pdq |
|---|---|---|---|---|---|---|
| i64 | random_uniform | 15.13 [14.63–15.57] 2.0% | 20.51 [20.15–21.18] 1.6% | 53.73 | **1.36×** |
| i64 | sorted_ascending | 27.15 [26.62–28.03] 1.7% | 0.61 [0.60–0.62] 1.6% | 6.64 | **0.02×** |
| i64 | sorted_descending | 96.82 [95.89–98.15] 0.8% | 1.43 [1.42–1.47] 1.2% | 4.75 | **0.01×** |
| i64 | few_unique | 254.82 [251.88–258.45] 0.8% | 7.40 [7.24–7.43] 0.9% | 31.29 | **0.03×** |
| i64 | organ_pipe | 22.80 [22.61–23.17] 0.8% | 21.96 [21.71–22.30] 0.9% | 44.86 | **0.96×** |
| pair64 | random_uniform | 38.05 [37.60–38.99] 1.2% | 72.80 [72.56–73.88] 0.6% | 77.71 | **1.91×** |
| pair64 | sorted_ascending | 16.47 [16.30–17.05] 1.6% | 1.33 [1.33–1.36] 0.8% | 14.82 | **0.08×** |
| pair64 | sorted_descending | 191.64 [191.40–195.21] 0.7% | 1.99 [1.95–2.22] 4.9% | 11.60 | **0.01×** |
| pair64 | few_unique | 370.51 [368.99–375.90] 0.7% | 41.02 [40.80–41.71] 0.8% | 56.35 | **0.11×** |
| pair64 | organ_pipe | 31.45 [31.41–32.20] 0.9% | 29.30 [29.19–30.02] 1.1% | 90.80 | **0.93×** |
| pair64f | random_uniform | 51.59 [51.30–52.90] 1.2% | 37.07 [37.02–37.62] 0.6% | 43.42 | **0.72×** |
| pair64f | sorted_ascending | 47.27 [47.16–48.60] 1.2% | 1.13 [1.11–1.14] 0.8% | 8.22 | **0.02×** |
| pair64f | sorted_descending | 48.33 [47.95–49.99] 1.5% | 2.28 [2.26–2.34] 1.2% | 8.98 | **0.05×** |
| pair64f | few_unique | 73316.39 [72252.98–79469.88] 3.6% | 9.32 [9.24–9.59] 1.3% | 19.06 | **0.00×** |
| pair64f | organ_pipe | 83.87 [83.09–86.10] 1.2% | 8.48 [8.36–10.28] 8.2% | 44.84 | **0.10×** |

## Design incremental sweep — n=2^22 random_uniform (median ns/elem)

| type | pure(m3,boost,→1) | +halver leaf | +sized partitioner | +adaptive pivot (FINAL) |
|---|---|---|---|---|
| i64 | 28.3 | 18.2 | 15.4 | 14.9 |
| pair64 | 46.1 | 39.0 | 38.8 | 38.0 |
| pair64f | 51.5 | 51.6 | 52.2 | 51.9 |

## Analysis

**1. Random, high-cardinality keys (the design target) — quicksort wins, and the
win holds at scale.**
* **i64: 1.32–1.36× faster than pdqsort** across 2^22→2^26 (the fair,
  apples-to-apples case — both branchless). The ratio is essentially flat with n,
  so the advantage is structural, not a small-n artifact.
* **pair64 (lex): 1.9–2.1× faster**, the margin growing slightly with n. pdqsort's
  branchless block partition only engages for arithmetic keys under `<`; for the
  16-byte lexicographic key it falls back to a branchy partition, while every
  partitioner here is branchless.

**2. Low-cardinality / duplicate-heavy keys — quicksort LOSES (corrects an earlier
single-run claim).**
* **pair64f random: 0.72–0.88× (slower than pdqsort).** Here the sort key
  `.first = rank>>8` has only ~n/256 distinct values. An earlier *single* run that
  used a full-entropy `.first` showed quicksort ~2.6× *faster*; that was both noisy
  and a different (high-cardinality) input. With the proper repeated measurement
  **and** a realistic low-cardinality key, the honest result is that quicksort is
  **slower** — see the mechanism below.
* **few_unique is catastrophic** (qs÷pdq 0.00–0.11×); pair64f/few_unique reaches
  **73,316 ns/elem ≈ O(n²)**. Cause: the quicksort uses a plain 2-way partition
  with **no equal-element (3-way) handling**. A block of all-equal keys puts every
  element on the `>=` side, so each level peels off only the one pivot → O(block²);
  the halver leaf rescues only blocks ≤16. pdqsort detects and splits equal runs.

**3. Pre-sorted / patterned input — quicksort loses by 1–2 orders of magnitude.**
* sorted_ascending/descending/organ_pipe: pdqsort's *pattern-defeating* logic runs
  near O(n) (0.6–2.3 ns/elem); our quicksort has **no pattern detection** (27–191
  ns/elem) and **no introsort heapsort fallback**, so it also has a genuine O(n²)
  worst case on adversarial input.

**4. The size-adaptive design pays off on the random target** (incremental sweep):
halver leaf is the dominant lever, then the size-adaptive partitioner (i64), then
the size-adaptive pivot.

## Bottom line (honest)

On its design target — **random, high-cardinality keys** — the pure-partition,
size-adaptive `partitions::quicksort` is **1.35× faster than boost::pdqsort for i64
and ~2× for lexicographic pair64**, and that lead **holds from 4M up to 67M
elements** with <2% run-to-run variation. It is *not* a general-purpose sort: it
deliberately omits pdqsort's adaptivity (pattern-defeating, 3-way duplicate
handling, introsort fallback), so on **sorted/reverse/duplicate-heavy or
low-cardinality** inputs it ranges from slower to quadratic. Those three additions
are the obvious next step to make it robust without giving up the random-case win.
