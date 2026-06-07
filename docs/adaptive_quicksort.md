# Size-adaptive pure-partition quicksort (`partitions::quicksort`)

The synthesis of the whole partition study: a quicksort built **only** from
partitioning (no insertion-sort / sorting-network leaf) that chooses **both the
partition algorithm and the pivot strategy by the size of the block** being
partitioned, and finishes the smallest blocks with a recursive **halver** (a
balanced rank-split — itself a branchless partition primitive). Total sort time is
the only criterion. Implementation: `include/partitions/quicksort.hpp`; verified
by `tests/test_quicksort.cpp`; benchmark `benchmarks/bench_quicksort.cpp`.

```
 block size n      pivot                partition primitive
 n <= 16           (none)               recursive halver (balanced split)
 16 < n <= 2048    median_of_3 (cheap)  algo::sized  (lomuto small / boost large)
 n  > 2048         ninther (9 samples)  algo::sized  (-> boost_block)
```

## The full picture — each size-adaptive choice, measured (total sort ns/elem)

```
 configuration                 i64            pair64          pair64f
                            2^20  2^22     2^20  2^22       2^20  2^22
 pure (m3, boost, ->size1)  25.7  27.6     31.7  34.5       25.8  28.5
 + halver leaf (T<=24)      16.2  17.9     25.2  27.3       19.7  22.5    (-35% / -21% / -22%)
 + sized partitioner        13.6  15.4     25.0  27.2       19.9  22.4    (size-adaptive partition)
 + halver T=16, m3          13.4  14.9     23.5  26.2       18.9  21.3
 + adaptive pivot  = FINAL  12.8  14.4     23.1  25.0       19.1  21.3    (size-adaptive pivot)
 quicksort.hpp (the impl)   13.4  14.8     23.3  25.3       18.4  21.2
```

The three size-dependent choices each pay off, largest first:
1. **Halver leaf** (≤16): the big one, **−21…35%** — recursing to size 1 with a
   pivot at every tiny node is very wasteful; a branchless balanced split is far
   cheaper.
2. **Size-adaptive partitioner** (`algo::sized`: branchless Lomuto for small/narrow
   blocks, pdqsort block partition for large): **−14%** on i64 (its Lomuto wins the
   mid-size band), neutral-to-small on the 16-byte pairs (which go straight to the
   block partition).
3. **Size-adaptive pivot** (ninther only for big nodes, median_of_3 for the many
   small ones): **−4…7%** — the cheap pivot keeps per-node cost down where most
   nodes are, the ninther's balance helps the few large nodes where its sampling
   amortises. (A single pivot can't do both: m3-everywhere pays balance at the top,
   ninther-everywhere pays sampling at every small node.)

## Versus boost::pdqsort and std::sort

See **`docs/quicksort_statistics.md`** for the rigorous numbers (5 runs, median ±
spread, CV ≤ 2%, sizes to 2^26, plus a distribution sweep). Summary of the honest
picture (qs÷pdq ratio, >1 = our quicksort faster):

```
 random_uniform     2^22   2^24   2^26     verdict
 i64               1.36×  1.32×  1.33×    WIN (fair: both branchless), holds at scale
 pair64 (lex)      1.91×  1.97×  2.08×    WIN (pdqsort stays branchy for 16-byte lex)
 pair64f (.first)  0.72×  0.79×  0.88×    LOSE — low-cardinality key (see below)
```

* **i64 is the fair, apples-to-apples case** (both branchless): we are **~1.35×
  faster than pdqsort** and the lead is flat from 4M→67M elements — structural, not
  a small-n artifact.
* **pair64 (lex) ~2×**: pdqsort's branchless block partition only engages for
  arithmetic keys under `<`; for the 16-byte lexicographic key it goes branchy,
  while all our partitioners are branchless.
* **pair64f we LOSE** (0.72–0.88×). This **corrects an earlier single-run claim of
  ~2.6× faster** — that run used a full-entropy `.first` and was noisy. The proper
  repeated measurement uses `.first = rank>>8` (~n/256 distinct), a
  **low-cardinality / duplicate-heavy** key, and our pure quicksort has **no 3-way
  duplicate handling**, so it degrades (catastrophically so on `few_unique`:
  ~O(n²)).

**Limitations (deliberate — pure partitioners only).** The quicksort has no
pattern-defeating, no 3-way/equal-element partition, and no introsort heapsort
fallback, so on sorted/reverse/duplicate-heavy/low-cardinality inputs it ranges
from slower to quadratic (see the distribution sweep). It is tuned for, and wins
on, **random high-cardinality** data.

## Bottom line

A purely partition-based quicksort, **size-adaptive in both partitioner and pivot**
with a halver finishing the smallest blocks, is **~1.35× faster than boost::pdqsort
on i64 and ~2× on lexicographic pair64**, holding from 4M to 67M elements. It is not
general-purpose: it loses on patterned, duplicate-heavy, or low-cardinality input
(no pattern-defeating / 3-way / introsort fallback). The dominant lever is the
halver leaf; the size-adaptive partitioner and pivot add the rest.
`partitions::quicksort` is that implementation.
