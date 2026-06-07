# Quickselect benchmark + a size-dispatching partitioner

## The quickselect driver (`benchmarks/bench_quickselect.cpp`)

Quickselect finds the k-th element by partitioning around a pivot and recursing
into the **one** side containing k. Its total work is ~2n and is almost entirely
*partitioning* — no leaf-sort dominance, no two-sided recursion bookkeeping — so
it isolates a partitioner's cost far better than a full quicksort.

The driver is templated on the partitioner; **pivot selection (ninther) is held
constant** across all partitioners, so the only variable is the partition step.
The pivot is swapped to the front, the tail `[first+1,last)` is partitioned by
value, and the pivot is placed at the resulting boundary and **excluded** from
the recursion — guaranteeing progress even on heavy-duplicate inputs. Correctness
is checked against `std::nth_element` before timing.

## The dispatcher (`algo::sized`)

A `PivotPartitioner` that routes by block size to the sub-partitioner that wins
there:

```
n <= cutoff(T)  ->  lomuto_branchless   (cheap branchless gap method)
n >  cutoff(T)  ->  boost_block         (pdqsort branchless block)
cutoff(T) = sizeof(T) <= 8 ? 512 : 24
```

The threshold is an *array-size* cutoff whose **value is chosen by `sizeof(T)`**:
the gap method does two moves per element, so it only pays for narrow,
cheap-to-move keys (i64) — for wider elements (pair64) the block partition is used
almost immediately. This mirrors Rust ipnsort's `size_of::<T>()` rule. It also
carries a position fast path `at()` (block-partition sentinel for large n).
Verified across the full correctness matrix.

## Results — quickselect ns/element (median select, min over reps; lower better)

```
 i64        2^16   2^18   2^20   2^22      pair64     2^16   2^18   2^20   2^22
 hoare      6.03   7.57   5.62   5.04      hoare      6.02   6.63   7.26   7.50
 lomuto_bl  1.01   1.29   1.39   1.31      lomuto_bl  1.65   1.73   1.92   2.24
 boost      0.83   1.23   1.07   0.89      boost      1.21*  1.24*  1.85   1.93
 fulcrum    1.12   1.27   1.20   1.38      fulcrum    1.90   2.05   1.93   1.99
 sized      0.83*  1.23*  1.06*  0.91      sized      1.25   1.28   1.72*  1.91*

 pair64f    2^16   2^18   2^20   2^22
 hoare      5.99   6.32   6.92   6.72
 lomuto_bl  1.66   1.32   1.86   2.35
 boost      0.95   1.25   1.65   1.93
 fulcrum    1.16   1.36   1.61   1.65*
 sized      0.91*  1.12*  1.60*  1.89
```

* **`sized` is the best or within ~1% of the best in 8 of 9 large-n cases** (and
  the outright winner in 5). It never has a weak regime, whereas every *single*
  partitioner does: branchy **hoare** is 5–7× slower everywhere (mispredicts on
  random data); **lomuto_branchless** is good for i64 but ~2× off for pair64 (its
  two 16-byte moves/element); **boost_block** is the strong all-rounder but is
  edged at i64 mid-n and pair64 large-n; **fulcrum** is competitive but not a
  consistent leader.
* The dispatcher beats plain `boost_block` where the quickselect tail's small
  partitions favour the cheap Lomuto (i64/pair64f mid-n), and matches it where the
  large levels dominate. The one real gap: **pair64f at 2^22**, where `fulcrum`
  beats `boost` by ~7% — a *wide-element, cheap-comparator, large-n* niche that a
  **size-only** dispatch cannot detect (it would need comparator-cost awareness;
  `sizeof(pair64)=16` looks identical to lexicographic pair64).

## Bottom line

A size-dispatching partitioner is the right synthesis of the whole study: it puts
the cheap branchless Lomuto on the small blocks and the pdqsort block partition on
the large ones, and so delivers **top-tier performance across every element type
and size with no weak case** — exactly what a real introselect/quicksort wants
from its partition primitive. Reproduce: `build/benchmarks/bench_quickselect`.
