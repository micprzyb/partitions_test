# Difficult cases for partition functions

This note records *why* the inputs in `include/partitions/distributions.hpp`
are there: what each one stresses, and which classic failure mode it targets.
A 2-way partition (`< pivot` left, `>= pivot` right) has two largely
independent quality axes:

1. **Throughput** — instructions, swaps/moves, and *branch mispredictions* per
   element. This is where scheme choice (Lomuto vs Hoare vs block) shows up.
2. **Balance** — where the partition point lands, which is decided entirely by
   the *pivot*, not by the partition algorithm. A bad split is what turns a
   quicksort built on the partition into an O(n²) disaster.

The distributions below probe both.

## Equal keys

* **all_equal**, **few_unique**, **binary**.

Equal elements are *the* classic partitioning hazard.

* **Lomuto** on all-equal input swaps every element (the predicate is always
  true), doing maximal work for zero benefit.
* **Hoare** on all-equal input swaps about every *pair*, and the two pointers
  meet in the middle — an even split and O(n log n) in a recursive sort.
* For a **2-way `< / >=` split, equal-to-pivot keys all land on the right.** No
  pivot choice can balance a block dominated by the pivot value; the
  `mean_equal` column of `balance_report` makes this explicit. (Three-way /
  Dutch-flag partitioning is the standard remedy, and is the natural next
  algorithm to add.)

Sources: Wikipedia, *Quicksort* (repeated-elements section); the BlockQuicksort
and branchless-Lomuto writeups below.

## Pre-sorted and nearly-sorted

* **sorted_ascending**, **sorted_descending**, **nearly_sorted**,
  **single_outlier**.

These break naive pivot rules: choosing the first or last element as pivot on
sorted input gives a maximally unbalanced 1-vs-(n−1) split every time. They
also expose fast-path bugs (`nearly_sorted` defeats "is it already sorted?"
shortcuts; `single_outlier` is the one-element-out-of-place boundary).

## Adversarial structure for median-of-k

* **organ_pipe** (ascending then descending) and **reverse_organ_pipe**.

The first/middle/last samples of a "mountain" are `0, max, 0`, so a
**median-of-3** pivot is near the extreme — a known trap. `balance_report`
shows median-of-3 degrading on `organ_pipe` while the ninther and
median-of-medians hold up, because they sample more widely.

* **median_of_3_killer** — Musser's permutation.

Constructed so that median-of-3 quicksort makes a near-useless split at *every*
level of recursion, forcing Θ(n²). For n = 20 it is the sequence

```
1 11 3 13 5 15 7 17 9 19 2 4 6 8 10 12 14 16 18 20
```

The first partition's median-of-3 of `{1, 2, 20}` is `2`, so the partition
point barely moves. The construction is reproduced exactly in
`dist::median_of_3_killer` and pinned by a unit test.

Source: D. Musser, *Introspective Sorting and Selection Algorithms*,
Software: Practice and Experience 27(8), 1997.

## The adaptive adversary (why no fixed input is enough)

McIlroy's *killer adversary* defeats **any** deterministic quicksort — even a
median-of-3 or randomized one — by deciding comparison outcomes *lazily* as the
sort runs, only committing to a consistent total order at the end. No fixed
array can do this; it requires a comparator that watches the algorithm.

This matters for a *single* partition only indirectly (the adversary attacks the
recursion), so this repository ships the static `median_of_3_killer` and
documents the adaptive attack here rather than implementing it. The defense is
also worth recording: pivot rules that inspect Θ(n) elements (median-of-medians)
provably resist the adversary, because it cannot make a linear-sample median
extreme.

Source: M. D. McIlroy, *A Killer Adversary for Quicksort*, Software: Practice
and Experience 29(4), 1999. <https://www.cs.dartmouth.edu/~doug/mdmspe.pdf>

## Constant-cost pseudo-median pivots (`pseudo15`, `median_of_5_medians_of_5`)

Pivot quality is a cost/balance trade-off. Sampling more of the block gives a
better-centred pivot but costs more; a Θ(n) full scan (`median_of_medians_5`,
`midpoint_min_max`) is the most balanced but is linear per partition. Between
median-of-3 and a full scan sit *constant-cost pseudo-medians* that inspect a
fixed number of elements:

* **`pseudo15`** — a 15-element pseudo-median computed by a fixed compare/
  select network, returned as a **value** pivot. ~15 reads, O(1).
* **`median_of_5_medians_of_5`** — the 5×5 analogue of Tukey's ninther: 25
  evenly-spaced samples, the median of each group of five, then the median of
  those five medians. Returned as a **position**. ~25 reads + tiny sorts, O(1).

Both are unfazed by the median-of-k traps that ruin a naive median-of-3,
because they sample widely across the block. Measured here (`balance_report`,
400 trials at n=4096; `mean` ≈ left-side share, `worst_side` = worst observed
`max(left,right)` split):

| input              | median_of_3 | ninther | pseudo15 | 5×5 medians | median_of_medians_5 (Θ(n)) |
|--------------------|-------------|---------|----------|-------------|----------------------------|
| random (stddev)    | 0.213       | 0.160   | 0.129    | **0.109**   | 0.005                      |
| random worst_side  | 0.992       | 0.909   | 0.879    | 0.887       | **0.514**                  |
| organ_pipe         | 0.0002 💀   | 0.250   | 0.533    | 0.583       | 0.500                      |
| median_of_3_killer | 0.0002 💀   | 0.250   | 0.333    | **0.417**   | 0.500                      |

Selection time (`bench_pivot`, i64, total ns per call — flat in n, confirming
O(1)): `pseudo15` ≈ 20–30 ns, `median_of_5_medians_of_5` ≈ 70–120 ns, vs
`median_of_medians_5` ≈ 5 ns/element (i.e. ~5 ms at n=2²⁰).

So the family ranks cleanly: **`median_of_5_medians_of_5` buys slightly better
balance than `pseudo15`** (tighter spread on random, 0.42 vs 0.33 on the
killer) by sampling 25 vs 15 elements, at **~3× the selection cost**; both crush
median-of-3's variance and neither collapses on the adversarial inputs; and both
are still beaten on balance by the Θ(n) median-of-medians, which is the price of
sampling everything. Which to pick depends on whether the partition's own work
or the pivot selection dominates — exactly what this bench is for measuring.

## Periodic / blocked structure

* **sawtooth** (fixed-period ramp) and **shuffled_blocks** (sorted runs in
  shuffled order).

These create many short runs and periodic patterns that interact badly with
block-based scanning and with pivot samplers that happen to align with the
period.

## Efficient schemes and where they win

* **Branchless Lomuto** (orlp's "gap" method) replaces the per-element branch
  with `i += predicate`, lowered to a conditional move. It wins decisively on
  random data with cheap elements (this repo measures ~3× over Lomuto/Hoare on
  small random `i32`), at the cost of two moves per element instead of one.
  Source: O. Peters, *Branchless Lomuto partitioning*,
  <https://orlp.net/blog/branchless-lomuto-partitioning/>.

* **BlockQuicksort / block partition** batches comparison results into small
  offset buffers, then performs only the necessary swaps — avoiding mispredicted
  branches in a Hoare-style scheme. It does a single write per moved element, so
  it overtakes branchless Lomuto on large inputs / wider elements where memory
  traffic dominates.
  Sources: Edelkamp & Weiss, *BlockQuicksort*, ESA 2016; Peters,
  *Pattern-defeating Quicksort*, <https://arxiv.org/pdf/2106.05123>.

## Summary table

| Input                | Primarily stresses                          |
|----------------------|---------------------------------------------|
| random_uniform       | baseline throughput, branch prediction      |
| few_unique / binary  | equal-key handling, three-way opportunity   |
| all_equal            | Lomuto worst case; unbalanceable 2-way split|
| sorted_*             | first/last pivot degeneration               |
| nearly_sorted        | fast-path correctness                       |
| organ_pipe           | median-of-3 sampling trap                   |
| median_of_3_killer   | recursive quadratic blowup                  |
| single_outlier       | one-out-of-place boundary                   |
| sawtooth / blocks    | periodic & run structure                    |
