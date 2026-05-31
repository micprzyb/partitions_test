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
