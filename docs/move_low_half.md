# `move_low_half` — move ≈ half of the below-key smallest elements to the end

**Goal.** Given an array `A[0,n)` and a `key`, let `S = #{x : comp(proj(x), key)}`
(the elements *below* the key). Rearrange `A` so the **last `k` positions hold the
`k` smallest elements** (any order), with `k ≈ S/2`. Those `k` smallest are
exactly the *lower half of the below-key elements* (all `< key`), so the operation
is "find the **median of the below-key elements** and stash its lower half at the
end." Order within the two parts is irrelevant; the call returns `k`.

Two refinements of the spec (from the requester):
- `k` need **not be exact** — being *close to* `S/2` is enough, and that buys
  speed.
- **Base case:** if the target half `S/2` would be `< 16`, take **all** `S`
  below-key elements (`k = S`) — it is not worth selecting a half of a tiny set.

Raw throughput is the only criterion. Measured on this host (Meteor Lake / Core
Ultra 7 165H, GCC 15.2, `-O3 -march=native`), min ns/elem, median of repeated
runs. Files: `include/partitions/move_low_half.hpp`,
`benchmarks/bench_move_low_half.cpp`.

> **This report has two rounds.** Round 1 (the next sections) explored the
> *strategies* (sample / key-select / count-select / descend-until) with every
> variant built on the **forward** partition plus a move-to-end epilogue.
> Round 2 ([below](#round-2--the-reversed-formulation-kill-the-move-pass))
> rebuilt the winners on the repo's **reversed** partition family, which makes
> the move pass disappear entirely — up to **1.8×** faster at high percentiles,
> never slower beyond one ±6% cell. The shipped functions are the round-2
> versions; round-1 numbers are kept for the strategy comparison (the *relative*
> ordering of strategies is unchanged).

## The shape of the problem

The `k` smallest are the bottom `S/2` order statistics, and `S` itself is
data-dependent (it is the rank of `key`). So a correct algorithm must, in some
form, (a) discover `S` (or `S/2`) and (b) place the bottom `S/2` at one end. The
**key percentile `p = S/n`** (where the key falls in the data) is the decisive
axis: it controls how much work each strategy does, and the strategies have
genuinely different sweet spots.

A subtle, recurring trap: a *single fixed-position pivot* (e.g. the ninther) is a
good median estimate **only on unstructured data**. After a partition, the array
has structure that the ninther's fixed sample offsets hit non-uniformly — so any
"partition then take one ninther split" shortcut is **biased**. This is the
single most important lesson below.

## Fixing the supplied example (Approach 1)

The starting point was a `partition_until_leq` that "partitions the range until
the partition point is ≤ the guard." Its bugs:
- `quicksort(begin, begin, end, …)` — three iterators; the repo `quicksort` takes
  two, and a full sort is overkill for the base case.
- `partition(begin, end)` returning `(mid, val)` does not exist — it must be built
  (`ninther_pos` pivot → swap to front → `algo::sized` partition of the tail →
  place the pivot → return `(mid, pivot_value)`).
- It returns a boundary pair but never **moves** anything to the end, and stopping
  at the *first* pivot `< key` leaves a bottom set whose size is essentially a
  random fraction of `S` (`k/S` measured 0.33–0.92 at small `n`), not `≈ S/2`.

The corrected version (`part_until` in the bench) descends **left** (`hi = mid`)
while `pivot ≥ key` — correct, because every below-key element is then in the left
part `[a, mid)` — keeps `lo == a` throughout (so the bottom set is always at the
front, ready to move), takes `[a, mid]` once a pivot drops below `key`, and uses
the take-all base case for ranges `≤ 32`. It is now **correct** (a valid
bottom-k) and fast, but its `k/S` is still **approximate and biased high**
(≈0.6–0.74 at large `n`): the descent overshoots, because discrete halving cannot
land exactly on `S/2` without knowing `S`.

## The ideas benchmarked

All reuse the repo's branchless `algo::sized` partition and the inline
`detail::ninther_pos` pivot.

| name | strategy | `k` |
|---|---|---|
| `ref` | oracle: count `S`, `std::nth_element` for rank `k`, swap to end | exact |
| `count_select` | count `S`, quickselect rank `k` over the **whole** array | exact |
| `key_select` | **(Approach 2)** partition by `key` → `S`, quickselect rank `k` over the below-key **subset**, swap | exact |
| `key_one_part` | partition by `key`, then **one** ninther-partition of the remainder | approx |
| `sample` | estimate the (S/2)-th value from a **stride sample of the original array**, then **one** partition around it | approx |
| `part_until` | **(Approach 1, fixed)** descend-left while `pivot ≥ key`, take the smaller part | approx |

## Methodology

`benchmarks/bench_move_low_half.cpp`. Types `i64`, `pair64` (lexicographic),
`pair64f` (by `.first`), generated with the **fair high-cardinality** `gen_data`
(so every type sorts a key of the same cardinality — see `docs/quicksort_lr.md`
for why this matters). Distributions: `random_uniform`, `few_unique`. The key is
chosen at percentile `p ∈ {0.10, 0.50, 0.90}` (so `S ≈ p·n`); `n` swept
2¹⁶…2²². **Every** output is verified — bottom-k (`max(last k) ≤ min(first n−k)`
by the projection) *and* multiset preserved — before timing, and the achieved
`k/S` is reported alongside ns/elem.

## Results

Random high-cardinality keys, n = 2²², median of 3, ns/elem; **bold = fastest**.

**`i64`** — `k/S` in the last column:

| variant | p=0.10 | p=0.50 | p=0.90 | `k/S` |
|---|---|---|---|---|
| **`sample`** (winner) | **0.45** | **0.60** | 0.73 | 0.48–0.52 |
| `key_select` (Approach 2, exact) | 0.59 | 1.27 | 1.71 | 0.500 |
| `part_until` (Approach 1) | 1.09 | 0.99 | **0.64** | 0.61–0.74 |
| `key_one_part` | 0.56 | 0.92 | 1.04 | 0.35–0.71 |
| `count_select` (exact) | 1.38 | 1.48 | 1.63 | 0.500 |
| `ref` (`std::nth_element`) | 6.25 | 4.99 | 6.26 | 0.500 |

**`pair64`** (lexicographic, 16-byte):

| variant | p=0.10 | p=0.50 | p=0.90 |
|---|---|---|---|
| **`sample`** | **0.69** | **1.04** | **1.25** |
| `key_select` | 0.91 | 1.90 | 2.67 |
| `part_until` | 1.56 | 1.52 | 1.14 |
| `count_select` | 2.22 | 2.44 | 2.72 |
| `ref` | 8.63 | 7.00 | 8.79 |

**`pair64f`** (by `.first`):

| variant | p=0.10 | p=0.50 | p=0.90 |
|---|---|---|---|
| **`sample`** | **0.84** | **1.08** | **1.30** |
| `key_select` | 0.94 | 2.02 | 2.61 |
| `part_until` | 1.65 | 1.59 | 1.19 |

The one place a *non-winner* is faster is `i64` at p=0.90, where `part_until`
(0.64) edges `sample` (0.73) — but at `k/S = 0.74` it is moving ~50 % more
elements than the requested half, whereas `sample` stays at 0.48. On accuracy
`sample` wins there too.

### Reading it

1. **The two supplied approaches are complementary.** `key_select` (Approach 2,
   exact) is fastest at **low `p`**: the key-partition isolates a small below-key
   set, so the subsequent select is tiny. `part_until` (Approach 1) is fastest at
   **high `p`**: one partition already splits the array near the answer. Neither
   dominates, and `part_until`'s `k` is only approximate.

2. **The winner is a third idea: `sample`.** Estimating the (S/2)-th value from a
   small **stride sample of the *original* (unsorted, hence unbiased) array** and
   doing **one** `algo::sized` partition around it is `~n` work for *every*
   percentile and keeps `k/S` within **±4 % of 0.5**. It beats both supplied
   approaches on the stated goal ("fast, `k ≈ S/2`"): roughly **2× `key_select`**
   at mid/high `p`, and more accurate than `part_until` everywhere.

3. **`key_one_part` is the cautionary tale.** "Partition by key, then one ninther
   split of the remainder" *looks* like a cheap accurate median, but its `k/S`
   swings **0.35–0.83**: the ninther is sampling a **partition-structured** array,
   where its fixed offsets are biased. `sample` is the same idea done right — it
   samples the array *before* any partitioning, so the estimate is unbiased.

4. **`std::nth_element` (`ref`) is 4–8× slower** than the custom quickselect
   (`count_select`/`key_select`), which uses the repo's branchless `algo::sized`
   partition — the same gap the rest of the repo finds between branchy introselect
   and the branchless block partition.

### Why `sample` is `~n` and accurate

- **Work.** A stride sample of `m = 512` keys + a count (a few hundred branchless
  ops) + **one** `algo::sized` partition over `n` ≈ `n`, *independent of `p`*.
  That is why it flattens the percentile curve that `key_select` (cost grows with
  `S`) and `part_until` (cost grows as the key drops) both have.
- **Accuracy.** The (S/2)-th value is estimated as the `(below/2)`-th smallest of
  the sample, where `below` = sampled keys under `key`. The sampling error of a
  quantile is ~`1/√below`; with `below ≈ p·512` this is a few percent for the
  healthy percentiles, and `below < 16` (very low `p`) falls back to the exact
  path (which is cheap there anyway).

## Small arrays (n = 32 … 1024)

Small `n` is a different regime: the array is L1-resident, **per-call overhead**
and the **take-all base case** (`k = S` when `S/2 < 16`) dominate, and the
`sample` winner is irrelevant — `move_low_half` falls back to the exact path
below `n ≈ 2048`, so the question is purely *which exact/approximate primitive is
cheapest at this size*. Tiny calls are timed **batched** — a pool split into
size-`n` blocks, every block processed, ns/elem reported over the pool (`bench_move_low_half small`).
A single global key gives each block `S ≈ p·n` below-key elements.

> **Batching caveat (verified).** Batching tiny calls risks GCC *fusing* adjacent
> blocks' loops (e.g. vectorising two blocks' counts together), which would
> understate the per-call cost. A `do_not_optimize` **memory-clobber barrier after
> every block call** prevents it; this was confirmed by triangulation — batched
> with-barrier ≈ a `[[gnu::noinline]]` per-call build ≈ batched *without* barrier
> (i.e. no fusion happens anyway, because the data-dependent partition/select
> breaks it), all within ~5 % run-to-run noise.

**`i64`, ns/elem (median of 3), `random_uniform`:**

| n | key_select | count_select | part_until | sort_take | ref |
|---|---|---|---|---|---|
| 32 | **0.88** | 2.80 | 1.01 | 6.27 | 8.33 |
| 64 | 1.47 | 2.52 | **1.40** | 7.40 | 7.54 |
| 128 | 1.68 | 1.97 | **1.03** | 8.01 | 6.62 |
| 256 | 1.47 | 1.69 | **0.90** | 8.51 | 6.13 |
| 512 | 1.30 | 1.54 | **0.81** | 9.14 | 6.00 |
| 1024 | 1.27 | 1.44 | **0.75** | 9.65 | 5.77 |

*(p = 0.50; `part_until`'s `k/S ≈ 0.66 here — approximate. `key_select`/`count_select`/`ref` are exact.)*

### Findings

1. **Never sort.** `sort_take` (full `quicksort` then take the front `k`) is
   **6–10× slower** than the selection-based methods, and `key_then_sort`
   (partition by key, then *sort* the below-key set) is dominated — equal to
   `key_select` only in the take-all regime (where neither sorts), and far worse
   once a real select is needed (O(S log S) vs O(S)). You only need a *split*, so
   paying for *order* is pure waste. This is the headline small-n lesson.

2. **`std::nth_element` (`ref`) is the slowest** real contender (6–8) — its branchy
   introselect has no answer at these sizes; the branchless quickselect is 2–6×
   faster.

3. **The take-all base case flattens the smallest sizes.** At `n = 32` every
   percentile has `S/2 < 16`, so `k = S`: no select happens and `key_select`
   collapses to *one key-partition + move* (~0.88, flat across `p`). The operation
   is trivial there.

4. **Exact winner is `p`-dependent**, not `n`-dependent:

   | exact best | p=0.25 | p=0.50 | p=0.90 |
   |---|---|---|---|
   | n=64 | key_select 0.76 | key_select 1.47 | **count_select 2.42** |
   | n=256 | key_select 1.06 | key_select 1.47 | **count_select 1.78** |
   | n=1024 | key_select 0.93 | key_select 1.27 | **count_select 1.50** |

   `key_select` wins at low/mid `p` (the key-partition isolates a small below-key
   set, so the select is cheap). At **high `p`** the below-key set is almost the
   whole array, so the key-partition buys nothing and its element *moves* cost more
   than `count_select`'s move-free counting scan — `count_select` (count +
   quickselect over the whole array) edges ahead by ~10 %. The crossover is around
   `p ≈ 0.7`.

5. **`part_until` is the fastest at every mid/high `p`** (0.6–1.0) because it is a
   single descend — but its `k/S` is biased (0.56–0.74), so it is only the right
   choice when an approximate `k` is acceptable. (At `n = 32` it is the take-all
   case and `k/S = 1.0`, exact by construction.)

6. **Per-call overhead shows up as `n → 32`** for the select-heavy cases: at
   `p = 0.9`, `key_select` rises from 1.74 (n=1024) to 2.73 (n=64) ns/elem as the
   fixed costs (the count, the pivot setup, the move) amortise over fewer elements.

7. **Quickselect leaf cutoff:** sweeping the leaf where the select stops
   partitioning and sorts a network shows **16–20 is marginally best** for these
   small selects (n=128/p=0.9: 2.21 at cut-16 vs 2.30 at cut-24, ~3 %), the gain
   shrinking with `n`. The shipped cutoff of 24 (the `small_sort` network max,
   shared with the large-`n` path) is within ~2–3 %, so it is kept for simplicity.

`pair64` and `pair64f` follow the same ordering with higher absolute ns (16-byte
moves / lexicographic compares): `part_until` fastest, `key_select` the exact
choice, sort-based non-competitive.

### Tuning summary

| regime | use | why |
|---|---|---|
| exact `k`, p ≲ 0.7 | `key_select` (= `move_low_half_exact`) | small isolated select |
| exact `k`, p ≳ 0.7 | `count_select` | skip the key-partition's moves |
| approximate `k` ok | `part_until` | one descend, ~0.6–1.0 ns/elem |
| any | **not** sort-based | order is wasted work |

The shipped `move_low_half` routes small `n` to `move_low_half_exact`
(= `key_select`), i.e. onto the exact small-n optimum for `p ≲ 0.7`;
`count_select` at high `p` is the only refinement, and a minor one. (Round 2
lowered the exact-routing threshold from 2048 to 1024 — see below.)

## Efficiency / assembler

The hot path of *every* variant is the single `algo::sized` partition pass — the
repo's branchless block/Lomuto partitioner, already studied in
`docs/partition_schemes.md`. Disassembly of `move_low_half` confirms the
stride-sample/count loop compiles to a branchless accumulate (`add` of the
compare result, no per-element jump) and the one O(n) pass is a call into
`branchless_partition`. Swapping the pivot to the inline branchless `ninther_pos`
is **neutral** here (the pivot is a negligible fraction of one partition) — the
same conclusion as the quicksort work: the partition, not the pivot, is the
budget.

## Round 2 — the reversed formulation (kill the move pass)

Round 1's variants all share one structural waste, pointed out by the requester:
they partition the small elements to the **front** and then pay a
`move_front_to_end` epilogue — `min(k, n−k)` extra swaps that exist *only*
because the partition put the bottom set at the wrong end. At `p = 0.9` that is
~45 % of `n` in 16-byte round-trip traffic for pairs. The repo already has the
mirror-image partition family (`algo_rev::sized_rev`, `reverse_partition.hpp`:
`[first, m) ≥ pivot | [m, last) < pivot`, measured at parity with forward —
role-exchanged scans, identical op count), so the fix is to make **every**
partition in the pipeline reversed; the bottom set then lands at the end as a
side effect and the epilogue vanishes:

- **`sample` (rev)**: stride sample → estimate the (S/2)-th value `t` → **one**
  `sized_rev` partition; the tail `[mid, last)` *is* the answer (`k = last −
  mid`, all `< t < key`). Bonus: the sample's rank-`r` element is now found with
  the repo's branchless quickselect instead of `std::nth_element` (worth ~3-7 %
  of the whole call at mid sizes).
- **`key_select` (rev)**: `sized_rev` by `key` lands **all** `S` below-key
  elements in the tail — the take-all base case is *already finished* at that
  point (one pass, zero extra work). Otherwise a new **reversed quickselect**
  (`detail::quickselect_rev`) splits the tail in place: each step parks the
  pivot at the left edge of its `≥` run (`[first, pp) ≥ key | *pp == key |
  [pp+1, last) < key`), recurses toward `nth = last − k`, and stops when `nth ∈
  {pp, pp+1}` (both are valid splits since the pivot equals the boundary
  value). The leaf is a **descending** `small_sort` network via the
  argument-swapped comparator — a strict weak order, unlike the negation
  `!comp`, which is reflexive and breaks sorting networks (the
  `small_halve_rev` lesson).
- **`rev_count_select`**: count `S` (move-free scan), then one
  `quickselect_rev` of the whole array — the high-`p` exact alternative.

Every output is verified as before (bottom-k + multiset), plus a dedicated CTest
suite (`tests/test_move_low_half.cpp`, ~24 000 checks: exact `k`, bottom-k,
multiset, all-below-key, take-all/empty/full edge cases, non-identity
projection).

### Large n (2²²), random high-cardinality, ns/elem

| | i64 p=.10 | p=.50 | p=.90 | pair64 p=.10 | p=.50 | p=.90 | pair64f p=.10 | p=.50 | p=.90 |
|---|---|---|---|---|---|---|---|---|---|
| `sample` fwd (round 1) | 0.42 | 0.60 | 0.73 | 0.69 | 1.04 | 1.24 | 0.92 | 1.10 | 1.34 |
| **`sample` rev (shipped)** | **0.43** | **0.44** | **0.48** | **0.65** | **0.74** | **0.72** | **0.67** | **0.78** | **0.76** |
| `key_select` fwd | 0.58 | 1.25 | 1.65 | 0.88 | 1.92 | 2.60 | 0.93 | 2.01 | 2.72 |
| **`key_select` rev (shipped)** | 0.53 | 0.90 | 1.34 | 0.86 | 1.29 | 2.01 | 0.99 | 1.38 | 2.17 |

The reversed `sample` is now essentially **flat in `p`** (the residual slope of
the forward version *was* the move pass): i64 0.43–0.48 ns/elem for *any* key
percentile, pair64 ≤ 0.74. At `p = 0.9` the win is **1.5–1.7×** for `sample`
and ~1.25× for the exact path; `k/S` accuracy is unchanged (same estimator).
Across the full sweep (4 sizes × 3 percentiles × 3 types × 2 distributions) the
reversed exact path wins or ties everywhere except two ~+6 % cells: `n = 2¹⁶,
p = 0.9` (i64 and pair64 — the array is L2-resident there, so the forward move
costs almost nothing and the reversed block partition gives back a few
percent) and pair64f `n = 2²², p = 0.10` (tiny select, 7-rep min of a big
partition — at `n = 2¹⁶` the same cell favours rev by 10 %). Localized,
measured exceptions, not worth a size special-case (at 2¹⁶ the *approximate*
path is 3× faster anyway).

### Small n (batched, pool 2²⁰, `random_uniform`), i64 ns/elem

| n | p | fwd_key_select | **key_select (rev)** | rev_count_select | **sample (shipped)** | part_until (biased) |
|---|---|---|---|---|---|---|
| 32 | .50 | 1.05 | **0.63** | 0.69 | 0.68 | 0.86 |
| 64 | .50 | 1.50 | **1.36** | 1.60 | 1.37 | 1.51 |
| 128 | .50 | 1.74 | **1.58** | 1.87 | 1.56 | 1.09 |
| 256 | .50 | 1.51 | **1.37** | 1.60 | 1.36 | 0.89 |
| 512 | .90 | 1.81 | 1.72 | **1.52** | 1.76 | 0.65 |
| 1024 | .50 | 1.21 | 1.25 | 1.37 | **1.01** | 0.78 |
| 1024 | .90 | 1.69 | 1.65 | 1.50 | **1.09** | 0.62 |
| 2048 | .50 | 1.18 | 1.10 | 1.35 | **0.90** | 0.75 |
| 2048 | .90 | 1.60 | 1.52 | 1.32 | **0.93** | 0.57 |

(`part_until` remains the raw-speed leader at mid/high `p` but its `k/S` is
0.56–0.67 — ~30 % over target; it stays rejected for the shipped functions.)

- **n = 32 is the headline small-n win (~1.6×, 1.05 → 0.63):** every percentile
  is in the take-all regime there, and take-all is now literally a single
  reversed partition pass — the entire round-1 cost beyond the partition (the
  move) was overhead.
- At sizes where a real select runs, the reversed exact path wins by ~3–10 %
  (the select dominates; only the move was removed). `rev_count_select` again
  edges it at high `p` (counting is move-free), same crossover `p ≈ 0.7` as
  round 1 — and since `p` is unknown a priori, `key_select` stays the shipped
  exact default.

### Sampling now starts at n = 1024 (was 2048)

With the move gone the sample path's only remaining costs are the sample+rank
(≈ `m` reads + an `m`-element select) and one partition, so a scaled-down sample
`m = min(n/4, 512)` was tested down to n = 256 (`rev_sample_n4` in the bench).
Speed: wins from n ≈ 256 at mid/high `p`. Accuracy is the constraint — measured
**per-call** `k/S` spread (2000 trials, i64; the batched `k/S` column hides
this because block errors average out):

| n (m) | p=0.25: 5th–95th pct | p=0.50 | p=0.90 |
|---|---|---|---|
| 256 (64) | 0.34–0.67 | 0.38–0.63 | 0.41–0.60 |
| 512 (128) | 0.38–0.63 | 0.41–0.59 | 0.43–0.57 |
| 1024 (256) | 0.41–0.59 | 0.44–0.56 | 0.46–0.55 |
| 2048 (512) | 0.44–0.56 | 0.46–0.54 | 0.47–0.53 |

The median is 0.500 everywhere (stride-sampling the *original* array is
unbiased; only the variance grows as `m` shrinks, ~`1/√(p·m)`). n = 256/512
spreads (±25 %/±15 % at mid `p`) were judged too loose for a default; from
n = 1024 the spread at healthy percentiles matches what the large-`n` path
already produces at its low-`p` edge. **Shipped routing:** exact below 1024,
`m = min(n/4, 512)` above — 10–35 % faster than round 1 at n = 1024–2048,
identical behaviour from n = 2048 up. (Callers wanting speed at 256–1024 with
±15–25 % `k` can lift `rev_sample_n4` from the bench.)

### Efficiency / assembler (round 2)

Disassembly of the shipped instantiations (i64, pair64, pair64-by-first;
`objdump -d` of `-O3 -march=native`):

- **`lomuto_branchless_rev`** (the small/mid exact hot loop): 12
  instructions/element, the pivot stays in a register for the whole loop (the
  by-value pivot rule — zero reloads), predicate is `cmp r9,rcx; setle` +
  conditional add; **no data-dependent branch**. Byte-identical to the forward
  loop except `setle` ↔ `setg`, confirming the role-exchange costs nothing.
- **`branchless_partition_rev`** (large-n fills): exactly **one** `vpcmpgtq`
  per vector block and **zero** `vpcmpeqq`/mask-inverts — the reversal does not
  add a single vector op.
- **Sample loop**: GCC vectorises the stride-4 case (n = 1024–2048) into
  `vpcmpgtq` + `vpsubq` mask-accumulation (a branchless vector count); larger
  strides compile to scalar `cmp/setcc/add` — load-bound either way.
- **Ninther**: fully inlined `setg`/cmov chains, no calls, no branches.

Nothing left on the table at the instruction level: the only O(n) work is the
one partition pass (plus the select's geometric tail for the exact path), and
that pass *is* the repo's measured-optimal partitioner.

## Shipped functions

`include/partitions/move_low_half.hpp` promotes the two round-2 winners (both
generic over `comp`/`proj`, verified across all types/sizes/percentiles by the
benchmark **and** by `tests/test_move_low_half.cpp`):

- **`move_low_half(first, last, key, comp, proj)`** — reversed `sample`: a
  stride sample of the original array (`m = min(n/4, 512)`, n ≥ 1024), repo
  quickselect for the sample rank, **one reversed partition**, done. `k ≈ S/2`
  (per-call 5th–95th pct ≈ ±5 % at large n, see the spread table). Falls back
  to the exact path for n < 1024 or when fewer than 16 samples are below the
  key.
- **`move_low_half_exact(first, last, key, comp, proj)`** — reversed
  `key_select`: one reversed partition by `key` (which *completes* the take-all
  base case by itself), then `detail::quickselect_rev` on the tail. `k = ⌊S/2⌋`
  exactly.

Both apply the take-all-when-small base case (`k = S` if `S/2 < 16`) and return
`k`.

## Caveats

- `sample` uses **stride** sampling (no RNG, deterministic). On *random* and
  *few-unique* inputs this is unbiased; on adversarially **periodic** input whose
  period aligns with the stride it could bias the estimate. A randomized offset
  would remove that at the cost of an RNG; it was not needed for the workloads
  here.
- Like the rest of this repo, the select/partition primitives have no 3-way path,
  so genuinely **degenerate-duplicate** keys lean on `algo::sized`'s
  pivot-exclusion for progress (covered by the `few_unique` rows, which stay
  well-behaved).
