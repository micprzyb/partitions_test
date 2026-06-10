# `quicksort_lr` — left-to-right, non-recursive, selection-leaf quicksort

**Goal.** Add a quicksort variant under three hard constraints, then make it as
fast as the constraints allow on the three target key types (`i64`, `pair64`
lexicographic, `pair64`-by-first):

1. **Selection-sort leaf.** Blocks below a threshold are finished by selection
   sort — *find the minimum of the unsorted suffix, swap it to the front,
   repeat* — and by no other method (no insertion sort, no sorting network, no
   halver). The only freedom is *how the minimum is found*.
2. **Left-to-right.** At each partition step the left side is sorted to
   completion before the right is touched; the array is finalised front-to-back.
3. **No recursion.** The recursion stack is materialised by hand.

Measured on this host (Meteor Lake / Core Ultra 7 165H, GCC 15.2,
`-O3 -march=native`), min ns/elem over repeated runs (the repo protocol). The
absolute numbers drift ±10% between sessions (frequency/thermal), so the robust
metric is the **intra-session ratio** vs a reference. Files:
`include/partitions/quicksort_lr.hpp`, `tests/test_quicksort_lr.cpp`,
`benchmarks/bench_quicksort_lr.cpp`.

## Where the time goes (and why "stop splitting early" is the lever)

A threshold quicksort has ~`n/Threshold` internal nodes, almost all of them one
partition away from a leaf. So the cost splits roughly: **partition ~75%**,
**selection leaf ~22%**, **pivot ~a few %**. Crucially the *per-node fixed*
overhead (pivot selection, the two pivot swaps, the stack op) is what makes a
*small* threshold lose despite doing fewer total comparisons — so making each
node cheap is what lets the threshold pay off. Every optimisation below was
chosen by measurement, and several promising ones were measured and **rejected**.

## Structure (constraints 2 & 3): the lean stack

The driver keeps an explicit array stack and, after partitioning `[lo,hi)` around
the pivot rank `pp`, **pushes** the right child and **continues** on the left —
so the left spine is descended first and the array is finalised left-to-right.

Because we always go left, **only the right's `lo` (= `pp+1`) is stored**: the
deferred rights tile the suffix adjacently (separated by the in-place pivots), so
a right's `hi` equals *(the `lo` of the entry below it) − 1*, or `last` at the
bottom — derived on pop. That halves the stack traffic vs storing `(lo,hi)`
pairs (a small win, but the elementary one).

This forbids the usual "recurse the smaller side" trick (it would sort the
smaller — possibly right — side first), so the stack depth is the *left-spine
height* rather than a guaranteed `O(log n)`. An all-inline **ninther** pivot
keeps it logarithmic on every distribution in the matrix (`depth` study, worst
**26** at n=2²² ≈ 1.1·log₂n), so `qslr_stack_cap = 192` has a >7× margin and is
asserted. Like the recursive `quicksort`, there is no 3-way equal partition, so
heavy-duplicate input degrades toward `O(n²)` — the same non-adversarial
contract.

## Making each node cheap (the per-node overhead)

### Pivot — inline, branchless ninther (the one real win)

Disassembly of the first version exposed the culprit: the library
`pivot::ninther` builds its median-of-3 through a `make_less` lambda that GCC
**outlines into an `.isra` clone and CALLS 4×/node** for the 16-byte `pair64`.
Two fixes, both measured in the `ab` study (one session, ratio vs the library
ninther, same driver/threshold/stack/leaf so frequency drift cancels):

1. **Inline** the ninther from a local median-of-3 (no lambda → no call).
2. Make that median-of-3 **branchless** — `max(min(a,b), min(max(a,b),c))` with
   the iterator carried beside its key, lowering to `cmp + cmov` instead of the
   nested-`if` that mispredicts ~50% on random keys. Disassembly: the inline
   ninther drops from **15→3** branches on `i64` (29 `cmov`), 12→3 on `pair64f`,
   64→26 on `pair64` (the residual = the well-predicted lex short-circuits).

| ns/elem, n=2²², vs library ninther (fair data) | i64 | pair64 lex | pair64f |
|---|---|---|---|
| inline branchless ninther | **+3.5%** | **+3.9%** | **+0.3%** |

Faster or tied on **all three** target types — the best of branchless / branchy /
library, so it is the raw-throughput choice. (An earlier run showed `pair64f`
−4%, but that was the low-cardinality `.first` generator artifact; on fair
high-cardinality keys it is neutral.)

### Median-of-3 size tier — measured and REJECTED

Using a cheaper median-of-3 for the many small nodes (ninther only for large) is
the obvious next idea. The `pivcut` sweep killed it: it is a **wash** on random
(once the ninther is inline the pivot is only ~3% of comparisons), and large
cutoffs are **unsafe** — median-of-3 degenerates on `sorted_descending` (the
m3/partition recursion interaction) into an `O(n²)` deep left-spine that
**overflows the fixed stack** (two hard crashes during the sweep). Kept only as a
reproducible study; disabled by default (`qslr_pivot_ninther_cutoff = 0`).

### Partitioner — `algo::sized` confirmed, alternatives REJECTED

Since most nodes are small, a setup-free branchless Lomuto might beat
`boost_block` (which cacheline-aligns two offset buffers). The `part` study says
no: for `pair64` lex, forcing Lomuto onto more nodes **hurts** (+10–14% — the
expensive lex compare wants boost's ILP), and `boost`-everywhere is the **worst**
for `i64` (+15%). `algo::sized` (branchless Lomuto for narrow/small, boost for
wide/large) is best or within ~2% for all three. No change.

## The improved find-minimum (the selection leaf, constraint 1)

The leaf is ~22% of the work and selection sort has no early-out, so the
per-compare constant of the min-scan is the whole game. Measured in the `min`
study and confirmed by disassembly:

- **Write the update branchy**, `if (comp(k,best)) { best=k; idx=j; }` — GCC
  lowers it to **1 `cmp` + 2 `cmovg`** (running min in a register). The
  "obviously branchless" `best = c?k:best; idx = c?j:idx;` clobbers `best` before
  the index predicate and so wastes a **second compare**.
- **`find_min` dispatches on `sizeof(K)`**: narrow keys (`i64`, `pair64f`) use a
  **two-accumulator** scan (even/odd interleave) that breaks the loop-carried
  `cmp→cmov→cmp` chain and wins from L≈16 (i64 L=64: 13.0 vs 18.6 ns/elem);
  the 16-byte `pair64` key uses a **single accumulator** because forcing the
  two-accumulator branchless blend pays a 16-byte conditional blend on *every*
  element while min-updates are rare — 2–3× slower.
- **Cost of the constraint:** an out-of-spec double-ended min+max selection (one
  scan, both ends) is ~1.5–1.9× faster, bounding what find-min-only costs.

## Why the selection leaf is the ceiling (latency vs throughput)

The natural objection is: a selection sort of an `L`-block does only `L`
min-searches + ≤`L` swaps, with *far fewer moves* than the halver leaf and "min
is easier than median" — so it ought to be *much* faster, and `quicksort_lr`
ought to beat `quicksort`. The `leaf` study (selection vs the halver leaf vs
insertion on random blocks, with op counts) shows why it does not — and the
answer is **instruction-level parallelism**, not the move/compare counts:

| L=24 block | i64 ns | cmp | mov | pair64 ns | cmp | mov |
|---|---|---|---|---|---|---|
| halver (quicksort) | **3.5** | 202 | 606 | 10.4 | 202 | 606 |
| selection (ours)   | 7.0 | 299 | **61** | **9.4** | 276 | **61** |
| insertion          | 6.2 | 276 | 828 | 55.4 | 276 | 828 |

Selection really does ~10× fewer moves (61 vs 606) and *fewer total operations*
(360 vs 808) — yet on `i64` it is **2× slower**. The reason:

- The halver is a **compare-exchange network**: its operations are
  **data-independent**, so the out-of-order core issues 4–6/cycle
  (throughput-bound). 808 parallel ops finish in 3.5 ns/elem.
- Selection's min-search is a **serial reduction** — `min ← cmov(min, a[j])` is a
  **loop-carried dependency** (each compare needs the previous running min), so it
  runs at ~1 step per `cmp+cmov` latency (~2–3 cycles), **latency-bound**,
  regardless of how few ops it is.

So the outcome is **move-cost dependent**: on cheap-move `i64` the halver's
parallelism wins decisively; on expensive-move 16-byte `pair64` selection's
move savings make it a *tie* (it even wins at L=16/24). End-to-end `lr/qs ≈ 0.95`
is exactly the i64 leaf losing the ILP race, partly offset by the pair64 leaf.

This is **not an implementation problem**: the shipped `find_min` already uses 2
accumulators to break the serial chain (without it selection is ~3× slower);
more accumulators were measured (`min` study `bl4`) and lose to register/merge
overhead. (The out-of-spec min+max double-selection — half the passes — is ~1.7×
faster, which is the measured cost of constraint 1.)

### Tournament / tree-selection: fewer operations, but *slower* (the `leaf` study)

The deeper objection is that the min-search need not be a linear scan at all: a
**tournament** finds the min in log-depth (parallel) and, with **reuse**, makes
the whole selection **O(L log L)** comparisons (recompute only the root path),
moving *indices* not elements. I implemented and measured three forms — a winner-
tree tournament, a **branchless** tournament (`+inf` sentinel, no removed-leaf
mispredict), and a 2-level **grouped** selection. The op-counts confirm the
theory; the timings invert it (i64, L=24, median of 5):

| method | ns/elem | comparisons |
|---|---|---|
| halver (data-*independent* network) | **3.6** | 202 |
| selection (linear, 2-acc) | 6.5 | 299 |
| grouped 2-level | 8.3 | 210 |
| tournament, branchless | 16.1 | 151 |
| tournament, full tree | 19.4 | **83** |

The tournament does **3.6× fewer comparisons** yet runs **3× slower**. Cause:
**fewer ops ≠ faster at small L** — the core is bound by dependency latency and
access pattern, not op-count. Reuse needs a position-indexed structure, so the
sift-up is a **serial chain of indexed `tree[]`/`key[]` loads** (the branchless
variant proves it is the *structure*, not the branches); the linear scan instead
streams the array **sequentially** with a short `cmov` chain, and the halver is
**data-independent** (no dependency at all → maximal ILP).

### Min-finding *networks* (`min_index<N>`): the register-pressure wall

The natural fix is to make the min-finder a **data-independent network** like the
halver: a fixed, branchless, **log-depth balanced-tree** reduction over the N
register-resident keys, tracking the winner index by `cmov` (`selnet` in the
`leaf` study). The disassembly is exactly what you'd hope **for scalar keys** —
and exactly the problem **for the wide key**:

| `min_index<N>` | insns | cmov | branches | stack-spill ops |
|---|---|---|---|---|
| n=24, i64 | 142 | 37 | 0 | **0** |
| n=24, pair64-by-first | 142 | 37 | 0 | **0** |
| n=24, pair64 lex | 570 | 122 | 62 | **95** |

Yet the network is **slower than the 2-accumulator scan** (i64 L=24: 9.1 vs 6.1;
pair64: 25.9 vs 9.7 ns/elem). Three measured reasons:

1. **Register pressure.** A log-depth tree keeps `O(width)` live `(key,index)`
   pairs; for 16-byte `pair64` that exceeds the GP registers → **95 spills** (+ 62
   short-circuit branches from the lex compare). The 2-accumulator scan keeps
   **two**. The `min` study already shows the trend: **4 accumulators lose to 2** —
   going wider trades depth for pressure and the optimum is **2-wide**. The tree
   is the wide extreme.
2. **Front-end bloat (scalar).** `selection_sort_n<L>` unrolls L networks
   (~1800 instructions at L=24); the core is decode-bound, while the 2-accumulator
   loop runs from the µop cache.
3. **The structural reason (decisive).** The halver leaf is **one** data-
   independent network — independent sub-halvings, no data-dependent address or
   control, fully pipelined. Selection sort is **L serial passes**: find-min →
   **swap** → find-min of the next suffix, which *reads the just-swapped array*.
   Each pass depends on the previous swap's store. A faster per-pass min-network
   cannot remove the serialization *between* passes. So the halver wins
   *structurally* (parallel network vs serial passes), and no selection-based
   method — linear, tournament, or network — can match it, because "find the
   minimum and swap, and proceed" is a serial chain of L passes by construction.

The ranking, then: **data-independence + parallel structure > low register
pressure > op-count**, with op-count last. The 2-accumulator linear scan is the
optimal *selection* leaf precisely because it is the lowest-register-pressure
2-wide network; the halver's edge is the parallel structure that constraint 1
forbids. (All variants — tournament, branchless tournament, grouped, and the
`min_index<N>` networks — are kept in the `leaf` study as documented negative
results.)

## Threshold sweep

Median of 3, n = 2²², ns/elem (fair high-cardinality data):

| type / dist | T8 | T12 | T16 | T20 | T24 | T32 | best |
|---|---|---|---|---|---|---|---|
| i64 random     | 17.8 | 16.8 | **16.5** | 16.5 | 16.7 | 17.2 | T16/20 |
| pair64 random  | 42.1 | **40.4** | 40.6 | 40.6 | 40.9 | 42.2 | T12–20 |
| pair64f random | 23.8 | 22.7 | **21.7** | 22.1 | 23.2 | 22.7 | T16 |

A flat plateau **T12–T20** on random for all three types — including `pair64f`,
which on fair high-cardinality data behaves like the others (the earlier "drifts
to larger thresholds" was the low-cardinality artifact). **`qslr_threshold = 20`**
sits in the plateau (within ~1% of each type's optimum). Selection sort being
non-adaptive, already-sorted input prefers a smaller leaf; 20 keeps that bounded.

## Benchmark fairness: the cardinality trap

The first cut of this head-to-head produced nonsense — `std::sort` *faster* on
`pair64f` than on `i64`, and both repo quicksorts *slower* on `pair64f` than on
`pair64` lex. Cause: a **data-generation artifact**, not the algorithms.
`dist::random_uniform` draws ranks in `[0,n]` and `make_value<pair64>(rank) =
{rank>>8, rank&0xff}`, so the `i64` key and the `pair64` *lexicographic* key are
both `rank` (**63% distinct**), but the `pair64`-**by-first** key `.first =
rank>>8` is only **0.4% distinct** — a hidden low-cardinality test. Comparing a
low-cardinality sort against high-cardinality ones is apples-to-oranges (the
no-3-way quicksorts degrade on the duplicates; `std::sort`/`pdqsort` *speed up*
on fewer distinct values). The benchmark now routes all generation through
`gen_data`, which puts the **full rank** in `.first` for the by-first case so
every type sorts a key of the **same** cardinality, isolating the real
difference (16-byte vs 8-byte element, identical `i64` compare). The numbers
below are the corrected, fair ones.

## Head-to-head

Median of 3, ns/elem, at the tuned default T20, vs the recursive halver-leaf
`partitions::quicksort`, `boost::pdqsort` and `std::sort` (`lr/x` = speedup of
`quicksort_lr`; random at n=2²², sorted at n=2²⁰):

| type / dist | quicksort_lr | quicksort | pdqsort | std::sort | lr/qs | lr/pdq |
|---|---|---|---|---|---|---|
| i64 random        | 16.5 | 15.6 | 20.2 | 53.5 | 0.94 | **1.23** |
| i64 sorted_asc    |  9.7 |  7.8 | 0.39 |  5.6 | 0.80 | 0.04 |
| pair64 random     | 40.1 | 38.0 | 73.0 | 77.0 | 0.95 | **1.82** |
| pair64 sorted_asc | 12.9 | 15.0 | 1.37 | 12.8 | **1.16** | 0.11 |
| pair64f random    | 21.5 | 22.0 | 53.5 | 57.2 | **1.03** | **2.49** |
| pair64f sorted_asc|  9.8 |  9.6 | 1.13 |  6.6 | 0.97 | 0.11 |

Now the ordering is sensible — `i64 < pair64f < pair64` lex for **every** sorter
(wider element, then expensive lex compare).

**Reading it.** On random high-cardinality input `quicksort_lr` is **1.23–2.49×
faster than `boost::pdqsort`** (both repo quicksorts use a branchless block
partition; pdqsort goes branchy on the 16-byte lex compare), and is **competitive
with the unconstrained recursive `quicksort`** — `lr/qs` 0.94–1.03, in fact
**ahead on `pair64f` (1.03)** and on sorted `pair64` (1.16). The one place it
trails `quicksort` is **cheap-move `i64`** (`lr/qs` 0.94 random, 0.80 sorted):
that is exactly the selection-vs-halver leaf gap from the `leaf` study — on
16-byte elements the selection leaf's far fewer moves cancel it, but on 8-byte
`i64` the halver's parallel network wins. The remaining weakness is **pattern
defeating** (no run detection → already-sorted input is ~20× pdqsort, which
*does* detect it); and genuinely **low-cardinality** input (the `few_unique`
distribution, not the now-fixed random `pair64f`) still degrades for want of a
**3-way** partition. Both are orthogonal to the three constraints and are the
obvious next steps.
