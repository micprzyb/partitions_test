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

| ns/elem, n=2²², vs library ninther | i64 | pair64 lex | pair64f |
|---|---|---|---|
| inline branchless ninther | **+3.0%** | **+3.8%** | −4.0% |

Summed over the three target types branchless nets **+2.8%**, the best of
branchless / branchy / library — so it is the raw-throughput choice. (`pair64f`
loses because the repo generator makes its `.first` low-cardinality and the
cheap, well-predicted median compares prefer the early-out `if` to a longer
`cmov` chain; on the two healthy types it is a clear win.)

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

## Threshold sweep

Median of 3, n = 2²², ns/elem:

| type / dist | T8 | T12 | T16 | T20 | T24 | T32 | best |
|---|---|---|---|---|---|---|---|
| i64 random        | 20.4 | 19.8 | 19.6 | **19.4** | 19.8 | 20.3 | T20 |
| pair64 random     | 48.6 | 47.3 | **46.6** | 46.6 | 47.0 | 48.3 | T16/20 |
| pair64f random    | 64.0 | 63.5 | 62.4 | 61.7 | 61.3 | 61.3 | →larger (low-card) |
| i64 sorted_asc    | **12.3** | 12.6 | 13.1 | 13.1 | 13.3 | 13.8 | T8 |

A flat plateau T16–T24 on random; **`qslr_threshold = 20`** (i64 peak, pair64
tied). Selection sort being non-adaptive, sorted input prefers a smaller leaf;
20 keeps that penalty bounded. `pair64f` drifts toward larger thresholds only
because of the low-cardinality `.first` generator.

## Head-to-head

Median of 3, ns/elem, at the tuned default T20, vs the recursive halver-leaf
`partitions::quicksort`, `boost::pdqsort` and `std::sort` (`lr/x` = speedup of
`quicksort_lr`; random at n=2²², sorted at n=2²⁰):

| type / dist | quicksort_lr | quicksort | pdqsort | std::sort | lr/qs | lr/pdq |
|---|---|---|---|---|---|---|
| i64 random        | 19.2 | 18.3 | 23.5 | 61.7 | 0.95 | **1.23** |
| i64 sorted_asc    | 12.0 |  9.4 | 0.52 |  6.7 | 0.78 | 0.04 |
| pair64 random     | 47.7 | 44.8 | 83.9 | 88.5 | 0.94 | **1.76** |
| pair64 sorted_asc | 15.6 | 18.5 | 1.88 | 16.6 | **1.18** | 0.12 |
| pair64f random    | 62.7 | 60.9 | 43.5 | 50.2 | 0.97 | 0.69 |
| pair64f sorted_asc| 56.9 | 53.8 | 1.56 |  8.5 | 0.95 | 0.03 |

**Reading it.** The rewrite lifted `lr/qs` on the healthy random cases from
~0.91–0.93 (the first version) to **0.94–0.97** — i.e. within ~5% of the
unconstrained recursive `quicksort`, the remaining gap being the selection leaf
vs its halver leaf (the halver does a balanced split with ~20–33% fewer
compare-exchanges, no index bookkeeping). It is still **1.23× faster than
`boost::pdqsort` on `i64`** and **1.76× on `pair64` lexicographic** (both repo
quicksorts use a branchless block partition where pdqsort goes branchy on the
16-byte lex compare), and it even beats the recursive `quicksort` on sorted
`pair64` (1.18×).

It inherits the repo quicksort's two documented weaknesses, both orthogonal to
the three constraints: no **3-way** equal partition (so the low-cardinality
`pair64f` generator costs ~1.4× vs pdqsort) and no **pattern defeating** (so
already-sorted input is ~20× pdqsort, which detects the run). Adding either would
help all three constraints equally and is the obvious next step.
