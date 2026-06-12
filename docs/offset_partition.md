# Offset partition: forward partition with a known all-`>=` prefix

**Problem.** Given `[first, last)`, a `key` and an `offset` with the
precondition that every element of `[first, first+offset)` is `>= key`,
rearrange the array so all elements **strictly smaller** than `key` are at the
front, and return their count. Only the suffix `[first+offset, last)` ever
needs a comparison; the prefix is pre-classified. `offset == 0` reduces to the
ordinary forward partition, and when the below-count `c` exceeds `offset` the
prefix slots run out and the task degenerates into a normal partition of the
remainder — a good algorithm must handle both ends without special cases.

Shipped as `partitions::offset_partition(first, last, offset, key, comp, proj)`
(returns the count) dispatching over the `partitions::algo_off` family
(`include/partitions/offset_partition.hpp`). Like `algo_rev`, the family does
not model the 5-argument `PivotPartitioner` concept and is exercised by its own
suite (`tests/test_offset_partition.cpp`, `benchmarks/bench_offset_partition.cpp`).

## Parameters of the problem

The cost regime is set by three quantities:

* `n` — array length (cache- vs DRAM-resident);
* `f = offset/n` — prefix fraction (how much is pre-known);
* `p = c/(n-offset)` — below fraction **of the suffix**.

`f` and `n` are known at call time; **`p` is not** — so the shipped dispatch
may use `n`, `f` and `sizeof(T)` only, and the best p-dependent variant per
cell is an oracle bound, not an implementable policy.

## Candidates

| name | idea | compares | moves |
|---|---|---|---|
| `scan_swap` | branchy Lomuto from `offset`, store cursor at `first` | `n-off` | 3 per below (swap) |
| `gap_off` | branchless Lomuto gap method ([orlp](https://orlp.net)) **started mid-stream**: lift `v[offset]`, store index from 0 — the precondition *is* the loop invariant after `offset` virtual iterations | `n-off` | 2 per **suffix** element, sequential |
| `part_swap` | `algo::sized` on the suffix, then one `swap_ranges` of `min(c, offset)` to bridge the front | `n-off` | partition swaps + 3·`min(c,off)` vectorised |
| `fused_block` | whole-array pdqsort block partition with compare-free identity fill inside the prefix | ~`n-off`…`n` | block-partition swaps |
| `prefix_fill` | **new**: scan the suffix from the right in branchless 128-blocks (pdqsort offset fill); rotate each found below directly into the next front slot (2 moves, no compare on the prefix); finish the remainder as `part_swap` | `n-off` (exact, no rescans) | 2 per below + remainder |
| `whole` | `algo::sized` over everything, precondition ignored (reference) | `n` | partition swaps |

`scan_swap`/`gap_off` correctness note: the Lomuto invariant
"`[first, store) < key`, `[store, cur) >= key`" holds at `store = first`,
`cur = first+offset` *by the precondition*, so both genuinely start mid-stream;
no warm-up pass exists in either.

## Results

Meteor Lake, GCC 15.2, `-O3 -march=native`, random high-cardinality keys
(unique shuffled ranks), suffix uniformly shuffled. Metric: **ns per suffix
element** = min ns / (n − offset) — the fair cross-`f` metric, since the prefix
is free by contract. (CSV column 6 has the whole-array view.) Selected cells;
full tables from `bench_offset_partition large` / `small`.

### i64, n = 2^22 (DRAM-resident)

| f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.1 | 1.32 | 0.52 | 0.49 | 0.49 | **0.45** | 0.50 |
| 0.1 | 0.5 | 3.30 | 0.55 | 0.54 | 0.53 | **0.51** | 0.55 |
| 0.1 | 0.9 | 1.35 | 0.63 | 0.54 | **0.50** | 0.53 | 0.53 |
| 0.5 | 0.1 | 1.33 | 0.52 | 0.50 | 0.83 | **0.45** | 0.83 |
| 0.5 | 0.5 | 3.27 | **0.58** | 0.80 | 0.90 | 0.62 | 0.98 |
| 0.5 | 0.9 | 1.41 | **0.65** | 0.98 | 0.96 | 0.80 | 1.13 |
| 0.9 | 0.1 | 1.27 | 0.52 | 0.50 | 3.75 | **0.45** | 3.73 |
| 0.9 | 0.5 | 3.27 | **0.56** | 0.77 | 3.83 | 0.61 | 3.92 |
| 0.9 | 0.9 | 1.36 | **0.61** | 0.86 | 3.93 | 0.79 | 4.08 |

### pair64 lexicographic, n = 2^22

| f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.5 | 3.80 | 1.07 | 0.79 | 0.77 | **0.75** | 0.77 |
| 0.5 | 0.1 | 1.54 | 0.88 | 0.82 | 1.26 | **0.70** | 1.28 |
| 0.5 | 0.5 | 3.78 | 1.06 | 1.33 | 1.39 | **1.03** | 1.46 |
| 0.5 | 0.9 | 1.79 | **1.27** | 1.95 | 1.44 | 1.31 | 1.50 |
| 0.9 | 0.1 | 1.54 | 0.80 | 0.76 | 5.74 | **0.70** | 5.73 |
| 0.9 | 0.5 | 3.74 | 1.01 | 1.21 | 6.27 | **0.97** | 6.07 |
| 0.9 | 0.9 | 1.72 | **1.19** | 1.63 | 5.95 | 1.26 | 6.03 |

pair64-by-first (16-byte element, 8-byte key) tracks the pair64 shape with
slightly lower absolute numbers (e.g. f=.5 p=.5: prefix_fill 1.04, gap 1.06,
part_swap 1.46).

### Pair of i64 compared by first coordinate — the primary case (mode `byfirst`)

This is the most important production shape, so it gets a dedicated study
(`bench_offset_partition byfirst`; it also runs **first** in the default
mode). Three variants per (n, f, p) cell, same data geometry:

* `p64f_proj` — projection formulation: `proj = first_key`, 8-byte key;
* `p64f_comp` — comparator formulation: `comp` reads `.first` only, the key
  is a full 16-byte `pair64` **passed by value**;
* `p64f_dup256` — projection formulation on low-cardinality firsts
  (256 distinct values, heavy ties).

**Findings:**

1. **The two formulations are interchangeable.** Over all 27 cells × 7
   algorithms, comp/proj time ratio: mean 1.004, median 1.002 (extremes
   ±13% are single-cell noise). The disassembly explains why: GCC keeps only
   the *live* 8-byte half of the by-value pair pivot, hoisted into a register
   (`cmp %rbx, -0x10(%rdx); setl; add` in the fill loop) — the `.second`
   half of the key is dead-code-eliminated. The by-value pivot rule
   (concepts.hpp) extends to 16-byte keys at zero cost.
2. **Low cardinality does not flip any regime.** `p64f_dup256` tracks
   `p64f_proj` within noise in every cell and the per-cell winners are the
   same. With a partition's single fixed key, the below-test outcome is
   ~Bernoulli(p) per element whether firsts are unique or drawn from 256
   values — ties change *which* elements are below, not the predictability
   of the branch. (The boost_block low-cardinality caveat needs equal keys
   *adjacent in scan order*, e.g. all-equal blocks, not just duplicates.)
3. **Winner structure for by-first pairs** matches pair64-lex: `prefix_fill`
   at p=.1 and at f=.1 (0.70–0.80 vs gap 0.85–1.13 at 2^22), `gap_off`
   closing to ~5% parity (occasionally ahead: f=.9 p=.9 2^22: 1.10 vs 1.19)
   once the prefix dominates at large n. The dispatcher's 16-byte routing
   (always `prefix_fill` above the cutoff) gives up at most ~5% in those
   cells and gains 1.4–1.5x at small f — the right p-blind trade.
4. **Correction recorded:** the "gap_off is 4–6 ns on 16-byte types" figure
   from the batched-small study is a small-block effect; at large single-call
   sizes by-first gap_off runs 0.7–1.2 ns/suf-elem. The dispatcher rationale
   in the header was updated accordingly.

### Batched small blocks (f = 0.5, pool 2^20)

i64: the dispatcher's `gap_off` routing wins nearly every cell up to suffix
512 (n=1024); e.g. n=256, p=.5: sized_off 0.60 vs part_swap 0.69, prefix_fill
0.72, scan_swap 3.23. pair64: `prefix_fill`/`part_swap` from n=256 up
(gap_off's 16-byte double moves run 4–6 ns/suf-elem); below that, per-call
overhead dominates and the cells are within ~1 ns of each other.

## What the data says

1. **Exploiting the precondition is worth up to 8x** (prefix_fill 0.45 vs
   whole 3.73, i64 2^22 f=.9 p=.1) — the floor is the `1/(1-f)` compare
   saving; beating `part_swap` on top of that needs single-pass moves.
2. **`fused_block` is a trap (kept as a negative result).** A forward block
   partition's *left* cursor stops at the boundary `c`; when `c < offset` the
   identity-fill never fires beyond `c` and the *right* scan re-compares the
   prefix from the other side. At f=.9 it matches `whole` to within noise.
   The prefix must be treated as swap *targets*, not scan input.
3. **`prefix_fill` is the right single-pass formulation.** Exactly `n-offset`
   compares (branchless fill, verified `cmp; setg; add` with no data-dependent
   jump), 2 moves per below element via a cyclic rotation, prefix never
   touched by a compare. It is the best or tied variant in most cells and the
   best *implementable* (p-blind) choice for 16-byte types everywhere and for
   narrow types at small `f`.
4. **`gap_off` is the p-flat workhorse**: 1 compare + 2 *sequential* moves per
   suffix element no matter what. It loses to prefix_fill at low p (moves
   everything for nothing) but wins high-p cells where prefix_fill's scattered
   per-below rotations cost more than gap's streaming stores. Averaged over
   uniform p it is the better narrow-type choice once the prefix dominates
   (`offset >= suffix`), which is the dispatcher's second rule.
5. **`part_swap` is an honest composition but moves belows twice.** Its
   epilogue is `min(c, offset)` AVX-vectorised swaps, so it ties prefix_fill
   when either `f` or `p` is small, and loses up to 1.5x when both are
   mid/high. It survives as prefix_fill's phase-2 (and the small-suffix
   remainder), where the window left to bridge is small by construction.
6. **`scan_swap` only wins when the branch is predictable** (p near 0 or 1 and
   cache-resident: 0.44 at f=.9 p=.9 n=2^16) — the same branchy-vs-branchless
   caveat as `algo::boost_block`'s low-cardinality note. Not worth a path:
   the predictable-p caller can pick it explicitly.

## Shipped dispatch (`sized_off` / `offset_partition`)

```
suffix <= sized_cutoff<T> (512 narrow / 24 wide)  ->  gap_off
sizeof(T) <= 8 && offset >= suffix                ->  gap_off   (expected-cost rule)
otherwise                                         ->  prefix_fill
```

Measured over the full 81-cell sweep, mean regret vs the per-cell oracle is
~5%; the worst structural cells are ~15% (i64 f>=.5 p=.1, where the
narrow-type rule picks gap over prefix_fill, and the predictable-branch
scan_swap cells it deliberately ignores).

## Zero-offset identity (mode `zero`)

`offset == 0` is, by contract, the ordinary forward partition — so the
dispatcher must cost the same as raw `algo::sized` there, or the offset
machinery has hidden overhead. Mode `zero` times every variant at f = 0
against `whole` (= raw `algo::sized`, offset ignored) on identical data:
3 types × {64…4096 batched, 2^16…2^22 single} × p ∈ {.1,.5,.9}.

**Result: the identity holds.** Over all 63 cells, `sized_off`/raw ratio:
mean **1.0015**, median 1.0012, σ 2.3%, extremes 0.93–1.07 *in both
directions* (paired noise, not a one-sided tax). `prefix_fill` alone at
offset 0, n ≥ 2^16: ratio 0.998 — statistically identical. Batched tiny
blocks: mean 1.011, a ~1% residual at the edge of noise.

**Why it holds, by construction (verified in the disassembly):**

* Large path: at offset 0, `sized_off` → `prefix_fill` (the narrow-type rule
  `offset >= suffix` is false); phase 1's guard `pfx_end - lo >= 128` fails
  on the first test, and phase 2 calls `algo::sized` on the full range. Both
  routes bottom out in the **same out-of-line `branchless_partition`
  instantiation** (one symbol in the probe object, called from both).
* The phase-2 bridge degenerates safely: slots = 0 forces the
  `swap_ranges(lo, pfx_end, …)` arm with an **empty** range — no spurious
  self-swap (the `c_rem <= slots` arm would have been
  `swap_ranges(first, first+c, first)`, a real in-place swap pass; the
  branch order avoids it for every `c_rem > 0`).
* Small path: `gap_off(offset=0)` is literally `lomuto_branchless` — same
  11-instruction branchless body (`2 moves; cmp; setg; add`); the only
  difference is scheduling (raw's loop software-pipelines the next `v[i]`
  load, gap reloads at the loop top; measured parity, see the gap_off
  columns at n ≤ 512).

**One asymmetry found in standalone codegen (compare assemblers):** compiled
as an out-of-line function, `sized_off` pays `prefix_fill`'s *hoisted* frame
before any dispatch check runs — 64-byte stack realignment, six register
pushes, a 0x140-byte frame for the offset buffer, and a stack-protector
canary (`mov %fs:0x28`) — while raw `algo::sized`'s small path runs
frameless (GCC keeps `branchless_partition` out of line there, so the lean
path stays lean). This is ~10–15 cycles per call, **not reproduced at real
(inlined) call sites** — the benchmark shows mean 1.0015 — and would matter
only for a non-inlinable wrapper called at high rate on tiny arrays.
Splitting `prefix_fill` behind `[[gnu::noinline]]` to fix the standalone
shape was considered and **not adopted**: it trades an unconditional call
into the hot large-n path for a cost the measurements cannot see (negative
result, recorded). For the same reason no `offset == 0` early-exit branch
was added to the dispatcher — there is nothing to win.

## Assembly notes (GCC 15.2, `-O3 -march=native`, objdump of noinline probes)

* `gap_off` i64 hot loop is 7 instructions: two 8-byte moves,
  `cmp; setg; movzbl; add`, pointer bump, backedge. **No data-dependent
  branch**; the key stays in a register the whole loop (by-value pivot rule —
  same finding as `concepts.hpp`).
* `prefix_fill`'s right-block fill is the intended pdqsort pattern: 8×
  unrolled `cmp; setg; add` + unconditional byte store into the
  cacheline-aligned offset buffer; descending loads handled by the hardware
  prefetcher.
* **Found and fixed:** the first rotation implementation reused
  `swap_offsets` with a constexpr identity table for the left side; GCC kept
  the per-element `movzbl identity[i]` load and its address dependency. A
  dedicated rotation indexing `lo[i]` directly (`detail::fill_slots`) removes
  it; the only remaining byte load is the genuine offsets buffer.
* `part_swap`'s `swap_ranges` epilogue vectorises to 32-byte `vmovdqu`
  pairs — why its double-move stays competitive at low `min(c, offset)`.
* pair64 lexicographic fills keep the **short-circuiting** compare
  (`cmp; je <second-key>; cmov…`): the equality branch is rarely taken on
  high-cardinality data and predicts perfectly — consistent with the
  documented negative result on fully-branchless lex compares for fixed-pivot
  partitions (`algorithms.hpp`).

## Negative results kept

* `fused_block` identity-fill block partition (see above) — wrong cursor.
* Generic `swap_offsets` + identity table for the rotation — dead load GCC
  cannot fold (fixed by `fill_slots`).
* Reversed-partition formulation (smalls to the END, as in `move_low_half`)
  does not fit this contract: the required front placement would re-introduce
  a `min(c, n-c)` move pass, which is never smaller than `part_swap`'s
  `min(c, offset)` bridge.
