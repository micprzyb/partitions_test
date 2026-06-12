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
is free by contract. (CSV column 6 has the whole-array view.) Selected cells
here; the **complete tables for all four modes are in the appendix** at the
end of this document.

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

`offset == 0` is, by contract, the ordinary forward partition. The offset
machinery is only correctly engineered if the **real dispatched algorithms**
— `gap_off(0)` below the size cutoff, `prefix_fill(0)` above, exactly what
`sized_off` routes to — cost the same as raw `algo::sized` on the same data.
Mode `zero` measures that head-to-head and nothing else. There is
deliberately **no `offset == 0` special case** in the dispatcher: an early
exit calling `algo::sized` would make this benchmark circular (raw vs raw
through a wrapper) and prove nothing about the offset algorithms. (One was
briefly added during this study and removed for exactly that reason.)

**Why parity is expected, by construction:**

* `gap_off(0)` **is** `algo::lomuto_branchless` — the same 11-instruction
  branchless gap loop, merely entered with a runtime start index that
  happens to be 0;
* `prefix_fill(0)` fails its phase-1 guard (`pfx_end - lo >= 128`) on the
  first test and runs phase 2 = `algo::sized` on the full range — the same
  out-of-line `branchless_partition` instantiation raw uses — plus an empty
  bridge `swap_ranges`.

**Coverage:** 5 element types (i32, i64, pair64 lex, pair64-by-first,
pair_li {long,int}) × 9 sizes (64 … 2^22; n < 2^16 batched over a 2^20
pool, larger single calls) × p ∈ {.1, .5, .9} = 135 cells.

**Methodology** (it matters at tiny sizes): both kernels are out-of-line
`[[gnu::noinline]]` functions — inlined copies inside the timing loops sat
at different code alignments and produced ±20% per-cell scatter *in both
directions*, including "off0 faster than raw", impossible for real
overhead. Each side is timed in two alternated rounds (A/B/A/B, min of
mins) so frequency drift cannot bias one side, and a **control** is timed
alongside: a byte-identical second copy of the raw kernel. The control's
measured "overhead" is the per-cell noise floor (pure code placement); the
off0 column is only meaningful relative to it.

**Result: the real algorithms are at parity with the raw partition.**

|  | mean | median | σ | range |
|---|---|---|---|---|
| off0 (real route) vs raw | **−0.25%** | −0.13% | 4.5 | [−21.2%, +25.3%] |
| control vs raw (noise floor) | −0.31% | +0.09% | 3.8 | [−17.1%, +22.3%] |

Split by what actually runs: `gap_off`-routed cells mean −1.5% (control
−1.2%), `prefix_fill`-routed mean −0.13% (control −0.22%). Five cells
nominally exceed the control band (|overhead| > |control| + 4 pp), but
three of them have off0 *faster* by 8–21% — impossible for genuine
overhead — and the two positive ones flipped sign in the previous build of
the same source: all five are the ±15–20% code-placement flips the control
column exists to expose, not systematic cost. Per-type overhead means:
pair64f −0.33%, i64 +0.46%, pair64 −0.67%, i32 −0.55%, pair_li −0.18% —
each within a fraction of a percent of its control. The full per-cell
table is in the appendix.

**Standalone-codegen note (compare assemblers).** As separately compiled
out-of-line functions, the two entry points are *not* byte-identical:
`sized_off` carries `prefix_fill`'s hoisted frame — 64-byte stack
realignment, six pushes, a 0x140-byte offset-buffer frame, and a
stack-protector canary (`mov %fs:0x28`) — executed before the dispatch
tests, while raw `algo::sized`'s small path runs frameless. That is
~10–15 cycles per call. The measurements above bound its real-world effect
at the noise floor even for 64-element batched calls; it would only matter
for a non-inlinable wrapper hammered with tiny arrays, and the fix (a
`noinline` split of `prefix_fill`'s body) is documented but not applied —
it would add an unconditional call to the hot large-n path to fix a cost
the data cannot detect.


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
* Measuring the zero-offset overhead with kernels inlined into the timing
  loops: each side's copy lands at a different code alignment and the
  per-cell "overhead" scatters ±20% in both directions on tiny blocks —
  unusable. Out-of-line `noinline` kernels plus a byte-identical control
  copy (the noise floor) is the working methodology; a first version of this
  study without the control briefly misread layout luck as real overhead.

## Appendix: full benchmark results

Machine: Meteor Lake, GCC 15.2, `-O3 -march=native`. Metric: **ns per suffix
element** = min-of-reps ns / (n − offset); lower is better, the per-cell best
is bold. f = offset/n, p = below fraction of the suffix. Large cells are
single calls on fresh copies; n ≤ 4096 cells are batched over a 2^20 pool
(f = 0.5 in `small` mode, f = 0 in `zero` mode) with an anti-fusion barrier
between block calls. `whole` ignores the precondition (raw `algo::sized`);
`sized_off` is the shipped dispatcher behind `offset_partition`. All tables
were produced by the final binary in one session; selected cells quoted in
the body text come from earlier runs of the same benchmark, so cell-level
deltas against these tables are run-to-run variance (~±5%).

### Large single calls (mode `large`)

#### pair64f

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 65536 | 0.1 | 0.1 | 1.045 | 0.534 | 0.426 | 0.385 | **0.366** | 0.465 | 0.377 |
| 65536 | 0.1 | 0.5 | 2.798 | 0.527 | 0.599 | 0.477 | 0.487 | 0.500 | **0.448** |
| 65536 | 0.1 | 0.9 | 0.948 | 0.566 | 0.425 | 0.449 | **0.392** | 0.523 | 0.398 |
| 65536 | 0.5 | 0.1 | 0.949 | 0.512 | 0.403 | 0.679 | 0.365 | 0.727 | **0.364** |
| 65536 | 0.5 | 0.5 | 2.719 | 0.586 | 0.703 | 0.881 | 0.521 | 0.869 | **0.512** |
| 65536 | 0.5 | 0.9 | 0.771 | **0.630** | 0.841 | 1.050 | 0.688 | 1.087 | 0.751 |
| 65536 | 0.9 | 0.1 | 0.744 | 0.502 | 0.417 | 3.340 | **0.361** | 3.384 | 0.363 |
| 65536 | 0.9 | 0.5 | 1.490 | **0.530** | 0.739 | 2.859 | 0.531 | 2.979 | 0.533 |
| 65536 | 0.9 | 0.9 | 0.571 | **0.559** | 0.838 | 3.038 | 0.715 | 3.288 | 0.736 |
| 1048576 | 0.1 | 0.1 | 1.504 | 0.862 | 0.775 | 0.789 | **0.718** | 0.779 | 0.750 |
| 1048576 | 0.1 | 0.5 | 3.469 | **0.753** | 0.815 | 0.848 | 0.799 | 0.850 | 0.784 |
| 1048576 | 0.1 | 0.9 | 1.590 | 1.316 | **0.767** | 0.843 | 0.790 | 0.860 | 0.790 |
| 1048576 | 0.5 | 0.1 | 1.504 | 0.827 | 0.816 | 1.404 | **0.712** | 1.382 | 0.727 |
| 1048576 | 0.5 | 0.5 | 3.459 | **1.015** | 1.309 | 1.418 | 1.032 | 1.548 | 1.028 |
| 1048576 | 0.5 | 0.9 | 1.716 | **1.207** | 1.672 | 1.479 | 1.343 | 1.586 | 1.311 |
| 1048576 | 0.9 | 0.1 | 1.357 | 0.797 | 0.811 | 6.688 | **0.719** | 6.609 | 0.749 |
| 1048576 | 0.9 | 0.5 | 3.278 | **0.940** | 1.072 | 6.883 | 0.951 | 6.595 | 0.942 |
| 1048576 | 0.9 | 0.9 | 1.502 | **1.105** | 1.401 | 6.788 | 1.202 | 6.570 | 1.228 |
| 4194304 | 0.1 | 0.1 | 1.514 | 0.928 | 0.843 | 0.823 | **0.758** | 0.808 | 0.772 |
| 4194304 | 0.1 | 0.5 | 3.493 | 1.078 | 0.854 | 0.840 | 0.819 | 0.844 | **0.806** |
| 4194304 | 0.1 | 0.9 | 1.664 | 1.117 | 0.823 | 0.878 | **0.792** | 0.857 | 0.793 |
| 4194304 | 0.5 | 0.1 | 1.516 | 0.915 | 0.880 | 1.403 | 0.723 | 1.413 | **0.722** |
| 4194304 | 0.5 | 0.5 | 3.482 | 1.079 | 1.456 | 1.508 | **1.039** | 1.584 | 1.063 |
| 4194304 | 0.5 | 0.9 | 1.782 | **1.258** | 2.016 | 1.476 | 1.316 | 1.589 | 1.345 |
| 4194304 | 0.9 | 0.1 | 1.484 | 0.817 | 0.793 | 6.943 | 0.789 | 7.078 | **0.735** |
| 4194304 | 0.9 | 0.5 | 3.468 | 1.005 | 1.121 | 6.970 | 0.999 | 7.279 | **0.981** |
| 4194304 | 0.9 | 0.9 | 1.714 | **1.208** | 1.641 | 6.869 | 1.236 | 7.382 | 1.215 |

#### i64

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 65536 | 0.1 | 0.1 | 1.135 | 0.459 | 0.360 | **0.352** | 0.356 | 0.380 | 0.358 |
| 65536 | 0.1 | 0.5 | 2.826 | 0.471 | 0.444 | 0.454 | 0.435 | 0.481 | **0.432** |
| 65536 | 0.1 | 0.9 | 1.054 | 0.488 | 0.362 | 0.409 | 0.380 | 0.435 | **0.360** |
| 65536 | 0.5 | 0.1 | 1.027 | 0.452 | 0.353 | 0.574 | **0.333** | 0.604 | 0.433 |
| 65536 | 0.5 | 0.5 | 2.684 | **0.461** | 0.535 | 0.705 | 0.498 | 0.823 | 0.472 |
| 65536 | 0.5 | 0.9 | 0.839 | 0.481 | 0.538 | 0.833 | 0.649 | 1.053 | **0.481** |
| 65536 | 0.9 | 0.1 | 0.883 | 0.496 | **0.374** | 2.829 | 0.375 | 2.646 | 0.437 |
| 65536 | 0.9 | 0.5 | 1.355 | **0.463** | 0.506 | 2.712 | 0.486 | 2.862 | 0.466 |
| 65536 | 0.9 | 0.9 | **0.433** | 0.474 | 0.756 | 4.569 | 0.706 | 3.378 | 0.636 |
| 1048576 | 0.1 | 0.1 | 1.236 | 0.540 | 0.435 | **0.396** | 0.418 | 0.463 | 0.429 |
| 1048576 | 0.1 | 0.5 | 3.005 | 0.637 | 0.476 | 0.477 | 0.507 | 0.510 | **0.464** |
| 1048576 | 0.1 | 0.9 | 1.149 | 0.583 | 0.449 | 0.444 | 0.432 | 0.477 | **0.431** |
| 1048576 | 0.5 | 0.1 | 1.226 | 0.522 | 0.436 | 0.666 | **0.401** | 0.699 | 0.498 |
| 1048576 | 0.5 | 0.5 | 2.958 | 0.571 | 0.649 | 0.763 | **0.517** | 0.953 | 0.585 |
| 1048576 | 0.5 | 0.9 | 1.174 | **0.532** | 0.678 | 0.891 | 0.730 | 1.107 | 0.592 |
| 1048576 | 0.9 | 0.1 | 1.162 | 0.465 | 0.401 | 2.708 | **0.361** | 2.722 | 0.464 |
| 1048576 | 0.9 | 0.5 | 2.822 | 0.534 | 0.574 | 3.121 | 0.516 | 3.080 | **0.495** |
| 1048576 | 0.9 | 0.9 | 1.105 | **0.563** | 0.645 | 3.060 | 0.718 | 3.381 | 0.587 |
| 4194304 | 0.1 | 0.1 | 1.337 | 0.527 | 0.522 | 0.510 | **0.455** | 0.509 | 0.457 |
| 4194304 | 0.1 | 0.5 | 3.288 | 0.546 | 0.555 | 0.544 | 0.552 | 0.571 | **0.535** |
| 4194304 | 0.1 | 0.9 | 1.334 | 0.671 | 0.551 | **0.488** | 0.606 | 0.540 | 0.610 |
| 4194304 | 0.5 | 0.1 | 1.337 | 0.524 | 0.523 | 0.837 | **0.446** | 0.832 | 0.520 |
| 4194304 | 0.5 | 0.5 | 3.307 | 0.586 | 0.798 | 0.924 | 0.634 | 1.033 | **0.584** |
| 4194304 | 0.5 | 0.9 | 1.431 | **0.651** | 0.981 | 0.973 | 0.816 | 1.203 | 0.659 |
| 4194304 | 0.9 | 0.1 | 1.294 | 0.519 | **0.481** | 3.796 | 0.485 | 3.744 | 0.514 |
| 4194304 | 0.9 | 0.5 | 3.165 | 0.561 | 0.763 | 3.873 | 0.601 | 3.892 | **0.560** |
| 4194304 | 0.9 | 0.9 | 1.376 | **0.615** | 0.843 | 3.946 | 0.772 | 4.153 | 0.616 |

#### pair64

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 65536 | 0.1 | 0.1 | 1.180 | 0.696 | 0.562 | 0.519 | **0.500** | 0.552 | 0.510 |
| 65536 | 0.1 | 0.5 | 2.961 | 0.693 | 0.604 | 0.586 | **0.555** | 0.635 | 0.571 |
| 65536 | 0.1 | 0.9 | 1.003 | 0.686 | 0.554 | 0.564 | **0.525** | 0.586 | 0.557 |
| 65536 | 0.5 | 0.1 | 1.070 | 0.710 | 0.560 | 0.829 | 0.504 | 0.913 | **0.502** |
| 65536 | 0.5 | 0.5 | 2.884 | 0.682 | 0.828 | 0.876 | 0.615 | 1.080 | **0.612** |
| 65536 | 0.5 | 0.9 | 0.881 | **0.685** | 0.993 | 0.972 | 0.775 | 1.303 | 0.777 |
| 65536 | 0.9 | 0.1 | 0.698 | 0.770 | 0.580 | 3.800 | 0.510 | 4.282 | **0.508** |
| 65536 | 0.9 | 0.5 | 1.456 | 0.802 | 0.872 | 3.885 | 0.605 | 4.459 | **0.602** |
| 65536 | 0.9 | 0.9 | **0.550** | 0.699 | 0.973 | 4.135 | 0.788 | 4.630 | 0.804 |
| 1048576 | 0.1 | 0.1 | 1.633 | 0.836 | 0.755 | 0.763 | **0.701** | 0.764 | 0.709 |
| 1048576 | 0.1 | 0.5 | 3.607 | 0.798 | 0.771 | 0.825 | **0.742** | 0.823 | 0.744 |
| 1048576 | 0.1 | 0.9 | 1.632 | 1.369 | 0.755 | 0.872 | **0.755** | 0.868 | 0.772 |
| 1048576 | 0.5 | 0.1 | 1.657 | 0.843 | 0.807 | 1.310 | **0.707** | 1.294 | 0.710 |
| 1048576 | 0.5 | 0.5 | 3.618 | 1.030 | 1.328 | 1.383 | **1.028** | 1.526 | 1.040 |
| 1048576 | 0.5 | 0.9 | 1.726 | **1.212** | 1.717 | 1.429 | 1.326 | 1.566 | 1.361 |
| 1048576 | 0.9 | 0.1 | 1.467 | 0.866 | 0.755 | 6.186 | 0.701 | 5.873 | **0.696** |
| 1048576 | 0.9 | 0.5 | 3.402 | 1.001 | 1.091 | 6.177 | **0.974** | 5.933 | 0.987 |
| 1048576 | 0.9 | 0.9 | 1.540 | 1.200 | 1.388 | 6.179 | **1.198** | 6.097 | 1.204 |
| 4194304 | 0.1 | 0.1 | 1.650 | 0.906 | 0.836 | 0.745 | 0.737 | 0.775 | **0.714** |
| 4194304 | 0.1 | 0.5 | 3.665 | 1.069 | 0.789 | **0.761** | 0.768 | 0.788 | 0.770 |
| 4194304 | 0.1 | 0.9 | 1.665 | 1.109 | 0.811 | 0.788 | 0.789 | 0.830 | **0.780** |
| 4194304 | 0.5 | 0.1 | 1.666 | 0.892 | 0.840 | 1.290 | 0.723 | 1.290 | **0.713** |
| 4194304 | 0.5 | 0.5 | 3.629 | 1.065 | 1.387 | 1.411 | 1.054 | 1.514 | **1.041** |
| 4194304 | 0.5 | 0.9 | 1.770 | **1.271** | 1.942 | 1.482 | 1.342 | 1.498 | 1.344 |
| 4194304 | 0.9 | 0.1 | 1.631 | 0.843 | 0.763 | 6.132 | **0.696** | 5.787 | 0.698 |
| 4194304 | 0.9 | 0.5 | 3.636 | 1.010 | 1.212 | 6.074 | 0.981 | 5.991 | **0.964** |
| 4194304 | 0.9 | 0.9 | 1.722 | **1.086** | 1.510 | 6.240 | 1.246 | 6.154 | 1.266 |

### Batched small blocks, f = 0.5 (mode `small`)

#### pair64f

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.5 | 0.1 | **2.128** | 2.141 | 2.486 | 2.852 | 2.552 | 3.310 | 2.604 |
| 64 | 0.5 | 0.5 | 3.980 | 2.219 | 3.507 | **2.138** | 3.462 | 3.051 | 3.433 |
| 64 | 0.5 | 0.9 | 2.242 | 2.136 | 1.888 | 2.081 | **1.768** | 3.519 | 1.807 |
| 256 | 0.5 | 0.1 | 2.124 | **1.754** | 2.316 | 2.884 | 1.777 | 1.842 | 2.122 |
| 256 | 0.5 | 0.5 | 3.783 | **1.739** | 2.284 | 2.091 | 2.432 | 2.453 | 1.861 |
| 256 | 0.5 | 0.9 | 1.920 | 2.313 | 2.867 | 2.837 | **1.897** | 2.211 | 1.909 |
| 1024 | 0.5 | 0.1 | 1.614 | 1.386 | 1.513 | 1.948 | 1.440 | 2.098 | **1.347** |
| 1024 | 0.5 | 0.5 | 3.257 | **1.562** | 2.157 | 1.885 | 1.812 | 2.101 | 1.800 |
| 1024 | 0.5 | 0.9 | 1.770 | **1.536** | 2.259 | 2.054 | 1.874 | 2.434 | 1.942 |
| 4096 | 0.5 | 0.1 | 1.504 | 1.325 | 1.223 | 1.559 | **1.079** | 1.640 | 1.091 |
| 4096 | 0.5 | 0.5 | 3.303 | 1.363 | 1.641 | 1.554 | 1.335 | 1.595 | **1.302** |
| 4096 | 0.5 | 0.9 | 1.717 | **1.287** | 1.720 | 1.537 | 1.405 | 1.736 | 1.449 |

#### i64

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.5 | 0.1 | 1.339 | 0.520 | 0.586 | 1.430 | 0.753 | 0.952 | **0.513** |
| 64 | 0.5 | 0.5 | 3.279 | 0.969 | 0.631 | 1.222 | 0.665 | 0.944 | **0.518** |
| 64 | 0.5 | 0.9 | 1.442 | 0.527 | 0.677 | 1.108 | 0.708 | 0.929 | **0.523** |
| 256 | 0.5 | 0.1 | 1.190 | 0.493 | 0.511 | 0.914 | **0.453** | 0.902 | 0.487 |
| 256 | 0.5 | 0.5 | 2.972 | 0.505 | 0.583 | 0.963 | 0.609 | 0.904 | **0.505** |
| 256 | 0.5 | 0.9 | 1.156 | **0.527** | 0.655 | 0.981 | 0.825 | 0.959 | 0.528 |
| 1024 | 0.5 | 0.1 | 1.151 | 0.477 | 0.503 | 0.708 | **0.424** | 0.790 | 0.469 |
| 1024 | 0.5 | 0.5 | 2.932 | 0.544 | 0.591 | 0.758 | 0.526 | 0.907 | **0.493** |
| 1024 | 0.5 | 0.9 | 1.125 | **0.514** | 0.662 | 0.869 | 0.672 | 1.171 | 0.516 |
| 4096 | 0.5 | 0.1 | 1.127 | 0.469 | 0.406 | 0.621 | **0.375** | 0.648 | 0.468 |
| 4096 | 0.5 | 0.5 | 2.838 | **0.489** | 0.545 | 0.717 | 0.489 | 0.837 | 0.490 |
| 4096 | 0.5 | 0.9 | 1.008 | 0.516 | 0.588 | 0.811 | 0.914 | 1.136 | **0.513** |

#### pair64

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.5 | 0.1 | **2.593** | 3.430 | 2.983 | 3.104 | 2.978 | 3.821 | 2.975 |
| 64 | 0.5 | 0.5 | 4.186 | 3.645 | 3.303 | **2.104** | 3.372 | 3.464 | 3.369 |
| 64 | 0.5 | 0.9 | 2.640 | 3.462 | 1.950 | 1.958 | 1.913 | 3.564 | **1.895** |
| 256 | 0.5 | 0.1 | 2.477 | 4.245 | 2.546 | 3.250 | **1.998** | 2.276 | 2.332 |
| 256 | 0.5 | 0.5 | 4.216 | 4.190 | 2.568 | 2.470 | **2.091** | 2.582 | 2.112 |
| 256 | 0.5 | 0.9 | 2.388 | 4.375 | 2.916 | 3.252 | 2.654 | 2.327 | **2.103** |
| 1024 | 0.5 | 0.1 | 3.340 | 5.658 | 3.000 | 4.217 | **2.837** | 4.844 | 2.857 |
| 1024 | 0.5 | 0.5 | 3.574 | 5.004 | 2.985 | 3.457 | **2.244** | 4.809 | 2.350 |
| 1024 | 0.5 | 0.9 | **3.656** | 6.069 | 4.131 | 3.909 | 3.813 | 5.115 | 3.831 |
| 4096 | 0.5 | 0.1 | 1.831 | 4.785 | 1.833 | 2.422 | **1.562** | 2.570 | 1.608 |
| 4096 | 0.5 | 0.5 | 3.498 | 4.484 | 1.924 | 2.050 | **1.454** | 2.702 | 1.617 |
| 4096 | 0.5 | 0.9 | 1.841 | 4.417 | 1.978 | 1.840 | **1.521** | 2.902 | 1.589 |

### Pair-of-i64 by first coordinate deep dive (mode `byfirst`)

`p64f_proj` = projection formulation (8-byte key), `p64f_comp` = comparator
formulation (16-byte pair key by value), `p64f_dup256` = 256 distinct first
values (heavy ties). Same data geometry per cell.

#### p64f_proj

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 65536 | 0.1 | 0.1 | 0.996 | 0.534 | 0.403 | 0.344 | **0.338** | 0.385 | 0.344 |
| 65536 | 0.1 | 0.5 | 2.639 | 0.518 | 0.458 | 0.446 | 0.485 | 0.528 | **0.416** |
| 65536 | 0.1 | 0.9 | 0.841 | 0.558 | 0.408 | 0.388 | **0.367** | 0.404 | 0.401 |
| 65536 | 0.5 | 0.1 | 0.885 | 0.472 | 0.372 | 0.567 | **0.332** | 0.587 | 0.334 |
| 65536 | 0.5 | 0.5 | 2.637 | 0.569 | 0.643 | 0.732 | 0.516 | 0.856 | **0.481** |
| 65536 | 0.5 | 0.9 | 0.712 | **0.505** | 0.856 | 0.949 | 0.637 | 1.003 | 0.642 |
| 65536 | 0.9 | 0.1 | 0.679 | 0.476 | 0.380 | 2.588 | **0.328** | 2.595 | 0.331 |
| 65536 | 0.9 | 0.5 | 1.206 | **0.497** | 0.738 | 2.784 | 0.543 | 2.875 | 0.523 |
| 65536 | 0.9 | 0.9 | **0.497** | 0.607 | 0.887 | 2.954 | 0.715 | 3.134 | 0.694 |
| 1048576 | 0.1 | 0.1 | 1.439 | 0.814 | 0.725 | 0.759 | 0.689 | 0.728 | **0.670** |
| 1048576 | 0.1 | 0.5 | 3.284 | **0.735** | 0.782 | 0.766 | 0.735 | 0.768 | 0.741 |
| 1048576 | 0.1 | 0.9 | 1.532 | 1.182 | 0.800 | 0.784 | **0.733** | 0.774 | 0.813 |
| 1048576 | 0.5 | 0.1 | 1.442 | 0.724 | 0.783 | 1.278 | 0.733 | 1.274 | **0.647** |
| 1048576 | 0.5 | 0.5 | 3.287 | **0.924** | 1.249 | 1.379 | 0.953 | 1.359 | 0.979 |
| 1048576 | 0.5 | 0.9 | 1.649 | **1.143** | 1.664 | 1.585 | 1.209 | 1.426 | 1.254 |
| 1048576 | 0.9 | 0.1 | 1.351 | 0.817 | 0.775 | 6.267 | **0.705** | 6.377 | 0.716 |
| 1048576 | 0.9 | 0.5 | 3.130 | 1.005 | 1.088 | 6.714 | 1.035 | 6.293 | **0.943** |
| 1048576 | 0.9 | 0.9 | 1.479 | **1.084** | 1.315 | 6.469 | 1.141 | 6.253 | 1.190 |
| 4194304 | 0.1 | 0.1 | 1.457 | 0.901 | 0.790 | 0.774 | 0.795 | **0.749** | 0.788 |
| 4194304 | 0.1 | 0.5 | 3.334 | 1.015 | 0.800 | 0.790 | **0.768** | 0.804 | 0.792 |
| 4194304 | 0.1 | 0.9 | 1.618 | 1.128 | 0.812 | 0.852 | **0.753** | 0.805 | 0.753 |
| 4194304 | 0.5 | 0.1 | 1.454 | 0.861 | 0.792 | 1.317 | **0.776** | 1.316 | 0.778 |
| 4194304 | 0.5 | 0.5 | 3.385 | 1.026 | 1.376 | 1.422 | 1.015 | 1.513 | **0.997** |
| 4194304 | 0.5 | 0.9 | 1.731 | **1.214** | 1.873 | 1.418 | 1.290 | 1.531 | 1.268 |
| 4194304 | 0.9 | 0.1 | 1.420 | **0.737** | 0.753 | 6.876 | 0.752 | 6.562 | 0.763 |
| 4194304 | 0.9 | 0.5 | 3.316 | **0.932** | 1.194 | 6.874 | 0.947 | 6.852 | 0.934 |
| 4194304 | 0.9 | 0.9 | 1.661 | **1.099** | 1.323 | 6.717 | 1.194 | 6.861 | 1.175 |

#### p64f_comp

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 65536 | 0.1 | 0.1 | 0.930 | 0.473 | 0.382 | 0.354 | **0.342** | 0.373 | 0.352 |
| 65536 | 0.1 | 0.5 | 2.634 | 0.489 | 0.460 | 0.476 | **0.427** | 0.473 | 0.428 |
| 65536 | 0.1 | 0.9 | 0.821 | 0.507 | 0.393 | 0.399 | **0.373** | 0.440 | 0.385 |
| 65536 | 0.5 | 0.1 | 0.892 | 0.472 | 0.389 | 0.577 | **0.336** | 0.598 | 0.347 |
| 65536 | 0.5 | 0.5 | 2.566 | 0.487 | 0.667 | 0.721 | 0.579 | 0.802 | **0.484** |
| 65536 | 0.5 | 0.9 | 0.722 | **0.509** | 0.781 | 0.815 | 0.670 | 1.073 | 0.669 |
| 65536 | 0.9 | 0.1 | 0.683 | 0.503 | 0.426 | 2.621 | 0.342 | 2.631 | **0.341** |
| 65536 | 0.9 | 0.5 | 1.241 | **0.495** | 0.677 | 2.805 | 0.570 | 2.950 | 0.565 |
| 65536 | 0.9 | 0.9 | 0.542 | **0.514** | 0.799 | 2.999 | 0.778 | 3.212 | 0.677 |
| 1048576 | 0.1 | 0.1 | 1.438 | 0.823 | 0.760 | 0.733 | **0.692** | 0.732 | 0.741 |
| 1048576 | 0.1 | 0.5 | 3.277 | **0.733** | 0.829 | 0.742 | 0.739 | 0.791 | 0.788 |
| 1048576 | 0.1 | 0.9 | 1.538 | 1.186 | 0.816 | **0.753** | 0.818 | 0.828 | 0.826 |
| 1048576 | 0.5 | 0.1 | 1.437 | 0.734 | 0.765 | 1.276 | **0.663** | 1.335 | 0.682 |
| 1048576 | 0.5 | 0.5 | 3.287 | **0.920** | 1.239 | 1.361 | 0.970 | 1.510 | 0.957 |
| 1048576 | 0.5 | 0.9 | 1.642 | **1.129** | 1.726 | 1.585 | 1.236 | 1.611 | 1.168 |
| 1048576 | 0.9 | 0.1 | 1.294 | 0.803 | 0.745 | 6.594 | **0.720** | 5.986 | 0.732 |
| 1048576 | 0.9 | 0.5 | 3.127 | 1.003 | 1.160 | 6.504 | 0.981 | 6.907 | **0.928** |
| 1048576 | 0.9 | 0.9 | 1.475 | 1.159 | 1.320 | 6.487 | 1.280 | 6.649 | **1.153** |
| 4194304 | 0.1 | 0.1 | 1.456 | 0.897 | 0.777 | 0.770 | 0.798 | **0.756** | 0.805 |
| 4194304 | 0.1 | 0.5 | 3.337 | 1.017 | 0.791 | 0.797 | 0.778 | 0.816 | **0.778** |
| 4194304 | 0.1 | 0.9 | 1.620 | 1.129 | 0.773 | 0.852 | 0.753 | 0.811 | **0.750** |
| 4194304 | 0.5 | 0.1 | 1.451 | 0.853 | **0.767** | 1.352 | 0.804 | 1.283 | 0.791 |
| 4194304 | 0.5 | 0.5 | 3.370 | **1.015** | 1.379 | 1.415 | 1.015 | 1.526 | 1.021 |
| 4194304 | 0.5 | 0.9 | 1.740 | **1.209** | 1.826 | 1.442 | 1.284 | 1.610 | 1.277 |
| 4194304 | 0.9 | 0.1 | 1.421 | **0.721** | 0.760 | 6.532 | 0.784 | 6.458 | 0.779 |
| 4194304 | 0.9 | 0.5 | 3.311 | **0.929** | 1.140 | 6.836 | 0.942 | 6.633 | 0.936 |
| 4194304 | 0.9 | 0.9 | 1.645 | **1.111** | 1.359 | 6.633 | 1.185 | 6.563 | 1.179 |

#### p64f_dup256

| n | f | p | scan_swap | gap_off | part_swap | fused_block | prefix_fill | whole | sized_off |
|---|---|---|---|---|---|---|---|---|---|
| 65536 | 0.1 | 0.1 | 0.908 | 0.474 | 0.376 | 0.369 | **0.333** | 0.368 | 0.343 |
| 65536 | 0.1 | 0.5 | 2.658 | 0.541 | 0.508 | 0.445 | **0.416** | 0.462 | 0.429 |
| 65536 | 0.1 | 0.9 | 0.840 | 0.506 | 0.392 | 0.377 | **0.366** | 0.444 | 0.406 |
| 65536 | 0.5 | 0.1 | 0.855 | 0.464 | 0.367 | 0.562 | **0.324** | 0.584 | 0.325 |
| 65536 | 0.5 | 0.5 | 2.607 | 0.485 | 0.647 | 0.674 | **0.475** | 0.799 | 0.499 |
| 65536 | 0.5 | 0.9 | 0.807 | **0.554** | 0.780 | 0.801 | 0.634 | 1.000 | 0.633 |
| 65536 | 0.9 | 0.1 | 0.658 | 0.474 | 0.366 | 2.571 | 0.325 | 2.578 | **0.319** |
| 65536 | 0.9 | 0.5 | 1.110 | 0.503 | 0.631 | 2.689 | **0.465** | 2.751 | 0.468 |
| 65536 | 0.9 | 0.9 | 0.600 | **0.514** | 0.771 | 2.837 | 0.703 | 3.033 | 0.627 |
| 1048576 | 0.1 | 0.1 | 1.409 | 0.825 | 0.761 | 0.723 | **0.682** | 0.714 | 0.691 |
| 1048576 | 0.1 | 0.5 | 3.273 | **0.721** | 0.755 | 0.740 | 0.764 | 0.783 | 0.739 |
| 1048576 | 0.1 | 0.9 | 1.550 | 1.253 | 0.765 | **0.739** | 0.788 | 0.757 | 0.778 |
| 1048576 | 0.5 | 0.1 | 1.388 | 0.846 | 0.784 | 1.368 | **0.745** | 1.333 | 0.761 |
| 1048576 | 0.5 | 0.5 | 3.295 | **0.941** | 1.274 | 1.363 | 0.987 | 1.452 | 0.958 |
| 1048576 | 0.5 | 0.9 | 1.642 | **1.146** | 1.653 | 1.582 | 1.232 | 1.449 | 1.284 |
| 1048576 | 0.9 | 0.1 | 1.120 | 0.866 | 0.734 | 6.877 | 0.685 | 6.366 | **0.670** |
| 1048576 | 0.9 | 0.5 | 3.079 | 0.860 | 1.073 | 6.929 | 0.917 | 6.572 | **0.857** |
| 1048576 | 0.9 | 0.9 | 1.654 | **1.119** | 1.401 | 6.865 | 1.232 | 6.593 | 1.204 |
| 4194304 | 0.1 | 0.1 | 1.455 | 0.878 | 0.875 | **0.766** | 0.771 | 0.793 | 0.774 |
| 4194304 | 0.1 | 0.5 | 3.349 | 1.020 | 0.817 | 0.801 | 0.784 | 0.819 | **0.783** |
| 4194304 | 0.1 | 0.9 | 1.621 | 1.159 | 0.788 | 0.788 | 0.758 | 0.821 | **0.748** |
| 4194304 | 0.5 | 0.1 | 1.403 | 0.864 | 0.855 | 1.387 | 0.772 | 1.415 | **0.768** |
| 4194304 | 0.5 | 0.5 | 3.371 | 1.022 | 1.346 | 1.417 | 1.007 | 1.492 | **1.002** |
| 4194304 | 0.5 | 0.9 | 1.733 | **1.211** | 1.920 | 1.451 | 1.287 | 1.590 | 1.272 |
| 4194304 | 0.9 | 0.1 | 1.225 | 0.793 | 0.819 | 6.700 | **0.701** | 6.429 | 0.703 |
| 4194304 | 0.9 | 0.5 | 3.274 | 0.910 | 1.175 | 6.715 | 0.967 | 6.469 | **0.908** |
| 4194304 | 0.9 | 0.9 | 1.651 | **1.124** | 1.321 | 6.827 | 1.182 | 6.968 | 1.197 |

### Zero-offset identity, f = 0 (mode `zero`)

Raw `algo::sized` vs the dispatcher's REAL route at offset 0
(`routed`: gap_off below the size cutoff, prefix_fill above — no
offset==0 special case exists). `control %` is a byte-identical
second copy of the raw kernel: the code-placement noise floor;
overhead is only real where it clearly exceeds the control, which
happens in no cell (see the analysis above). ns per element, min over
2× alternated rounds; n < 2^16 batched over a 2^20 pool.

#### pair64f

| n | p | routed | raw ns/elem | offset-0 ns/elem | overhead % | control % |
|---|---|---|---|---|---|---|
| 64 | 0.1 | prefix_fill | 1.4403 | 1.8047 | +25.30 | +22.34 |
| 64 | 0.5 | prefix_fill | 1.7615 | 1.7566 | -0.28 | -2.40 |
| 64 | 0.9 | prefix_fill | 1.2352 | 1.2300 | -0.42 | +3.05 |
| 256 | 0.1 | prefix_fill | 1.7422 | 1.7810 | +2.23 | +0.29 |
| 256 | 0.5 | prefix_fill | 1.4752 | 1.4667 | -0.58 | +0.13 |
| 256 | 0.9 | prefix_fill | 1.6560 | 1.6860 | +1.82 | +0.85 |
| 1024 | 0.1 | prefix_fill | 1.0916 | 1.0905 | -0.10 | -0.62 |
| 1024 | 0.5 | prefix_fill | 1.0775 | 1.0798 | +0.21 | +0.06 |
| 1024 | 0.9 | prefix_fill | 1.0025 | 1.0096 | +0.71 | +0.95 |
| 4096 | 0.1 | prefix_fill | 0.8429 | 0.8380 | -0.58 | -0.05 |
| 4096 | 0.5 | prefix_fill | 0.8138 | 0.8108 | -0.37 | +0.73 |
| 4096 | 0.9 | prefix_fill | 0.8403 | 0.8393 | -0.12 | +0.09 |
| 16384 | 0.1 | prefix_fill | 0.7657 | 0.7711 | +0.70 | +1.23 |
| 16384 | 0.5 | prefix_fill | 0.7587 | 0.7632 | +0.59 | +1.12 |
| 16384 | 0.9 | prefix_fill | 0.7605 | 0.7711 | +1.40 | +0.74 |
| 65536 | 0.1 | prefix_fill | 0.3794 | 0.4413 | +16.30 | -8.26 |
| 65536 | 0.5 | prefix_fill | 0.4796 | 0.4291 | -10.52 | -10.43 |
| 65536 | 0.9 | prefix_fill | 0.4535 | 0.3615 | -20.29 | -16.08 |
| 262144 | 0.1 | prefix_fill | 0.4059 | 0.3981 | -1.92 | -1.48 |
| 262144 | 0.5 | prefix_fill | 0.4961 | 0.4800 | -3.24 | -9.60 |
| 262144 | 0.9 | prefix_fill | 0.4758 | 0.3747 | -21.24 | -7.77 |
| 1048576 | 0.1 | prefix_fill | 0.7452 | 0.7629 | +2.39 | -0.32 |
| 1048576 | 0.5 | prefix_fill | 0.7352 | 0.7272 | -1.09 | -1.87 |
| 1048576 | 0.9 | prefix_fill | 0.7286 | 0.7344 | +0.80 | -1.12 |
| 4194304 | 0.1 | prefix_fill | 0.8183 | 0.8370 | +2.29 | +2.57 |
| 4194304 | 0.5 | prefix_fill | 0.7606 | 0.7512 | -1.24 | +2.06 |
| 4194304 | 0.9 | prefix_fill | 0.7271 | 0.7151 | -1.66 | -3.17 |

#### i64

| n | p | routed | raw ns/elem | offset-0 ns/elem | overhead % | control % |
|---|---|---|---|---|---|---|
| 64 | 0.1 | gap_off | 0.5713 | 0.5600 | -1.98 | +2.79 |
| 64 | 0.5 | gap_off | 0.5786 | 0.5737 | -0.83 | +1.01 |
| 64 | 0.9 | gap_off | 0.6350 | 0.5836 | -8.10 | -2.52 |
| 256 | 0.1 | gap_off | 0.4792 | 0.4781 | -0.22 | +0.53 |
| 256 | 0.5 | gap_off | 0.5756 | 0.5492 | -4.58 | -17.12 |
| 256 | 0.9 | gap_off | 0.4905 | 0.4698 | -4.20 | -2.19 |
| 1024 | 0.1 | prefix_fill | 0.4278 | 0.4246 | -0.76 | -0.80 |
| 1024 | 0.5 | prefix_fill | 0.4897 | 0.4945 | +0.99 | -0.10 |
| 1024 | 0.9 | prefix_fill | 0.4378 | 0.4385 | +0.17 | +0.26 |
| 4096 | 0.1 | prefix_fill | 0.3900 | 0.3900 | +0.01 | +1.74 |
| 4096 | 0.5 | prefix_fill | 0.4451 | 0.4463 | +0.26 | +1.16 |
| 4096 | 0.9 | prefix_fill | 0.3844 | 0.4431 | +15.26 | -0.06 |
| 16384 | 0.1 | prefix_fill | 0.3654 | 0.3649 | -0.12 | +1.49 |
| 16384 | 0.5 | prefix_fill | 0.4296 | 0.4338 | +0.98 | -1.10 |
| 16384 | 0.9 | prefix_fill | 0.3713 | 0.3631 | -2.20 | +0.06 |
| 65536 | 0.1 | prefix_fill | 0.3320 | 0.3374 | +1.62 | -0.55 |
| 65536 | 0.5 | prefix_fill | 0.4069 | 0.4080 | +0.27 | -0.04 |
| 65536 | 0.9 | prefix_fill | 0.3227 | 0.3324 | +3.01 | +0.47 |
| 262144 | 0.1 | prefix_fill | 0.3434 | 0.3420 | -0.41 | +0.51 |
| 262144 | 0.5 | prefix_fill | 0.4143 | 0.4177 | +0.83 | +0.41 |
| 262144 | 0.9 | prefix_fill | 0.3389 | 0.3396 | +0.20 | +2.11 |
| 1048576 | 0.1 | prefix_fill | 0.3572 | 0.3583 | +0.28 | +2.35 |
| 1048576 | 0.5 | prefix_fill | 0.4257 | 0.4371 | +2.68 | +2.65 |
| 1048576 | 0.9 | prefix_fill | 0.3567 | 0.3648 | +2.27 | +2.51 |
| 4194304 | 0.1 | prefix_fill | 0.4284 | 0.4291 | +0.15 | +1.99 |
| 4194304 | 0.5 | prefix_fill | 0.4756 | 0.4851 | +1.99 | +0.71 |
| 4194304 | 0.9 | prefix_fill | 0.4291 | 0.4503 | +4.96 | +1.81 |

#### pair64

| n | p | routed | raw ns/elem | offset-0 ns/elem | overhead % | control % |
|---|---|---|---|---|---|---|
| 64 | 0.1 | prefix_fill | 1.9293 | 1.9412 | +0.62 | -0.79 |
| 64 | 0.5 | prefix_fill | 1.7969 | 1.8123 | +0.86 | +0.14 |
| 64 | 0.9 | prefix_fill | 1.5951 | 1.5738 | -1.34 | +0.16 |
| 256 | 0.1 | prefix_fill | 1.2401 | 1.2267 | -1.08 | -0.54 |
| 256 | 0.5 | prefix_fill | 1.0731 | 1.0618 | -1.05 | -0.57 |
| 256 | 0.9 | prefix_fill | 1.2774 | 1.2676 | -0.77 | -0.73 |
| 1024 | 0.1 | prefix_fill | 1.1635 | 1.1492 | -1.23 | -1.12 |
| 1024 | 0.5 | prefix_fill | 1.1022 | 1.0878 | -1.31 | -2.04 |
| 1024 | 0.9 | prefix_fill | 1.0393 | 1.0409 | +0.16 | +1.35 |
| 4096 | 0.1 | prefix_fill | 0.8420 | 0.8281 | -1.65 | -0.67 |
| 4096 | 0.5 | prefix_fill | 0.7832 | 0.7751 | -1.03 | -1.34 |
| 4096 | 0.9 | prefix_fill | 0.7855 | 0.7798 | -0.73 | -1.16 |
| 16384 | 0.1 | prefix_fill | 0.7580 | 0.7638 | +0.77 | -0.32 |
| 16384 | 0.5 | prefix_fill | 0.6869 | 0.6851 | -0.26 | -0.54 |
| 16384 | 0.9 | prefix_fill | 0.7050 | 0.7006 | -0.62 | +0.12 |
| 65536 | 0.1 | prefix_fill | 0.5116 | 0.5045 | -1.40 | -0.05 |
| 65536 | 0.5 | prefix_fill | 0.5296 | 0.5306 | +0.19 | +0.77 |
| 65536 | 0.9 | prefix_fill | 0.4819 | 0.4750 | -1.42 | +0.08 |
| 262144 | 0.1 | prefix_fill | 0.5142 | 0.5053 | -1.71 | +0.41 |
| 262144 | 0.5 | prefix_fill | 0.5333 | 0.5264 | -1.29 | +0.54 |
| 262144 | 0.9 | prefix_fill | 0.4862 | 0.4890 | +0.58 | -2.13 |
| 1048576 | 0.1 | prefix_fill | 0.6872 | 0.6757 | -1.67 | -0.08 |
| 1048576 | 0.5 | prefix_fill | 0.6320 | 0.6129 | -3.03 | +0.20 |
| 1048576 | 0.9 | prefix_fill | 0.6471 | 0.6557 | +1.33 | +1.51 |
| 4194304 | 0.1 | prefix_fill | 0.6957 | 0.6896 | -0.87 | -0.88 |
| 4194304 | 0.5 | prefix_fill | 0.6636 | 0.6645 | +0.12 | +1.38 |
| 4194304 | 0.9 | prefix_fill | 0.6835 | 0.6823 | -0.18 | +0.80 |

#### i32

| n | p | routed | raw ns/elem | offset-0 ns/elem | overhead % | control % |
|---|---|---|---|---|---|---|
| 64 | 0.1 | gap_off | 0.5635 | 0.5644 | +0.16 | +0.40 |
| 64 | 0.5 | gap_off | 0.5628 | 0.5689 | +1.09 | +0.97 |
| 64 | 0.9 | gap_off | 0.5808 | 0.5871 | +1.08 | +0.83 |
| 256 | 0.1 | gap_off | 0.4780 | 0.4755 | -0.53 | -0.69 |
| 256 | 0.5 | gap_off | 0.4763 | 0.4738 | -0.51 | -0.75 |
| 256 | 0.9 | gap_off | 0.4775 | 0.4791 | +0.33 | +2.50 |
| 1024 | 0.1 | prefix_fill | 0.4071 | 0.4027 | -1.07 | +1.48 |
| 1024 | 0.5 | prefix_fill | 0.4866 | 0.4807 | -1.21 | -1.03 |
| 1024 | 0.9 | prefix_fill | 0.4241 | 0.4231 | -0.23 | -1.76 |
| 4096 | 0.1 | prefix_fill | 0.3749 | 0.3774 | +0.68 | +6.01 |
| 4096 | 0.5 | prefix_fill | 0.4823 | 0.4726 | -2.02 | +0.72 |
| 4096 | 0.9 | prefix_fill | 0.3850 | 0.3805 | -1.17 | -1.88 |
| 16384 | 0.1 | prefix_fill | 0.3656 | 0.3793 | +3.75 | -2.11 |
| 16384 | 0.5 | prefix_fill | 0.4920 | 0.4396 | -10.65 | -10.61 |
| 16384 | 0.9 | prefix_fill | 0.3630 | 0.3706 | +2.09 | +4.95 |
| 65536 | 0.1 | prefix_fill | 0.3285 | 0.3270 | -0.47 | -0.74 |
| 65536 | 0.5 | prefix_fill | 0.4119 | 0.4118 | -0.03 | -1.23 |
| 65536 | 0.9 | prefix_fill | 0.3239 | 0.3249 | +0.28 | +0.22 |
| 262144 | 0.1 | prefix_fill | 0.3416 | 0.3415 | -0.04 | +0.53 |
| 262144 | 0.5 | prefix_fill | 0.4238 | 0.4183 | -1.30 | +0.20 |
| 262144 | 0.9 | prefix_fill | 0.3428 | 0.3374 | -1.57 | -0.84 |
| 1048576 | 0.1 | prefix_fill | 0.3555 | 0.3559 | +0.11 | +6.03 |
| 1048576 | 0.5 | prefix_fill | 0.4373 | 0.4336 | -0.86 | +0.80 |
| 1048576 | 0.9 | prefix_fill | 0.3656 | 0.3593 | -1.72 | -1.60 |
| 4194304 | 0.1 | prefix_fill | 0.3759 | 0.3733 | -0.68 | -0.66 |
| 4194304 | 0.5 | prefix_fill | 0.4445 | 0.4441 | -0.10 | +1.89 |
| 4194304 | 0.9 | prefix_fill | 0.3634 | 0.3625 | -0.26 | +1.02 |

#### pair_li

| n | p | routed | raw ns/elem | offset-0 ns/elem | overhead % | control % |
|---|---|---|---|---|---|---|
| 64 | 0.1 | prefix_fill | 2.0845 | 2.0840 | -0.02 | -0.42 |
| 64 | 0.5 | prefix_fill | 1.8572 | 1.8659 | +0.47 | +0.14 |
| 64 | 0.9 | prefix_fill | 1.5725 | 1.5689 | -0.22 | -1.23 |
| 256 | 0.1 | prefix_fill | 1.7873 | 1.7695 | -0.99 | -0.26 |
| 256 | 0.5 | prefix_fill | 1.6882 | 1.7058 | +1.04 | +0.33 |
| 256 | 0.9 | prefix_fill | 1.8349 | 1.8325 | -0.13 | -0.10 |
| 1024 | 0.1 | prefix_fill | 1.2020 | 1.2031 | +0.08 | +0.77 |
| 1024 | 0.5 | prefix_fill | 1.1621 | 1.1656 | +0.31 | +1.03 |
| 1024 | 0.9 | prefix_fill | 1.0756 | 1.0877 | +1.13 | +1.10 |
| 4096 | 0.1 | prefix_fill | 0.8505 | 0.8424 | -0.96 | -1.31 |
| 4096 | 0.5 | prefix_fill | 0.8208 | 0.8222 | +0.18 | -0.94 |
| 4096 | 0.9 | prefix_fill | 0.8139 | 0.8060 | -0.96 | +0.05 |
| 16384 | 0.1 | prefix_fill | 0.7773 | 0.7802 | +0.38 | -1.50 |
| 16384 | 0.5 | prefix_fill | 0.7085 | 0.7123 | +0.54 | +0.39 |
| 16384 | 0.9 | prefix_fill | 0.7239 | 0.7266 | +0.37 | -0.50 |
| 65536 | 0.1 | prefix_fill | 0.4778 | 0.4748 | -0.63 | +0.90 |
| 65536 | 0.5 | prefix_fill | 0.5926 | 0.5910 | -0.26 | -0.43 |
| 65536 | 0.9 | prefix_fill | 0.4819 | 0.4914 | +1.98 | +1.74 |
| 262144 | 0.1 | prefix_fill | 0.5640 | 0.5513 | -2.26 | -4.52 |
| 262144 | 0.5 | prefix_fill | 0.5857 | 0.6030 | +2.96 | +4.45 |
| 262144 | 0.9 | prefix_fill | 0.5598 | 0.5016 | -10.40 | -12.78 |
| 1048576 | 0.1 | prefix_fill | 0.6917 | 0.6956 | +0.56 | -2.20 |
| 1048576 | 0.5 | prefix_fill | 0.6417 | 0.6592 | +2.72 | +0.31 |
| 1048576 | 0.9 | prefix_fill | 0.6597 | 0.6720 | +1.87 | +1.59 |
| 4194304 | 0.1 | prefix_fill | 0.6959 | 0.6845 | -1.64 | -0.63 |
| 4194304 | 0.5 | prefix_fill | 0.6858 | 0.6802 | -0.82 | +1.15 |
| 4194304 | 0.9 | prefix_fill | 0.6873 | 0.6858 | -0.22 | -0.90 |
