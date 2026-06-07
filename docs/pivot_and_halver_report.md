# Why sorted input was slow (pivot bug) + halver-network comparison

Two investigations triggered by the distribution sweep showing `partitions::quicksort`
*slower* on sorted/reverse than on random — which is backwards: near-median pivots on
sorted data should make it **faster**.

## 1. Root cause: median-of-3 is a pivot *killer* on sorted/reverse input

Instrumenting the recursion (total partition work = Σ node sizes; per-node split
balance) on i64, n=2^20:

```
                       work/elem   split behaviour
 random_uniform          18.3      ~50/50 (healthy)
 sorted_ascending        54.9      top 50/50 but 97% of nodes <10% balance
 sorted_descending      500.1      peels ONE element per node (L=1098,R=1)
```

The top levels (n>2048, which used `ninther`) split 50/50 — but every node ≤2048 used
the cheap **`median_of_3`**, and there it picks **near-extreme** pivots, peeling one
element per level → O(n²) within each ≤2048 chunk. Confirming by forcing one pivot
everywhere (work/elem):

```
 pivot \ dist          sorted_descending     random
 median_of_3 EVERY      131073 (= n/8, O(n²))   19.1
 ninther     EVERY          16 (O(n log n))      17.5   <- even less work on random
```

So it is **not** an imbalance from bad luck and **not** "no pattern-defeating" — it is
the textbook *median-of-3 killer*: `median_of_3` samples 3 fixed positions, and the
partitioner's deterministic output on structured input keeps feeding those positions
near-extreme values. (The earlier "median_of_3 is the best pivot" finding was measured
on **random only** — true there, catastrophic in general.)

### Fix and the median-of-3/5 + threshold study

`quicksort.hpp` now uses **ninther (median-of-9) at every node**. We tested the
"m3/m5 with a different threshold" idea directly (i64, n=2^20, ns/elem):

```
 policy                   rand    asc      desc       organ
 m3 everywhere           13.71  25.59  21678.79(O(n²)) 36.15
 m3≤256 / m5>256         13.80  25.58    16.14        26.97   m3 unsafe even ≤256
 m3≤64  / m5>64          14.31  10.89    11.05        17.51   bounded but poor on patterns
 m5 everywhere           15.44  10.36    10.67        16.10   slower than ninther everywhere
 ninther everywhere      14.53   7.83     8.12        13.08   <-- BEST
 m5≤1024 / ninther>1024  15.22  10.39    10.12        14.16
```

Conclusions: **m3 is unsafe at any threshold > ~64**; **m5 is robust but slower than
ninther on every column** (ninther's 9 spread samples give better balance, and the
sampling cost is amortised); a size-adaptive m3/m5 never beats plain ninther. So
**ninther everywhere** is the choice — fastest on sorted/organ, within noise on random.

Verified after the fix (i64, n=2^18, single run, ns/elem): sorted_ascending **27→7.5**,
sorted_descending **96→8.3** — now *faster than random* (13.7), as intuition predicts.
(`few_unique` stays slow — that is the separate, unaddressed lack of 3-way duplicate
partitioning.) All 6 test suites pass.

## 2. Halver-network comparison (`benchmarks/bench_halver_compare.cpp`)

Candidates verified by exhaustive 0/1 enumeration, then timed (ns per network app).

### Correctness
* `h8_new` (14 comparators, depth 4) is a correct halver @ split 4 — one fewer
  comparator and two shallower than `h8` (15, depth 6).
* `h7_alt` (11 comparators, the posted layered net) is a halver **only for split@4**
  as originally posted (counterexample `1110100 → 0100111` for split@3). **Reflecting
  the wires `i → 6−i` makes it split@3** (reflection swaps split@k ↔ split@(n−k);
  7−4 = 3), matching `halve_n<7>`'s `first+3` boundary. That reflected 11-comparator
  net is what is now in the header.

### The i64 "+43%" was a benchmark artifact, not a real cost

The first measurement (batch-apply the net to many blocks in a tight loop) showed
`h8_new` **+43% slower on i64** — despite fewer comparators, lower depth, and (in the
single-block disassembly) *fewer* instructions/cmov and **no spills**. Disassembling
the **batched loop** explained it: GCC **auto-vectorises the i64 batch loop with AVX2**
(`vpblendvb` for branchless min/max on packed longs) and then **spills ymm registers
hard** — a 552-byte stack frame with ~50 `vmovdqa …(%rsp)` spills. `h8_new`'s wide
depth-4 layers keep more vector values live → more spilling → slower.

This **does not happen in the real quicksort**: `halve()` is called on one block per
recursion step, not in a batch loop — a single `halve()` disassembles to **0**
vectorised-spill ops (lean scalar cmov). Forcing the benchmark to match real use
(`-fno-tree-vectorize`) flips the result and speeds up *both* nets:

```
 i64 batched, ns/block      h8_old   h8_new
 vectorised (artifact)       4.90     7.32   (h8_new 1.49x slower)
 scalar (= real use)         3.11     2.91   (h8_new 0.93x — FASTER, and both ~40-60% faster)
```

### Representative (scalar / real-use) net costs, ns/block

```
            h7_alt vs h7(12)     h8_new vs h8_old(15)
 i64           −8.0%                  −6.5%
 pair64        +4.3%                  −0.2%
 pair64f       −8.2%                  −6.6%
```

### Decisions & improvements

1. **Adopt `h8_new`** — `halve_n<8>` now uses it (clean win or neutral on every type).
2. **Adopt the reflected split@3 `h7_alt`** — `halve_n<7>` now uses it (−8% i64/pair64f;
   a small +4.3% on pair64-lex, on the infrequent n=7 blocks → negligible total).
3. **Fixed the halver benchmarks** (`bench_halver_compare`, `bench_small_halve`) with
   `[[gnu::optimize("no-tree-vectorize")]]` on the batch appliers, so they measure the
   scalar per-block cost the quicksort actually pays. (The earlier batch-vectorised
   numbers were unrepresentative — a lesson: benchmark a small kernel the way it is
   really invoked.)
4. **Future:** for the cheap-cmov i64 case the scalar comparator *count* is what wins,
   so depth-minimal nets help only if they don't raise register pressure; the search
   that found `h8_new`/`h7_alt` can target "few comparators AND low scalar live-set".

## 3. The correct way to optimise a small network (`python/halver_optimize.py`)

#comparators and depth are the *wrong* objective. The real cost is set by register
pressure (→ spills, especially when the net is **fused/inlined** into a sort),
critical-path *latency* (weighted by per-comparator cost, which differs for i64 vs
16-byte pairs), throughput/port pressure, and the compiler's own codegen (including
counter-productive auto-vectorisation). Crucially, a **wire permutation preserves the
DAG** — same count, depth and critical path — yet changes which physical registers the
compiler picks and how it schedules, so it changes speed **for reasons no static metric
can see**. A halver stays correct under any permutation that maps the bottom-k and top-k
wires to themselves (a symmetry group of size k!·(n−k)!), and under any topological
re-schedule of its comparators. The only reliable way to pick among them is to measure.

`python/halver_optimize.py` is an empirical autotuner over that space: for a target
(n, split k, element type) it takes base halvers, generates split-preserving wire
permutations + alternative schedules, **verifies every candidate** (0/1 enumeration),
emits one C++ TU applying each as a **scalar (no-tree-vectorize, = real per-block use)**
applier *and* a **"fused" variant** that keeps 6 values live across the loop to expose
register pressure, compiles once with the real toolchain, and times them.

Findings from running it (which correct the earlier write-up):

* **`h7_alt` is faster-or-tied for all three types**, not slower on pair64 — the earlier
  "+4.3% pair64-lex" was harness noise (a permutation can't change the DAG; such tiny
  flips are pure measurement/codegen). It also has **lower register pressure (6 vs 7)**.
* **`h8_new` wins for every type in BOTH plain and fused modes**, even though its peak
  register pressure is *higher* (8 vs h8_old's 7). So minimising register pressure is
  *also* not the right objective on its own — 8 live wires still fit in the 16 GP
  registers without spilling even when fused, and h8_new's fewer comparators + lower
  depth dominate. (The concern is real; it just doesn't bite at n=8. The "fused" column
  is exactly how you'd catch it when it does.)
* Static metrics mispredict the winner (h8_new wins i64 *despite* higher reg-pressure;
  among permutations the best for pair64f differs from the best for i64) — confirming
  measurement, not counting, is the optimisation criterion. Pipeline:
  `halver_finder.py` (find few-comparator base nets) → `halver_optimize.py` (tune the
  arrangement for the actual CPU/compiler/element-type).
