# Plan12 — Optimizing the 12-input perfect halver

Goal: make `small_halve::halve_n<12>` (and, where it helps, `halve_reg<12>`) run
as fast as possible on a modern CPU. **Raw performance is the only criterion.**

Test element types (exactly the bench's three):
1. `i64`     — long integer.
2. `pair64`  — pair of longs, **lexicographic** order.
3. `pair64f` — pair of longs, compare on **first coordinate only** (projection).

Correctness gate: the halver property (`max(bottom 6) <= min(top 6)`) must hold
for all 2^12 = 4096 0/1 inputs (Knuth 0-1 principle). Verified by
`tools/verify_small_halve.cpp` and, during search, by an in-tool 4096 enumeration.

---

## Measurement methodology

- Machine: AMD Ryzen 9 5950X (Zen 3), AVX2/BMI2/FMA, no AVX-512. `perf` denied,
  governor not writable. So: **pin `taskset -c 4`, min-over-reps, treat ns ±5%.**
- Tool: `build/benchmarks/bench_small_halve 12`. Columns: `halve_ns, halve_reg_ns,
  sort_ns` (ns/element, min over 400 reps, batch ≈ 65536/12 blocks). Only
  `halve_ns` (and `halve_reg_ns`) are the optimization targets; `sort_ns` is the
  reference upper bound.
- Re-run 3× per change; report the median of the per-line minimums.
- Rebuild: `cmake --build build -j --target bench_small_halve verify_small_halve`.
- Always re-run `build/tools/verify_small_halve` after any network change.

### Baseline (commit 70c35e7, current h12 = 29 CE, depth 9)

| type    | halve_ns | halve_reg_ns | sort_ns |
|---------|----------|--------------|---------|
| i64     | 0.540    | 0.616        | 0.720   |
| pair64  | 2.330    | 2.647        | 3.045   |
| pair64f | 1.243    | 1.369        | 1.640   |

Observations:
- `halve_n` (in-place) beats `halve_reg` (register-blocked) for all three types
  in this batch loop. So the in-place path is the one to optimize.
- pair64 is ~4.3× the i64 cost (16-byte lex CE is the dominant primitive).
- pair64f sits in between (1 first-key compare + 16-byte carry/blend per CE).

---

## Cost model / where the time goes

Per block of 12: the network is fully unrolled, scalar (bench forces
`no-tree-vectorize`), branchless `cswap` per comparator.

- **i64 cswap** = 1 `cmp` + 2 `cmov` (parallel min/max form) ≈ 3 ALU µops.
  29 CE ≈ 87 ALU µops + 12 loads + 12 stores. Roughly ALU-throughput bound, but
  one block (~120 µops) is ~½ the ROB, so the per-block critical path (depth 9,
  with a serial depth-4 tail) is partly exposed.
- **pair64 cswap (lex)** = lex predicate (cmp first, cmp-eq first, cmp second,
  OR/AND combine) + `half_swap` (4 cmov over the two 64-bit halves) ≈ 8–10 µops.
  29 CE ≈ ~270 µops/block — one block exceeds the ROB, so this is the most
  throughput-bound AND its depth-9 critical path is fully exposed. Both CE count
  and depth matter most here.
- **pair64f cswap** = 1 first-key `cmp` + `half_swap` (4 cmov) ≈ 5 µops.

**Conclusion: the two primary levers are (1) total comparator count and
(2) network depth (esp. the serial tail), in that priority. Per-CE primitive
cost is a secondary lever, mostly for pair64.**

---

## Optimization ideas (ranked)

### E1 — Find a smaller / shallower perfect halver network  [PRIMARY]
The current 29-CE/depth-9 net came from greedy pruning of the 39-CE sorter; it is
**not proven minimal**. A 12-halver may need fewer than 29 CE, and almost
certainly admits a shallower arrangement (the depth-4 serial tail
`{6,7}{4,6}{5,7}{5,6}` is suspicious). Plan:
- Write a standalone search tool (`tools/search_halve12.cpp`):
  - validity = 4096-input 0/1 enumeration, split@6;
  - greedy comparator deletion with randomized order + random restarts, seeded
    from the 39-CE sorter, the 29-CE halver, and random supersets;
  - perturb-and-reprune to escape local minima;
  - among smallest CE, minimize depth; also separately hunt min-depth.
- Candidates → drop into `nets::h12`, re-verify, benchmark. A net with fewer CE
  helps all three types ~proportionally; a shallower net helps pair64 most.

### E2 — Reorder comparators within the chosen net for codegen/ILP
Within data-dependency layers, present independent CEs so register allocation and
the OOO scheduler expose maximum parallelism; minimize the exposed serial tail.
Likely marginal for throughput-bound i64 (compiler reschedules) but may help
pair64 latency. Cheap to try once a net is fixed.

### E3 — Per-CE primitive improvements (secondary, mostly pair64)
- i64: confirm the parallel min/max (1 cmp + 2 cmov) is optimal vs alternatives.
- pair64 lex: try to shave the predicate/`half_swap` µops (fewer XOR/cmov).
- pair64f: the carry is 16 bytes but only `.first` is compared — confirm codegen
  isn't recomputing the predicate or over-copying.
- These live in `small_sort.hpp::cswap`; must not regress other N or the sorter.

### E4 — SIMD across independent blocks (SoA)  [HIGH UPSIDE, HIGHER EFFORT]
The per-block network has lane-crossing comparators, so within-block SIMD needs
shuffles. But **processing K independent blocks in SIMD lanes** turns each
comparator (i,j) into a vector min/max with NO shuffles:
- i64: AVX2 has no 64-bit integer min/max, emulate with `cmpgt`+`blendv`
  (2–3 µops) over 4 blocks at once → potential ~ up to 4× on i64.
- pair64: 2 blocks per 256-bit reg; lex compare via vector cmp/blend.
Requires a batched entry point; the current bench calls `halve_n` per block.
Evaluate whether a batched variant is in-scope/beneficial and, if so, add it and
a bench mode. Treat as a later, separate exploration.

### E5 — Misc
- Test whether `halve_reg<12>` can be made to win (it currently loses); if not,
  leave the in-place path as the target.
- Check alignment / load-store patterns.

---

## Results log

(Updated immediately after every attempt — newest first. No exceptions.)

### E1 — Network search: found 27-CE / depth-7 halvers (was 29-CE / depth-9)
**Done. Big win available. Not yet committed to the header.**

Tooling built:
- `tools/search_halve12.cpp` — standalone iterated-local-search (greedy deletion
  + perturb-reprune), validity by 4096-input 0/1 truth-table-column evaluator
  (wire = 4096-bit mask; CE = AND→min/OR→max; valid iff `(OR bottom)&~(AND top)==0`).
- `benchmarks/bench_h12_candidates.cpp` — benches arbitrary nets passed as a
  compile-time reference NTTP (must be compile-time so CE indices bake in exactly
  like `halve_n`; a runtime ref inflates time ~3×). Build:
  `g++ -O3 -march=native -std=c++23 -Iinclude -Ibenchmarks
  benchmarks/bench_h12_candidates.cpp -o /tmp/benchcand`.

Search outcome: every seed/restart (thousands of ILS iters, multiple seeds)
floors at **27 comparators, depth 7**. 26 never found → 27 is the practical
minimum. The current 29-CE net is beatable by 2 CE *and* 2 depth levels.

Candidate bench (min over 8 reps, ns/elem; lower = better):

| net  | i64    | pair64 | pair64f | sum   |
|------|--------|--------|---------|-------|
| base | 0.5538 | 2.3339 | 1.2394  | 4.127 |
| s1   | 0.5094 | 2.2252 | **1.0902** | 3.825 |
| s2   | 0.5072 | 2.2121 | 1.1419  | 3.861 |
| s3   | 0.5094 | 2.2237 | 1.1454  | 3.879 |
| s4   | 0.5269 | 2.3235 | 1.1296  | 3.980 |
| d1   | 0.5271 | 2.2803 | 1.1170  | 3.924 |
| d2   | 0.5095 | 2.3694 | 1.1038  | 3.983 |
| d3   | 0.5274 | **2.0838** | 1.1274  | **3.739** |
| d4   | **0.4918** | 2.2364 | 1.1257  | 3.854 |

All 27-CE nets beat base on all three types. No single net wins all three:
- **d3** wins pair64 decisively (2.084, ~11% under base) and has the best *total*.
- **d4** wins i64 (0.492, ~11% under base).
- **s1** wins pair64f (1.090, ~12% under base), best geomean, most balanced.

The per-net spread among 27-CE nets is real (esp. d3's pair64), and likely traces
to depth/tail structure of the heavy 16-byte lex CE — worth understanding.

Networks (application order):
- s1 = {2,6},{0,7},{3,11},{5,9},{0,1},{2,5},{3,4},{6,9},{7,8},{10,11},{0,2},{9,11},{4,6},{5,7},{1,4},{3,5},{6,8},{2,9},{7,10},{2,5},{6,9},{1,7},{4,10},{4,6},{5,7},{4,7},{5,6}
- d3 = {1,7},{3,11},{2,6},{5,9},{0,1},{2,5},{3,4},{6,9},{7,8},{10,11},{0,2},{9,11},{1,10},{4,6},{5,7},{4,7},{1,4},{2,9},{3,5},{6,8},{7,10},{2,5},{6,9},{4,6},{5,7},{4,7},{5,6}
- d4 = {1,7},{2,6},{3,11},{5,9},{0,1},{2,5},{3,4},{6,9},{7,8},{10,11},{0,2},{9,11},{4,6},{4,10},{5,7},{1,7},{1,4},{3,5},{6,8},{2,9},{7,10},{2,5},{6,9},{4,6},{5,7},{5,6},{4,7}

### E1 (cont.) — Pool-bench → committed p44 as the new `nets::h12`. **DONE.**
To pick the best *single* net data-drivenly (not by guesswork), extended the
search to collect distinct 27-CE minima and benched 150+ of them
(`bench_h12_pool.cpp`, generated `/tmp/h12_pool.inc`). High-rep stability bench
(`bench_h12_top.cpp`, min over 4 runs) of the leaders:

| net  | i64    | pair64 | pair64f | sum    | depth |
|------|--------|--------|---------|--------|-------|
| base | 0.5317 | 2.2999 | 1.2283  | 4.060  | 9     |
| **p44**  | 0.5095 | **2.0122** | 1.1201 | **3.642** | **8** |
| p118 | 0.5071 | 2.0289 | 1.1067  | 3.643  | 10    |
| p123 | 0.4935 | 2.1227 | 1.0977  | 3.714  | 8     |
| d4   | 0.4949 | 2.2318 | 1.1349  | 3.862  | 7     |

Key finding: **depth barely matters for this throughput-bound batch** — p118
(depth 10) ties p44 (depth 8); the i64-fastest net (p123/d4) sacrifices pair64,
which is the dominant absolute cost. **Chose p44**: best pair64 (the heaviest +
task-emphasised case), tied-best total, shallowest (depth 8, no serial tail →
robust if reused latency-bound), clean layer profile [6,5,3,4,3,2,2,2].

Committed p44 to `nets::h12`. Re-verified BOTH forward
(`verify_small_halve`) and reverse (`verify_small_halve_rev`, which reflects the
same array) — both pass exhaustive 0/1.

**Real `bench_small_halve 12` (the canonical tool), median of 3:**

| type    | baseline halve_ns | new halve_ns | improvement |
|---------|-------------------|--------------|-------------|
| i64     | 0.540             | 0.500        | **−7.4%**   |
| pair64  | 2.330             | 2.010        | **−13.7%**  |
| pair64f | 1.243             | 1.117        | **−10.1%**  |

`halve_reg` also improved (pair64 2.647→2.37) but still loses to in-place `halve`.

New baseline for subsequent experiments = p44 numbers above.

### E2 — Reorder p44 within dependency layers. **DONE — keep search order.**
Tested the committed search order vs valid topological reorders (all identical,
verified networks): layer-grouped (within-layer asc / desc), and within-layer
random shuffles. Min over 3 runs (ns/elem):

| order   | i64    | pair64 | pair64f | sum   |
|---------|--------|--------|---------|-------|
| **search (p44)** | 0.5161 | **2.0101** | 1.1264 | **3.653** |
| rnd1    | 0.5274 | 2.2070 | 1.1840  | 3.918 |
| layer   | 0.5272 | 2.2503 | 1.1436  | 3.921 |
| layerR  | 0.5272 | 2.2730 | 1.1377  | 3.938 |
| rnd2    | 0.5272 | 2.3071 | 1.1613  | 3.996 |

**Finding (counterintuitive, valuable):** layer-grouping — which "exposes ILP" —
is ~10% SLOWER on pair64 and worse on every type. Emitting all independent
layer-1 CEs first makes ~all 12 elements (×2 words = 24 GPRs for pair64) live at
once → register spills / store-reload traffic. The greedy/search order is
depth-first-ish, keeping the live set small. **Lesson: do not layer-group these
networks; preserve the search/greedy emission order.** p44 stays as committed.
No code change. (Also explains E1's depth-insensitivity: register pressure +
port throughput, not comparator-layer depth, governs the scalar per-block cost.)

### Validation — full suite green
`ctest` (all 12 suites) passes with p44 committed; `verify_small_halve` and
`verify_small_halve_rev` both pass exhaustive 0/1. No regressions.

### E3 — Per-CE primitive. **Examined, no in-scope win.**
The dominant cost (esp. pair64) is `small_sort::cswap`. It is already
heavily tuned and SHARED by every N and by the sorter: i64 uses the optimal
parallel min/max (1 cmp + 2 cmov); pair64 lex uses a single-compare-derived
predicate (`seta`+`sete` off one `cmp`) + branchless `half_swap` (4 cmov). There
is no *halver-specific* shortcut (comparisons are between arbitrary positions, not
a fixed pivot, so no pivot trick). The pair64f "compare keys only, carry 16B"
idea decomposes to the same ~5 µops as the current `half_swap`. Changing cswap
risks every other consumer for no clear gain → left untouched. No code change.

### E4 — SoA SIMD across blocks (AVX2). **Prototyped, REJECTED with data.**
Built an i64 halver that runs 4 independent blocks in AVX2 lanes (`p44` applied
lane-wise: `cmpgt_epi64` + 2×`blendv` per CE), including the AoS↔SoA transpose
(`/tmp/e4.cpp`). Verified correct (halver property holds per lane). Result on
Zen 3:

| variant            | i64 ns/elem |
|--------------------|-------------|
| scalar p44         | 0.519       |
| SoA-4 AVX2 (+transpose) | 0.584  | → **0.89× (slower)** |

Why it loses on Zen 3: (1) AVX2 has **no native 64-bit integer min/max** — must
emulate with `cmpgt`+`blendv`, and `blendv` is a 2-µop, port-limited op; (2) the
AoS↔SoA transpose needs 6× cross-lane `permute2x128` (3-cycle latency) plus
unpacks; the transpose overhead alone exceeds the scalar network. pair64 SoA
(only 2 lanes/reg, harder lex predicate) would fare worse. **Also out of scope
for the real consumer:** the only user is `quicksort.hpp::halve_sort`, a
*recursive single-block* halver — there is no batch of independent same-size
blocks to vectorise. The scalar per-block network is the correct target.

---

## FINAL SUMMARY

**Single change shipped:** `nets::h12` replaced (29 CE / depth 9 → **27 CE /
depth 8**, network "p44"), found by ILS search over the 4096-input 0/1 validity
check and chosen by direct benchmark from the pool of 27-CE minima. Verified
correct (forward + reverse, exhaustive 0/1); full test suite green.

**Real `bench_small_halve 12` improvement (median of 3, Zen 3):**

| type    | before | after | speedup |
|---------|--------|-------|---------|
| i64     | 0.540  | 0.500 | **1.08×** (−7.4%) |
| pair64  | 2.330  | 2.010 | **1.16×** (−13.7%) |
| pair64f | 1.243  | 1.117 | **1.11×** (−10.1%) |

`halve_reg<12>` (and the reverse halver) inherit the gain automatically.

**Key lessons (carry to other N):**
1. The shipped 29-CE nets are prunable — search for fewer comparators first
   (biggest lever; 27 is the floor for n=12, 26 never found).
2. For this scalar throughput-bound regime, **comparator-layer depth barely
   matters**; register pressure + port throughput dominate.
3. **Do not layer-group** the comparators — it inflates the live set (16-byte
   pair values spill) and costs ~10% on pair64. Keep the greedy/search emission
   order, which is naturally low-register-pressure.
4. **SIMD-across-blocks does not pay** on Zen 3 (no epi64 min/max; transpose
   cost) and doesn't fit the recursive single-block consumer.
5. Per-CE primitive (`cswap`) is already optimal and shared; not a halver lever.

**Reproducibility:**
- Search (generic over N — supersedes the original search_halve12.cpp):
  `g++ -O3 -march=native -std=c++23 optimizers/search_halve.cpp -o /tmp/sh
  && /tmp/sh 12 <seed> <restarts> <iters> <poolK>` (writes distinct minima to
  `/tmp/h12_pool.inc`).
- Candidate/ordering/SIMD micro-benches were scratch (`/tmp/*.cpp`), removed from
  the tree; methodology above is sufficient to regenerate. The canonical measure
  is `build/benchmarks/bench_small_halve 12` under `taskset -c 4`.

**Status: complete for n=12.** No further scalar lever identified with positive,
in-scope expected value. Ready to apply the same method to other N on request.

### E1-recheck (later pass) — confirmed 27 is the floor. **No change.**
Re-ran the search with the generic `optimizers/search_halve.cpp` (deeper: 4 seeds ×
400 restarts × 2000 iters, plus the n14/15/16 batch's seeds). Still floors at
**27 comparators** (depth-7 variants exist; 26 never found). The committed p44
remains the chosen net. No header change. (Done as part of the n=14/15/16 batch,
which the user listed alongside n=12.)
