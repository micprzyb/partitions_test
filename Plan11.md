# Plan11 — Optimizing the 11-input perfect halver

Same method and goal as Plan12.md: make `small_halve::halve_n<11>` as fast as
possible. **Raw performance is the only criterion.** Test types: `i64`,
`pair64` (lex), `pair64f` (first-coordinate projection).

n=11 is **odd**: `halve_n<11>` returns `first + 5`, so the split is @5 — the 5
smallest in `[0,5)`, the 6 largest in `[5,11)` (asymmetric). Validity (0-1
principle, 2^11 = 2048 inputs): `max(bottom 5) <= min(top 6)`, i.e.
`(OR_{0..4}) & ~(AND_{5..10}) == 0`. The reverse (descending) halver reuses the
same array via the negation identity (`cswap_rev ≡ -forward(-x)`), valid for any
split, so a new forward halver keeps `halve_rev_n<11>` valid (re-verified).

Measurement: `build/benchmarks/bench_small_halve 11`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`.

## Baseline (committed h11 = 25 CE, depth 8)

Canonical `bench_small_halve 11`, median of 3 (ns/elem):

| type    | halve_ns | halve_reg_ns | sort_ns |
|---------|----------|--------------|---------|
| i64     | 0.523    | 0.606        | 0.708   |
| pair64  | 2.199    | 2.485        | 3.221   |
| pair64f | 1.156    | 1.286        | 1.591   |

## Method (identical to n=12 — see Plan12.md for rationale)
1. ILS search (`optimizers/search_halve.cpp`, generic over N): greedy CE deletion +
   perturb-reprune; validity by the 2048-input truth-table-column evaluator;
   seeds = current h11 (25 CE) and the best-known n11 *sorter* (35 CE).
2. Collect distinct minimal nets across many seeds; benchmark them directly
   (pool bench, stride 11) for the three types; pick by measured time, not by
   CE/depth heuristics.

## Results log (newest first)

### E1 — Network search → committed p66 (23 CE). **DONE.**
Every seed/restart floors at **23 comparators** (was 25; **22 never found** →
23 is the practical minimum). Depth of the fastest 23-CE net is 9 (the previous
net was depth 8) — but, as established for n=12, **depth barely matters** in this
throughput-bound batch; selection is by measured time.

High-rep bench (min over 5 runs, ns/elem) of the leading 23-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.5096 | 2.1775 | 1.1409  | 3.828  |
| **p66**  | 0.4824 | 1.8547 | 1.0253  | **3.3624** |
| p146 | 0.4807 | 1.8390 | 1.0573  | 3.377  |
| p9   | 0.4822 | 1.9028 | 1.0129  | 3.398  |
| p142 | 0.4641 | 1.9324 | 1.0230  | 3.420  |

Chose **p66** — best total, near-best pair64 (the heaviest case), strong on all
three. Committed to `nets::h11`; re-verified forward + reverse (exhaustive 0/1).

p66 = {0,9},{1,6},{0,1},{3,5},{4,10},{2,8},{1,10},{6,9},{1,3},{4,7},{8,10},{0,4},{1,2},{3,7},{5,9},{4,5},{7,8},{2,4},{3,6},{6,7},{3,4},{5,6},{4,5}

**Real `bench_small_halve 11` (median of 3):**

| type    | baseline | new (p66) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.523    | 0.485     | **1.08× (−7.3%)** |
| pair64  | 2.199    | 1.865     | **1.18× (−15.2%)** |
| pair64f | 1.156    | 1.028     | **1.12× (−11.1%)** |

`halve_reg<11>` and the reverse halver inherit the gain.

## FINAL SUMMARY
`nets::h11`: **25 CE/depth 8 → 23 CE** ("p66"), found by ILS + chosen by direct
benchmark. Verified (fwd+rev exhaustive 0/1); full test suite green. Gains:
i64 −7%, pair64 −15%, pair64f −11%. Same lessons as n=12 apply (CE count is the
big lever; depth ~irrelevant here; keep search/greedy emission order — do not
layer-group; SIMD-across-blocks and per-CE primitive offer no in-scope win).

Reproducibility: `g++ -O3 -march=native -std=c++23 optimizers/search_halve.cpp -o
/tmp/sh && /tmp/sh 11 <seed> <restarts> <iters> <poolK>` (writes distinct minima
to `/tmp/h11_pool.inc`). Canonical measure: `bench_small_halve 11`.
