# Plan10 — Optimizing the 10-input perfect halver

Same method and goal as Plan12.md: make `small_halve::halve_n<10>` as fast as
possible. **Raw performance is the only criterion.** Test types: `i64`,
`pair64` (lex), `pair64f` (first-coordinate projection).

n=10 split@5 (balanced): the 5 smallest in `[0,5)`, 5 largest in `[5,10)`.
Validity (0-1 principle, 2^10 = 1024 inputs): `max(bottom 5) <= min(top 5)`, i.e.
`(OR_{0..4}) & ~(AND_{5..9}) == 0`. Reverse halver reuses the same array via the
negation identity (re-verified).

Measurement: `build/benchmarks/bench_small_halve 10`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`.

## Baseline (committed h10 = 21 CE, depth 6)

Canonical `bench_small_halve 10`, median of 3 (ns/elem):

| type    | halve_ns | halve_reg_ns | sort_ns |
|---------|----------|--------------|---------|
| i64     | 0.490    | 0.557        | 0.664   |
| pair64  | 2.030    | 2.347        | 2.760   |
| pair64f | 1.094    | 1.252        | 1.488   |

## Method (identical to n=12/n=11)
ILS search (`optimizers/search_halve.cpp`, generic over N): greedy CE deletion +
perturb-reprune; validity by the 1024-input truth-table-column evaluator; seeds
= current h10 (21 CE) and the best-known n10 *sorter* (29 CE). Collect distinct
minimal nets across many seeds; benchmark them directly (pool bench, stride 10);
pick by measured time.

## Results log (newest first)

### E1 — Network search → committed p22 (20 CE). **DONE.**
Every seed/restart floors at **20 comparators** (was 21; **19 never found** →
20 is the practical minimum), depth 6 (unchanged).

High-rep bench (min over 5 runs, ns/elem) of the leading 20-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.4812 | 2.0182 | 1.0881  | 3.5875 |
| **p22**  | 0.4671 | 1.7470 | 1.0052  | **3.2193** |
| p37  | 0.4879 | 1.7635 | 0.9816  | 3.2330 |
| p58  | 0.4667 | 1.8145 | 0.9591  | 3.2403 |
| p48  | 0.4454 | 1.8188 | 0.9768  | 3.2410 |

Chose **p22** — best total and best pair64 (the heaviest case). (Nets like p48
shave i64 further but lose more on pair64; the dominant absolute cost is pair64,
so p22 wins overall.) Committed to `nets::h10`; re-verified fwd + rev (exhaustive
0/1).

p22 = {0,8},{1,9},{2,7},{0,2},{4,5},{1,4},{3,6},{5,8},{7,9},{2,4},{5,7},{8,9},{0,1},{1,5},{2,3},{4,8},{6,7},{3,5},{4,6},{4,5}

**Real `bench_small_halve 10` (median of 3):**

| type    | baseline | new (p22) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.490    | 0.490     | ~tied (i64 is load/loop-bound at this size; the 1-CE cut shows in the pool bench but is within canonical noise) |
| pair64  | 2.030    | 1.761     | **1.15× (−13.3%)** |
| pair64f | 1.094    | 1.007     | **1.09× (−8.0%)** |

`halve_reg<10>` and the reverse halver inherit the gain. No regression on i64.

## FINAL SUMMARY
`nets::h10`: **21 CE → 20 CE/depth 6** ("p22"), found by ILS + chosen by direct
benchmark. Verified (fwd+rev exhaustive 0/1); full test suite green. Gains:
pair64 −13%, pair64f −8%, i64 flat (no regression). Only one CE could be removed
(20 is the floor), so the headroom here is smaller than n=11/n=12, but the
heavy-CE types (pair64/pair64f) still improve markedly. Same lessons as n=12.

Reproducibility: `g++ -O3 -march=native -std=c++23 optimizers/search_halve.cpp -o
/tmp/sh && /tmp/sh 10 <seed> <restarts> <iters> <poolK>` (writes distinct minima
to `/tmp/h10_pool.inc`). Canonical measure: `bench_small_halve 10`.
