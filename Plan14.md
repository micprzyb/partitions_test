# Plan14 — Optimizing the 14-input perfect halver

Same method/goal as Plan12.md. **Raw performance is the only criterion.** Test
types: `i64`, `pair64` (lex), `pair64f` (first-coordinate projection).

n=14 split@7 (balanced): the 7 smallest in `[0,7)`. Validity (0-1 principle,
2^14 = 16384 inputs): `(OR_{0..6}) & ~(AND_{7..13}) == 0`. Reverse halver reuses
the same array via the negation identity (re-verified).

Measurement: `build/benchmarks/bench_small_halve 14`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`.

## Method (identical to n=10/11/12)
ILS search (`optimizers/search_halve.cpp`, generic over N; seeds read from
`/tmp/seeds_14.txt` = current 38-CE halver + best-known n14 sorter): greedy CE
deletion + perturb-reprune; validity by the 16384-input truth-table-column
evaluator. Collect distinct minimal nets across 6 seeds; benchmark them directly
(pool bench, stride 14); pick by measured time.

## Results log (newest first)

### E1 — Network search → committed p41 (34 CE). **DONE.**
Every seed/restart floors at **34 comparators** (was 38; **−4**), depth 8 (was
10). High-rep bench (min over 5 runs, ns/elem) of leading 34-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.6089 | 2.6276 | 1.4173  | 4.654  |
| **p41**  | 0.5614 | 2.2046 | 1.2409  | **4.007** |
| p33  | 0.5539 | 2.2084 | 1.2626  | 4.025  |
| p46  | 0.5603 | 2.2729 | 1.2318  | 4.065  |

Chose **p41** — best total and best pair64. Committed to `nets::h14`; re-verified
fwd + rev (exhaustive 0/1).

p41 = {0,1},{8,9},{2,3},{4,5},{6,7},{10,11},{6,10},{12,13},{0,2},{4,8},{5,9},{10,12},{3,7},{11,13},{2,8},{7,9},{0,6},{1,5},{7,13},{8,12},{2,10},{3,11},{4,6},{1,3},{5,11},{6,7},{3,10},{3,6},{5,8},{7,10},{5,6},{7,8},{6,7},{1,12}

**Real `bench_small_halve 14` (median of 3):**

| type    | baseline | new (p41) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.609    | 0.560     | **1.09× (−8.0%)** |
| pair64  | 2.619    | 2.216     | **1.18× (−15.4%)** |
| pair64f | 1.434    | 1.249     | **1.15× (−12.9%)** |

`halve_reg<14>` and the reverse halver inherit the gain.

## FINAL SUMMARY
`nets::h14`: **38 CE/depth 10 → 34 CE/depth 8** ("p41"). Verified (fwd+rev
exhaustive 0/1); full suite green. Gains: i64 −8%, pair64 −15%, pair64f −13%.
The previous 38-CE net (pruned from the 51-CE Batcher-ish sorter) left 4 CE on
the table. Same lessons as n=12 apply.

Reproducibility: `optimizers/search_halve.cpp` with `/tmp/seeds_14.txt`; canonical
measure `bench_small_halve 14`.
