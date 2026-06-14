# Plan13 — Optimizing the 13-input perfect halver

Same method/goal as Plan12.md. **Raw performance is the only criterion.** Test
types: `i64`, `pair64` (lex), `pair64f` (first-coordinate projection).

n=13 is **odd**: `halve_n<13>` returns `first + 6`, split@6 — 6 smallest in
`[0,6)`, 7 largest in `[6,13)`. Validity (0-1 principle, 2^13 = 8192 inputs):
`(OR_{0..5}) & ~(AND_{6..12}) == 0`. Reverse halver reuses the same array via the
negation identity (re-verified).

Measurement: `build/benchmarks/bench_small_halve 13`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`.

## Method (identical to n=10..16)
ILS search (`optimizers/search_halve.cpp`; seeds `/tmp/seeds_13.txt` = current 33-CE
halver + best-known n13 sorter); validity by the 8192-input flat truth-table-
column evaluator; **24 parallel jobs** (distinct seeds → distinct pool files,
merged). Pick from the pool of minimal nets by direct benchmark (stride 13).

## Results log (newest first)

### E1 — Network search → committed p41 (30 CE). **DONE.**
24 parallel searches (200 restarts × 1000 iters each) → 1315 distinct minima;
all floor at **30 comparators** (was 33; **−3**; 29 never found). High-rep bench
(min over 5 runs, ns/elem) of leading 30-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.5748 | 2.4362 | 1.2767  | 4.288  |
| **p41**  | 0.5356 | 2.1281 | 1.1533  | **3.817** |
| p67  | 0.5194 | 2.2206 | 1.1294  | 3.869  |
| p32  | 0.5355 | 2.2063 | 1.1393  | 3.881  |

Chose **p41** — best total and best pair64. Depth 11 (was 9) — irrelevant for
this throughput-bound regime (established in Plan12). Committed to `nets::h13`;
re-verified fwd + rev (exhaustive 0/1).

p41 = {0,12},{1,10},{2,9},{3,7},{5,11},{6,8},{1,6},{2,3},{4,11},{7,9},{8,10},{3,6},{7,8},{9,10},{5,9},{8,11},{1,2},{3,8},{4,7},{8,12},{5,8},{2,5},{6,9},{5,7},{0,4},{3,4},{6,8},{4,5},{6,7},{5,6}

**Real `bench_small_halve 13` (median of 3):**

| type    | baseline | new (p41) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.585    | 0.540     | **1.08× (−7.7%)** |
| pair64  | 2.474    | 2.144     | **1.15× (−13.3%)** |
| pair64f | 1.310    | 1.158     | **1.13× (−11.6%)** |

`halve_reg<13>` and the reverse halver inherit the gain.

## FINAL SUMMARY
`nets::h13`: **33 CE → 30 CE** ("p41"). Verified (fwd+rev exhaustive 0/1); full
suite green. Gains: i64 −8%, pair64 −13%, pair64f −12%. Same lessons as n=12.

Reproducibility: `optimizers/search_halve.cpp` with `/tmp/seeds_13.txt` (24 parallel
seeds); canonical measure `bench_small_halve 13`.
