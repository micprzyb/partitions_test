# Plan15 — Optimizing the 15-input perfect halver

Same method/goal as Plan12.md. **Raw performance is the only criterion.** Test
types: `i64`, `pair64` (lex), `pair64f` (first-coordinate projection).

n=15 is **odd**: `halve_n<15>` returns `first + 7`, split@7 — 7 smallest in
`[0,7)`, 8 largest in `[7,15)`. Validity (0-1 principle, 2^15 = 32768 inputs):
`(OR_{0..6}) & ~(AND_{7..14}) == 0`. Reverse halver reuses the same array via the
negation identity (re-verified).

Measurement: `build/benchmarks/bench_small_halve 15`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`.

## Method (identical to n=10/11/12/14)
ILS search (`optimizers/search_halve.cpp`; seeds `/tmp/seeds_15.txt` = current 43-CE
halver + best-known n15 sorter); validity by the 32768-input truth-table-column
evaluator; 6 seeds; pick from the pool of minimal nets by direct benchmark
(stride 15).

## Results log (newest first)

### E1 — Network search → committed p13 (39 CE). **DONE.**
Every seed/restart floors at **39 comparators** (was 43; **−4**). High-rep bench
(min over 5 runs, ns/elem) of leading 39-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.6383 | 2.9941 | 1.5449  | 5.177  |
| **p13**  | 0.5948 | 2.3922 | 1.3315  | **4.319** |
| p3   | 0.5950 | 2.4545 | 1.3285  | 4.378  |
| p44  | 0.5977 | 2.4552 | 1.3569  | 4.410  |

Chose **p13** — best total and best pair64. Committed to `nets::h15`; re-verified
fwd + rev (exhaustive 0/1).

p13 = {1,2},{4,14},{5,8},{6,13},{9,11},{1,5},{2,8},{3,7},{6,9},{10,12},{11,13},{0,7},{1,6},{2,9},{4,10},{5,11},{8,13},{12,14},{0,6},{2,4},{3,5},{7,14},{7,11},{8,10},{9,12},{1,14},{0,3},{4,7},{5,9},{6,8},{4,6},{7,9},{3,5},{10,12},{8,10},{5,6},{2,11},{7,8},{6,7}

**Real `bench_small_halve 15` (median of 3):**

| type    | baseline | new (p13) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.658    | 0.583     | **1.13× (−11.4%)** |
| pair64  | 3.020    | 2.394     | **1.26× (−20.7%)** |
| pair64f | 1.561    | 1.333     | **1.17× (−14.6%)** |

`halve_reg<15>` and the reverse halver inherit the gain.

## FINAL SUMMARY
`nets::h15`: **43 CE → 39 CE** ("p13"). Verified (fwd+rev exhaustive 0/1); full
suite green. Largest relative win of the batch — pair64 −21%. The previous 43-CE
net left 4 CE on the table. Same lessons as n=12.

Reproducibility: `optimizers/search_halve.cpp` with `/tmp/seeds_15.txt`; canonical
measure `bench_small_halve 15`.
