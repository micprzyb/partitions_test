# Plan16 — Optimizing the 16-input perfect halver

Same method/goal as Plan12.md. **Raw performance is the only criterion.** Test
types: `i64`, `pair64` (lex), `pair64f` (first-coordinate projection).

n=16 split@8 (balanced): the 8 smallest in `[0,8)`. Validity (0-1 principle,
2^16 = 65536 inputs): `(OR_{0..7}) & ~(AND_{8..15}) == 0`. Reverse halver reuses
the same array via the negation identity (re-verified).

Measurement: `build/benchmarks/bench_small_halve 16`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`.

## Method (identical to n=10/11/12/14/15)
ILS search (`optimizers/search_halve.cpp`; seeds `/tmp/seeds_16.txt` = current 47-CE
halver + best-known n16 sorter); validity by the 65536-input truth-table-column
evaluator; 6 seeds; pick from the pool of minimal nets by direct benchmark
(stride 16).

## Results log (newest first)

### E1 — Network search → committed p79 (44 CE). **DONE.**
Every seed/restart floors at **44 comparators** (was 47; **−3**). High-rep bench
(min over 5 runs, ns/elem) of leading 44-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.6554 | 3.0288 | 1.5641  | 5.248  |
| **p79**  | 0.6229 | 2.6092 | 1.4409  | **4.673** |
| p73  | 0.6235 | 2.6681 | 1.4469  | 4.739  |
| p5   | 0.6246 | 2.7113 | 1.4252  | 4.761  |

Chose **p79** — best total and best pair64. Committed to `nets::h16`; re-verified
fwd + rev (exhaustive 0/1).

p79 = {2,15},{3,14},{4,8},{7,11},{9,10},{0,5},{1,7},{2,9},{3,4},{6,13},{8,14},{10,15},{0,1},{2,3},{10,11},{14,15},{0,2},{7,9},{1,3},{4,10},{5,11},{6,7},{8,9},{12,14},{3,13},{13,15},{1,2},{3,12},{10,12},{4,6},{5,7},{8,10},{9,11},{13,14},{2,6},{5,8},{7,10},{9,13},{3,6},{9,12},{8,9},{6,8},{7,9},{7,8}

**Real `bench_small_halve 16` (median of 3):**

| type    | baseline | new (p79) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.671    | 0.626     | **1.07× (−6.7%)** |
| pair64  | 3.065    | 2.621     | **1.17× (−14.5%)** |
| pair64f | 1.583    | 1.448     | **1.09× (−8.5%)** |

`halve_reg<16>` and the reverse halver inherit the gain.

## FINAL SUMMARY
`nets::h16`: **47 CE → 44 CE** ("p79"). Verified (fwd+rev exhaustive 0/1); full
suite green. Gains: i64 −7%, pair64 −15%, pair64f −9%. Same lessons as n=12.

Reproducibility: `optimizers/search_halve.cpp` with `/tmp/seeds_16.txt`; canonical
measure `bench_small_halve 16`.
