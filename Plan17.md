# Plan17 — Optimizing the 17-input perfect halver

Same method/goal as Plan12.md. **Raw performance is the only criterion.** Test
types: `i64`, `pair64` (lex), `pair64f` (first-coordinate projection).

n=17 is **odd**: `halve_n<17>` returns `first + 8`, split@8 — 8 smallest in
`[0,8)`, 9 largest in `[8,17)`. Validity (0-1 principle, 2^17 = 131072 inputs):
`(OR_{0..7}) & ~(AND_{8..16}) == 0`. Reverse halver reuses the same array via the
negation identity (re-verified).

Measurement: `build/benchmarks/bench_small_halve 17`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`.

## Method (identical to n=10..16)
ILS search (`optimizers/search_halve.cpp`, flat truth-table evaluator — the per-check
reset is one contiguous memcpy, important at n=17's NW=2048 words/wire);
**24-way parallel**, two waves:
- wave 1: seeds = current 56-CE halver + best-known n17 sorter, 24 jobs.
- wave 2: reseeded from the wave-1 48-CE minima (seed file = those nets), 24 jobs,
  more iters, to push the floor.

Pick from the pool of minimal nets by direct benchmark (stride 17). (A defensive
guard was added to the seed parser to skip out-of-range indices, after a buggy
non-destructive size filter once produced garbage seeds → segfault.)

## Results log (newest first)

### E1 — Network search → committed p22 (48 CE). **DONE — largest win so far.**
Both waves floor at **48 comparators** (was 56; **−8!**; 47 never found across
~3100 distinct minima). High-rep bench (min over 5 runs, ns/elem) of leading
48-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.7292 | 3.2675 | 1.7477  | 5.744  |
| **p22**  | 0.6398 | 2.7014 | 1.4919  | **4.833** |
| p72  | 0.6298 | 2.7638 | 1.4496  | 4.843  |
| p52  | 0.6391 | 2.8050 | 1.4301  | 4.874  |

Chose **p22** — best total and best pair64. Committed to `nets::h17`; re-verified
fwd + rev (exhaustive 0/1).

p22 = {0,11},{1,15},{2,10},{7,11},{6,14},{4,5},{4,6},{8,12},{3,5},{9,16},{13,14},{0,6},{1,8},{2,8},{5,15},{3,7},{4,9},{6,16},{10,11},{0,9},{12,14},{0,2},{1,4},{5,6},{7,13},{8,9},{10,12},{11,14},{15,16},{2,5},{6,11},{9,13},{12,15},{3,4},{5,10},{7,8},{3,7},{4,8},{6,12},{6,9},{2,7},{4,5},{10,12},{5,7},{8,10},{6,8},{7,9},{7,8}

**Real `bench_small_halve 17` (median of 3):**

| type    | baseline | new (p22) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.740    | 0.643     | **1.15× (−13.1%)** |
| pair64  | 3.284    | 2.711     | **1.21× (−17.4%)** |
| pair64f | 1.756    | 1.498     | **1.17× (−14.7%)** |

`halve_reg<17>` and the reverse halver inherit the gain.

## FINAL SUMMARY
`nets::h17`: **56 CE → 48 CE** ("p22"), an 8-comparator (14%) reduction — by far
the largest of the batch. The old 56-CE net (greedy-pruned from the 71-CE n17
sorter) was very loose. Verified (fwd+rev exhaustive 0/1); full suite green.
Gains: i64 −13%, pair64 −17%, pair64f −15%. Same lessons as n=12; the bigger N is
where the most CE could be shaved, and the 24-way parallel + reseeded-wave search
was what reached the floor cheaply.

Reproducibility: `optimizers/search_halve.cpp` with `/tmp/seeds_17.txt` (24 parallel
seeds; wave 2 reseeded from 48-CE minima); canonical measure `bench_small_halve 17`.
