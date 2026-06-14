# Plan24 — Optimizing the 24-input perfect halver

Same method/goal as Plan12.md. **Raw performance is the only criterion.** Test
types: `i64`, `pair64` (lex), `pair64f` (first-coordinate projection).

n=24 split@12 (balanced): the 12 smallest in `[0,12)`. Validity (0-1 principle):
a net is a valid halver iff it halves every input with exactly 12 zeros —
**C(24,12) = 2,704,156** inputs (the monotonicity reduction; ~6.3× fewer than
2²⁴ = 16.8M). Reverse halver reuses the same array via the negation identity
(re-verified).

Measurement: `build/benchmarks/bench_small_halve 24`, `taskset -c 4`, min over
reps, ns ±5% (Zen 3). Verify: `verify_small_halve` + `verify_small_halve_rev`
(exhaustive 2²⁴ — the trusted gate for the heuristic-found net).

## Why a different search tool here
At n=24 the full 2²⁴ truth-table state is ~48 MB; re-materialising it per
validity check is memory-bandwidth-bound — **measured ~36 ms/check** with the
flat exhaustive method (`search_halve.cpp`), so one ILS restart ≈ 30 min:
intractable for the iteration counts needed. So n=24 used
`optimizers/search_halve_cegar.cpp`:
- **CEGAR**: a cheap witness pre-filter (counterexample inputs) rejects almost
  all invalid trials; only survivors pay the full sweep.
- **Monotonicity / k-zero reduction** (idea borrowed from the SAT finders in
  `python/`): the full sweep checks only the C(24,12) balanced inputs, packed as
  bit-columns. **Measured 6.35× faster** than the full 2²⁴ sweep (11.9 ms → 1.9 ms).
- Working set ~4.5 MB (L2-resident) → 30-way parallel scales near-linearly.

The flat exhaustive method (the small-n approach) was also evaluated per request:
correct and simple, memory is fine (48 MB), but ~100× too slow *in time* for the
search at this scale — hence CEGAR. The flat exhaustive check IS still used, via
the repo's trusted `verify_small_halve`, as the final gate.

## Method
ILS (greedy CE deletion + perturb-reprune), 30 parallel jobs, **three reseeded
waves** — each wave reseeds `/tmp/seeds_24.txt` from the previous wave's frontier
(best-size nets), which is what kept breaking the floor:
- wave 1 (seeds = 88-CE h24 + 120-CE n24 sorter): floor **84**.
- wave 2 (reseed from 84-CE): floor **82** (passed through 83).
- wave 3 (reseed from 82-CE, 8×1500 iters × 30): **1035** distinct 82-CE nets,
  **no 81 found** → 82 is the practical floor.

## Results log

### E1 — Network search → committed p83 (82 CE). **DONE.**
Floor **82 comparators** (was 88; **−6**; 81 not found across 3 waves / ~90 jobs).
High-rep bench (min over 5 runs, ns/elem) of leading 82-CE candidates:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.8451 | 3.7977 | 2.0280  | 6.671  |
| **p83**  | 0.8071 | 3.2467 | 1.7906  | **5.844** |
| p12  | 0.7731 | 3.2739 | 1.8167  | 5.864  |
| p78  | 0.7766 | 3.3043 | 1.8070  | 5.888  |

Chose **p83** — best total and best pair64. Depth 13 (was 12) — irrelevant for
this throughput-bound regime. Committed to `nets::h24`; **verified by exhaustive
2²⁴ enumeration, forward AND reverse** (the heuristic only proposes; the trusted
verifier is the gate — it passed).

p83 = {0,20},{1,12},{2,16},{3,23},{4,6},{5,10},{7,21},{11,22},{13,18},{0,3},{17,19},{1,11},{2,7},{4,17},{5,13},{6,19},{10,18},{12,22},{16,21},{20,23},{0,1},{2,4},{1,4},{3,12},{5,8},{6,9},{8,15},{15,18},{7,10},{11,20},{13,16},{14,17},{11,13},{19,21},{22,23},{2,5},{4,8},{7,14},{9,16},{15,19},{18,21},{3,9},{8,22},{0,5},{3,14},{9,20},{10,12},{3,4},{6,10},{6,11},{8,15},{9,14},{13,17},{10,13},{12,17},{16,23},{1,7},{18,22},{19,20},{4,7},{14,18},{16,19},{7,9},{10,11},{12,13},{14,16},{7,8},{5,11},{9,11},{12,14},{13,19},{15,16},{5,10},{8,9},{13,18},{14,15},{4,10},{10,12},{11,13},{9,12},{11,14},{11,12}

**Real `bench_small_halve 24` (median of 3):**

| type    | baseline | new (p83) | speedup |
|---------|----------|-----------|---------|
| i64     | 0.854    | 0.797     | **1.07× (−6.7%)** |
| pair64  | 3.831    | 3.276     | **1.17× (−14.5%)** |
| pair64f | 2.046    | 1.787     | **1.15× (−12.7%)** |

`halve_reg<24>` and the reverse halver inherit the gain.

## FINAL SUMMARY
`nets::h24`: **88 CE → 82 CE** ("p83"). Verified (fwd+rev exhaustive 2²⁴); full
suite green. Gains: i64 −7%, pair64 −15%, pair64f −13%. The old 88-CE net
(greedy-pruned from the 120-CE n24 sorter) had 6 CE of slack. n=24 needed the
CEGAR + k-zero tool (the flat method is ~100× too slow here) and multiple
reseeded waves to reach the floor.

Reproducibility: `optimizers/search_halve_cegar.cpp` with `/tmp/seeds_24.txt`
(30 parallel jobs; reseed from the frontier between waves), pool benched via
`optimizers/{make_pool.py,bench_pool.cpp}`; canonical measure `bench_small_halve 24`.
