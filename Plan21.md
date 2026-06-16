# Plan21 — Optimizing the 21-input perfect halver

Same method/goal as Plan22.md / Plan23.md. **Raw performance is the only
criterion.** Test types: `i64`, `pair64` (lex), `pair64f` (first-coordinate
projection). Flagged as a follow-up in Plan22 (n=21's committed 69 CE was *larger*
than the new n=22 at 68 CE — same Batcher slack).

## Setup
The committed `nets::h21` is **69 CE** and Batcher-derived (`{0,1},{2,3},{4,5},...`
opening) — never CEGAR-searched.

n=21 is **odd**, split@10 — 10 smallest in `[0,10)`, 11 largest in `[10,21)`.
Validity (0-1 principle): a valid split@10 halver iff it halves every input with
exactly 10 zeros / 11 ones — **C(21,11) = 352,716** inputs (k-zero reduction), the
cheapest sweep of the family.

The **reverse halver `halve_rev_n<21>` reuses the same `nets::h21` array**
(duality) — any new net must be re-verified **both** forward
(`verify_small_halve`) and reverse (`verify_small_halve_rev`), exhaustive 2²¹.

## Hardware / method
- Intel Core Ultra 7 165H. **6 P-cores**; search pinned one job per physical core
  → CPUs **{0,1,3,6,8,10}**. Benchmark on a quiet 5 GHz P-core (CPU 4) after the
  search stops, min over reps.
- Tool: `optimizers/search_halve_cegar2.cpp` (`/tmp/shb2`) — plateau-walking ILS +
  CEGAR + C(21,11) k-zero validity. `g++ -O3 -march=native -std=c++23`.
- Seeds: `make_seeds.py 21` → h21 (69 CE) + n21 sorter (99 CE).
- Bench pool: `optimizers/{make_pool.py,bench_pool.cpp}`; canonical
  `bench_small_halve 21`. Gate: `verify_small_halve{,_rev}` (exhaustive 2²¹).

## FINAL SUMMARY
`nets::h21`: **69 CE → 65 CE** ("p1160", depth 16) — a 4-comparator (6%) reduction.
The old 69-CE net was Batcher-derived and loose (larger than the freshly-optimised
n=22 at 68). Found by `search_halve_cegar2.cpp` (plateau-walking ILS + CEGAR +
C(21,11) k-zero sweep), 6 P-cores × 4 reseeded waves (69→66→65; 64 never found
across **1181** distinct 65-CE minima), selected by direct benchmark (median of 9
over the full 1181-net floor; p1160 won sum + pair64). Verified fwd+rev exhaustive
2²¹; full suite green (12/12). Gains (canonical `bench_small_halve 21`, median of
3): **i64 −5.8%, pair64 −15.7%, pair64f −12.7%**. Reverse halver and `halve_reg<21>`
inherit the gain (shared array). Depth rose 14→16 — irrelevant in this
throughput-bound regime (per project notes / Plan24).

Reproducibility: `optimizers/search_halve_cegar2.cpp` with `/tmp/seeds_21.txt`
(6 jobs pinned to P-cores {0,1,3,6,8,10}; reseed from the floor between waves),
pool benched via `optimizers/{make_pool.py,bench_pool.cpp}`; canonical
`bench_small_halve 21` on a quiet P-core (CPU 4).

## Results log (newest first)

### E7 — Commit p1160, verify, canonical benchmark. **DONE.**
Committed p1160 to `nets::h21` (69 → 65 CE, depth 16). **Exhaustive 2²¹ verify
PASSED both forward and reverse** (reverse reuses the same array). Canonical
`bench_small_halve 21`, `taskset -c 4`, `halve_ns`, median of 3:

| type    | baseline (69 CE) | new (65 CE) | speedup |
|---------|------------------|-------------|---------|
| i64     | 0.6645           | 0.6262      | **1.06× (−5.8%)** |
| pair64  | 3.3796           | 2.8498      | **1.19× (−15.7%)** |
| pair64f | 1.5634           | 1.3642      | **1.15× (−12.7%)** |

Full `ctest`: 12/12 passed.

### E6 — High-rep confirm of the leaders (median of 9 runs). **DONE.**
Focused pool re-benched 9× on CPU 4, **median per type**:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.6665 | 3.4569 | 1.5545  | 5.6779 |
| **p1160** | 0.6522 | 2.8436 | 1.3379  | **4.8337** |
| p1119 | 0.6525 | 2.8726 | 1.3450  | 4.8701 |
| p1162 | 0.6525 | 2.9124 | 1.3317  | 4.8966 |

**Winner: p1160** (65 CE, depth 16) — best sum (~0.7% clear) and best pair64,
competitive on i64/pair64f. (p1047, the single-pass pair64f leader, collapsed to
3.60 on pair64 under medians — single-run noise; the median step was needed.)
Next: commit, verify fwd+rev, ctest, canonical bench.

p1160 = {6,7},{2,3},{4,5},{16,20},{0,1},{5,7},{8,9},{10,11},{17,18},{9,11},{14,15},{0,2},{12,13},{4,6},{8,10},{16,17},{0,8},{1,3},{12,14},{4,12},{17,19},{13,15},{1,9},{7,15},{5,13},{6,14},{3,11},{3,7},{8,12},{2,10},{3,12},{10,14},{7,19},{3,17},{7,20},{7,17},{9,13},{13,17},{12,20},{2,6},{6,13},{6,9},{3,16},{7,10},{2,16},{9,16},{4,8},{10,18},{0,8},{11,13},{1,5},{5,10},{5,6},{10,18},{6,12},{11,15},{11,14},{1,7},{6,8},{7,8},{10,16},{11,12},{10,11},{9,10},{8,10}

### E5 — Pool benchmark, full 1181-net floor. **DONE.**
Compile of 1182 nets: 7.5 min. `taskset -c 4`, min over 300 reps. Baseline h21
(69 CE): 0.7061 / 3.3560 / 1.5573 (sum 5.619). Top by sum:

| net   | i64    | pair64 | pair64f | sum    |
|-------|--------|--------|---------|--------|
| base  | 0.7061 | 3.3560 | 1.5573  | 5.6194 |
| **p1160** | 0.6523 | 2.8431 | 1.3365  | **4.8320** |
| p1162 | 0.6260 | 2.9131 | 1.3298  | 4.8689 |
| p1119 | 0.6520 | 2.8734 | 1.3442  | 4.8696 |

Per-type winners: i64 p911 (0.6254), pair64 p1160 (2.8431), pair64f p1047
(1.2905). **p1160 leads sum + pair64.** Next: focused median-of-9 confirm of
{p1160,p1162,p1119,p1158,p1130,p769,p861,p1166,p723,p1047,p911,p204}.

### E4 — Wave 4: 6 jobs × 36 restarts × 5000 iters, reseeded from 36 basins (final 64 attempt). **DONE.**
~12 min wall. **Floor held at 65 — no 64.** Distinct census **65→1181** (66→393,
67→320) — as conclusive as n=22 (1192). **Conclusion: 65 is the practical floor
for n=21** (69 → 65, −4). Next: benchmark the 65-CE pool, pick fastest by median.

### E3 — Wave 3: 6 jobs × 30 restarts × 5000 iters, reseeded from 30 of 240 frontier basins. **DONE.**
~10 min wall. **Floor held at 65 — no 64.** Distinct census **65→709**, 66→393,
67→320. Since 65 first appeared only in wave 2, run one more wave (W4) before
concluding. Next: wave 4, final 64 attempt.

### E2 — Wave 2: 6 jobs × 24 restarts × 4000 iters, reseeded from 24 of 160 frontier basins. **DONE.**
~6 min wall. Floor **66 → 65 CE**: 3 jobs at 65. Distinct census **65→240, 66→393,
67→320**. So 69 → 65 (**−4**); floor still moving. Next: wave 3 (reseed from 65-CE
frontier) to chase 64.

### E1 — Wave 1: 6 jobs × 10 restarts × 5000 iters, header seeds. **DONE.**
Cores {0,1,3,6,8,10}, ~3 min wall (300k iters). Floor **67 → 66 CE**: 2 jobs at
66, rest 67. 480 distinct nets (66→160, 67→320). So 69 → 66 (**−3**). Next: reseed
wave 2 from the 66-CE frontier to chase 65.

### E0 — Smoke test (1 job, 4 restarts × 600 iters, CPU0). **DONE.**
~9 s wall. Floor immediately **69 → 67 CE** — confirms ≥2 CE slack. Next:
multi-core reseeded waves to find the true floor, then benchmark.
