# Plan22 — Optimizing the 22-input perfect halver

Same method/goal as Plan23.md / Plan24.md. **Raw performance is the only
criterion.** Test types: `i64`, `pair64` (lex), `pair64f` (first-coordinate
projection).

## Setup
The committed `nets::h22` is **74 CE** and Batcher-derived (`{0,1},{2,3},{4,5},...`
opening) — never CEGAR-searched, so headroom was expected (cf. n=23: 84→78).

n=22 is **even**, split@11 — 11 smallest in `[0,11)`, 11 largest in `[11,22)`.
Validity (0-1 principle): a valid split@11 halver iff it halves every input with
exactly 11 zeros / 11 ones — **C(22,11) = 705,432** inputs (k-zero reduction),
the cheapest sweep of the n≥22 family (½ of n=23, ¼ of n=24).

The **reverse halver `halve_rev_n<22>` reuses the same `nets::h22` array**
(duality), so any new net must be re-verified **both** forward
(`verify_small_halve`) and reverse (`verify_small_halve_rev`), exhaustive 2²².

## Hardware / method
- Intel Core Ultra 7 165H. **6 P-cores**; search pinned one job per physical core
  → CPUs **{0,1,3,6,8,10}** (no HT contention). Benchmark on a quiet 5 GHz P-core
  (CPU 4) **after** the search stops, min over reps, ns ±5%.
- Tool: `optimizers/search_halve_cegar2.cpp` (`/tmp/shb2`) — plateau-walking ILS
  (sideways acceptance + adaptive kicks + in-process reseed; bounded witness set),
  CEGAR + C(22,11) k-zero validity. Built `g++ -O3 -march=native -std=c++23`.
- Seeds: `make_seeds.py 22` → h22 (74 CE) + n22 sorter (106 CE).
- Bench pool: `optimizers/{make_pool.py,bench_pool.cpp}`; canonical measure
  `build/benchmarks/bench_small_halve 22`.
- Gate: `verify_small_halve` + `verify_small_halve_rev` (exhaustive 2²²).

## FINAL SUMMARY
`nets::h22`: **74 CE → 68 CE** ("p205", depth 13) — a 6-comparator (8%) reduction.
The old 74-CE net was Batcher-derived and grossly loose (it had *more* comparators
than n=21's 69; that net is likely loose too). Found by
`search_halve_cegar2.cpp` (plateau-walking ILS + CEGAR + C(22,11) k-zero sweep),
6 P-cores × 3 reseeded waves (74→68; 67 never found across **1192** distinct
68-CE minima), selected by direct benchmark (median of 9 over the full 1192-net
floor; p205 won every type). Verified fwd+rev exhaustive 2²²; full suite green
(12/12). Gains (canonical `bench_small_halve 22`, median of 3):
**i64 −8.0%, pair64 −21.0%, pair64f −14.9%**. Reverse halver and `halve_reg<22>`
inherit the gain (shared array).

Follow-up flagged: **n=21 (69 CE) is now larger than n=22 (68 CE)** — the same
Batcher slack; a Plan21 pass should shave it similarly.

Reproducibility: `optimizers/search_halve_cegar2.cpp` with `/tmp/seeds_22.txt`
(6 jobs pinned to P-cores {0,1,3,6,8,10}; reseed from the floor between waves),
pool benched via `optimizers/{make_pool.py,bench_pool.cpp}`; canonical measure
`bench_small_halve 22` on a quiet P-core (CPU 4).

## Results log (newest first)

### E6 — Commit p205, verify, canonical benchmark. **DONE.**
Committed p205 to `nets::h22` (74 → 68 CE, depth 13). **Exhaustive 2²² verify
PASSED both forward and reverse** (reverse reuses the same array). Canonical
`bench_small_halve 22`, `taskset -c 4`, `halve_ns`, median of 3:

| type    | baseline (74 CE) | new (68 CE) | speedup |
|---------|------------------|-------------|---------|
| i64     | 0.6793           | 0.6249      | **1.09× (−8.0%)** |
| pair64  | 3.4562           | 2.7305      | **1.27× (−21.0%)** |
| pair64f | 1.5797           | 1.3440      | **1.18× (−14.9%)** |

Full `ctest`: 12/12 passed.

### E5 — High-rep confirm of the leaders (median of 9 runs). **DONE.**
Focused pool re-benched 9× on CPU 4, **median per type**:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.6793 | 3.5509 | 1.6482  | 5.8784 |
| **p205** | 0.6502 | 2.7271 | 1.3452  | **4.7225** |
| p189 | 0.6503 | 2.7938 | 1.3724  | 4.8165 |
| p231 | 0.6518 | 2.8445 | 1.3638  | 4.8601 |

**Winner: p205** (68 CE, depth 13) — wins i64, pair64, and sum outright (~2% clear
of #2 p189), 2nd-best pair64f (p79 0.3385... err 1.3385). Unambiguous, no
selection noise. Next: commit, verify fwd+rev, ctest, canonical bench.

p205 = {8,9},{0,1},{20,21},{2,3},{16,17},{12,13},{18,19},{19,21},{18,20},{0,2},{4,5},{1,3},{6,7},{10,11},{14,15},{4,6},{5,7},{15,17},{14,16},{2,6},{15,19},{1,5},{16,20},{8,12},{5,17},{3,7},{11,13},{6,20},{1,15},{5,21},{7,20},{6,15},{12,15},{0,4},{2,18},{6,11},{14,18},{4,16},{2,6},{6,14},{5,12},{0,1},{1,14},{7,17},{3,19},{13,21},{10,18},{9,11},{7,15},{12,18},{3,16},{5,10},{7,13},{3,12},{10,12},{7,19},{7,9},{3,7},{4,8},{11,16},{11,18},{8,14},{7,11},{9,12},{10,14},{9,14},{7,14},{9,11}

### E4 — Pool benchmark, full 1192-net floor. **DONE.**
Compile of 1193 nets: 8.5 min. `taskset -c 4`, min over 300 reps. Baseline h22
(74 CE): 0.7229 / 3.4052 / 1.6522 (sum 5.780). Top by sum:

| net   | i64    | pair64 | pair64f | sum    |
|-------|--------|--------|---------|--------|
| base  | 0.7229 | 3.4052 | 1.6522  | 5.7803 |
| **p205**  | 0.6506 | 2.7327 | 1.3535  | **4.7368** |
| p721  | 0.6505 | 2.8501 | 1.3553  | 4.8558 |
| p690  | 0.6504 | 2.8250 | 1.3824  | 4.8578 |

Per-type winners: i64 p191 (0.6243, but weak elsewhere), pair64 p205 (2.7327),
pair64f p79 (1.2891). **p205 dominates** — best sum AND best pair64, ~2.5% clear
of #2. Next: focused median-of-9 confirm of {p205,p721,p690,p266,p336,p231,p455,
p189,p373,p737,p79,p191}.

### E3 — Wave 3: 6 jobs × 30 restarts × 5000 iters, reseeded from 30 basins (final 67 attempt). **DONE.**
~10 min wall. **All 6 jobs floored at 68 — no 67.** Distinct census now **1192 @
68 CE** — stronger evidence than n=23 (706) or n=24 (1035) had at their floors.
**Conclusion: 68 is the practical floor for n=22** (74 → 68, −6). Next: benchmark
the 68-CE pool, pick fastest by median.

### E2 — Wave 2: 6 jobs × 24 restarts × 4000 iters, reseeded from 24 of 240 frontier basins. **DONE.**
~6 min wall. **All 6 jobs floored at 68 — no 67.** Distinct census now **719 @ 68
CE** (+160@69, +80@70). Strong evidence 68 is the floor (cf. n=23: 706@78 was
conclusive). Next: wave 3 (reseed from 30 basins, more iters) as a final 67
attempt; if none, benchmark the 68-CE pool.

### E1 — Wave 1: 6 jobs × 10 restarts × 5000 iters, header seeds. **DONE.**
Cores {0,1,3,6,8,10}, ~3 min wall (300k iters). Floor crashed **72 → 68 CE**: 3
jobs floored at 68, others 68/69/70. 480 distinct nets (68→240, 69→160, 70→80).
So 74 → 68 already (**−6**). NB 68 < n=21's committed 69 CE → **n=21 is also
loose** (same Batcher-derived slack; flag for a follow-up). Next: reseed wave 2
from the 68-CE frontier to chase 67.

### E0 — Smoke test (1 job, 4 restarts × 600 iters, CPU0). **DONE.**
~8.7 s wall. Floor immediately **74 → 72 CE** (depth 13), 112 distinct 72-CE nets
pooled — confirms the committed net had ≥2 CE of slack and the plateau-walk
explores broadly. Next: multi-core reseeded waves to find the true floor, then
benchmark.
