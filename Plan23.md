# Plan23 — Optimizing the 23-input perfect halver

Same method/goal as Plan12.md / Plan24.md. **Raw performance is the only
criterion.** Test types: `i64`, `pair64` (lex), `pair64f` (first-coordinate
projection).

---

# PLAN B — Fresh ideas to break the n=23 floor (78 CE) [PROPOSED — not yet run]

## The anomaly
CE ladder: h20:63, h21:65, h22:68, **h23:78**, h24:82. The +10 from n=22→n=23
(same split@11, one extra top wire) dwarfs the +2/+3 elsewhere; n=23→n=24 is only
+4. Either 78 is a search artifact, or the odd split@11 with a 12-wire top is
genuinely expensive. The existing engine (greedy-delete ILS + plateau-walk)
plateaued hard at 78 (706→ then more distinct 78-CE nets, no 77).

## Reconnaissance done (analysis only, no search run)
- **Line-removal from h24 (force a wire to +∞, delete its ∞-path, → split@12 23-halver, reverse → split@11):**
  the ∞-token ratchets to the top fast, so **only 4 comparators come off → a ~78-CE seed.** No free win.
- **Extend h22 (68) by folding in a 23rd wire (max-of-{[0,11),w22}→slot 11):** ~68+11 = **~79-CE seed.**
- ⇒ Both cross-constructions land at **~78–79**, *corroborating 78 as a real plateau* rather than
  pure seed-bias. So the fresh idea must be an engine that can **leave** the plateau, not a better seed.
- `python/halver_finder*.py` already drives **CaDiCaL** (SAT) with a cardinality/`pysat` encoding — a
  real exact-method path exists.

## Tracks (ranked by effort vs payoff)

### Track 1 — Diverse-basin seeding + plateau-walk  *(cheap, ~30 min, low risk)*
Build a *mixed* seed pool of structurally-different **valid** 23-halvers, each validated by the
k-zero sweep before use, then run the existing `shb2` plateau-walk waves from it:
  (a) h24 line-removal → split@12 → reverse to split@11 (~78);
  (b) h22 + one-wire fold (~79);
  (c) from-scratch bitonic / odd-even-merge and Parberry pairwise networks, greedy-pruned;
  (d) **symmetric split@12 search** (optimise the complementary halver, reverse the winner — a
      different greedy basin for the *same* problem).
Goal: reach 77 by starting greedy-delete from non-Batcher basins. If it works, it's nearly free.

### Track 2 — Violation-annealing engine  *(medium, the real fresh idea, ~1–2 h dev)*
New tool `optimizers/anneal_halve.cpp`. The current search is **always-valid greedy descent** — it
can never cross an invalid net, so it's trapped on the 78 plateau. Instead optimise over **all** nets
(any size, valid or not) with objective
        f(net) = |net| + λ · (#k-zero counterexamples)
computed by the **bit-packed k-zero sweep extended to popcount the violation columns** (reuse
`full_cex`'s machinery — cheap). Moves: add / delete / retarget-endpoint of a comparator;
Metropolis acceptance with a cooling schedule; whenever violations hit 0, attempt aggressive
size-reduction (greedy-delete) then continue annealing at the smaller size. This can **tunnel**
through temporarily-invalid configurations the greedy ILS cannot. 6-way multistart on the P-cores.
Goal: 77/76 for n=23 (and the same engine retargets n=24→81).

### Track 3 — SAT-CEGAR existence / bound  *(high effort, moonshot)*
Adapt `python/halver_finder*.py`: encode "∃ a 23-halver with ≤K comparators in ≤D layers?" and drive
CaDiCaL with **lazy counterexample (CEGAR) addition** of k-zero inputs (don't pre-encode all 1.35M).
Binary-search K downward from 78. SAT ⇒ extract the net (and validate exhaustively); UNSAT at depth D
⇒ a depth-bounded lower bound (evidence, or proof if depth-unbounded). First do a **timeboxed
feasibility probe** (small K margin, tight timeout) before committing — SAT may not scale to n=23.

## Selection / verification (all tracks, unchanged)
Any candidate net is only a *proposal*: gate every committed net through `verify_small_halve` +
`verify_small_halve_rev` (exhaustive 2²³), pick the fastest 77-CE (or lower) net by `bench_pool`
median-of-N, then canonical `bench_small_halve 23`. Same applies to n=24 (target 81) — the engines
are size-agnostic.

## Recommendation
Run **Track 1 then Track 2** (Track 1 may crack 77 for free; Track 2 is the engine most likely to
break a true plateau). Hold Track 3 as a moonshot / optimality-evidence step.

## DECISION (2026-06-15): target **n=23 (→77)**, run **Track 1 + Track 2**.

## PLAN-B FINAL SUMMARY
**`nets::h23`: 78 CE → 74 CE ("p411", depth 15); 84 → 74 overall (−10).** The 78
floor the previous CEGAR search hit was a **Batcher-seed-bias local optimum** — not
the real floor. The fresh idea that cracked it (Track 1): seed the search from a
**structurally different basin** — the optimised n=24 halver with a wire forced to
+∞ and its incident comparators dropped → a valid split@12 23-halver, reversed to
split@11. From that basin the plateau-walk descended 78→75→74. A new
**violation-annealing engine** (Track 2, `optimizers/anneal_halve.cpp`) then gave
strong evidence **73 is infeasible** (best 73-net mis-sorts 260/1.35M k-zero inputs
across ~78 strong restarts) → 74 is the practical floor. Winner picked by median-of-9
over the 550-net 74-CE pool. Verified fwd+rev exhaustive 2²³; full suite green.
Gains vs the committed 78-CE (canonical `bench_small_halve 23`, median of 3):
**i64 −1.1%, pair64 −3.3%, pair64f −8.7%**; vs the original 84-CE: ≈ −12% / −20% / −19%.

Reusable lesson: **floors found by always-valid greedy search are seed-relative;
cross-construction seeds (prune/extend a well-optimised neighbour at n±1) + a
violation-tolerant engine are what break them.** The same recipe should be tried on
the rest of the ladder (n=18–20, and a fresh attempt at n=24→81 via diverse seeds).

## PLAN-B execution log (newest first)

### B7 — Commit p411 (74 CE), verify, canonical benchmark. **DONE.**
Committed to `nets::h23` (78 → 74 CE, depth 15). **Exhaustive 2²³ verify PASSED
forward AND reverse** (reverse reuses the array). `bench_small_halve 23`,
`taskset -c 4`, `halve_ns`, median of 3:

| type    | committed 78 CE | new 74 CE | speedup | (vs orig 84 CE) |
|---------|-----------------|-----------|---------|------------------|
| i64     | 0.6844          | 0.6766    | −1.1%   | ≈ −12% |
| pair64  | 3.0851          | 2.9830    | −3.3%   | ≈ −20% |
| pair64f | 1.5923          | 1.4542    | −8.7%   | ≈ −19% |

Full `ctest`: 12/12 (see below).

### B6 — Pool benchmark, full 550-net 74-CE floor. **DONE.**
Compile 551 nets: 3.6 min. `taskset -c 4`, min/300. Baseline = committed h23 (78 CE):
0.7282 / 3.0864 / 1.5999 (sum 5.414). Top by sum:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base(78) | 0.7282 | 3.0864 | 1.5999 | 5.4144 |
| **p532** | 0.6766 | 2.9058 | 1.4536  | **5.0361** |
| p411 | 0.6768 | 2.9915 | 1.4454  | 5.1137 |
| p62  | 0.6769 | 3.0069 | 1.4356  | 5.1194 |

Per-type winners: i64 p275 (0.6494), pair64 p532 (2.9058), pair64f p3 (1.3985).
**p532 leads sum + pair64.** The 74-CE nets beat the 78-CE baseline on all axes.
Next: median-of-9 confirm of {p532,p411,p62,p277,p409,p242,p170,p404,p331,p275,p3}.

### B5 — Track 2 annealer, target=73, attempt 2 (reheats + polish + randomised starts). **DONE — 74 is the floor.**
6 jobs × 5 restarts × 200k steps. Best **V=260** (was 455); 5/6 jobs converge to
~260, none to 0. A robust violation barrier. **Conclusion: 73 is (almost
certainly) infeasible; 74 is the practical floor for n=23.** Evidence: plateau-walk
stalled at 74 (550 distinct, no 73) AND a strong annealer can't get a 73-net below
260/1.35M violations across ~78 diverse starts. (Definitive settling of 73 would
need Track 3 SAT — not authorised; left as future.) Result so far: **84 → 74
(−10); −4 below the previously-"stuck" 78.** Next: benchmark the 74-CE pool, pick
fastest, commit, verify.

### B4 — Track 2 annealer, target=73, attempt 1. **DONE — no 73 (yet).**
6 jobs × 8 restarts × 80k steps from the 74-CE frontier, varied T0/cooling. Best
**V=455** violating k-zero inputs (one job 262) — i.e. the best 73-net mis-sorts
only 455 / 1.35M inputs (0.03%), close but not valid. The repeated 455 across 48
starts = funneling, not a proven barrier (262 shows lower is reachable). Next:
strengthen the annealer (full reheats, randomized start-deletion, more steps) and
retry target=73.

### B3 — Track 1 wave 3: 6 jobs × 30 restarts × 6000 iters, reseeded from 74-CE frontier. **DONE.**
~15 min. **Floor held at 74 — no 73.** 550 distinct 74-CE nets (75→478, 77→320,
78→80). Plateau-walk stalled at 74 (84 → 74, −10). Hand off to **Track 2 (the
violation-annealing engine)** to break 74 → 73. Next: anneal target_size=73 from
the 74-CE frontier.

### B2 — Track 1 wave 2: 6 jobs × 28 restarts × 5000 iters, reseeded from 75-CE frontier. **DONE.**
~12 min. **Floor 75 → 74 CE** (job 750023). Census 74→80, 75→478, 77→320, 78→80.
So **84 → 74 (−10)**, still dropping one level/wave. Next: wave 3 from 74 frontier.

### B1 — Track 1 first wave: 6 jobs × 12 restarts × 6000 iters from diverse seeds. **DONE — PLATEAU BROKEN.**
Seeds = {seedA (h24 line-removal, n24-basin), current h23, n23 sorter}. ~7 min.
**Floor 78 → 75 CE!** (job 7709 hit 75; others 77; census 75→80, 77→320, 78→80).
⇒ **78 was a Batcher-seed-bias local optimum after all** — the earlier 3-wave
search never left that basin; the n24-derived seed descends to 75 immediately.
Track 1 alone already beat the target (77) by 2. So **84 → 75 (−9)**; floor may go
lower. Next: reseed from the 75/77 frontier and push (still Track 1) for 74; then
benchmark. (Track 2 annealer built & compiled at `/tmp/anneal`, held in reserve.)

### B0 — Track 1 setup: build diverse, *validated* seeds. **DONE.**
seedA = h24 with wire-23 forced +∞ (drop its (·,23) comparators) → valid split@12,
reversed → split@11; validated by the C(23,12) k-zero sweep (78 CE, n24 basin).
Seeds file = {seedA, h23, n23 sorter}.
Generate non-Batcher valid 23-halvers and seed the plateau-walk from them:
(a) h24 force-wire-23=+∞ then drop its (·,23) comparators → valid split@12, reverse → split@11
    (proved valid; ~78 CE, n24 basin); (b) h22 + one-wire fold (construct, **validate or drop**);
plus the n23 sorter and current h23 for contrast. Each candidate checked by the C(23,12) k-zero
sweep before use.

## The anomaly that motivates this
The committed `nets::h23` has **84 comparators** — *more* than `nets::h24`'s 82.
A halver on fewer inputs must not need more comparators; the n=23 net is
plainly Batcher-derived (`{0,1},{2,3},{4,5},...` opening) and greedy-pruned,
never run through the CEGAR search that took n=24 to its floor. So large
headroom was expected.

n=23 is **odd**: `halve_n<23>` returns `first + 11`, split@11 — 11 smallest in
`[0,11)`, 12 largest in `[11,23)`. Validity (0-1 principle): a net is a valid
split@11 halver iff it halves every input with exactly **11 zeros / 12 ones** —
**C(23,12) = 1,352,078** inputs (the k-zero / monotonicity reduction), ~2×
fewer than n=24's C(24,12)=2.70M, so the CEGAR sweep is even cheaper here.

The **reverse halver `halve_rev_n<23>` reuses the same `nets::h23` array**
(comparator-network duality, `small_halve_rev.hpp:120`) — so any new net must be
re-verified **both** forward (`verify_small_halve`) and reverse
(`verify_small_halve_rev`), exhaustive 2²³.

## Hardware / method
- Box: Intel Core Ultra 7 165H (Meteor Lake). **6 P-cores** = physical cores
  0–5, logical CPUs paired `{0,5},{1,2},{3,4},{6,7},{8,9},{10,11}`. Search is
  pinned one job per physical P-core → CPUs **{0,1,3,6,8,10}** (no HT contention).
  E-cores (12–19) and LP-E (20–21) left idle. Benchmark on a quiet 5 GHz P-core
  (CPU 2 or 4) **after** the search stops, min over reps, ns ±5%.
- Tool: `optimizers/search_halve_cegar.cpp` (ILS = greedy CE deletion +
  perturb-reprune; CEGAR witness pre-filter + full C(23,12) k-zero sweep).
  Built `g++ -O3 -march=native -std=c++23`.
- Seeds: `make_seeds.py 23` → h23 (84 CE) + n23 sorter (114 CE).
- Bench pool: `optimizers/{make_pool.py,bench_pool.cpp}`; canonical measure
  `build/benchmarks/bench_small_halve 23`.
- Gate: `verify_small_halve` + `verify_small_halve_rev` (exhaustive 2²³).

## FINAL SUMMARY
`nets::h23`: **84 CE → 78 CE** ("p335", depth 13) — a 6-comparator (7%)
reduction. The old 84-CE net was Batcher-derived and grossly loose (it had *more*
comparators than n=24's 82); it had never been run through the CEGAR search. The
new net was found by `search_halve_cegar.cpp` (ILS + CEGAR + C(23,12) k-zero
sweep), 6 P-cores × 3 reseeded waves (84→80→78; 77 not found across 706 distinct
78-CE minima ≈ n=24's evidence), then selected by direct benchmark (median of 7
over the full 706-net floor — single-run ranking was noise). Verified fwd+rev
exhaustive 2²³; full suite green (12/12). Gains (canonical `bench_small_halve 23`,
median of 3): **i64 −7.2%, pair64 −17.0%, pair64f −12.2%**. The reverse halver and
`halve_reg<23>` inherit the gain automatically (shared array).

Reproducibility: `optimizers/search_halve_cegar.cpp` with `/tmp/seeds_23.txt`
(6 parallel jobs pinned to P-cores {0,1,3,6,8,10}; reseed from the floor between
waves), pool benched via `optimizers/{make_pool.py,bench_pool.cpp}`; canonical
measure `bench_small_halve 23` on a quiet P-core (CPU 4).

## Results log (newest first)

### E7 — Commit p335, verify, canonical benchmark. **DONE.**
Committed p335 to `nets::h23` (84 → 78 CE, depth 13) with a Plan-style comment.
**Exhaustive 2²³ verify PASSED both forward (`verify_small_halve`) and reverse
(`verify_small_halve_rev`)** — the reverse halver reuses the same array, so both
gated. Canonical `bench_small_halve 23`, `taskset -c 4`, `halve_ns` column,
median of 3:

| type    | baseline (84 CE) | new (78 CE) | speedup |
|---------|------------------|-------------|---------|
| i64     | 0.7683           | 0.7128      | **1.08× (−7.2%)** |
| pair64  | 3.7127           | 3.0821      | **1.20× (−17.0%)** |
| pair64f | 1.7949           | 1.5757      | **1.14× (−12.2%)** |

`halve_reg<23>` and the reverse halver inherit the gain. Full `ctest` suite:
**12/12 passed** (29 s).

### E6 — High-rep confirm of the leaders (median of 7 runs). **DONE.**
Focused pool {p26,p335,p477,p99,p607,p635,p391,p660,p42,p539,p480,p674}, each
re-benched 7× on CPU 4, **median per type**:

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.7378 | 3.7169 | 1.7972  | 6.2519 |
| **p335** | 0.7131 | 3.0848 | 1.6107  | **5.4086** |
| p477 | 0.7292 | 3.0704 | 1.6147  | 5.4143 |
| p635 | 0.7134 | 3.1277 | 1.6271  | 5.4682 |
| p607 | 0.7129 | 3.1571 | 1.6006  | 5.4706 |
| p26  | 0.7138 | 3.2410 | 1.6173  | 5.5721 |

The single-run E5 leader **p26 fell to 5.57** under medians (confirming ~1% noise
made E5's ranking unreliable) — high-rep was necessary. **Winner: p335** (78 CE,
depth 13): best sum, best-tied i64 (0.7131, vs p607 0.7129), 2nd-best pair64,
good pair64f. p477 wins pair64 alone but loses ~2% on i64 (the primary type), so
p335's balanced profile is the pick (same rationale as n=24's p83).

p335 = {0,1},{2,3},{4,5},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},{0,2},{4,6},{5,7},{8,10},{9,11},{12,14},{20,22},{13,15},{16,18},{17,19},{21,22},{0,4},{1,5},{2,6},{8,12},{9,13},{10,14},{11,15},{17,21},{18,20},{19,22},{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{15,22},{5,18},{18,21},{9,16},{6,7},{2,9},{16,19},{7,19},{3,20},{10,18},{12,17},{13,20},{6,16},{14,15},{1,12},{3,17},{13,16},{3,4},{5,8},{9,12},{11,21},{17,18},{7,13},{14,16},{3,9},{4,10},{11,17},{8,10},{7,11},{10,12},{13,18},{14,17},{6,9},{7,10},{11,13},{12,14},{8,9},{11,12},{4,9},{9,10},{10,11}

Next: commit to `nets::h23`, verify fwd+rev (exhaustive 2²³), full ctest, then
canonical `bench_small_halve 23` speedup vs the old 84-CE net.

### E5 — Pool benchmark, full 706-net floor. **DONE.**
Compile of 707 nets: 5.5 min. `taskset -c 4`, min over 300 reps. Baseline h23
re-measured 0.7847 / 3.7811 / 1.8675 (sum 6.433). Top by sum:

| net   | i64    | pair64 | pair64f | sum    |
|-------|--------|--------|---------|--------|
| base  | 0.7847 | 3.7811 | 1.8675  | 6.433  |
| **p26**   | 0.7131 | 3.1824 | 1.6017  | **5.4973** |
| p335  | 0.7282 | 3.1461 | 1.6325  | 5.5068 |
| p477  | 0.7450 | 3.1341 | 1.6424  | 5.5214 |
| p99   | 0.7288 | 3.1787 | 1.6533  | 5.5608 |

Per-type winners: i64 p42 (0.7129), pair64 p477 (3.1341), pair64f p539 (1.5907).
**p26 is the best all-rounder** — best sum, ~best i64, best pair64f of the
leaders, top-tier pair64. Leaders cluster within ~1% (single-run noise), so next:
focused high-rep multi-run confirm of {p26,p335,p477,p99,p42,p539,...} to pick
robustly. NB net indices here are the full-pool ordering; resolve the actual
comparator list from `/tmp/hpool.inc` before committing.

### E4 — Pool benchmark, pass 1 (120-net sample of the 706 78-CE floor). **DONE.**
`bench_pool` HN=23, `taskset -c 4` (5 GHz P-core), min over 300 reps. Compile of
121 nets: 44 s. Baseline = committed h23 (84 CE).

| net  | i64    | pair64 | pair64f | sum    |
|------|--------|--------|---------|--------|
| base | 0.7847 | 3.7229 | 1.8288  | 6.336  |
| **p99**  | 0.7135 | 3.1769 | 1.6203  | **5.511** |
| p89  | 0.7283 | 3.2284 | 1.6344  | 5.591  |
| p51  | 0.7131 | 3.2576 | 1.6398  | 5.611  |

Per-type winners: i64 p80 (0.7128), pair64 p99 (3.1769), pair64f p65 (1.5406).
**p99 wins both sum and pair64** — provisional leader (i64 −9.1%, pair64 −14.7%,
pair64f −11.4% vs base). But only 120/706 nets sampled (cap order). Next: bench
the **full 706-net floor** to confirm the global fastest, then high-rep confirm
the top few.

### E3 — Wave 3: 6 jobs × 30 restarts × 5000 iters, reseeded from 78-CE (final 77 attempt). **DONE.**
Reseeded from a shuffled 30-net subset of the 78-CE frontier (so each job's 30
restarts cover all basins), iters cranked to 5000, distinct RNG. **All 6 jobs
floored at 78 — no 77.** Distinct-net census now **706 @ 78 CE** (+ 300 @ 79),
comparable in magnitude to n=24's 1035 @ 82 before it declared 82 the floor.
**Conclusion: 78 is the practical floor for n=23** (84 → 78, −6; 77 not found
across 3 waves / 18 job-runs / ~1.4M ILS iters). Next: benchmark the 78-CE pool
and pick the fastest by direct measurement.

### E2 — Wave 2: 6 jobs × 16 restarts × 3000 iters, reseeded from 78-CE. **DONE.**
Reseeded `/tmp/seeds_23.txt` from the 11 wave-1 78-CE nets; new seeds
{110017,220019,330023,440029,550037,660043}, ~6 min wall. **All 6 jobs floored at
78** — no 77 found. Distinct-net census now **370 @ 78 CE, 300 @ 79 CE**. Strong
(not yet n=24-strength) evidence 78 is the practical floor. Next: wave 3 (reseed
from the 370-net 78-CE frontier, more iters) as a final 77 attempt; if none,
benchmark the 78-CE pool.

### E1 — Wave 1: 6 jobs × 12 restarts × 2000 iters, header seeds. **DONE.**
Cores {0,1,3,6,8,10}, seeds {10007,20011,30013,40031,50033,60041}, ~3 min wall.
Floor **80 → 78 CE**: 5 jobs floored at 79, job seed 60041 reached **78** with 11
distinct 78-CE nets. 311 distinct nets total (counts: 78→11, 79→300). So 84→78
(**−6** already, matching n=24's depth of optimisation). Next: reseed wave 2 from
the 78-CE frontier to chase 77.

### E0 — Smoke test (1 job, 4 restarts × 300 iters, CPU0). **DONE.**
~11.7 s wall. Floor immediately **84 → 80 CE** (passed through 82, 81). Confirms
the committed net had ≥4 CE of pure slack and the search is fast here (~10
ms/iter avg, falling as the net shrinks). 17 distinct 80-CE nets pooled.
