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

---

# PLAN B — Fresh approach for 81 (2026-06-15)

After the n=23 win (78→74) proved the "floor" was Batcher-seed-bias, retry n=24→81
with the same fresh ideas. Tools reused: `optimizers/anneal_halve.cpp`
(violation-annealing) and diverse-basin seeding. The 906 distinct 82-CE nets from
the earlier effort are saved at `/home/mrp/halver24_frontier.txt` (restored to
`/tmp/seeds_24.txt`).

**Lead with the annealer (decisive):** the greedy/plateau search found ~2000
distinct 82-nets but could never *test* whether an 81-net exists, because it only
holds valid nets. The annealer fixes size=81 and minimises the violation count V
over the 906 seeds (each reduced 82→81); if any run hits V=0 we have an 81-halver;
if V sticks well above 0 across many strong restarts, that's strong evidence 82 is
optimal (exactly how it settled n=23's 73 as infeasible). Then, if needed, Track-1
diverse seeds (Parberry pairwise / bitonic pruned, n=23-extended) to escape the
Batcher basin.

## PLAN-B log (newest first)

### PB6 — Lock in the 80-CE win: benchmark + commit "p1057". **DONE.**
Benchmarked the full 1091-net 80-CE pool, median-of-9 confirm; winner **p1057**
(80 CE, depth 18): best sum and best pair64. Committed to `nets::h24` (82 → 80).
**Exhaustive 2²⁴ verify PASSED forward AND reverse.** Canonical
`bench_small_halve 24`, `taskset -c 4`, `halve_ns`, median of 3:

| type    | 82 CE (old) | 80 CE (p1057) | Δ |
|---------|-------------|---------------|------|
| i64     | 0.6904      | 0.7013        | **+1.6%** (small regression — all 80-CE nets share it; depth 18 hurts the latency-bound i64 path) |
| pair64  | 3.3010      | 3.0572        | **−7.4%** |
| pair64f | 1.5688      | 1.5471        | **−1.4%** |
| sum     | 5.560       | 5.306         | **−4.6%** |

Total throughput clearly better (driven by pair64); committed. Full ctest: **12/12 passed**.

### PB7 — VERIFY THE ANNEALER on a known-feasible target (81 from 82-seeds). **DONE — ANNEALER INVALIDATED.**
81 provably exists (plateau-walk found 160+ distinct 81-CE nets). The annealer
**failed**: stuck at **V≥283** (never 0) on this feasible target, and 5/6 jobs
logged *zero* restarts in ~80 min — trapped in the `polish()` steepest-descent
sweep, which is O(|net|·C(N,2)) ≈ 22k violation-evals (~45 s/sweep at n=24) and
fires on every reheat once V≤200. **Verdict: the annealer cannot find nets that
demonstrably exist → its "infeasible" verdicts for n=23/73 and n=24/79 are
meaningless** (false negatives). Drop it as a solver; the diverse-seed
plateau-walk is the only trustworthy tool. (Caveat: n=23's 74 floor still stands
on the plateau-walk evidence alone, which is solid; the annealer "confirmation"
there was illusory.)

### PB8 — Is an 81-CE net faster than the committed 80-CE p1057? **DONE — NO; 80-CE stays.**
Benchmarked all 559 distinct 81-CE nets vs p1057 (committed 80-CE); median-of-9 on
the top contenders:

| net   | CE | i64    | pair64 | pair64f | sum    |
|-------|----|--------|--------|---------|--------|
| **p1057** | 80 | 0.6738 | 3.0332 | 1.5145  | **5.2215** |
| p107  | 81 | 0.7099 | 3.1098 | 1.5215  | 5.3412 |
| p158  | 81 | 0.7097 | 3.2621 | 1.4841  | 5.4559 |

**p1057 (80 CE) is fastest on every axis** — i64, pair64, pair64f, and sum
(best 81-CE ~2.3% slower on sum). So fewer comparators *does* win here once the
right 80-CE net is chosen; the earlier "i64 regression" was largely single-run
noise (p1057 also has the best i64 under medians). **Commit stands: h24 = 80 CE.**

### PB9 — Fresh-basin seeds for the 79 hunt (user: "new construction seeds"). (in progress)
Generated + k-zero-validated diverse VALID split@12 24-halvers from non-h24/Batcher
constructions: **Bose–Nelson sorter (157 CE, valid)**, seedX (h23-extend, 86),
insertion sorter (276, valid but dropped — too slow to prune). Seed plateau-walk
from {Bose–Nelson, seedX} (pure new-basin escape attempt), deep waves. Also (user
follow-up): bring in **median/approximate-median networks** from
ehw-fit/approximate-medians — read the docs to parse the `.cha` format, convert a
median/selection net into a valid halver seed.

### PB10 — Median/approximate-median networks as seeds (user follow-up). **SEED OBTAINED.**
Read the docs + `mom.py` to decode the `.cha` format: SSA/dataflow netlist
`{inputs,1,ops,1,2,2,seed}([ida,idb]ina,inb,fn)...(output)` — each op consumes
wire-IDs ina,inb and emits two NEW IDs ida,idb (fn∈{1,2} = orientation), trailing
(output)=median wire. **De-SSA** (track wire-ID→position, emit in-place CE on the
two positions) verified on `s_9_1` (median lands at pos 4 for all 2⁹). Parsed the
exact-median **N=25** net (99 ops, median→pos 12); de-SSA'd, **line-removed wire 24**
(force +∞, drop its 5 comparators) → and the result is **already a VALID 94-CE
split@12 24-halver** (V=0, no repair needed — an exact median net inherently halves
around the median). A genuine **median-evolved basin** seed, saved at
`/home/mrp/median24_seed.txt`. (Built a C++ greedy repair-to-valid tool
`/tmp/repair_halve.cpp` in case repair were needed; here it wasn't.) Next: plateau-walk
from this seed (+ PB9 Bose–Nelson) to see if either descends below 80.

### PB11 — Optimal-sorting-network seeds (user: Bert Dobbelaere's page). **SEEDS READY.**
Fetched best-known sorters from bertdobbelaere.github.io. Findings:
- **N=24 size-optimal (120 CE)** = *identical* to the repo's `n24` → it's the 82
  basin the original search used; useless as a "new" seed.
- **N=24 depth-optimal (122 CE, 12 layers)** = a *different* network (validated
  24-sorter, valid split@12, ≠ n24) → genuine new basin. Saved
  `/home/mrp/sorter24d_seed.txt`.
- **N=25 size-optimal (130 CE)** → line-removed to 24 gives a valid 126-CE halver,
  BUT it has the same Batcher/odd-even opening as n24 → same basin → **EXCLUDED**
  (user: waste of time, ~= n24).

**New-basin seed set for the 79 hunt:** median-evolved (94), N=24 depth-optimal
sorter (122), seedX/h23-extend (86), Bose–Nelson (157, in PB9). Exclude n24 and
n25-derived (same basin).

### PB12 — Combined new-basin wave (median + depth-opt-sorter + seedX + p1057), hunt ≤79. **PAUSED (user break).**
PB9 (Bose–Nelson 157 + seedX) floored at **81** (worse than committed 80) — that
basin doesn't help. Launched the combined wave from {median-evolved 94, N24
depth-opt sorter 122, seedX 86, committed p1057 80} but **interrupted early** (1–5
restarts/job; the 94/122-CE seeds were still pruning, partial floors 82–87) — no
conclusion. **Committed h24 stays at 80 CE (verified, benchmark-fastest).**

**Resume state (durable; /tmp is wiped on reboot):**
- New-basin seeds: `/home/mrp/median24_seed.txt` (94), `/home/mrp/sorter24d_seed.txt`
  (122); full PB12 seed file `/home/mrp/pb12_seeds_24.txt` (med, depth-opt-sorter,
  seedX, p1057). seedX = h23(0..22) + {(11,i):i=12..23}; p1057 = committed h24.
- Tools: `optimizers/search_halve_cegar2.cpp` (`/tmp/shb2`), `/tmp/repair_halve.cpp`.
- Excluded basins (same as n24): N=24 size-optimal sorter (==n24), N=25-derived.
- Annealer = invalidated (don't use). Only the plateau-walk is trustworthy.

### PB13 — PB12 resumed (deeper 30×8000) + final reseeded wave (40×8000). **DONE — 80 is the floor.**
Resumed combined new-basin wave: all 6 jobs floored at 80 (540 distinct 80-CE
nets). Final deep wave reseeded from those 540 diverse-basin 80-nets (40×8000):
**held at 80, 1072 distinct 80-CE nets, no 79.** **Five+ independent basins**
(Batcher→82, n23-extend→80, Bose–Nelson→81, median-evolved→80, depth-opt-sorter→80)
all converge to ≥80. With the annealer invalidated (can't test feasibility), this
convergent plateau-walk evidence is the strongest available: **80 is the practical
floor for n=24.**

**To resume the 79 hunt:**
```bash
cd /home/mrp/partitions_test
g++ -O3 -march=native -std=c++23 optimizers/search_halve_cegar2.cpp -o /tmp/shb2
cp /home/mrp/pb12_seeds_24.txt /tmp/seeds_24.txt
rm -f /tmp/h24_*_pool.inc
CORES=(0 1 3 6 8 10); S=(7901 7903 7907 7909 7911 7913)
for i in 0 1 2 3 4 5; do taskset -c ${CORES[$i]} /tmp/shb2 24 ${S[$i]} 30 8000 90 \
  >/tmp/pb12_${S[$i]}.out 2>/tmp/pb12_${S[$i]}.log & done; wait
# then reseed from the frontier (see PB3-PB5 snippets) for deeper waves; census any sub-80.
```

## PLAN-B FINAL SUMMARY (n=24)
**`nets::h24`: 82 → 80 CE (p1057, committed), verified fwd+rev exhaustive 2²⁴,
ctest 12/12, benchmark-fastest (beats all 559 distinct 81-CE nets).** The original
82 "floor" was Batcher-seed-bias — the fresh idea (n=23-style) of seeding from a
structurally different basin (h23 extended to 24) cracked it: 82→81→80.

**79 hunt: exhausted; 80 is the practical floor.** Tried many fresh basins —
n23-extend, Bose–Nelson sorter, median-evolved (de-SSA'd exact-median N=25 netlist,
`.cha` format decoded), N=24 depth-optimal sorter — across ~13 reseeded deep waves;
**all converge to ≥80** (Bose–Nelson only 81; the rest 80), with >2600 distinct
80-CE nets seen and no 79. The annealer was **invalidated** (couldn't reach a
provably-existing 81 — false-negative engine), so feasibility of 79 can't be
machine-tested, but the convergent plateau-walk evidence from 5+ independent basins
strongly indicates 80 is the floor.

Speed gains vs the old 82-CE (canonical `bench_small_halve 24`, median of 3):
pair64 −7.4%, pair64f −1.4%, i64 ≈ flat; sum −4.6%.

Net deliverable: **n=24 82→80 (−2)**, beating the original 81 target.
Reusable lessons recorded: (1) always-valid-search floors are seed-relative — break
them with cross-construction seeds (neighbour halvers extended, optimal/median nets
de-SSA'd); (2) the violation-annealer as written is an unreliable solver (verify any
solver on a known-feasible target before trusting "infeasible").
After this: (i) **verify the annealer** on a known-feasible target (repair 82→81)
— the partial target-79 run stuck at V≈387, so the engine's "can't find" verdicts
need validating; (ii) **median/approximate-median network** seeds
(ehw-fit/approximate-medians, exact N=25) as a fresh basin (annealer-fed, since a
median net is an invalid halver start). Then resume the 79 hunt.

### PB5 — Deep wave 4: 6 jobs × 40 restarts × 7500 iters, reseeded from 80 frontier. **DONE.**
~50 min. **Floor held at 80** — 1091 distinct 80-CE nets, no 79. Plateau-walk
firmly stalled at 80. Next (deeper): annealer target=79 from the 1091-net frontier
(strong budget) to test 79 feasibility; plus a fresh-construction seed for a new
basin if the annealer can't reach V=0.

### PB4 — Deep wave 3: 6 jobs × 35 restarts × 7000 iters, reseeded from 80 frontier. **DONE.**
~40 min. **Floor held at 80** — 558 distinct 80-CE nets, no 79. Plateau-walk
stalled at 80 (82 → 80, −2). Going deeper per user steer: bigger wave 4 (≥40×7500)
from the broad 80 frontier, then annealer target=79 + a fresh construction seed if
needed. Next: deep wave 4.

### PB3 — PB wave 2: 6 jobs × 28 restarts × 5000 iters, reseeded from 81 frontier. **DONE.**
**Floor 81 → 80 CE** (job 810011). Census 80→80, 81→559, 82→316. So 82 → 80 (−2),
still cascading like n=23. Per user steer (n=24 is harder → explore deeper than
n=23): subsequent waves use a much larger budget (≥35×7000) and more waves before
concluding. Next: deep wave 3 from the 80 frontier, chase 79.

### PB2 — PB1 wave 1 result. **DONE — 81 FOUND! Target reached.**
6 jobs × 12 restarts × 6000 iters from {seedX 86, h24 82, n24 120}. **Floor 82 → 81**
(jobs 8101, 8113); 160 distinct 81-CE nets, 316 @ 82. So **82 was Batcher-seed-bias
too** — the non-Batcher n23-extended seed broke it on the *first* wave, exactly as
the user predicted (use the n=23-winning settings, not the annealer). The earlier
~2000-net plateau search and the annealer never escaped because every seed shared
the Batcher basin. Floor may go lower (n=23 went 78→75→74). Next: reseed wave 2
from the 81 frontier, chase 80.

### PB1 — Pivot to the n=23-winning recipe: diverse-seed `shb2` plateau-walk. **DONE.**
Correction: for n=23 the *successful* lever was the diverse-seed **plateau-walk**,
not the annealer (which only proved 73 infeasible). Mirror it exactly here — same
`shb2` settings (12×6000 then reseeded waves), seeded from a **structurally
different basin**: the new **h23 (74)** extended to a split@12 24-halver
(`h23(0..22)` then min of {11..22,23}→pos 11; **86 CE, validated**) — a non-Batcher
seed, analogous to n=23's seedA (which came from its neighbour h24). Seed file =
{seedX 86, Batcher h24 82, n24 sorter 120}. Annealer (target 81) result from the
earlier mis-led attempt: see PB-annealer note; superseded by this.

### PB0 — Setup. **DONE.** Restored 906-net 82 frontier → `/tmp/seeds_24.txt`;
`/tmp/anneal`, `/tmp/shb2` built; h24 confirmed 82 CE.

---

# Addendum — pushing for 81 comparators (2026-06-14)

**Goal:** beat the 82-CE floor; find an **81-CE** valid halver for n=24.
6-P-core box (Core Ultra 7 165H), search pinned to physical P-cores
{0,1,3,6,8,10}.

The original wave search (3 waves / ~90 jobs) found 1035 distinct 82-CE nets and
**no 81** — strong but unproven. The original ILS perturbs a single fixed `best`,
so it re-probes one net's neighbourhood; to find a doorway off the size-82
plateau you want to **walk the plateau widely**. New tool
`optimizers/search_halve_cegar2.cpp` (CEGAR + k-zero validity **copied verbatim**
— still gated by the exhaustive verifier) changes only the driver: sideways
acceptance to a different same-size net (p=0.35), adaptive kick strength
(escalate on stagnation), in-process reseed from a pooled net after 600 stagnant
steps, and a bounded witness set (8000, reservoir replace) so `quick_ok` stays
cheaper than a full sweep at high iteration counts.

## A-log (newest first)

### A3 — Final maximal wave. **POSTPONED (computer power-off) — to resume.**
Launched (6 jobs × 30 restarts × 3000 iters, reseeded from 30 of 906 frontier
basins) but **stopped early** at the user's request (power-off) after only 3–5 of
30 restarts per job. All jobs were at **82, no 81**; notably each job's in-process
pool had already grown to **1500–2300 distinct 82-CE nets** before being killed —
the plateau walk explores very broadly, yet 81 stays absent.

**State saved for resume (the `/tmp` pools are wiped on reboot):**
- 906 distinct 82-CE nets persisted to **`/home/mrp/halver24_frontier.txt`**
  (one net per line, durable / outside the repo).
- Tool `optimizers/search_halve_cegar2.cpp` (build: `g++ -O3 -march=native
  -std=c++23 ... -o /tmp/shb2`).

**To resume A3 after reboot:**
```bash
cd /home/mrp/partitions_test
g++ -O3 -march=native -std=c++23 optimizers/search_halve_cegar2.cpp -o /tmp/shb2
# reseed /tmp/seeds_24.txt from a shuffled subset of the saved frontier:
python3 - <<'PY'
import random
fr=[l.strip() for l in open('/home/mrp/halver24_frontier.txt') if l.strip()]
random.seed(303030); random.shuffle(fr)
open('/tmp/seeds_24.txt','w').write("\n".join(fr[:30])+"\n")
PY
CORES=(0 1 3 6 8 10); SEEDS=(70101 70203 70305 70407 70509 70611)
for i in 0 1 2 3 4 5; do
  taskset -c ${CORES[$i]} /tmp/shb2 24 ${SEEDS[$i]} 30 3000 80 \
    >/tmp/a3_${SEEDS[$i]}.out 2>/tmp/a3_${SEEDS[$i]}.log & done; wait
```
Then re-census the `/tmp/h24_*_pool.inc` files for any sub-82 (see A2 snippet).

### A2 — Plateau-walk wave 2: 6 jobs × 24 restarts × 2500 iters, reseeded from 24 frontier basins. **DONE.**
~15 min wall (360k iters). **All 6 jobs floored at 82 — no 81.** Distinct 82-CE
census now **906** (mine) — on top of the original effort's 1035. Next: one final
maximal-exploration wave (A3), then conclude.

### A1 — Plateau-walk wave 1: 6 jobs × 2 restarts × 20000 iters, header seeds. **DONE.**
Cores {0,1,3,6,8,10}, seeds {811..866}, ~10 min wall (240k iters total). **All 6
jobs floored at 82 — no 81.** Collected **475 distinct 82-CE nets**. Next: reseed
wave 2 from this diverse frontier (the plateau-walk gives many basins to start
from) with a bigger iteration budget.

### A0 — Build + smoke of the plateau-walk tool. **DONE.**
`/tmp/shb2`, n=24 smoke (2 restarts × 400 iters, CPU0, 12.4 s): valid, stayed at
82, and pooled **48 distinct 82-CE nets in 800 iters** — far broader plateau
coverage than the original (~20 in similar budget), i.e. many more doorways
tried per unit time. ~15 ms/iter. Next: big multi-core wave hunting 81.
