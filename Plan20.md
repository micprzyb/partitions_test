# Plan20 — optimize the n=20 perfect halver (split@10)

Goal: reduce `nets::h20` (currently **63 CE**, depth 12) for raw throughput. Same
proven recipe as Plan21–24: diverse valid seeds → `search_halve_cegar2` plateau-walk
ILS + CEGAR/k-zero(C(20,10)=184,756) → floor pool → cycle-accurate benchmark select
→ exhaustive 2^20 verify fwd+rev → commit. Search pinned to P-cores {0,1,3,6,8,10};
benchmark on quiet CPU 4. **Document every attempt here before the next.**

Context: neighbors h19=59, **h20=63 (old, pre-optimization)**, h21=65, h22=68,
h23=74, h24=80. The 63 predates the recent rounds, so it is expected to be loose.

## Seed construction (line-removal from optimized neighbors)
`/tmp/mkseeds20.py` emits candidates; `/tmp/filt20.cpp` keeps valid distinct 20-halvers
(full 2^20 enum). Sources: current h20; h21 minus each wire; h22 minus each wire-pair.

**Result: line-removal alone already beats 63.** 7 valid distinct seeds:
sizes {60, 60, 61, 62, 62, 63, 63}. The two **60-CE** seeds come from h22 (the
freshly-optimized 68-CE net), the 61 from h21. So the practical floor starts at **60**
before any search — the old 63 was indeed loose. Seeds written to `/tmp/seeds_20.txt`.

## FINAL SUMMARY
`nets::h20`: **63 CE → 60 CE** ("p382", depth 13) — a 3-comparator (~5%) reduction.
The old 63-CE net was pruned from the size-optimal n=20 sorter and was loose; the
freshly-optimised n=22 (68 CE) line-removed straight to 60, and both line-removal
and pruned-optimal-sorter basins bottom out at 60 ⇒ **60 is the practical floor**
(59 never reached across W1+W2; same pattern as n=24→80). Selected from **988**
distinct 60-CE minima by benchmark: ns median-of-25 (p382 best sum + best pair64,
twice) then **cycle-accurate `perf`** confirmation. Verified fwd+rev exhaustive
2^20; full suite green (12/12).

Gains vs committed 63-CE base (cycle-accurate `optimizers/bench_cycles.sh 20`,
median of 7, cyc/element, frequency-invariant):
| type    | base (63 CE) | p382 (60 CE) | speedup |
|---------|-------------:|-------------:|---------|
| i64     | 3.281 | 3.132 | **−4.8%** |
| pair64  | 16.656 | 13.273 | **−25.5%** (IPC 3.82→4.32, 63.5→57.3 ins/el) |
| pair64f | 7.229 | 6.467 | **−11.8%** |

Reverse halver and `halve_reg<20>` inherit the gain (shared `nets::h20` array via
`cswap_rev`). Depth 12→13 — irrelevant in this throughput-bound regime.

Reproducibility: seeds `seeds/h20_60pool.txt` (988 verified 60-CE nets) +
`/tmp/seeds_20.txt` (line-removal + sorter seeds); search
`optimizers/search_halve_cegar2.cpp` (6 P-cores {0,1,3,6,8,10}); select via
`optimizers/bench_pool_par.sh` (ns shortlist) then `optimizers/bench_cycles.sh`
(perf cycles/IPC). p382 = `{13,15},{7,8},{16,17},{11,12},{18,19},{1,2},{16,18},{17,19},{15,18},{0,2},{5,6},{3,4},{9,10},{10,12},{13,14},{1,5},{4,6},{3,5},{14,17},{2,6},{1,16},{0,4},{7,11},{5,18},{0,14},{12,18},{3,15},{4,19},{5,14},{11,14},{13,16},{1,7},{4,11},{5,10},{2,17},{6,19},{9,16},{8,10},{12,14},{6,12},{11,16},{2,15},{4,9},{3,13},{0,5},{9,11},{6,17},{2,11},{10,15},{5,13},{6,8},{2,6},{10,16},{9,13},{7,13},{6,10},{8,11},{8,13},{6,13},{8,10}`

## Results log (newest first)

### D1 — hunt for a depth-12 60-CE net. **DONE — exists, but SLOWER; not committed.**
The plain search never found depth-12 (no depth pressure). New tool
`optimizers/search_halve_depth.cpp`: fixed size 60, minimize depth, **depth-aware
greedy-delete** (among valid single-removals take the min-resulting-depth one) +
depth-driven plateau walk, seeded from the 988 depth-13 nets. 6 P-core jobs —
**all 6 found a depth-12 60-CE halver** (so depth 12 at 60 CE is feasible and the
depth-aware greedy was the missing lever). 6 distinct, all re-verified valid; saved
to `seeds/h20_60_depth12.txt`.

**But depth-12 is slower.** Benched the 6 d12 nets vs committed p382 (d13):
- ns median-25: every d12 net loses on pair64 (2.99–3.20 vs **2.68**); sum 5.01–5.19
  vs p382 **4.67**.
- cycle-accurate `perf` (best d12 vs p382): i64 −0.9%, **pair64 +10.7% slower**,
  pair64f −2.4%. The d12 net has *fewer* instructions (55.8 vs 57.3 ins/el on
  pair64) yet costs more cycles because **IPC collapses 4.31→3.79** — the depth-
  packing crowds back-to-back dependencies and serialises the multi-cycle pair64
  compare-exchange (same mechanism as h24 p75; see `docs/h24_p75_vs_p1057.md`).

**Conclusion:** depth ≠ speed in this throughput regime. Keep the depth-13 **p382**.
A shallower net trades scheduling slack for dependency density and loses on the
dominant pair64 workload. (`search_halve_depth.cpp` retained — reusable for any
`<N> <floorS> <Dtarget>` depth-targeting.)

### W3 — select fastest 60-CE net. **DONE — committed p382 (60 CE, depth 13).**
988-net pool benched (`bench_pool_par.sh 20`, median 5): top by sum p218/p382/p217.
High-rep ns confirm (median 25, ×2): **p382 wins sum (4.624/4.630) AND pair64**
(p175's single-run p64f lead collapsed under medians — why medians are required).
Cycle-accurate `perf` (now re-enabled): p382 vs p218 — wins pair64 (the dominant
~13 cyc/el cost) +0.9%, ties i64, −1.1% p64f → p382. vs base: i64 −4.8%, pair64
−25.5%, p64f −11.8%. Committed to `nets::h20`; verify_small_halve{,_rev} N=20
PASS (2^20 fwd+rev); ctest 12/12.

### W3 — select fastest 60-CE net (pool benchmark). RUNNING.
988 distinct 60-CE candidates pooled across W1+W2. Building pool (base = committed
h20, 63 CE) and benchmarking; shortlist by ns sum, confirm top few cycle-accurate.

**All 988 saved to `seeds/h20_60pool.txt`** for further (deeper) search — each
independently re-verified valid via the bitset k-zero sweep (`/tmp/verify_pool20`:
988 nets, 0 invalid, 0 wrong-size). Format matches `seeds/h24_80pool.txt`
(one comma-joined comparator list per line); the search seed-parser reads it directly.

### W2 — fresh basin: prune the optimal n=20 sorters. **DONE — floor still 60 CE.**
Observation: the size-optimal n=20 sorter (91 CE) first layer == h20's first layer
minus (2,5) — **h20 was pruned from that sorter**, so it shares the W1 basin. The
**depth-optimal** sorter (93 CE, Green-style first layer (0,12),(1,13),…) is a
different basin. Added both sorters to `/tmp/seeds_20.txt`; 6 jobs
`/tmp/shb2 20 {201..206} 20 6000 80`. **All 6 converged to 60 CE again**, including
from both fresh sorter basins. Two independent basin families (h22 line-removal +
pruned optimal sorters) both bottom out at 60 ⇒ **60 is the practical floor** for
n=20 (same pattern as n=24→80; 59 not reached anywhere). 988 distinct 60-CE minima.

### W1 — first search wave from line-removal seeds. **DONE — floor = 60 CE.**
6 jobs `/tmp/shb2 20 {101..106} 16 5000 80`, pinned to P-cores, distinct RNG seeds,
seeds sorted smallest-first (60-CE h22-line-removal basin). **All 6 converged to
60 CE (depth 13); none reached 59.** Hundreds of distinct 60-CE minima pooled per
job (`/tmp/h20_*_pool.inc`). 60 is a strong plateau from the line-removal basin.
