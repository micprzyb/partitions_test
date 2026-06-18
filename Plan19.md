# Plan19 — optimize the n=19 perfect halver (split@9)

Goal: reduce `nets::h19` (currently **59 CE**, depth 12) for raw throughput. Same
recipe as Plan20–24: diverse valid seeds (line-removal from optimized neighbors) →
`search_halve_cegar2` plateau-walk ILS + CEGAR/k-zero(C(19,9)=92,378) → floor pool →
ns shortlist + cycle-accurate `perf` select → exhaustive 2^19 verify fwd+rev →
commit. Search pinned to P-cores {0,1,3,6,8,10}; benchmark on quiet CPU 4.
**Document every attempt here before the next.**

Context: neighbors h18=53, **h19=59 (old)**, h20=60 (just optimized 63→60),
h21=65, h22=68. h20's big drop suggests h19 may also be loose; line-removal from the
fresh h20 (60) is the prime seed.

## Seed construction (line-removal from optimized neighbors)
`/tmp/mkseeds.py 19` (reads nets live from the header) + `/tmp/filt.cpp` (full 2^19
validity, dedupe, sort smallest-first). Sources: current h19; h20 minus each wire;
h21 minus wire-pairs; h22 minus wire-triples (sampled).

**Result: line-removal alone crushes 59.** 13 valid distinct seeds, sizes
{56×4, 57×2, 58×4, 59×3}. **Four 56-CE** seeds (from the freshly-optimized h20=60).
Practical floor starts at **56** before any search — the old 59 was very loose.

## FINAL SUMMARY
`nets::h19`: **59 CE → 56 CE** ("p123", depth 13) — a 3-comparator (~5%) reduction.
The old 59-CE net was pruned from the size-optimal n=19 sorter and was loose; the
freshly-optimised h20 (60 CE) line-removed straight to 56, and both line-removal and
pruned-optimal-sorter basins bottom out at 56 ⇒ **56 is the practical floor** (55
never reached; same pattern as n=20→60, n=24→80). Selected from **956** distinct
56-CE minima: ns median-25 (p123 best sum + best pair64) then cycle-accurate `perf`.
Verified fwd+rev exhaustive 2^19; full suite green (12/12).

Gains vs committed 59-CE base (cycle-accurate `optimizers/bench_cycles.sh 19`,
median 7, cyc/element, frequency-invariant):
| type    | base (59 CE) | p123 (56 CE) | speedup |
|---------|-------------:|-------------:|---------|
| i64     | 3.258 | 3.109 | **−4.8%** |
| pair64  | 16.493 | 13.217 | **−24.8%** (IPC 3.66→4.07, 60.4→53.8 ins/el) |
| pair64f | 6.967 | 6.290 | **−10.8%** |

Reverse halver and `halve_reg<19>` inherit the gain (shared `nets::h19` via
`cswap_rev`). Depth 12→13 — irrelevant in this throughput-bound regime (cf. Plan20
D1: a depth-12 variant would be slower).

Reproducibility: seeds `seeds/h19_56pool.txt` (956 verified 56-CE nets);
`/tmp/mkseeds.py 19` + `/tmp/filt.cpp` for line-removal seeds; search
`optimizers/search_halve_cegar2.cpp`; select via `bench_pool_par.sh` then
`bench_cycles.sh`. p123 = `{12,14},{10,11},{15,16},{17,18},{12,13},{15,17},{12,15},{16,18},{4,5},{8,9},{14,17},{0,1},{9,11},{2,3},{3,5},{13,16},{1,5},{0,3},{2,4},{6,7},{4,17},{0,13},{6,10},{2,14},{3,18},{4,13},{5,11},{7,16},{10,13},{4,9},{3,10},{13,16},{8,15},{5,18},{7,9},{1,14},{5,17},{10,15},{0,4},{3,8},{2,4},{8,10},{1,7},{5,13},{4,12},{8,12},{9,14},{1,6},{5,7},{7,10},{6,12},{9,15},{5,7},{5,9},{7,12},{7,9}`

## Results log (newest first)

### W3 — select fastest 56-CE net. **DONE — committed p123 (56 CE, depth 13).**
956-net pool benched (`bench_pool_par.sh 19`, median 5): p123 best sum + best
pair64. ns median-25 confirm: p123 sum 4.674 / pair64 2.712 (clear lead). Cycle-
accurate `perf`: p123 vs p623 — wins dominant pair64 +2.0%, ties rest; vs base —
i64 −4.8%, pair64 −24.8%, p64f −10.8%. Committed to `nets::h19`;
verify_small_halve{,_rev} N=19 PASS (2^19 fwd+rev); ctest 12/12.

### W3 — select fastest 56-CE net. RUNNING.
956 distinct 56-CE candidates pooled (W1+W2), saved to `seeds/h19_56pool.txt` (all
re-verified valid, full 2^19). Benchmarking (base = committed h19, 59 CE); ns
shortlist then cycle-accurate `perf` confirm.

### W2 — fresh basin: prune optimal n=19 sorters. **DONE — floor still 56 CE.**
Size-optimal sorter (85 CE) shares h19's basin (h19 pruned from it). Depth-optimal
(87 CE, Batcher-style) is different. Both prepended to seeds; 6 jobs
`/tmp/shb2 19 {201..206} 20 7000 80`. **All 6 converged to 56 CE again**, incl. from
both sorter basins ⇒ **56 is the practical floor** for n=19 (55 never reached; same
pattern as n=20→60, n=24→80). 956 distinct 56-CE minima.

### W1 — search wave from line-removal seeds. **DONE — floor = 56 CE.**
6 jobs `/tmp/shb2 19 {101..106} 16 6000 80`, seeds smallest-first (56-CE basin).
**All 6 converged to 56 CE (depth 13); none reached 55.** 56 is a strong plateau.
