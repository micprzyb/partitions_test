# Plan18 — optimize the n=18 perfect halver (split@9)

Goal: reduce `nets::h18` (currently **53 CE**, depth 11) for raw throughput. Same
recipe as Plan19–24: line-removal seeds from optimized neighbors →
`search_halve_cegar2` ILS + CEGAR/k-zero(C(18,9)=48,620) → floor pool → ns shortlist
+ cycle-accurate `perf` select → exhaustive 2^18 verify fwd+rev → commit.
P-cores {0,1,3,6,8,10}; bench on quiet CPU 4. **Document every attempt before next.**

Context: h17=48, **h18=53 (old)**, h19=56 (just 59→56), h20=60 (just 63→60). The
cascade: optimizing h19/h20 unlocks fresh line-removal seeds for h18.

## Seed construction (line-removal)
`/tmp/mkseeds.py 18` + `/tmp/filt.cpp` (full 2^18, dedupe, sort). Sources: current
h18; h19 minus each wire; h20 minus wire-pairs; h21 minus wire-triples (sampled).
**Result: 10 valid seeds, sizes {52×4, 53×2, 54×3, 55}.** Four **52-CE** (from the
fresh h19=56). Floor starts at 52 before search.

## FINAL SUMMARY
`nets::h18`: **53 CE → 52 CE** ("p173", depth 13) — a 1-comparator (~2%) reduction.
The old 53 was pruned from the size-optimal n=18 sorter; line-removal from the fresh
h19 (56) gave 52, and both line-removal and pruned-sorter basins bottom out at 52 ⇒
**52 is the practical floor** (51 never reached; same pattern as n=19→56, n=20→60).
Selected from **959** distinct 52-CE minima: ns median-25 (p173 best sum + pair64)
then cycle-accurate `perf`. Verified fwd+rev exhaustive 2^18; suite green (12/12).

Gains vs committed 53-CE base (cycle-accurate `optimizers/bench_cycles.sh 18`,
median 7, cyc/element):
| type    | base (53 CE) | p173 (52 CE) | speedup |
|---------|-------------:|-------------:|---------|
| i64     | 3.106 | 3.034 | **−2.4%** |
| pair64  | 14.641 | 13.402 | **−9.3%** |
| pair64f | 6.642 | 6.009 | **−10.5%** |

(ns median-25 had base marginally ahead on i64 — noise; cycle-accurate perf shows
p173 faster on all three.) Reverse halver and `halve_reg<18>` inherit the gain
(shared `nets::h18` via `cswap_rev`). Depth 11→13 — irrelevant in this regime.

Reproducibility: seeds `seeds/h18_52pool.txt` (959 verified 52-CE nets); search +
select as Plan19. p173 = `{12,14},{0,1},{6,7},{16,17},{1,2},{15,16},{12,13},{10,11},{14,17},{4,5},{8,9},{9,11},{6,10},{3,5},{1,4},{13,16},{12,15},{7,16},{2,5},{0,3},{4,17},{0,13},{4,13},{10,13},{1,14},{2,14},{4,9},{3,10},{5,17},{8,15},{7,9},{0,12},{10,15},{4,12},{1,6},{6,12},{9,15},{3,8},{8,10},{5,11},{5,13},{8,12},{2,10},{5,16},{5,7},{2,5},{9,14},{5,9},{7,10},{7,12},{5,12},{7,9}`

## Results log (newest first)

### W3 — select fastest 52-CE net. **DONE — committed p173 (52 CE, depth 13).**
959-net pool benched (median 5): p173 best sum + pair64. ns median-25: p173 sum
4.518 / pair64 2.692 (clear). Cycle-accurate `perf`: p173 vs p507 ≈ tie (p173 edges
pair64); vs base — i64 −2.4%, pair64 −9.3%, p64f −10.5%. Committed to `nets::h18`;
verify_small_halve{,_rev} N=18 PASS (2^18 fwd+rev); ctest 12/12.

### W3 — select fastest 52-CE net. RUNNING.
959 distinct 52-CE candidates pooled (W1+W2), saved to `seeds/h18_52pool.txt` (all
re-verified valid, 2^18). Benchmarking (base = committed h18, 53 CE).

### W2 — fresh basin: prune optimal n=18 sorters. **DONE — floor still 52 CE.**
Size-opt sorter (77 CE) shares h18's basin (h18 pruned from it). Depth-opt (78 CE)
different. Both prepended; 6 jobs `/tmp/shb2 18 {201..206} 20 7000 80`. **All 6
converged to 52 CE** ⇒ **52 is the practical floor** for n=18 (51 never reached).
959 distinct 52-CE minima.

### W1 — search wave from line-removal seeds. **DONE — floor = 52 CE.**
6 jobs `/tmp/shb2 18 {101..106} 16 6000 80`, seeds smallest-first (52-CE basin).
**All 6 converged to 52 CE** (depth 12–13); none reached 51.
