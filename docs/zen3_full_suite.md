# Full benchmark suite on Zen 3 vs the documented Meteor Lake numbers

Every `bench_*` binary rerun on the AMD Ryzen 9 5950X (Zen 3), GCC 16,
`-O3 -march=native` (→ znver3, AVX2/BMI2/FMA, no AVX-512), pinned to one core,
min-of-reps. The docs were all measured on an Intel Core Ultra 7 165H (Meteor
Lake). Raw CSV/text for each is in `docs/results/*_zen3.txt`. Environment caveats
(no frequency pinning, `perf` locked) are in `docs/zen3_benchmarks.md`.

**Headline: every documented design conclusion reproduces qualitatively.** The
only structural cross-vendor divergence is the one analysed in
`docs/zen3_benchmarks.md` + `docs/zen3_amd_block_partition.md`: `boost_block`'s
scalar byte-store offset-fill is ~6–22 % slower per element on Zen 3, which (a)
makes `fulcrum` overtake it for i64 mid/large-n, and (b) is fixed by the new
`block_simd_amd` (AVX2 fill), now the fastest i64 partition here.

## Coverage matrix

| benchmark | doc | documented finding | Zen 3 | Zen-3 delta |
|---|---|---|---|---|
| `bench_partition` | partition_schemes.md | branchless-Lomuto small / block large; SOTA split | ✓ | **boost_block rel. slow; fulcrum & new block_simd_amd win i64 1024–2²²** |
| `bench_partition_efficiency` | partition_efficiency.md | ninther best large-n pivot; mom O(n) trap; sort_mid/nin_lomuto small-n | ✓ | balance cols identical; crossover n≈24 (doc 21) |
| `bench_partition_methods` | algorithms.hpp interface study | by-value/comp-proj ≥ opaque predicate | ✓ | **stronger**: i64 2²⁰ value 0.68 vs pred 1.19 (doc 0.64 vs 0.71) |
| `bench_pivot` | (pivot quality) | balance per strategy; select cost | ✓ | balance arch-independent (matches); select-ns differs |
| `bench_pivot_total` | partition_efficiency.md | exchange pivot ≥ find pivot (worse) | ✓ | find ≤ swap confirmed (m3m3 0.46 vs 0.47 @4096) |
| `bench_large_n` | (pivot quality) | mom O(n) cold-expensive; ninther balance | ✓ | balance/worst_side match; cold select-ns differs |
| `bench_quickselect` | quickselect_dispatch.md | `sized` top-tier, no weak case | ✓ | **fulcrum beats boost/sized at i64 2¹⁸–2²⁰**; hoare faster on Zen 3 |
| `bench_quicksort` | pure_quicksort.md / adaptive_quicksort.md | design_adaptive best; halver leaf; ninther | ✓ | adaptive 16.4 ns/elem i64 2²², 3× std_sort |
| `bench_quicksort_lr` | quicksort_lr.md | qslr ≈ recursive qs (ratio≈1); std_sort 3× | ✓ | qslr/qs = 1.06 i64, 0.97 pair64f (doc <3%) |
| `bench_move_low_half` | move_low_half.md | `sample` wins; ref slowest | ✓ | sample 0.49/0.53/0.53 (doc 0.45/0.60/0.73) — faster at high p |
| `bench_halver_compare` | pivot_and_halver_report.md | adopt h7_alt, h8_new | ✓ | h7_alt −12 % i64; h8_new neutral i64, +3 % pair64 |
| `bench_pivot_tiers` | quicksort_pivot_tiers.md | T24 ≥ T16; m5m5>64k free; m3 sorted-unsafe | ✓ | **T24 helps more on Zen 3** (i64 −3 %); m3 sorted +55 % (doc +30 %) |
| `bench_reverse_partition` | reverse_partition_report.md | rev_rw≈rev_neg≈fwd; rev_view slower | ✓ | parity holds; **rev_view ~2× slower** (doc 5–30 %) |
| `bench_offset_partition` | offset_partition.md | prefix_fill/sized_off low-p, gap_off p-flat | ✓ | sized_off at/near per-cell best (i64 2²⁰ f.1 p.1: 0.383) |
| `bench_offset_low_half` | offset_low_half.md | approximate ≪ exact ≪ ref | ✓ | pivot_first/two_phase 0.35–0.37 vs exact 0.46 vs ref 5.6 |
| `bench_small_halve` | small_partition_design.md | halve faster than sort (ratio<1) | ✓ | halve/sort ≈ 0.7–0.85 for all N≤24 |
| `bench_small_merge` | (small merge study) | extend_sorted fastest | ✓¹ | extend_sorted 2.96 < scan 4.92 < bsearch 6.83 |

¹ `bench_small_merge`, `bench_quicksort` (2²⁴/2²⁶), `bench_reverse_partition`
(2²²) hit the per-binary timeout in the batch run and are truncated; the
completed cells (quoted above) cover the documented comparison sizes.

## Selected numeric comparisons

### bench_quickselect — i64, ns/elem (Zen 3 | Meteor Lake doc)

```
            2^16          2^18          2^20          2^22
 hoare    5.37 | 6.03   6.51 | 7.57   4.71 | 5.62   3.98 | 5.04   (Zen 3 faster: better BP)
 boost    1.00 | 0.83   1.40 | 1.23   1.21 | 1.07   0.98 | 0.89   (Zen 3 ~20% slower)
 fulcrum  1.09 | 1.12   1.23 | 1.27   1.14 | 1.20   1.33 | 1.38
 sized    1.00 | 0.83   1.41 | 1.23   1.21 | 1.06   0.98 | 0.91
```

Same shape as `bench_partition`: branchy `hoare` is *faster* on Zen 3 (its
branch predictor eats random data better), the `boost_block`-family is ~20 %
slower, and **`fulcrum` overtakes `boost`/`sized` at 2¹⁸–2²⁰** — the exact
inversion that motivated `block_simd_amd`.

### bench_quicksort_lr — random, ns/elem (qslr / recursive qs ratio)

```
 i64    2²²: qslr 18.22  qs 17.22  (1.06)   std_sort 50.23
 pair64 2²²: qslr 41.30  qs 39.11  (1.06)   std_sort 69.85
 pair64f 2²²: qslr 22.16 qs 22.81  (0.97)   std_sort 51.60
```

Within a few percent of the recursive quicksort and ~3× faster than `std::sort`,
as documented; `quicksort_lr` is faster on pair64f and on sorted pair64.

## Zen-3-specific deltas (collected)

These are second-order — the rankings are unchanged — but consistent and worth
recording for anyone tuning on this µarch:

1. **`boost_block` (pdqsort block) is ~6–22 % slower per element** at
   compute-bound sizes; root cause + fix in `docs/zen3_amd_block_partition.md`.
2. **Branchy schemes (`hoare`, `std_partition`, `std::sort`) are relatively
   faster on Zen 3** for random data — its branch predictor handles the ~50 %
   mispredict partition branch better than Meteor Lake. They are still far behind
   the branchless winners; the gap is just smaller.
3. **The opaque-predicate interface penalty is larger on Zen 3** (`pred` 1.19 vs
   `value` 0.68 at i64 2²⁰ = +75 %, vs the doc's +11 %) — more reason to keep the
   by-value comp/proj primitive.
4. **`rev_view` (reverse-iterator partition) is ~2× slower** (doc 5–30 %); the
   backward pointer walk hurts Zen 3's prefetch more.
5. **The halver cutoff T24 helps i64/pair64 more on Zen 3** (−3 % vs doc ~flat);
   **m3-on-sorted is more punishing** (+55 % vs +30 %).
6. **i64 at 2²² degrades less** for the streaming partitioners — a 32 MB array
   fits Zen 3's 32 MB per-CCD L3 but spills Meteor Lake's 24 MB.

## Reproduce

`bash` the runner used here (pinned, bounded): each binary is
`taskset -c <core> build/benchmarks/<name>` with the default (full) args; the
three long ones above want a higher timeout or a size cap (`<name> <maxsize>`).
Raw outputs: `docs/results/<name>_zen3.txt`.
