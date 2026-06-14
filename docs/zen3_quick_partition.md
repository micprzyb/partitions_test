# `quick_partition`: a size/width/ISA-dispatched partition (Zen 3 + portable)

Builds on `docs/zen3_amd_block_partition.md`. That note vectorised the
*offset-fill* of the pdqsort block partition (`block_simd_amd`). This one adds
the two remaining pieces the task asked for and ties them together:

1. an **i32 8-wide** fast path (the narrowest cheap key — the best SIMD case);
2. a **vectorised swap pass** — realised as a single-pass **compaction**
   partition (`block_compress_amd`) that has no separate swap pass at all;
3. **`quick_partition`** — one `PivotPartitioner` that dispatches by array size,
   element width, key/comparator cost and ISA to the fastest of the above, and
   stays good on Intel and on future architectures.

All new code is additive (`algorithms_amd.hpp`, `quick_partition.hpp`); the
Intel-tuned `boost_block`/`fulcrum`/`sized` are untouched. Raw throughput on
`random_uniform` is the only criterion. Method: Zen 3 (Ryzen 9 5950X), GCC 16,
`-O3 -march=native`, pinned core, min-of-reps via the repo harness (setup/restore
untimed). All correctness is checked against `std::partition` (the 12 ctest
suites pass with the new partitioners in the registry, plus standalone sweeps of
14k+ cases over sizes around every cutoff/block boundary × random/low-card/
all-equal/present-and-absent pivots, for both i32 and i64).

## 1. i32 8-wide fill

The fill from the previous note generalises from 4-wide (i64: `vpcmpgtq` +
`movmskpd`, 16-entry LUT, 4-byte store) to 8-wide (i32: `vpcmpgtd` + `movmskps`,
256-entry LUT, 8-byte store) — twice the lanes per vector. It is the biggest
single-instruction win because i32 is the narrowest key:

```
 i32 random, block_simd_amd vs boost_block (ns/elem)
 n        256    1024   4096   16384  65536  2^20
 boost    0.74   0.49   0.38   0.40   0.45   0.45
 simd_amd 0.78   0.28   0.19   0.22   0.25   0.27     (-35..-55% from n=1024)
```

## 2. Vectorising the swap: a single-pass compaction partition

The block partition is inherently two-pass (fill an offset buffer, then swap).
Rather than vectorise the gather/scatter swap (AVX2 has gather but **no
scatter**), the swap is *eliminated*: `block_compress_amd` compacts and commits
in one pass — the vectorised analogue of the `fulcrum` double write.

For each loaded vector: `vpcmpgtd/q` → `movemask` → a **`vpermd` LUT** gathers the
`< pivot` lanes to the front and the `>=` lanes to the back → the compacted
vector is **stored at both the left and the right write cursor**, which advance by
`popcount` and its complement. The overwritten halves are reclaimed as the
cursors converge; the two end-vectors are preloaded so their slots are free, and
reads are kept ahead of writes by always refilling the side with less slack. No
offset buffer, no swap pass. (LUTs: 256 × `vpermd` controls for i32; 16 for i64,
expressed as int32 pairs so one `vpermd` permutes the 64-bit lanes.)

This is the fastest small/mid partition measured here:

```
 i64 random (ns/elem), bench_partition harness
 n          64    256   1024  4096  2^16   2^18   2^20
 boost     0.82  0.72  0.50  0.45  0.47   0.46   0.49
 simd_amd  0.83  0.73  0.43  0.37  0.40   0.34   0.39
 compress  0.32* 0.30* 0.28* 0.28* 0.28*  0.79   0.92    <- cliff past L2
```

## 3. The crossover is L2 residency (the portable rule)

`block_compress_amd` double-stores **every** element — ~2× the store traffic.
While the block is **L2-resident** that is free (store-port-bound, and it runs
the fewest instructions), so it wins. Once the block spills L2, the extra writes
hit the lower L3/DRAM bandwidth and the few-write `block_simd_amd` (fill + a swap
that touches only the ~half misplaced elements) wins. The cliff is sharp and
lands exactly at L2: i64 wins through **2¹⁶ = 512 KiB = Zen 3's L2** and cliffs
at 2¹⁸; i32 (half the bytes/elem) wins to ~2¹⁷.

So the dispatch threshold is simply **`kCompressMax = L2_bytes / sizeof(T)`** —
measured Zen-3 truth *and* a portable rule. On Meteor Lake (2 MiB P-core L2) the
same formula keeps compaction winning a **4× larger element range**; a future
build can read `kL2Bytes` from the target (cpuid / `-D`).

## 4. `quick_partition` — the dispatcher

```
 cheap-compare narrow int key (i32/i64, identity proj, `<`), contiguous, AVX2:
     n < 64                    -> lomuto_branchless   (no setup; ~0.6 ns/elem)
     n <= L2_bytes/sizeof(T)   -> block_compress_amd  (L2-resident compaction)
     else                      -> block_simd_amd      (fill + few-write swap)
 else (wide / lexicographic / projected key, or no AVX2):
     -> algo::sized            (the portable Intel-tuned Lomuto/pdqsort dispatch)
```

The four dispatch axes and why each matters:

* **element width** — the SIMD paths are gated to 4/8-byte integer keys (8/4
  lanes); wider elements get too little parallelism and use the block partition.
* **key / comparator cost** — a cheap fixed-pivot integer `<` is what makes the
  vector compare+permute win. An *expensive* comparator (pair64 lexicographic)
  wants the block partition's 8×-unrolled compare ILP, so it routes to `sized`.
  A small key inside a wide element (pair64f: 8-byte key, 16-byte element) also
  routes to `sized`/boost — its cheap compare already feeds that ILP, and a
  2-elements-per-vector compaction would barely vectorise.
* **array size** — the L2-residency crossover above.
* **ISA** — AVX2 is common to current Intel and AMD, so the SIMD paths help both;
  the capability/eligibility constants are the single extension point for future
  ISAs (AVX-512 `vpcompressd` would replace the permute-LUT compaction outright).

### Result: fastest or tied at every size, ~1.4–2.1× over the old dispatcher

```
 i64 random (ns/elem), bench_partition harness
 n              4     64    256   1024  4096  2^16  2^18  2^20  2^22
 sized        0.63  0.62  0.64  0.50  0.45  0.47  0.46  0.49  0.53
 quick        0.65  0.30  0.30  0.28  0.28  0.28  0.33  0.39  0.50
 quick uses:  lom   cmp   cmp   cmp   cmp   cmp   simd  simd  simd
 speedup       ~1x  2.1x  2.1x  1.8x  1.6x  1.7x  1.4x  1.3x  ~1x
```

For **i32** the win is larger (compaction is 8-wide): `quick` is ~2–3× the old
`sized` across the L2-resident range. For **pair64 / pair64f / keyed** `quick`
routes to `sized` and matches it (no regression — verified across the sweep).

### Honest caveats

* A narrow i64 band (16 Ki–32 Ki) where compaction still edges `simd_amd` ~20 %
  in one harness but loses to it at 4 Ki–8 Ki in another is left to whichever
  side the L2 rule picks; a single monotonic, cache-grounded threshold cannot
  capture a non-monotonic measurement, and both choices beat the old baseline.
* Thresholds and `kL2Bytes` are Zen-3 values. The *shape* (L2 rule, width gating)
  is portable; the constant should be re-pinned per target. I could not measure
  on Meteor Lake — the formula is the predicted-good default there, not a
  measured one.
* `perf` is locked (`paranoid=4`); µarch reasoning is from assembly + the timing
  curves, not hardware counters.

## Reproduce

```
cmake --build build -j
ctest --test-dir build -R 'partition_correctness|pivot_partition'   # correctness
build/benchmarks/bench_partition           # quick_partition vs all, every type/dist
```

Raw data: `docs/results/bench_partition_zen3_v2.csv`. The new code lives in
`include/partitions/algorithms_amd.hpp` (i32 fill, `block_compress_amd`) and
`include/partitions/quick_partition.hpp` (the dispatcher).
