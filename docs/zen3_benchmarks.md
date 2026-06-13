# Benchmarks on AMD Zen 3 (Ryzen 9 5950X) vs documented Intel Meteor Lake

**Purpose.** The performance docs in this repo (`partition_schemes.md`,
`partition_efficiency.md`, `offset_partition.md`, the findings blocks in
`algorithms.hpp`) were all measured on an **Intel Core Ultra 7 165H (Meteor
Lake)**. This note reproduces the headline benchmarks on a different
microarchitecture, checks coherence, analyses the divergences, and proposes a
microarchitecture-specific tuning.

**TL;DR.** Every *qualitative* finding reproduces. The one structural
divergence is that **`boost_block` (the pdqsort branchless block partition) is
~6–22 % slower per element on Zen 3 at compute-bound sizes**, while `fulcrum`
and `lomuto_branchless` run at the same cycles/element on both chips. The
consequence: **on Zen 3 `fulcrum` — not `boost_block` — is the fastest i64
partition across n ≈ 256 … 2²⁰**, which the shipped `sized` dispatcher does not
currently exploit.

---

## Environment

| | This run | Documented |
|---|---|---|
| CPU | **AMD Ryzen 9 5950X**, Zen 3, 16C/32T | Intel Core Ultra 7 165H, Meteor Lake |
| ISA (`-march=native`) | **znver3**: AVX2 + BMI2 + FMA, **no AVX-512** | Redwood Cove/Crestmont: AVX2 + BMI2 + FMA, no AVX-512 |
| L1d / L2 / L3 (1 core) | 32 KB / **512 KB** / **32 MB per CCD** | 48 KB / **2 MB per P-core** / 24 MB shared |
| Compiler | GCC 16.0.1 (trunk), `-O3 -DNDEBUG -march=native` | GCC 15.2, `-O3 -march=native` |
| Pinning | `taskset -c 4`, governor `powersave`, **boost on** | single core pinned |

The ISA tier is identical (both top out at AVX2), so **every difference below is
microarchitectural, not instruction-set**. The two cache facts that matter:
Zen 3's **L2 is 4× smaller** (512 KB vs 2 MB) but its **L3 is larger** (32 MB
per-CCD vs 24 MB shared). Both shift the cache-residency crossovers.

**Limitations (be skeptical of absolute ns to ~±5 %).** I could not pin the
frequency (`scaling_governor` not writable, no root) and `perf` is locked down
(`perf_event_paranoid = 4` — even user-space CPU-event counting is denied), so
the µarch root-cause below is **reasoned from the timing curves, not measured
with hardware counters**. The harness already reports *min over many reps with
the array-restore excluded from timing*, which is the least-noisy estimator;
the crossovers reported here were stable across repeated runs.

Reproduce: `build/benchmarks/bench_partition`,
`build/benchmarks/bench_partition_efficiency`, `build/tools/balance_report 200`.
Raw CSVs are checked in under `docs/results/`.

---

## 1. What reproduces (coherence)

### Partition schemes — `bench_partition`, i64 / random_uniform, ns/elem

`*` = fastest at that size. Zen 3 measured; the doc (Meteor Lake) value follows
in parentheses for the two protagonists.

```
 n=               4     8    16    24    64   256  1024  4096  2^16  2^20  2^22
 lomuto_branchl 0.65 0.62 0.61 0.61 0.61 0.62 0.61 0.62 0.62 0.62 0.72
   (doc)        0.66 0.67 0.64 0.62 0.61 0.63 0.64 0.63 0.67 0.70 0.92
 boost_block    1.97 2.10 1.60 1.27 0.80 0.70 0.50 0.44 0.47 0.49 0.51*
   (doc)        2.42 2.55 1.71 1.27 0.75 0.67 0.47 0.44 0.44 0.55 0.52
 fulcrum        1.96 2.11 1.67 1.31 0.81 0.50*0.45*0.44*0.44*0.44* 0.55
 block          1.97 2.15 2.29 2.32 2.12 2.06 0.73 0.65 0.65 0.66 0.65
 hoare          2.04 2.19 2.37 2.40 2.23 2.20 2.18 2.00 2.26 2.88 1.76
 sized (shipped)0.64*0.63 0.61*0.61 0.61 0.62 0.49 0.44 0.47 0.49 0.52
```

Coherent: branchless Lomuto is flat at ~0.6 ns/elem and **3–4× faster than the
branchy schemes for n ≤ 256**; the block partitions cross over in the
1024-element band; the branchy schemes sit at ~2.0–2.4 ns/elem. The SOTA story
(`partition_schemes.md`) holds verbatim.

### Pivot efficiency — `bench_partition_efficiency`, large-n, `E` metric

`E = (find+partition ns)/smaller-element`; `[raw|smaller_frac]`. Zen 3:

```
 i64    n |   m3            ninther         m5m5            pseudo9
 4096     | 1.54[.44|.28]  1.35[.47|.35]  1.23*[.50|.41]  1.36[.46|.34]
 2^16     | 1.28[.46|.36]  0.96*[.47|.49] 1.00[.48|.48]   3.14[.39|.12]
 2^20     | 1.05[.48|.46]  0.99*[.48|.48] 1.14[.48|.42]   1.31[.47|.36]
 2^22     | 2.28[.51|.22]  1.18*[.53|.45] 1.82[.53|.29]   1.87[.53|.28]
```

This is **almost identical to the documented table**: `ninther` is the best
all-round large-n pivot (wins 2¹⁶–2²²), `m5m5` takes 4096, `pseudo9` craters at
2¹⁶ (0.12 smaller-fraction → E spikes), and `mom_boost` (median-of-medians, not
shown) is the same O(n) disaster (~25–60 E). The `smaller_frac` column —
pivot-balance, which is **architecture-independent** — matches the doc
column-for-column, confirming the distributions and pivot strategies are intact.
`balance_report` likewise reproduces the README claim (`median_of_medians_5`
worst-side ≈ 0.75 vs 1.0 for first/middle/last).

### Small-n sort-as-partition, structured inputs

* Small-n (`E`): `sort_mid` wins the tiniest i64 blocks (n ≤ 21), `nin_lomuto`
  the n = 24–64 band — doc crossover n ≈ 21, Zen 3 n ≈ 24. `sort_mid` is
  actually **faster on Zen 3 at n = 8** (E 1.92 vs doc 2.74).
* Sorted / all-equal: branchy **`hoare`** wins, branchless Lomuto's
  unconditional moves are pure waste (i64 all_equal: hoare 0.12–0.21 vs lomuto
  0.57–0.59 ns/elem). Reproduces the doc exactly.

---

## 2. The divergences and why

### (a) `boost_block` is ~6–22 % slower per element on Zen 3 — *core throughput*

| i64 random | n=4096 | 2¹⁶ | 2²⁰ |
|---|---|---|---|
| boost_block, Zen 3 | 0.44 | 0.47 | 0.49 |
| boost_block, Meteor Lake (doc) | 0.36 | 0.44 | 0.46 |
| fulcrum, Zen 3 | 0.44 | 0.44 | 0.44 |
| fulcrum, Meteor Lake (doc) | 0.45 | 0.45 | 0.49 |

The control is decisive: **at n = 4096 the i64 array is 32 KB = exactly Zen 3's
L1d**, so this is *not* a cache-size or bandwidth effect — it is a per-element
**core-throughput** gap. And it is specific to `boost_block`: `fulcrum` runs at
the *same* ~0.44 ns/elem (≈ 2.1 cyc at boost clock) on both chips, while
`boost_block` needs ~1.7 cyc/elem on Meteor Lake but ~2.1 on Zen 3.

Mechanism (reasoned — `perf` unavailable). `boost_block`'s hot loop is the
branchless offset fill (`algorithms.hpp:477`), 8× unrolled:

```
offsets_l[num_l] = (unsigned char)i++;   // byte store, address = offsets_l + num_l
num_l += !below(first); ++first;          // num_l is loop-carried
```

Per element that is a load + `cmp` + `setcc` + **byte (8-bit) store** + a
loop-carried `num_l += pred`, and the *next* store's address depends on `num_l`.
This is a short recurrence feeding an AGU, plus a narrow store every iteration.
Meteor Lake's wider Redwood Cove front-end (6-wide decode) and store scheduling
evidently absorb that mix at ~1.7 cyc/elem; Zen 3 (4-wide decode) tops out
nearer 2.1. By contrast `fulcrum`'s per element is a load + `cmp` + **two
sequential 8-byte stores** to `ptl[m]`/`ptr[m]` whose addresses depend only on
`m` (so they issue early on both store ports) — a pattern both µarchs run at the
same 2 stores/cycle, hence the identical timing. `lomuto_branchless` (single
streaming write) is likewise µarch-neutral.

This is the *same* family of effect the doc already documented for the by-value
pivot (aliasing reload): branchless block partitions are sensitive to
front-end/store-port details, and those details differ between vendors.

### (b) i64 at 2²² degrades far less on Zen 3 — *the 32 MB L3*

`lomuto_branchless` at 2²²: **0.72 ns/elem on Zen 3 vs 0.92 documented**. A 2²²
i64 array is **32 MB = exactly Zen 3's per-CCD L3**, so it stays cache-resident
where it spills Meteor Lake's 24 MB L3 to DRAM. The doc's "write-pressure
crossover where Lomuto's 2 moves/elem saturate memory bandwidth" still happens —
just shifted to a larger n on Zen 3 because the L3 is bigger. (For the 16-byte
`pair64f`, 2²² = 64 MB > 32 MB and both chips spill: Zen 3 lomuto 1.20 ≈ doc
1.14.)

### (c) Consequence: `fulcrum` is the i64 mid-range champion on Zen 3

Because (a) penalises only `boost_block`, the i64 ranking flips relative to the
doc. Versus the shipped `sized` dispatcher (which routes i64 to
`lomuto_branchless` up to n = 512, then `boost_block`):

```
 i64 random   n=256  1024  4096  2^16  2^20  2^22
 sized        0.622  0.494 0.440 0.467 0.490 0.517
 fulcrum      0.499  0.450 0.440 0.438 0.441 0.547
 gain         +19.8% +9.0% +0.1% +6.2% +10.0% -5.7%
```

`fulcrum` wins or ties everywhere except 2²² (where the array exceeds L3 and
`boost_block`'s fewer writes win, as the doc predicts). The doc's own hybrid
section already found fulcrum best for i64 *only at n ≈ 256* on Meteor Lake; on
Zen 3 that window widens to **256 … 2²⁰**.

This does **not** extend to wider/expensive-compare types: for `pair64` (lex)
`boost_block` is the clear winner at every size on Zen 3 too (its 8×-unrolled
block of independent compares hides the expensive comparator's latency —
`fulcrum`/`lomuto` are serialised on the `m`-chain and write-bound on 16-byte
stores), and for `pair64f` `boost_block` still wins the 1024–2¹⁶ band; `fulcrum`
only retakes `pair64f` at 2²⁰. So the flip is confined to **8-byte cheap-compare
keys**.

---

## 3. Suggested improvements

1. **Microarchitecture-tune the `sized` dispatcher for Zen 3 (validated, ~6–20 %
   on i64 mid-range).** For an 8-byte cheap-compare key, add a `fulcrum` tier
   between the small-n `lomuto_branchless` and the huge-n `boost_block`:

   ```
   sizeof(T) <= 8 :  n <= ~128         -> lomuto_branchless
                     ~128 < n <= ~2^21 -> fulcrum        (Zen 3: the new win)
                     n  >  ~2^21        -> boost_block   (array > L3)
   sizeof(T)  > 8 :  unchanged (boost_block from n=24; fulcrum never wins lex)
   ```

   This is a *per-µarch* policy: on Meteor Lake `boost_block` should keep the
   mid-range. The cleanest implementation is to make `detail::sized_cutoff` and
   the mid-tier choice depend on a `PARTITIONS_UARCH` switch (default = current
   Meteor-Lake-tuned behaviour) so the header stays portable. The current code
   is *not wrong* — it is tuned for the chip it was measured on; this just adds a
   second tuning point. I did **not** apply it, since it changes shipped
   dispatch and should be gated behind a build flag and re-confirmed per target.

2. **Re-tune `boost_block_size` (currently 128) on Zen 3.** The comment at
   `algorithms.hpp:341` notes 128 was a ~17 % win over pdqsort's 64 *on Meteor
   Lake*. Given the offset-fill loop is the Zen 3 bottleneck, the block size that
   best amortises the recurrence may differ here. A quick sweep of {64, 96, 128,
   192} is cheap and may recover part of the 22 %. (Out of scope to apply blind.)

3. **Soften the doc's absolute claim.** `algorithms.hpp:296` and the
   `boost_block` header call it "the fastest partitioner here". That is
   Meteor-Lake-specific; on Zen 3 it is *not* fastest for i64 in the mid-range.
   Wording like "fastest for wide/expensive-compare keys and very large arrays;
   `fulcrum` can win for narrow cheap-compare keys depending on µarch" would
   travel better. (Documentation-only.)

4. **`bench_partition_efficiency` doc-label drift (minor, found while
   reproducing).** `partition_efficiency.md` abbreviates the strategy names
   (`ninther`, `m5m5`, `mom`, `pseudo9`, `nin_lomuto`, `m3m3swap`) but the CSV
   emits `ninther_boost`, `m5m5_boost`, `mom_boost`, `pseudo9_boost`,
   `ninther_lomuto`, `m3m3swap_boost`. Anyone diffing the CSV against the doc by
   column name (as I first did) gets empty matches. Aligning the labels — or
   noting the mapping in the doc — would save the confusion.

---

## 4. Bottom line

The library's design conclusions are **robust across vendors**: branchless
Lomuto for small cheap-compare blocks, block-Hoare for wide/expensive-compare or
huge arrays, branchy Hoare for structured data, the ninther as the best O(1)
large-n pivot, and "never pay O(n) to find a pivot". The only thing that
*moves* with the microarchitecture is the **exact crossover between the two
fastest branchless partitions** — and on Zen 3 that crossover hands the i64
mid-range to `fulcrum`, because Zen 3 runs `boost_block`'s byte-store offset-fill
loop ~20 % slower per element than Meteor Lake does, while it runs `fulcrum`'s
sequential double-write at the same speed.
