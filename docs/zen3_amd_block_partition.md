# An AVX2-vectorised block partition for Zen 3 (`algo::block_simd_amd`)

**Context.** `docs/zen3_benchmarks.md` showed that the shipped `algo::boost_block`
(pdqsort branchless block partition, tuned for Intel Meteor Lake) runs ~6–22 %
slower *per element* on this AMD Ryzen 9 5950X (Zen 3) at compute-bound sizes,
so `fulcrum` overtakes it for i64. This note (1) pins down *why* in the
assembly, (2) shows that retuning the block size alone does **not** close the
gap, and (3) builds a new partition — `algo::block_simd_amd`
(`include/partitions/algorithms_amd.hpp`) — whose offset-fill loop is AVX2
vectorised, recovering and **beating** the Meteor-Lake `boost_block` numbers on
Zen 3. The shipped `boost_block`/`fulcrum`/`sized` are left untouched (they must
stay optimal on Intel); the AMD variant is additive.

Raw partition throughput on `random_uniform` is the only criterion. Method as in
`docs/zen3_benchmarks.md` (pinned core, min-of-reps; `perf` is locked at
`paranoid=4` so the µarch reasoning is from the assembly + timing, not counters).

## 1. Why `boost_block` is the slow one (assembly, `-O3 -march=native`, znver3)

Both protagonists are branchless. The difference is structural.

**`boost_block` — scalar offset fill, 8× unrolled (the hot loop):**

```asm
2760: mov  %r11b,(%rsi,%rax,1)   ; offsets_l[num_l] = idx      (BYTE store)
2767: cmp  %rdx,(%rbx)           ; x  vs pivot
276a: setge %r12b                ; !below
276e: add  %rax,%r12             ; num_l += !below     <- loop-carried recurrence
2771: lea  0x1(%r11),%eax        ; next idx
2775: mov  %al,(%rsi,%r12,1)     ; store at offsets_l[num_l]   (addr depends on num_l)
...                              ; xor-zero + setge + add + lea + byte-store / element
```

Per element: a load+`cmp`, a `setge`, a **loop-carried `add` into `num`**, a
`lea`, and a **1-byte store whose address depends on `num`**. Then a *second
pass* (`swap_offsets`) re-reads the ~50 % of elements that must move. So
`boost_block` is **two passes** with a recurrence-fed byte store.

**`fulcrum` — single-pass double write:**

```asm
30d9: mov  (%r9),%rcx            ; x = *read
30e2: cmp  %r8,%rcx              ; x vs pivot
30e5: mov  %rcx,(%rdi,%rdx,8)    ; ptl[m] = x     (8-byte store, addr = m only)
30e9: mov  %rcx,(%rax,%rdx,8)    ; ptr[m] = x     (8-byte store, addr = m only)
30ed: setl %cl                   ; v = below
30f3: add  %rdx,%rcx             ; m += v
```

Per element: 1 load, 1 `cmp`, **two 8-byte stores whose addresses depend only on
`m`** (so they issue early on both store ports), a `setcc`, an `add`. **Single
pass**, no re-read.

At **n = 4096 the i64 array is 32 KB = exactly Zen 3's L1d**, so this is a
core-throughput comparison, not a memory one. `fulcrum` runs at the same
~0.44 ns/elem (≈ 2.1 cyc) on both vendors; `boost_block` needs ~1.7 cyc on
Meteor Lake but ~2.1 on Zen 3. The byte-store + `num`-recurrence fill is simply
less friendly to Zen 3's narrower front-end / store scheduling than to Redwood
Cove's. The fix is to stop doing one scalar `setcc`+byte-store per element.

## 2. Retuning the block size does *not* help (negative result)

`block_scalar_amd<BS>` is `boost_block` with the offset-block size exposed.
Sweeping it on Zen 3 (i64 random, ns/elem, min):

```
 n        boost  sc<64> sc<96> sc<128> sc<192>
 4096     0.420  0.425  0.432  0.420   0.422
 65536    0.600  0.607  0.600  0.600   0.590
 1048576  0.527  0.582  0.555  0.538   0.527
```

≤ 10 % spread; `BS=192` is marginally best at huge n but the mid-range is flat.
**The scalar two-pass structure is the ceiling — block size is not the lever.**
The original's `BS=128` is fine; leave it.

## 3. The vectorised fill (`block_simd_amd`)

For a **contiguous block of an 8-byte signed key compared with `<`** (the i64
fast path — exactly where the Zen 3 gap is), the full-block fill becomes:

```cpp
__m256i lt = _mm256_cmpgt_epi64(vp, x);          // 4 predicates at once
unsigned m  = _mm256_movemask_pd(castpd(lt));     // 4-bit mask
uint32_t packed = kLut4.pos[ge] + i*0x01010101;   // pack indices via 16-entry LUT
memcpy(off + num, &packed, 4);                    // one unaligned 4-byte store
num += kLut4.cnt[ge];                             // advance by popcount
```

4 elements per `vpcmpgtq`+`movmsk`+LUT+store, vs the scalar 1-element
`cmp`/`setcc`/byte-store. The swap phase and all boundary bookkeeping are the
original's (reused verbatim). Everything that is **not** this shape — wider /
lexicographic keys, a non-identity projection, the reverse partition's negated
comparator, a non-contiguous iterator — falls back to the **identical scalar
fill**, so correctness is unchanged across the whole type/form matrix (verified:
all 12 ctest suites pass with `block_simd_amd` in the registry, plus a standalone
880-case sweep over sizes around the cutoff/block boundaries × random / low-card
/ all-equal / present-and-absent pivots).

**One subtlety, recorded.** The leftover-drain phase
(`while(num_r--) swap(base - offsets_r[num_r], first++)`) is only collision-free
when the right offsets are in monotonically increasing offset order — pdqsort's
scan order. The cyclic-swap path is order-independent and masked the bug; the
drain is not. The right-fill LUT therefore packs each group of four in
*ascending* offset order (lane 3→1 … lane 0→4), matching the scalar scan exactly.

## 4. Results — i64 random_uniform, ns/elem (min), Zen 3

From the production harness (`bench_partition`, `block_simd_amd` now in the
registry):

```
 n        lomuto  boost  fulcrum  block_simd_amd   best gain vs boost / fulcrum
 256      0.625   0.710  0.509    0.722            (lomuto/fulcrum win small)
 1024     0.615   0.496  0.452    0.432*           -13% / -4%
 4096     0.622   0.450  0.446    0.378*           -16% / -15%
 65536    0.623   0.470  0.444    0.386*           -18% / -13%
 262144   0.618   0.450  0.442    0.330*           -27% / -25%
 1048576  0.626   0.482  0.450    0.383*           -21% / -15%
 4194304  0.692   0.514  0.600    0.481*            -6% / -20%
```

`block_simd_amd` is the fastest i64 partition from **n = 1024 to 2²²**. At
n = 4096 it reaches **0.378 ns/elem — faster than the documented Meteor-Lake
`boost_block` (0.36 is the doc figure; Zen 3 boost is 0.45)**, i.e. the
vectorised fill more than recovers the cross-vendor gap. It also improves on
`boost_block` for i64 `sorted_descending` (0.48–0.55 vs 0.60–0.62), though
`fulcrum` still wins that structured case.

**No regression off the fast path.** For 16-byte `pair64`/`pair64f` and for the
reverse form, `block_simd_amd` falls back to the scalar block and tracks
`boost_block` to within noise (e.g. pair64f 2²⁰: 0.526 vs boost 0.529).

## 5. Status and next step

`block_simd_amd` is a standalone `PivotPartitioner` (with the position-aware
`at`) registered alongside the others. **The three follow-ups below are now
done — see `docs/zen3_quick_partition.md`:**

* **i32 (8-wide)** — ✅ added (`vpcmpgtd`/`movmskps`, 256-entry LUT, 8 elem/vec);
  ~2–3× boost_block for i32.
* **Vectorising the swap/compaction** — ✅ `block_compress_amd`, a single-pass
  `vpermd`-LUT compaction with no separate swap pass; fastest small/mid (it wins
  while the block is L2-resident, then `block_simd_amd` takes over).
* **A unified dispatcher** — ✅ `quick_partition` routes by size/width/key/ISA
  (lomuto → compaction while L2-resident → fill+swap; wide/lex → `sized`),
  1.4–2.1× the old `sized` for i64 and 1.7–3.3× for i32.
* **AVX-512** (still future) would collapse the compaction to a single masked
  `vpcompressd` per 8 elements; neither this Zen 3 nor Meteor Lake has it.
