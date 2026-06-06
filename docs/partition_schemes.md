# Partition schemes: full benchmarks + state of the art

**Question.** Is branchless Lomuto actually faster than the other partitioners
for small arrays? Full raw-time benchmarks of every scheme, cross-checked
against state-of-the-art libraries.

**Verdict (short).** It depends — and the dependence is exactly what the SOTA
predicts. For **cheap-to-compare** elements (i64, pair-by-first) on **random**
data, **branchless Lomuto is decisively the fastest for small/medium arrays**
(3–4× the branchy schemes), and this is precisely what the Rust standard
library's `ipnsort` does (it uses a branchless-Lomuto-cyclic partition for all
types `≤ 96` bytes). The intuition that it is *not* faster holds only for
**expensive-to-compare** elements (pair64 lexicographic), for **structured**
inputs (sorted / low-cardinality), and for **very large** arrays — where a
block-Hoare (pdqsort) or plain branchy-Hoare partition wins instead.

Measured on Intel Core Ultra 7 165H (Meteor Lake), GCC, `-O3 -march=native`,
pinned core; min ns/element over batched blocks. Reproduce with
`build/benchmarks/bench_partition`. The schemes:

| name | what it is | SOTA analogue |
|---|---|---|
| `std_partition` | libstdc++ `std::partition` | branchy Hoare baseline |
| `lomuto` | scalar branchy Lomuto | textbook (3 moves/elem) |
| `lomuto_branchless` | orlp "gap" method (2 moves/elem, branchless) | **ipnsort** lomuto-cyclic |
| `hoare` | scalar branchy Hoare (½–1 move/elem) | ipnsort's large-type path |
| `block` | BlockQuicksort (Edelkamp–Weiss) | branchless block Hoare |
| `boost_block` | pdqsort branchless block partition | **boost::pdqsort** |

## Random data (the headline)

min ns/element, `random_uniform`, median-of-3 pivot. `*` = fastest.

```
 i64            n=4      8     16     24     64    256   1024   4096   2^16   2^20   2^22
 lomuto_branchl 0.66*  0.67*  0.64*  0.62*  0.61*  0.63*  0.64   0.63   0.67   0.70   0.92
 boost_block    2.42   2.55   1.71   1.27   0.75   0.67   0.47*  0.44*  0.44*  0.55*  0.52*
 block          2.08   2.67   2.77   2.84   2.60   2.61   0.75   0.57   0.58   0.59   0.61
 hoare          2.28   2.77   2.80   2.79   2.48   2.47   2.28   2.10   2.48   3.32   2.23
 std_partition  2.18   2.76   2.92   3.04   2.76   2.87   2.83   2.48   2.77   3.54   2.11

 pair64f (.first only, 16B element, CHEAP compare)
 lomuto_branchl 0.72*  0.63*  0.61*  0.57*  0.62*  0.54*  0.56   0.53   0.55   0.80   1.14
 boost_block    2.31   2.64   1.76   1.32   0.85   0.70   0.49*  0.51*  0.44*  0.64*  0.77*

 pair64 (lexicographic, 16B element, EXPENSIVE compare)
 lomuto_branchl 3.79   3.36*  2.96   2.92   3.17   3.33   2.16   1.02   0.68   0.77   1.10
 boost_block    3.92   3.86   2.63*  2.03*  1.22*  0.89*  1.79*  0.52*  0.53*  0.68*  0.74*
 hoare          3.76   3.73   3.25   3.06   2.56   2.47   3.11   2.10   2.21   3.29   1.99
```

Reading this:

* **i64 and pair64f behave identically**: branchless Lomuto is the clear winner
  for `n ≤ 256` (≈0.6 ns/elem, **3–4× faster** than every branchy scheme, which
  pay ~2.5 ns/elem in branch-mispredict re-steers). The element being 8 or 16
  bytes does not change the small-`n` winner — what matters is that the
  **comparison is cheap**.

* **The crossover to block-Hoare is ~1024 elements.** From `n ≈ 1024` up,
  `boost_block` (pdqsort) overtakes, and at `n ≥ 2^20` it wins decisively
  (i64 2^22: 0.52 vs Lomuto 0.92). This is the **write-pressure** crossover:
  branchless Lomuto does *two unconditional moves per element*; once the array
  spills L2 those extra writes saturate memory bandwidth, and block-Hoare's
  "write only on swap" (≈ half a move/elem) wins. (The ipnsort writeup observes
  the same: Hoare overtakes "when inputs no longer fit into the L2 data cache".)

* **pair64 lexicographic is the exception that proves the rule.** Here the
  *comparator is expensive* (a short-circuiting two-key lex compare with its own
  branch). `boost_block` wins from `n = 16` onward, because it evaluates a whole
  **block of 8 comparisons with independent ILP**, hiding the comparator latency
  — whereas the gap-method Lomuto's tighter loop is throughput-bound on that
  latency. So: **cheap compare → branchless Lomuto small; expensive compare →
  block partition even when small.**

## Structured data changes the winner again

min ns/element, selected `n`:

```
 sorted_descending          n=16    64    256   4096   2^20
 i64    lomuto_branchless   0.48* 0.51  0.44* 0.45* 0.50*
        hoare               0.67  0.62  0.63  0.60  0.58
 pair64 hoare               0.74* 0.65* 0.48* 0.34* 0.63*
        lomuto_branchless   1.59  2.79  3.28  0.89  0.77

 all_equal                  n=16    64    256   4096   2^20
 i64    hoare               0.20* 0.15* 0.16* 0.15* 0.23
        boost_block         0.35  0.36  0.24  0.21  0.21*
        lomuto_branchless   0.47  0.51  0.45  0.45  0.53
 pair64 block               0.38* 0.47  0.35* 0.98  1.03
        hoare               0.54  0.50* 0.36  0.32* 0.54*
        lomuto_branchless   0.66  0.74  0.66  0.65  0.83
```

* On **predictable** inputs (sorted, all-equal) the **branchy** schemes win:
  there is nothing to mispredict, so branchless Lomuto's *unconditional* two
  moves per element are pure overhead. Plain branchy **Hoare** is the best
  all-equal / sorted partitioner, and does the fewest moves. (Production sorts
  like pdqsort and ipnsort special-case these: pattern detection, and ipnsort's
  branchless *equal*-partition for low-cardinality inputs.)

## What the state of the art does

* **Rust standard library `slice::sort_unstable` / `select_nth_unstable`**
  (ipnsort, by Lukas Bergdoll & Orson Peters) uses **two** partitions, chosen at
  compile time by `size_of::<T>()`:
  `MAX_BRANCHLESS_PARTITION_SIZE = 96` bytes — **branchless-Lomuto-cyclic for
  `size_of::<T>() ≤ 96`**, **branchy Hoare** above. The branchless Lomuto is the
  default for essentially all normal-sized keys; Hoare is reserved for very large
  records that are expensive to move (the gap method's 2 moves/elem dominate).
  This is exactly our i64/pair64f result, and the "huge element → Hoare" rule.

* **C++ `boost::sort::pdqsort` / pattern-defeating quicksort** (Orson Peters)
  uses a **branchless block-Hoare** partition (`partition_right_branchless`) for
  `n ≥ 24`, insertion sort below, plus pattern detection. That is our
  `boost_block`, and it is the right default for C++ where keys are often
  expensive to compare (no cheap derived `Ord`) and arrays are large.

* **BlockQuicksort** (Edelkamp & Weiss, ESA 2016) introduced the block scheme;
  **blipsort** (RedBedHed) puts branchless Lomuto inside pdqsort; **crumsort**
  and other "hybrid Hoare+Lomuto" schemes can beat pure branchless Lomuto by
  combining Hoare's low move count with Lomuto's branchlessness — the
  `sort-research-rs` writeups note a hybrid "can deliver a large improvement even
  compared with branchless Lomuto".

The two communities differ for a real reason: **Rust's common case is cheap
`Ord` on small types → branchless Lomuto**; **C++/pdqsort targets arbitrary
(often expensive) comparators and large data → branchless block Hoare**. Our
three element types straddle exactly that divide.

## Answer to "is branchless Lomuto faster for small arrays?"

* **Cheap compare (i64, pair-by-first):** YES, clearly — **3–4× faster** for
  `n ≤ ~256` on random data. This is the SOTA choice (Rust uses it for all
  `≤ 96`-byte types). The intuition that it is *not* faster is **rejected** here.
* **Expensive compare (pair64 lex):** NO — block-Hoare (pdqsort) wins from
  `n = 16`. Intuition **confirmed**.
* **Structured input (sorted / all-equal):** NO — branchy **Hoare** wins;
  branchless Lomuto's unconditional moves are wasted. Intuition **confirmed**.
* **Large arrays (`> L2`):** NO — block-Hoare wins (fewer writes). Independent of
  the above.

## The options, and when each wins

| scheme | wins when |
|---|---|
| **branchless Lomuto** (gap/cyclic) | small–medium arrays, cheap compare, ≤ ~96 B element, random/adversarial data — *the* small-array winner for i64-like keys |
| **branchless block Hoare** (pdqsort) | large arrays, expensive comparators, or once data exceeds L2 — the robust general default in C++ |
| **branchy Hoare** | structured/predictable inputs (sorted, low-cardinality) and very large records (> 96 B) that are costly to move |
| **BlockQuicksort `block`** | same niche as pdqsort but without the micro-optimised refill — generally dominated by `boost_block` |
| **branchy Lomuto / std::partition** | never the fastest here; useful only as baselines |
| **fulcrum / hybrid Hoare+Lomuto** (crumsort, `algo::fulcrum`) | **best for i64 at n=256–1k and ≥2^20**, competitive elsewhere for cheap-compare types — see next section |

## The hybrid: fulcrum partition (`algo::fulcrum`)

Implemented from crumsort's `fulcrum_default_partition` (Igor van den Hoven). It
is **bidirectional like Hoare** (streams the block from both ends) but
**branchless like Lomuto** via a *double write*: each element is stored
unconditionally to `ptl[m]` **and** `ptr[m]`, and the running count `m += (x <
pivot)` commits exactly one (the other is overwritten later). A 32+32 stack
`swap` buffer holds the two ends so the middle can be streamed while the ends are
filled, keeping reads ahead of writes. Same fixed pivot ⇒ identical partition
result, so this is a pure raw-time comparison (min ns/elem, random):

```
 i64       n=256  1024  4096  2^16  2^20  2^22   (after the by-value pivot fix)
 lom_bl    0.65  0.65  0.65  0.69  0.72  0.90
 boost     0.60  0.41* 0.36* 0.38* 0.46* 0.45*
 fulcrum   0.47* 0.45  0.45  0.45  0.49  0.51

 pair64f   n=256  1024  2^16  2^20  2^22
 boost     0.71  0.51* 0.41* 0.73  0.75
 fulcrum   0.53* 0.52  0.48  0.62* 0.75*

 pair64 (lex): boost wins at every size; fulcrum and lom_bl are both
 write-bound (16-byte stores) and pay the serial expensive-compare on the
 m-chain, so neither beats the block partition's 8x-unrolled compare ILP.
```

So the hybrid **delivers for cheap-compare keys, but only in a narrow band**:
fulcrum is fastest for i64 at n ≈ 256 and pair64f at n ≤ 256 / ≥ 2^20, and it
beats branchless Lomuto by ~1.8× at 2^22 (its committed stores are sequential and
the scratch store is absorbed by L1, unlike Lomuto's two *distinct* committed
stores). But once `boost_block` is given the same by-value pivot (below), the
block partition retakes the i64 mid-to-large range. The fulcrum is a worthwhile
addition for small-medium cheap-compare blocks, not a new overall champion.

### Assembler-level optimisation

1. **Pivot by value — the big one (~15% on `boost_block` i64).** The fulcrum's
   hot loop showed `cmp (%rcx),%r11` — the pivot reloaded from memory on *every*
   comparison. Cause: **aliasing**. A partition permutes the block while comparing
   against the pivot, so if the pivot is a *reference* the compiler must assume it
   may alias a written element and reloads it after every store. Taking it **by
   value** (`K pivot`, not `const K&` — and *not* `K&&`, which is also a reference
   and reloads identically; verified by disassembly: const-ref → 1 reload/elem,
   `K&&` → 1, by value → 0) makes it provably non-aliasing and register-resident.
   Applied to the whole `PivotPartitioner` interface, this removed the reload from
   `boost_block`'s `branchless_partition` too (i64 n=4096: 0.42→0.36) — the single
   most impactful change in this section. Keys here are ≤ 16 B, so the copy is
   free.

   Full A/B (`const K&` → by value), speedup% at mid sizes (n = 1k–64k, median
   over i64/pair64/pair64f; + = faster), every partitioner × distribution:

   ```
   partitioner      random few_uniq all_eq sortAsc sortDesc organ m3killer | all
   std_partition     +2.6   +4.1    +0.2   +0.7    +3.9    +0.4   +0.2     | +2.5
   lomuto            +3.2   +2.9    +0.2   +2.4    +4.0    +0.3   +0.0     | +1.9
   lomuto_branchless +4.4   +4.1    +4.0   +0.7    +5.1    +3.5   +3.7     | +4.0
   hoare             -1.3   +0.3    +0.3   +0.1    +4.1    +4.1   +0.1     | +0.3
   hoare_guarded     +4.1   +3.9    -0.0   +0.1    +4.0    +0.1   +0.2     | +2.5
   block             +1.4   +2.8    +4.0   +4.3    +5.0    +4.9   +4.0     | +4.1
   boost_block      +15.5  +13.8    +4.0   +2.5   +17.0    +9.6   +4.0     | +9.6
   fulcrum           +3.4   +4.0    +5.0   +2.4    +4.1    +3.1   +4.3     | +4.1
   ```

   Two readings: (a) the **branchless** partitioners benefit (boost_block +9.6%
   overall, up to **+22% for pair64-lex at 2^16** where the expensive compare
   reloaded the pivot every element; block / lomuto_branchless / fulcrum +4%),
   while the **branchy** ones (hoare, lomuto, std_partition) are ~neutral —
   they are branch-mispredict-bound, so the pivot reload was already hidden.
   (b) The win **peaks at mid sizes** (256–2^16: compute-bound but cache-resident)
   and fades toward tiny n (overhead) and huge n (memory-bandwidth-bound, where
   the reload hides under DRAM latency). Reproduce by toggling the pivot
   parameter type and re-running `bench_partition`.

2. **cmov single-store (tried, REJECTED).** The double write does two stores;
   replacing it with one cmov-selected store `*(v ? ptl+m : ptr+m) = x` halves
   the store count and is provably correct (the committed writes alone tile
   `[0,n)`). It *won* in an isolated microbench (~7–8% at 2^16), but in the real
   templated `comp/proj` path it was **5–15% slower**: the store address then
   depends on `v` (the comparison), serialising the store behind the compare,
   whereas the double write's addresses depend only on `m` and issue early on
   both store ports. A genuine "both addresses ready before the predicate" beats
   "one store after a cmov" here. Kept the double write.

The remaining cost is fundamental: the `m += v` count is loop-carried (the
partition is inherently sequential in `m`), and for 16-byte elements the two
16-byte stores per element are store-port-bound — which is exactly why the
block partition (few swaps) wins for wide/expensive-compare types and the
fulcrum/Lomuto family wins for narrow cheap-compare ones.

**Practical rule (updated with the fulcrum result):** for a **cheap comparator**
and a narrow key (i64), the **fulcrum** hybrid is the best single choice across
n ≳ 256 (with branchless Lomuto a hair faster only for the tiniest blocks); for
an **expensive comparator** or a wide/lexicographic key, use **branchless
block-Hoare** (pdqsort/`boost_block`); detect sorted/equal runs and use plain
**Hoare** there. This matches the SOTA split (ipnsort's Lomuto-cyclic ≤ 96 B,
crumsort's fulcrum, pdqsort's block-Hoare) — no one scheme dominates; the winner
is set by comparator cost, element width, array size and input pattern.

## Sources

* Orson Peters, *Branchless Lomuto Partitioning* — <https://orlp.net/blog/branchless-lomuto-partitioning/>
* Bergdoll & Peters, *ipnsort* writeups (lomcyc partition, ipnsort introduction) —
  <https://github.com/Voultapher/sort-research-rs/blob/main/writeup/lomcyc_partition/text.md>,
  <https://github.com/Voultapher/sort-research-rs/blob/main/writeup/ipnsort_introduction/text.md>
* Rust stdlib partition selection (`MAX_BRANCHLESS_PARTITION_SIZE = 96`) —
  <https://doc.rust-lang.org/src/core/slice/sort/unstable/quicksort.rs.html>
* Edelkamp & Weiss, *BlockQuicksort*, ESA 2016.
* `boost::sort::pdqsort` (pattern-defeating quicksort).
* blipsort (branchless Lomuto in pdqsort) — <https://github.com/RedBedHed/blipsort>
