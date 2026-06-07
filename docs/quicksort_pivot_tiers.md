# Quicksort pivot tiers + halver-cutoff sweep

Question: with `ninther` confirmed better than `m3`/`m5m5` *everywhere*, does a
**size-tiered** pivot (cheap small / ninther medium / expensive `m5m5` for the few
huge nodes) win?  And does a **bigger halver cutoff** (>16) pay off?  Total sort
time only.  Benchmark: `benchmarks/bench_pivot_tiers.cpp` (partitioner fixed =
`algo::sized`); measured on `random_uniform` **and** `sorted_descending` so any
config that reintroduces the median-of-3 killer (→ O(n²)) is caught.

## Results — ns/elem, vs `ninther` everywhere @ halver-cutoff 16 (median of runs, 2^20 & 2^22)

```
 config                    i64 rand   pair64 rand   pair64f rand   sorted_desc
 ninther.T16  (baseline)     0.0%        0.0%          0.0%        fast (O(n log n))
 ninther.T20                ~0%         -2.0%         -0.8%        fast
 ninther.T24                +2%/-0.8%   -0.4%/-2.0%   -0.8%        fast   ← within noise
 m5  (<=256) | ninther      +6%         +1.2%         +3.2%        fast   ← WORSE
 ninther | m5m5 (>64k)      -2.0..-2.6% ~0% / -1.9%   -2.5..-3.5%  fast   ← FREE WIN
 m3  (<=64)  | ninther|m5m5 -5..-7%     -3.8..-5%     ~0%          +30% (bounded)
```

## Findings

1. **Halver cutoff: raise 16 → 24 (small, type-dependent win).** A careful per-type
   sweep (5 runs @2^20 + 3 @2^22, median) — *not* the noisier 2-run check that first
   suggested "no gain" — shows:
   ```
                T8     T12    T16    T20      T24
    i64       +13%   +3.4%   0.0%  +0.1%    ±0.4%   (flat above 16)
    pair64    +5.9%  +2.1%   0.0%  -0.5%   -0.9..-1.2%
    pair64f   +4.1%  +0.3%   0.0%  -1.3%   -1.3..-1.6%
   ```
   So **T=24 is ~1.0–1.6% faster on the 16-byte pair types and neutral on i64** —
   on the 17–24 nodes the branchless, perfectly-balanced halver beats
   `ninther`+partition, and the *current* 17–24 halvers already capture it (the
   lever is the cutoff value, not the net quality). **Cutoff raised to 24.**

   Per-node micro (`halve_n<N>` vs `ninther`+one partition, in isolation) said the
   halver split is *cheaper* for i64 at every N≤24 yet *more expensive* for pair64 —
   the OPPOSITE of the total-time result. Lesson: an isolated hot-loop micro does
   not predict the recursive (cold-cache) sort; trust the end-to-end sweep.

2. **`m5m5` for the few huge nodes (>65536): a free ~2–3% win.** The 25-sample cost
   amortises completely over the top O(n) partitions, and the better balance shaves
   work off every level below — net −2.0…−2.6% on i64 and −2.5…−3.5% on pair64f,
   neutral on pair64, and **no penalty on sorted** (the big sorted node is still
   split well). **Adopted in `quicksort.hpp`** (`qs_m5m5_cutoff = 1<<16`).

3. **`m5` for small nodes: worse (+3–6%).** It costs more than ninther per node
   without enough balance benefit where most nodes are. Rejected.

4. **`m3` for tiny nodes (≤64): the biggest random win (~5%)** — cheapest pivot for
   the many small nodes — **but +30% on sorted** (bounded, confirmed O(n log n) at
   2^22, not quadratic, because m3 only acts on ≤64-element blocks → O(64·n) extra).
   A genuine raw-random-vs-adversarial trade-off; **left out** of the default since
   it degrades structured input, but it is the config to choose if the workload is
   guaranteed random and ~5% matters more than a bounded sorted slowdown.

## Net change

`partitions::quicksort` now uses: **halver ≤24** → ninther (25 … 65536) → **m5m5
(>65536)**, partitioner `algo::sized`.  ~2–3% from the m5m5 top tier on large
i64/pair64f, plus ~1–1.6% on the pair types from the cutoff=24; free, no robustness
loss.  Reproduce: `build/benchmarks/bench_pivot_tiers [n]`.
