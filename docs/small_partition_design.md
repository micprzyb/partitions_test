# Small-array partition via HALVERS (n ≤ 24, focus n ≤ 8): corrected design

**Goal.** A fast branchless "partition" primitive for small blocks inside a
quicksort. Element types `i64`, `pair64` (lex), `pair64f`. Raw performance on a
modern out-of-order CPU is the only criterion.

> **Correction to an earlier claim.** A previous draft argued "a fixed
> compare-exchange partition network must be a sorting network (0‑1 principle)."
> That is true only for *exact partition around an arbitrary pivot **value*** (the
> boundary may land anywhere for every value, which does force full sorting). It
> is **wrong for the object we actually want**, which is a **halver** — a network
> that splits by **rank** into a bottom half (smaller ranks) and a top half. The
> 0‑1 principle only requires `max(bottom) ≤ min(top)` of a halver, **not** that
> it group/sort all 0/1 inputs, so a halver is **not** a sorter and is much
> cheaper. This also vindicates the earlier "pseudo15-with-exchanges" idea: a
> median/selection network applied as compare-**exchanges** is an *approximate
> halver*.

---

## 1. Definitions

* **(ε)-halver** on m wires (m even): comparator network; output is two halves of
  m/2. For every k ≤ m/2, of the k smallest-rank inputs at most **εk** end up in
  the *top* half (and symmetrically for the k largest). **ε = 0** ⇒ *exact*
  halver: the m/2 smallest occupy the bottom half (unordered within), the m/2
  largest the top half. This is precisely a balanced **partition by median rank**.
* **(n, t)-selection network**: first t outputs carry the t smallest. An exact
  halver is the (n, n/2)-selection done both ways.
* **Approximate halver**: ε > 0; some elements land on the wrong side, bounded by
  εk. Cheap; needs downstream cleanup (recursion + leaf sort) to fully sort.

## 2. Why halvers are the right (and cheaper) primitive

* **Cost.** ε-halvers are **O(m)** comparators for constant ε (the AKS building
  block — though AKS's constants are huge *asymptotically*; for small m explicit
  halvers are small, e.g. a 1/4-halver on 12 wires is depth 4, **17** comparators).
  Exact median selection by a network is **sub-`O(n log n)`**, hence cheaper than a
  sorting network, and the gap **grows with n**:

  ```
  n              8    12   16
  sort S(n)      19   39   60     (optimal sorting-network size)
  exact halver   15   29   47     (MEASURED: greedy-prune of S(n), 0/1-verified,
                 -21% -26% -22%    proven NOT a sorter; true optimum is smaller)
  eps-halver     O(n), small constant (approximate; cheapest)
  ```
  (Greedy pruning is only *locally* minimal; a SAT/exhaustive search à la
  arXiv:1807.05377 finds smaller exact halvers, and ε>0 halvers are smaller again.
  Even greedily, an exact halver is ~20–26% fewer compare-exchanges than a full
  sort across n = 8–16, and the bound is loose.)

* **It is exactly what quicksort wants.** A halver yields a *balanced* split
  (perfect for ε=0) **with no pivot-selection step at all** — it splits by element
  comparisons directly. Compare the current path: choose a pivot (median-of-k,
  with balance *variance* at small n) + partition by value. A halver removes the
  pivot step and gives *guaranteed* balance.

* **It is branchless and shallow → ILP.** Like the `small_sort` networks, every
  comparator is a `cswap` (cmov / xor-mask, no data branch). A halver has fewer
  comparators **and** fewer layers than a sorter, so on a wide OoO core the
  independent comparators in each layer issue in parallel.

* **Connection to the repo's pseudo-medians.** `pivot::pseudo9/pseudo15` are
  approximate-median *selection* DAGs that **compute a value** (min/max combine,
  discarding element identity). Turning the same idea into a **permutation
  network** whose comparators compare-exchange the real elements gives an
  **approximate halver** that *produces the split* instead of just a pivot value.

## 3. Reconciling with the partition contract (important)

The repo's `PivotPartitioner` partitions around a pivot **value** and returns the
exact boundary `[<pivot | >=pivot]`. A halver instead splits by **rank** (fixed
n/2) and is **pivot-free**. So a halver is a *different primitive*, used by a
*different quicksort shape*:

* **Exact halver** ⇒ a "median-split quicksort": recurse on the two halves; no
  pivot value, guaranteed balance, O(n log n) deterministically. Drop in a
  `small_sort` network at the leaves (n ≤ some cutoff).
* **Approximate halver (ε>0)** ⇒ split is *almost* balanced; recursion depth stays
  O(log n) for bounded ε, and the leaf sort fixes the ε-imperfections. Cheapest,
  but only valid in this recurse-and-clean structure, **not** as a drop-in for the
  exact value-partition API.

This is a real fork: pursuing halvers means adding a *balanced-split* primitive
(and a quicksort that uses it), not another `PivotPartitioner`.

## 4. Modern-CPU performance analysis (the deciding factor)

Comparator count is the classical metric, but on a modern CPU what matters is:

1. **Moves dominate for wide elements.** Every comparator that fires does a swap.
   For `pair64` (16 B) the `cswap` is the cost; a halver does *fewer* comparators
   than a sorter, so fewer 16‑B swaps → a real win for the pair types (where the
   value-partition family is already move-bound).
2. **Compares dominate for expensive comparators** (`pair64` lex): a halver does
   fewer element comparisons than a sorter, and (unlike value-partition) needs
   **no pivot-selection comparisons** — a double saving.
3. **Branchless + shallow = ILP.** Fewer layers than a sorter shortens the
   critical path; independent comparators fill the OoO width.
4. **Guaranteed balance helps the efficiency metric** `time / |smaller part|`:
   the denominator is fixed at n/2 (vs median-of-3's variance), so even at equal
   time a halver scores better — and `sort_mid` already won that metric at small n
   *despite* doing a full sort; a halver is "sort_mid but with fewer comparators",
   so it should win outright.
5. **Caveat — approximate halvers cost a cleanup.** An ε-halver is cheapest but its
   ε-misplaced elements must be repaired by the recursion/leaf-sort; the net win
   depends on ε vs cleanup cost. Exact halvers avoid that but cost more
   comparators. The sweet spot is empirical.

**Hypothesis to test:** for small n a *branchless exact halver* beats both
`sort_mid` (fewer comparators, same perfect balance) and the value-partition
family (no pivot step, guaranteed balance), most decisively for `pair64`/`pair64f`
(move/compare-bound) and at larger small-n (16–24, where halver ≪ sort).

## 5. What was measured already (still valid)

* `pext`-permute compaction (a *value* partitioner): correct but **slower** than
  the branchless gap method for i64 at n ≤ 8 (BMI2 latency + gather). Negative.
* AVX2 left-pack: a *value* compaction; the SIMD lever for i32/i64 at n=16–24, no
  help for pair64. Orthogonal to the halver direction.
* Baseline value-partition at n=8 i64 ≈ 0.56 ns/elem (`lomuto_branchless`).

## 6. Plan to design & validate optimal small halvers

1. **Construct exact halvers** for n ∈ {4, 8, 16, 24} (and a few odd n) and
   **verify by 0/1 enumeration** — the halver property "for all 0/1 inputs,
   `max(bottom) ≤ min(top)`" extends `tools/verify_small_sort`'s exhaustive check.
   Source candidates: minimal (n, n/2)-selection networks (SAT/known optimal for
   small n; arXiv:1807.05377, arXiv:1502.04551), or prune a sorting network to a
   selection network. Record comparator counts vs S(n).
2. **Implement** as branchless `cswap` networks (reuse `small_sort::cswap`,
   including the pair64 half/word-swap specialisations) → a `halve<N>` routine
   returning the n/2 boundary.
3. **Approximate halvers**: also build a cheap ε-halver (e.g. a few brick-wall
   layers, or the pseudo-median network as compare-exchanges) and measure the
   imbalance distribution + speed.
4. **Benchmark** (raw ns/elem **and** efficiency `time/|smaller|`): `halve<N>` vs
   `sort_mid` vs (median-of-3 + value partition), for i64 / pair64 / pair64f, n =
   4…24. Confirm where the halver wins.
5. If it wins, add a **median-split** path (exact halver at the node, `small_sort`
   at the leaf) and benchmark a full small-array sort/partition pipeline.

## 7. Results — implemented & verified

Built: `include/partitions/small_halve.hpp` (halver networks for n=2..24, applied
with `small_sort::cswap`), `tools/verify_small_halve.cpp` (exhaustive 0/1 proof —
**all 24 halvers verified correct**, `max(bottom) ≤ min(top)` over every 2^N
input, and confirmed *not* sorters), `benchmarks/bench_small_halve.cpp`.

Halver vs full sort to produce the **same balanced median-rank split** (raw
ns/elem; ratio = halve/sort, **< 1 = halver faster**; stable across runs and two
independent harnesses):

```
 n        4    6    8   10   12   16   20   24
 i64    0.91 0.70 0.88 0.73 0.74 0.80 0.78 0.72
 pair64 0.73 0.72 0.94 0.67 0.76 0.83 0.69 0.80
 pair64f0.87 0.85 0.82 0.76 0.75 0.78 0.67 0.66
```

**The halver beats the full sort at every n for all three types — by ~6–34%
(typically 20–30%)**, consistent with its ~20–33% lower comparator count. Fewer
compare-exchanges ⇒ proportionally less work, uniformly.

**Two low-level fixes were required to get here (assembler-driven):**

1. **Eliminate an internal-lambda outlining (the big one).** `halve_n` originally
   dispatched the chosen network through a local `[&]` lambda. That lambda is not
   `always_inline`; GCC **outlined it into a `.isra` clone and *called* it** for
   some element types (pair64) while inlining it for others (i64). The out-of-line
   call destroyed inlining/vectorisation **and** made micro-timing wildly
   context-dependent — the source of an earlier *phantom* "i64 large-n regression"
   and "pair64 slower" that do not exist. Calling `apply_network` **directly** in
   each `if constexpr` branch guarantees inlining; disassembly after: halve_n<16,
   pair64> is 974 insns / 188 cmov, fully inlined (0 calls), vs the sort's 1209 /
   240 — halver wins on op count, as expected.

2. **Layer-order the comparators.** Each halver is re-emitted in topological
   *layer* order (a function-preserving reorder that keeps wire-sharing comparators
   in sequence but groups independent ones). Adjacent independent compare-exchanges
   are what GCC's SLP vectoriser can pack; the regular sort networks vectorise for
   this reason (pair64 n=16 sort: ~30 SIMD register ops, 360 `vpcmpgtq`), and
   layer-ordering gives the halver the same opportunity. Re-verified as still a
   valid halver by 0/1 enumeration.

**Methodology note (kept as a caution).** These networks are branchless and
data-independent, so their timing *must* be identical regardless of input —
meaning any run-to-run or harness-to-harness variation is pure **code-layout /
inlining** artifact, not algorithm. Two harnesses disagreeing by 2× was the tell
that exposed the lambda-outlining bug; after the fix both agree to ±2%. Lesson:
for sub-ns/elem branchless kernels, trust op-count + asm over a single timing
harness, and require cross-harness agreement.

* **`halve_reg` (register-blocked) is not faster** (often slower): for n>16 the
  block exceeds the GP register file and spills; for n≤16 GCC already
  scalar-replaces the in-place network. Kept as the "guarantee" variant (per the
  same finding in `small_sort`), not a default.

**Bottom line.** A verified branchless **halver** is the faster way to produce a
balanced small-array split than a full sort — **uniformly, for i64, pair64 and
pair64f at every n ≤ 24** (~20–30%), exactly tracking its lower comparator count.
There is **no** i64 large-n regression; the earlier one was a codegen artifact now
fixed. (An earlier "store-to-load-forwarding" explanation was a wrong diagnosis of
that artifact — corrected.)

## Sources
- ε-halvers / AKS building block & small optimal halvers (SAT): arXiv:1807.05377;
  Jeřábek, *A sorting network in bounded arithmetic* (AKS exposition),
  <https://users.math.cas.cz/~jerabek/papers/aks.pdf>.
- Construction of halvers: ScienceDirect S0020019099000320.
- Selection networks / cardinality (small (n,t) sizes): arXiv:1502.04551.
- Optimal sorting-network sizes S(9)=25, S(10)=29 etc.: arXiv:1310.6271;
  "25 comparators optimal for 9 inputs".
- Median-filter sorting-network architectures (selection networks in practice);
  AxMED approximate medians: arXiv:2502.18174 (cf. the repo's pseudo9/pseudo15).
- 0‑1 principle: Knuth TAOCP Vol. 3 §5.3.4.
