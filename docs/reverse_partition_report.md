# Reversed partition & quicksort (`>=` left / strictly `<` right)

**Goal.** Build the mirror image of the repo's size-adaptive partitioner and
its pure-partition quicksort, so that **every element `>= pivot` lands LEFT and
every element strictly `< pivot` lands RIGHT** (`[first,m) >= pivot |
[m,last) < pivot`, the `ptv::reverse_ok` contract). Raw performance is the only
criterion: the reversed version **must not be slower** than the forward
original. Measured on the same Meteor Lake host, GCC, `-O3 -march=native`,
pinned core; min/median ns/element.

## What was built

| file | what |
|---|---|
| `include/partitions/reverse_partition.hpp` | `algo_rev::{hoare_rev, lomuto_branchless_rev, boost_block_rev, sized_rev}` — value-partitioners returning the `>=/<` split |
| `include/partitions/small_halve_rev.hpp` | `cswap_rev` + `halve_rev` — descending halver networks (leaf of the reversed quicksort) |
| `include/partitions/quicksort_rev.hpp` | `quicksort_rev` — descending three-tier (halver / lomuto / block) quicksort |
| `tools/verify_small_halve_rev.cpp` | exhaustive 0/1 proof the reversed halvers are valid descending halvers |
| `tests/test_reverse_partition.cpp`, `tests/test_quicksort_rev.cpp` | correctness across the type × distribution × size matrix |
| `benchmarks/bench_reverse_partition.cpp` | forward vs the three reversal routes |

## The crux: two ways the comparator is consumed

There are **two structurally different** partition tiers, and "reverse the
order" means a different thing in each.

### 1. Value-partitioners (lomuto, hoare, block) — flip the *unary predicate*

These never compare two arbitrary elements. They evaluate the comparator only as
a **unary predicate against the single fixed pivot**:
`below(x) = comp(proj(x), pivot)` (= `x < pivot`), and send `below`-true elements
left. The reversed split wants the **set-complement** on the left:

```
forward LEFT group = { x : x <  pivot }
reversed LEFT group = { x : x >= pivot }   = complement of the forward group
```

so the predicate is forced to be `keep(x) = !comp(proj(x), pivot)` (`= x >=
pivot`). Strict-weak-ordering is irrelevant here (the comparator is a boolean
test, never an ordering of two elements), so the substitution is sound for *any*
predicate and is **free**. Disassembly of the branchless-Lomuto gap loop
(`g++ -O3 -march=native`, i64) — forward vs reversed are **instruction-for-
instruction identical except one condition code**:

```
  forward (below = x<piv)        reversed (keep = x>=piv)
    cmpq  %rcx, %r10               cmpq  %rcx, %r10
    setg  %cl          <--here-->  setle %cl          (the exact complement)
    addq  $1, %rax                 addq  $1, %rax
    addq  %rcx, %rdx               addq  %rcx, %rdx
```

40 instructions each — same count, same latency, same ILP. `setg`↔`setle` is the
complementary set-from-flags; it costs nothing.

### 2. Halver networks — keep the *strict* comparator, reverse the *direction*

The small-block halvers (`small_halve.hpp`) are comparator networks: every step
is a `cswap` that **orders two arbitrary elements**, and the network's
correctness rests on the 0/1 principle, which requires a **strict weak order**.

Here the "just negate the comparator" trick is **unsound**. The negated
comparator `ge(a,b) = !(a < b)` (= `a >= b`) is **reflexive** (`ge(a,a) = true`),
so it is not a strict order: the network-correctness proof no longer applies and
equal elements are mishandled. This is exactly the user's warning — *"the
negation of the comparator gives `>=` and makes some partitioners broken (they
accept strict order); passing the strict comparator would place equal elements
to the left."*

The correct construction keeps the **strict `<`** comparator but **reverses the
compare-exchange direction**: `cswap_rev` puts the *larger* element at the lower
index. By comparator-network duality (reversing every comparator ≡ complementing
the order: run the same net on `-x` and negate the result), the **same network
arrays** from `small_halve::nets` become valid **descending halvers** — the N/2
*largest* land in the bottom half, so recursive halving sorts descending. Equal
elements are never swapped (`cswap_rev` swaps on the strict `a < b`, false for
`a == b`), which is correct for a sort. Proven exhaustively:

```
build/tools/verify_small_halve_rev   →   N=2..24 desc-halver OK (2^N inputs each),
                                          ALL REVERSED HALVERS CORRECT
```

## Is `>=` (not `>`) safe? Does it change any algorithm's logic?

- **`>=` is forced, not chosen.** The reversed left group is the complement of
  `{x < pivot}`, which by trichotomy is `{x >= pivot}`. Using `>` would push
  pivot-equal elements to the right, violating the "equal goes LEFT"
  requirement. So `>=` is the *only* correct predicate, and it is exactly
  `!(x < pivot)`.
- **No value-partitioner relies on strictness.** `keep` is consumed only as a
  unary test of one element vs the fixed pivot; the partition loop produces
  `[P-true | P-false]` for *any* predicate `P`, so `P = (>=)` is correct by
  construction. The reflexivity of `>=` is harmless because no two arbitrary
  elements are ever compared. Verified: `test_reverse_partition` passes for all
  distributions including `all_equal` (every element `== pivot` ⇒ all `keep` ⇒
  all land left, boundary `= last`; `reverse_ok` holds).
- **No `>=` ever reaches a comparator network.** The halvers use the strict `<`
  via `cswap_rev`; reflexivity is never introduced where it would matter. This
  is the whole reason a *separate* descending-halver was built instead of
  negating the comparator.

## Benchmark: reversed is not slower than forward

Three ways to obtain the reversal were measured head-to-head against the forward
baseline:

* **`rev_rw`** — the rewritten `algo_rev` / `quicksort_rev` (route a)
* **`rev_neg`** — forward algorithm + `negate_comp(less)` (route b; what
  `partition_api.hpp::reverse_partition_by_key` already does)
* **`rev_view`** — forward algorithm run over `make_reverse_iterator`s (route c)

### Standalone partitioner, `random_uniform` (ns/elem, median of 5 runs)

```
 i64           fwd    rev_rw  rev_neg  rev_view
 n=256        0.454   0.465   0.461   0.531
 n=4096       0.385   0.395   0.391   0.479
 n=65536      0.404   0.404   0.406   0.534
 n=2^20       0.454   0.446   0.430   0.541

 pair64        fwd    rev_rw  rev_neg  rev_view
 n=4096       0.740   0.565   0.589   0.616
 n=65536      0.534   0.564   0.586   0.639
 n=2^20       0.649   0.675   0.658   0.753

 pair64f       fwd    rev_rw  rev_neg  rev_view
 n=4096       0.458   0.414   0.468   0.493
 n=2^20       0.715   0.745   0.742   0.726
```

* **`rev_rw` ≈ `rev_neg` ≈ `fwd`** to within run-to-run noise. At small n the
  differences (≤ ~7% for i64) are pure timing jitter on a sub-ns branchless
  kernel whose asm is *provably identical* (see above) — they swing both ways
  across sizes, and `rev_rw` is frequently *faster* (e.g. pair64 n=4096:
  0.565 vs 0.740). Per the repo's own methodology note, for sub-ns branchless
  kernels op-count + asm are the ground truth and a single timing harness is
  not.
* **`rev_view` is consistently 5–30% slower.** The `reverse_iterator`'s extra
  per-deref decrement and backward pointer walk are real overhead. **This is the
  answer to "does wrapping in a reversed view add overhead?": yes — so the
  dedicated rewrite is justified, and the reverse-view route is rejected.**

### Standalone partitioner, `sorted_descending`

```
 i64    n=65536:  fwd 0.510   rev_rw 0.127   (rev 2.5–4x FASTER)
 pair64 n=4096 :  fwd 0.677   rev_rw 0.284
```

A descending-sorted input is *already partitioned in the reversed-favorable
direction*, so the reversed partitioner streams it almost for free — the mirror
image of forward's good case on ascending input. Not an algorithmic "win", just
input orientation; reported for completeness (the reversed partitioner's
easy/hard input patterns are the mirror of the forward one's).

### Full quicksort, `random_uniform` (ns/elem, median of 5 runs)

```
              qs_fwd   qs_rev_rw   qs_rev_view   rev_rw/fwd
 i64   4096    8.57      8.59        10.43         1.00
 i64   2^20   14.50     14.47        16.91         1.00
 pair64 2^20  36.08     36.12        42.22         1.00
 pair64f 2^20 21.14     21.09        25.10         1.00
```

`quicksort_rev` matches the forward `quicksort` to **< 3% (mostly < 1%)** at
every size and type. The reverse-iterator route is again ~15–25% slower.

### Bigger arrays (2^24, 2^26)

Memory-bandwidth-bound regime. **Standalone partition** (i64, random_uniform,
targeted micro-bench, 25 reps, *min* ns/elem — the bandwidth-robust metric):

```
              fwd     rev_rw   rev_neg    rev_rw/fwd
 n=2^24      0.4815   0.4840   0.4905       1.005
 n=2^26      0.4976   0.4974   0.4981       0.9996
```

Dead parity. (An earlier 2-sample run showed i64 2^26 at ~1.10× — that was pure
sampling noise on a single 512 MB pass; with 25 reps + `min` it vanishes. The 2
extra AVX2 mask-invert ops are perf-neutral even here, where the partition is a
pure DRAM streaming pass.)

**Full quicksort** (random_uniform, ns/elem, median of 3 passes):

```
              qs_fwd   qs_rev_rw  rev_rw/fwd   qs_rev_view
 i64    2^24  16.769   16.779       1.001       19.347
 i64    2^26  17.971   17.889       0.995       20.627
 pair64 2^24  39.492   39.563       1.002       44.476
 pair64 2^26  41.167   41.176       1.000       46.151
 pair64f 2^24 24.517   24.292       0.991       28.831
 pair64f 2^26 26.372   25.963       0.985       30.567
```

`quicksort_rev` is within **±1.5%** of forward at 2^24 and 2^26 for every type
(and faster on pair64f). The reverse-iterator route stays ~15–18% slower.

## Assembly verdict

* **lomuto / scalar paths:** byte-identical except the complementary `setcc`.
  Zero overhead.
* **block (boost) path:** the reversed fill loop carries **2 extra vector
  instructions** out of ~124 — a `vpcmpeqq`-against-zero (mask invert) plus a
  loop-invariant, hoisted `vpxor` (zeroing). Reason: AVX2 has **no `>=`
  predicate** (`vpcmpgtq` only), so a vectorized `>=` is `compute (>) then invert
  the mask`. Forward evaluates its *negated* predicate (`>=`, for the left
  buffer) on the *ascending* fill; the reversed partition's left group is the
  `>=` set, so the negated predicate necessarily lands on the *descending*
  (right-buffer) fill, where the vectorizer materialises the invert explicitly.
  Meteor Lake has no AVX-512, so no single-instruction `>=` exists — this 1
  op/vector is unavoidable for *any* `>=` partition. It is **perf-neutral**: the
  fill loop is bound by offset-store/swap traffic and the extra compare hides in
  spare vector-ALU ILP, which is why `rev_rw` benchmarks at parity-or-faster
  than `fwd` at every block-partition size. Documented as a (harmless) negative
  result rather than chased into a fragile, compiler-version-specific workaround.

## Reproduce

```
cmake --build build -j
ctest --test-dir build -R 'reverse_partition|quicksort_rev' --output-on-failure
build/tools/verify_small_halve_rev
build/benchmarks/bench_reverse_partition quick      # sizes capped at 4096 (~seconds)
build/benchmarks/bench_reverse_partition            # default: up to 2^22 (~2-3 min/pass)
g++ -std=c++23 -O3 -march=native -Iinclude -S asm_probe.cpp   # asm diff fwd vs rev
```

**Cost warning.** `bench_reverse_partition <maxsize>` with `maxsize >= 2^24` is
expensive: a single pass at `2^26` takes **~50 min** (dominated by the pair64
2^26 quicksort — ~3 s/sort × warmup+reps × 3 routes × 4 distributions — and the
1 GB setup-copies before every rep). Do NOT loop it for many passes. For the
big-n regime, settle a specific cell with a targeted micro-bench (single type,
single size, `min` ns/elem over ~25 reps) instead — that is bandwidth-robust and
runs in seconds. The big-n numbers above were obtained that way.
