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

### 1. Value-partitioners (lomuto, hoare, block) — same `<`, route the other way

These never compare two arbitrary elements. They evaluate the comparator only as
a **unary predicate against the single fixed pivot**:
`below(x) = comp(proj(x), pivot)` (= `x < pivot`), and send `below`-true elements
left. The reversed split sends the **complement** to the left:

```
forward LEFT group = { x : x <  pivot }
reversed LEFT group = { x : x >= pivot }   = complement of the forward group
```

**This does NOT force a `>=` comparison.** A partition decides, per element,
*which side it goes to*; the reversed version can run the **identical**
`x < pivot` comparison and merely route the elements the other way. Forward
itself already evaluates the complement on one of its two scans (it sends
`x < pivot` left and the rest right): forward and reversed are mirror images, and
*which* of `<`/`>=` is "the negated one" is only a choice of **where the
complement lands**. On the scalar paths that choice is free — disassembly of the
branchless-Lomuto gap loop (`g++ -O3 -march=native`, i64) shows forward vs
reversed are **instruction-for-instruction identical except one condition code**:

```
  forward (route x<piv left)     reversed (route x>=piv left)
    cmpq  %rcx, %r10               cmpq  %rcx, %r10
    setg  %cl          <--here-->  setle %cl          (the exact complement)
    addq  $1, %rax                 addq  $1, %rax
    addq  %rcx, %rdx               addq  %rcx, %rdx
```

40 instructions each — same count, same latency, same ILP. `setg`↔`setle` is the
complementary set-from-flags; it costs nothing. So the reversed Lomuto *is*
literally "the same `<` computation, routed to the other side" at zero cost.
(The only place a literal mask-invert appears is the AVX2-vectorised block fill —
see the Assembly verdict; it is the cheapest lowering of the complement there,
not a forced semantic, and it is measured free.)

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

## Does using `>=` change any algorithm's logic? (No — and it isn't even forced)

- **`<` vs `>=` is only a choice of where the complement lands.** A reversed
  partition runs the bare strict `x < pivot` (the same comparison as forward) and
  routes elements to the opposite side, achieved by **exchanging the two scans'
  roles** — the forward left scan/fill works with one of `below`/`!below` and the
  right with the other; the reversed version swaps which is which. Forward and
  reversed therefore issue the *same* operations (one `below`, one `!below`). The
  scalar paths show this directly (the free `setg`↔`setle` flip above).
- **`>=` (where it is used) does not change any routing decision.** It must be
  `>=`, not `>`: the reversed left group is the set-complement of `{x < pivot}`,
  which is `{x >= pivot}`; `>` would wrongly send pivot-equal elements right. And
  it is harmless: the value-partitioners consume the routing test only as a unary
  predicate against the fixed pivot, so the loop yields `[goes-left | goes-right]`
  for *any* rule — reflexivity is irrelevant because two arbitrary elements are
  never compared. Verified on `all_equal` (all `== pivot` ⇒ all land left).
- **No `>=`/negated comparator ever reaches a comparator network.** The halvers
  use the strict `<` via `cswap_rev` (direction reversed, *not* comparator
  negated), so the 0/1-principle proof holds. This is the one place where the
  strict-vs-reflexive distinction genuinely matters, and it is respected — which
  is exactly why a separate descending halver was built rather than negating.

## Benchmark: reversed is not slower than forward

Three ways to obtain the reversal were measured head-to-head against the forward
baseline:

* **`rev_rw`** — the rewritten `algo_rev` / `quicksort_rev`: the same strict
  `comp` as forward, with the two scans' roles exchanged (route a; the default)
* **`rev_neg`** — forward algorithm + `negate_comp(less)` (route b; what
  `partition_api.hpp::reverse_partition_by_key` already does)
* **`rev_view`** — forward algorithm run over `make_reverse_iterator`s (route c;
  a *wrong* approach kept only to quantify its cost — GCC won't vectorise it)

### Standalone partitioner, `random_uniform` (ns/elem, median of 3 runs, final code)

```
 i64           fwd    rev_rw  rev_neg  rev_view
 n=256        0.454   0.467   0.466   0.531
 n=4096       0.384   0.385   0.387   0.490
 n=65536      0.400   0.415   0.407   0.539

 pair64        fwd    rev_rw  rev_neg  rev_view
 n=256        0.843   0.810   0.817   0.900
 n=4096       0.715   0.556   0.557   0.602
 n=65536      0.542   0.535   0.557   0.609

 pair64f       fwd    rev_rw  rev_neg  rev_view
 n=256        0.723   0.728   0.722   0.788
 n=4096       0.402   0.408   0.409   0.505
 n=65536      0.423   0.425   0.480   0.594
```

(Large-n, 2^20–2^26, is bandwidth-bound; a single 8 MB+ block is too noisy for a
3-pass median, so it is reported separately from the 25-rep `min` micro-bench
below — there `rev_rw/fwd = 0.9996` at 2^26.)

* **`rev_rw` ≈ `rev_neg` ≈ `fwd`** to within run-to-run noise — they swing both
  ways across sizes, and `rev_rw` is frequently *faster* (e.g. pair64 n=4096:
  0.556 vs 0.715). The scalar tier's asm is *provably identical* to forward
  (`setg`↔`setle`); per the repo's methodology note, for sub-ns branchless
  kernels op-count + asm are the ground truth and a single timing harness is not.
* **`rev_view` is consistently 5–30% slower.** The `reverse_iterator`'s extra
  per-deref decrement and backward pointer walk are real overhead. **This is the
  answer to "does wrapping in a reversed view add overhead?": yes — so the
  dedicated rewrite is justified, and the reverse-view route is rejected.**

### Standalone partitioner, `sorted_descending`

```
 i64    n=65536:  fwd 0.509   rev_rw 0.126   (rev ~4x FASTER)
 pair64 n=4096 :  fwd 0.715   rev_rw 0.283
```

A descending-sorted input is *already partitioned in the reversed-favorable
direction*, so the reversed partitioner streams it almost for free — the mirror
image of forward's good case on ascending input. Not an algorithmic "win", just
input orientation; reported for completeness (the reversed partitioner's
easy/hard input patterns are the mirror of the forward one's).

### Full quicksort, `random_uniform` (ns/elem, median of 3 runs, final code)

```
              qs_fwd   qs_rev_rw   qs_rev_view   rev_rw/fwd
 i64   4096    8.51      8.67        10.35         1.02
 i64   2^20   14.55     14.57        17.01         1.00
 pair64 2^20  35.83     36.21        41.37         1.01
 pair64f 2^20 21.77     20.79        25.10         0.95
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
sampling noise on a single 512 MB pass; with 25 reps + `min` it vanishes. Any
`>=` mask-invert GCC may emit is perf-neutral even here, where the partition is a
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

The reversed partitioners are built by **exchanging the roles of the two scans**
using the *same strict `comp`* as forward — never a negated comparator, never
reverse iteration. So forward and reversed perform the **identical work**: each
issues exactly one `below` (`x < pivot`) test and one `!below` test; they only
differ in which scan plays which role.

* **lomuto / scalar paths:** byte-identical except the complementary `setcc`
  (`setg`↔`setle`). Zero overhead; `<` vs `>=` is a free choice.

* **block (boost) path:** identical *source* operations to forward (one `below`
  fill, one `!below` fill). GCC's auto-vectoriser lowers the `!below` (`>=`) fill,
  when it chooses to vectorise it, as `vpcmpgtq` + a mask-invert (`vpcmpeqq`
  against zero, plus a hoisted `vpxor` to materialise the zero) — because AVX2
  has **no `>=`/`<=` compare** and this Meteor Lake has no AVX-512. Two facts
  make this a non-issue:
  - It is a **context-dependent codegen choice, not inherent.** Compiled
    out-of-line GCC vectorises the *direct* `below` fill instead and emits **no
    invert at all** (1 `vpcmpgtq`, 0 `vpcmpeqq`/`vpxor`); inlined into a tiny
    wrapper it picks the `!below` fill and emits the invert. Same source, same
    comparison — only the optimiser's fill choice varies. On an AVX-512 target
    the `>=` would be a single masked compare with no invert regardless.
  - When present it is **free**: the fill loop is bound by the offset-store/swap
    traffic, and the extra compare hides in spare vector-ALU ILP. `rev_rw`
    benchmarks at parity-or-faster than `fwd` at every block size, including the
    2^26 numbers above and the targeted 25-rep `min` measurement (i64 2^26 =
    0.9996×).

  **Verdict:** there is no inherent extra operation and no different comparison —
  the reversed partition is the forward partition with the two scans' roles
  exchanged, and it is not slower. The occasional mask-invert is a free AVX2
  lowering detail of `>=`, not a property of the algorithm.

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
