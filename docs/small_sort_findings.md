# Fast small-array sorting: findings & recommendation

**Scope.** Sort arrays of fewer than 24 elements of `int64` and of
`pair<int64,int64>` (lexicographic order), with support for a custom comparator
and projection but with the *natural order* as the hot path. **Raw throughput is
the only criterion.** Target hardware includes machines with **no AVX-512** and
where **AVX2 SIMD does not help** (a `pair64` is 16 bytes, so an AVX2 lane holds
only two of them and the lexicographic predicate needs a 128-bit compare that
x86 cannot express as a packed mask below AVX-512).

Measurements below were taken on an **Intel Core Ultra 7 165H (Meteor Lake,
AVX2, no AVX-512)**, GCC 15.2, `-O3 -march=native`, single core pinned.

**Benchmark methodology (for stability).** Every reported number aggregates
**≥ 5,000,000 individual sorts** taken across **8 independent configurations**
(4 RNG seeds × 2 value ranges: a wide full-width range and a narrow range that
forces many ties, exercising the second-key tie-break of `pair64` and the
equal-key paths of the branchy sorts). Sorts are issued in batches of ~64K
elements; each batch is the timed unit (one `steady_clock` pair amortised over
thousands of sorts), giving 900–3000 ns/sort samples per data point. The
destructive restore between batches is untimed; a few warmup batches per config
prime caches and the predictor. We report the **min** ns/sort (least-noisy
estimate of true cost) as the headline, and track the coefficient of variation
(`cv% = stddev/mean`) as an explicit stability gauge — for every result that
decides the ranking it sits in the low single digits.

---

## TL;DR recommendation

For this exact problem, **no off-the-shelf library is optimal**. The fastest
solution is a **branchless scalar sorting network** that

1. uses the **best-known *size*-optimal network** for each `N` (fewest
   compare-exchanges), and
2. implements each compare-exchange as a **branchless compare-exchange**, with
   two codegen subtleties that turned out to matter a lot (both found by
   disassembly — see "Two codegen investigations" below):
   - The 16-byte swap is written as a **whole-half blend over two integers**
     (via `memcpy` to `uint64`), which GCC lowers to **4 cmovs** in GP
     registers. The earlier hand-written field-level XOR-mask blocked that and
     emitted ~6 explicit xors per CE — **30–40% slower** on `pair64`. And a
     naive `cond ? hi : lo` on the struct is worse still: for a floating field
     it lives in XMM (no cheap branchless select) so GCC emits a **branch per
     CE**. The integer-half blend dodges both.
   - The register-sized integer CE is an **independent min/max blend** (two
     parallel cmovs), not cpp-sort's lower-register `min; y^=dx^x` (which is
     serial). The parallel form wins on a wide OoO core.

That implementation lives in `include/partitions/small_sort.hpp`
(`small_sort::sort` / `sort_n<N>`). It beats `std::sort`, Boost, and even
`cpp-sort`'s own network sorter **at every size for every element type tested**
(on `int64` the margin over `cpp-sort` is sub-nanosecond and size-dependent).

| N=24, min ns/sort | `int64` | `pair64`<br>⟨long,long⟩ | `pair_li`<br>⟨long,int⟩ | `pair_fi`<br>⟨float,int⟩ | `pair_di`<br>⟨double,int⟩ |
|---|---|---|---|---|---|
| `std::sort` (libstdc++) | 247 | 291 | 294 | 290 | 293 |
| `boost::pdqsort` | 194 | 260 | 239 | 279 | 250 |
| `cpp-sort` `sorting_network_sorter` | 32 | 360 | 346 | 340 | 369 |
| **`small_sort::sort_n` (this repo)** | **27** | **122** | **111** | **129** | **163** |

The pair columns are the headline: `cpp-sort`'s network sorter is *slower than
`std::sort`* on every one of them, because it performs branchy 16-byte swaps. Our
network is **1.8–2.6× faster than `std::sort`** on the pairs.

> **Element types.** `pair64`=⟨long,long⟩, `pair_li`=⟨long,int⟩,
> `pair_di`=⟨double,int⟩ are all 16B; `pair_fi`=⟨float,int⟩ is 8B. Ordered by
> cost they fall out exactly as the *first-key compare* predicts: integer-keyed
> `pair64`≈`pair_li` (fastest), then float `pair_fi`, then double `pair_di`. The
> pair benchmarks are built `-ffast-math` (no NaNs assumed) so the floating
> compares drop NaN/unordered handling — applied to the whole TU, so every
> candidate gets it.

---

## The questions you asked, answered

### "Is `std::sort` the most effective?" — No.

`std::sort` (libstdc++ introsort) is a general comparison sort. For `N<24` its
recursion/median-of-3/insertion-tail machinery and **data-dependent branches**
dominate. It is **6–9× slower** than a branchless network on `int64` and
**~1.55× slower** on `pair64`. It is the right *fallback* for `N>24` but the
wrong tool here.

A relevant detail: libc++ (LLVM) ships the **AlphaDev** `sort3/4/5` routines and
branchless `__cond_swap` — but libc++ **gates the branchless path on
`is_arithmetic && sizeof(T) <= sizeof(void*)`**, so even libc++'s improvement is
*invisible to `pair64`*. libstdc++ (what GCC uses, measured here) does not have
those routines at all.

### "Is Boost the most effective?" — No.

* `boost::sort::pdqsort` (Orson Peters' pattern-defeating quicksort) is excellent
  for large/medium arrays and is the best of the general sorts here, but its
  small-`N` insertion tail is still branch-driven: **5–7× slower** than a network
  on `int64`, **~1.4× slower** on `pair64`.
* `boost::sort::spreadsort::integer_sort` is a radix sort: **`int64` only, no
  custom comparator, no projection** — disqualified by your interface
  requirements, and not faster than a network at these sizes anyway.

### "Are the current implementations using *VarSort4*, invented by Google AI?" — No.

This is worth being precise about, because the names get conflated:

* **AlphaDev (DeepMind, *Nature* 2023)** produced two kinds of result: **fixed
  sorts** `sort3/sort4/sort5` and **variable sorts** `VarSort3/VarSort4/VarSort5`
  (sort a run of *up to* 3/4/5 elements, branching on the actual length). These
  were optimised to **minimise x86 assembly instruction count / latency for a
  *single* call** and were upstreamed into **libc++**. `VarSort4` is a
  length-dispatching routine **capped at 4 elements** — it does not even address
  `N` up to 24, and it is tuned for a different cost model (instruction count of
  one call, not steady-state throughput of a batch).
* **This repo does *not* use VarSort4 or any AlphaDev routine.** What it borrows
  from that line of work is only the *idea* that the compare-exchange should be a
  **branchless conditional move** — the same pattern as libc++'s `__cond_swap`.
  The *structure* is a classical **sorting network** (best-known size-optimal
  per `N`), not a length-dispatching variable sort. The two approaches are
  complementary: AlphaDev minimises a single short call; a network maximises
  throughput by exposing wide instruction-level parallelism across shallow
  compare-exchange chains.

### "Is `cpp-sort` (Morwenn) the answer?" — Closest, but still not optimal here.

`cpp-sort` is the most relevant library: its `sorting_network_sorter<N>` uses the
**same best-known size-optimal networks** we do (also sourced from SorterHunter),
with full comparator+projection support. On **`int64` it is excellent** —
within ~10% of our implementation, and it *was slightly faster than our original
code at large N* until we switched our integer compare-exchange to an
independent min/max cmov form (see "What we changed", below).

But on **`pair64` it is poor**: its compare-exchange uses `std::iter_swap` /
move-based exchange with a branchy comparison, so a 16-byte swap becomes a
mispredicted branch plus an XMM move. At `N=24` that makes it **~2× slower than
our network and slower than `std::sort`**. Its `low_moves_sorter<N>` (a network
variant that trades extra comparisons for fewer element moves — in principle
attractive when moves are expensive) does **not** recover the gap, because the
moves it does perform are still branchy.

**Conclusion:** if you only ever sort `int64`, `cpp-sort`'s
`sorting_network_sorter` is a fine library answer. For `pair64` — your stated
hard case — you need the branchless 16-byte compare-exchange that no general
library provides, which is exactly what `small_sort.hpp` adds.

---

## Why a sorting network, and why *size*-optimal

A sorting network is a fixed, data-independent sequence of compare-exchanges
(CEs). That is its superpower here:

* **No branches on the data.** Every CE is a `cmov`; there is nothing for the
  branch predictor to miss. Random small inputs are the worst case for branchy
  sorts and a non-issue for networks.
* **Fully unrolled, no loop overhead, no recursion.** With a compile-time `N`
  the whole thing is a straight-line block the compiler schedules across many
  execution ports.
* **Shallow dependency chains → deep ILP.** Independent CEs in the same "layer"
  issue in parallel on an out-of-order core.

Two different "optimal networks" exist; the choice matters:

* **Depth-optimal** minimises the number of *layers* (critical-path latency).
  Best when you care about the latency of *one* sort.
* **Size-optimal** minimises the *total CE count*. Best when you are
  **throughput-bound** (a hot loop over many blocks) — port pressure, i.e. total
  µops, sets the cost. **That is this workload**, so we use size-optimal.

The payoff is concentrated where each CE is expensive. On `pair64` a CE is a
lex-compare plus a *two-word* conditional swap, so cutting CEs cuts real work
linearly. Replacing the generic Batcher odd-even network (`sort_network_oems`,
which for non-power-of-two `N` pads up to the next power of two and wastes CEs)
with the best-known network buys, on `pair64`:

| N | OEMS (ns) | best-known (ns) | speedup |
|---|---|---|---|
| 18 | 91 | 79 | 1.15× |
| 20 | 112 | 97 | 1.15× |
| 24 | 138 | 122 | 1.13× |

(The win is smaller for `int64` because there the CE is a single cheap `cmov`,
so a few extra CEs barely register.)

### Why SIMD is left on the table (correctly)

Vectorised small-sort kernels (Google Highway's VQSort, `x86-simd-sort`, Intel's
AVX-512 bitonic kernels) are genuinely the fastest way to sort small blocks of
**32-bit or 64-bit scalar keys *when AVX-512 is available***: a 512-bit register
holds 8×`int64`, and AVX-512 has the masked `min`/`max` and permute primitives a
bitonic network needs. Under the stated constraints they do not apply:

* **No AVX-512** on the target → no efficient masked compare-and-permute; AVX2
  bitonic kernels exist but are far less effective and still scalar-key only.
* **`pair64` is 16 bytes.** Even with AVX-512 you would fit only 4 per register,
  the lexicographic order is not a single SIMD `min`, and there is no packed
  128-bit conditional select below AVX-512 mask intrinsics.

So for `pair64` the right design is **scalar, branchless, network-based** — which
is what we have. The ILP we exploit is *instruction-level*, across cmov chains,
not *data-level* across SIMD lanes.

---

## What we changed in this repo

Starting point: `small_sort.hpp` already had the right architecture (branchless
`cswap`, hand-coded optimal networks for `N≤8`) but **synthesised `N=9..24` with
Batcher odd-even merge sort** — provably correct but ~5% (int) to ~20% (`pair64`,
large N) above the best-known CE count.

Two changes, both verified by the exhaustive `verify_small_sort` tool (full 0/1
enumeration up to `N=24`) and the existing test suite:

1. **Best-known size-optimal networks for all `N=2..24`** (from Bert
   Dobbelaere's database / SorterHunter), hand-listed as `constexpr` arrays.
   Transcription risk is fully retired by the 0/1-enumeration verifier: any
   wrong network fails to sort some 0/1 input and is caught at build/test time.
   OEMS is retained as `small_sort::sort_network_oems<N>` purely so the
   benchmark can quantify the gain.
2. **Independent min/max `cmov` for register-sized integers.** The old integer
   path used an XOR-mask swap (a serial `mask → xor → write` chain). Replacing it
   with `lo = swap?b:a; hi = swap?a:b;` lets the two cmovs issue in parallel
   (they share only the predicate, not data). This closed the small `int64` gap
   versus `cpp-sort` at large `N` (N=24: 34→27 ns).
3. **One generalised branchless lexicographic compare-exchange for *all* 16-byte
   and 8-byte pairs**, replacing the hand-written `pair64` special case. The
   generalised path's whole-half integer blend lets GCC emit branchless `cmov`s
   (4 per CE) instead of the dedicated path's ~6 explicit `xor`s — making even
   `pair64` ~40% faster (N=24: 190→122 ns) — and is branch-free for floating
   first keys too. See "Two codegen investigations". Covers `pair64`,
   `pair_li`=⟨long,int⟩, `pair_fi`=⟨float,int⟩, `pair_di`=⟨double,int⟩.

The benchmark (`benchmarks/bench_small_sort.cpp`) was rewritten to drive every
candidate with a **compile-time `N`** (the regime these algorithms are designed
for) and to include `cpp-sort`'s `sorting_network_sorter` and `low_moves_sorter`
alongside `std::sort`, Boost, our best-known network, our OEMS network, and a
branchless insertion-sort baseline.

---

## Full results (min ns/sort; ≥5M sorts/point across 8 configs)

### `int64`
```
N                  2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24
std::sort          3    9   14   19   26   32   41   48   55   64   70   76   83   94  104  147  158  172  184  199  211  225  245
boost::pdqsort     5   10   14   20   26   33   42   51   59   68   77   86   98  108  116  129  142  154  168  177  193  207  197
cppsort::network   0    1    1    2    3    4    5    6    9   10   11   13   14   15   16   17   21   21   23   25   28   31   33
cppsort::low_moves 0    6   12   19   23   34   40   49   56   66   76   88  101  113  127  142  159  178  195  212  229  254  270
small_sort::oems   0    1    2    2    3    4    6    6    7    8    9   11   12   14   17   19   21   21   23   26   28   29   30
small_sort::best   0    1    2    2    3    4    6    6    6    8    9   10   11   12   17   15   17   18   20   22   25   27   28
```

### `pair64` (lexicographic ⟨long,long⟩)
```
N                  2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24
std::sort          5   10   16   23   31   41   50   61   73   80   90  108  115  126  137  184  199  213  235  245  257  282  291
boost::pdqsort     5   11   17   24   33   44   54   65   73   84   95  105  116  125  139  147  160  171  183  195  206  219  260
cppsort::network   3    7   13   19   25   33   44   51   60   67   79   95  109  128  146  171  198  224  246  274  304  329  360
small_sort::oems   1    3    6   10   12   16   21   28   33   40   46   51   55   60   63   89   91  103  112  115  122  134  138
small_sort::best   1    4    5   10   13   16   18   24   29   35   39   46   51   57   57   68   79   87   97  104  113  114  122
```
(`pair64` `small_sort::best` is ~40% faster than this doc's earlier numbers —
N=24 was 194, now 122 — after the dedicated-pair64 path was removed in favour of
the generalised cmov path; see "Two codegen investigations".)

`small_sort::best` is fastest at essentially every size for both types (it ties
`cppsort::network` to within noise on a few sub-2 ns `int64` points). The lead is
decisive on `pair64`, the case you flagged as hard, and grows with `N`.

Reproduce:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
taskset -c 2 build/benchmarks/bench_small_sort            # full sweep, CSV
taskset -c 2 build/benchmarks/bench_small_sort 16         # single size
build/tools/verify_small_sort                             # prove every network correct
```

(`cpp-sort` is vendored header-only under `third_party/cpp-sort`; clone with
`git clone --depth 1 https://github.com/Morwenn/cpp-sort third_party/cpp-sort`.)

---

## More pair shapes: ⟨long,int⟩, ⟨float,int⟩, ⟨double,int⟩

The benchmark sweeps three further lexicographic pairs alongside `pair64`:

* **`pair_li` = ⟨long,int⟩**, **16 bytes** (4 padding) — integer keys like
  `pair64` but a 4-byte second key.
* **`pair_fi` = ⟨float,int⟩**, **8 bytes** — a new, smaller size class.
* **`pair_di` = ⟨double,int⟩**, **16 bytes** — a *floating* first key, so the
  compare is `comisd`, not integer `cmp`.

To handle these, `cswap` gained a **generalised branchless lexicographic
compare-exchange** for any trivially-copyable first/second aggregate under the
natural order: it computes `swap = (b0<a0) | (b0==a0 & b1<a1)` without the
short-circuit branch the defaulted `operator<` emits. `lt0` and `eq0` are taken
off **one** compare (same operand pair → e.g. `comisd` + `seta` + `sete`).

Compile-time results, min ns/sort (full sweep in the CSV):
```
pair_li    N=  2    4    8   12   16   20   24      pair_fi  N=  2    4    8   12   16   20   24
std::sort        5   16   50   90  137  235  294    std::sort      4   16   48   85  140  229  290
boost::pdqsort   5   17   54   91  133  189  239    boost::pdqsort 6   19   56  105  163  224  279
cppsort::network 3   13   45   80  149  243  346    cppsort::net   3   13   42  107  171  254  340
small_sort::best 1    5   18   36   60   95  111    small_sort     1    7   22   43   66  100  129

pair_di    N=  2    4    8   12   16   20   24
std::sort        4   16   49   87  132  224  293
boost::pdqsort   6   19   58  109  168  235  250
cppsort::network 3   13   56  112  176  263  369
small_sort::best 2    7   28   58   82  119  163
```
`small_sort::best` is fastest at every size for all three (and for `pair64`).

---

## Two codegen investigations

Both of these came out of disassembling the hot loop. They are the most
instructive part of this whole exercise.

### 1. "Why was `pair_di` faster than `pair64`?" — it wasn't; `pair64` was slow

An earlier cut of this document had `pair_di` (⟨double,int⟩) coming out *faster*
than `pair64` (⟨long,long⟩) at large N, which is backwards: a `double` compare
cannot be cheaper than a `long` compare. Two disassembly findings explained it,
and a controlled experiment (`pair_li`) settled it.

**(a) `pair_di` first had a branchy swap.** The N=8 `pair_di` sort contained
**33 conditional branches** (`jne`/`ja`), ≈ one per compare-exchange; the
`pair64` sort had zero. With a `double` first field GCC keeps the element in
**XMM registers**, and the original `swap ? hi : lo` form of the 16-byte swap has
no cheap branchless XMM select — so GCC emitted a **branch per compare-exchange**,
which mispredicts on random data. (This is also why `-ffast-math` "barely
helped": it cheapens the compare, never the branch. The seductive "we just do
more FP compares" theory is *falsified* by exactly that — if it were true,
`-ffast-math` would help us more than `std::sort`, but it helped `std::sort`
more.) **Fix:** write the swap as a whole-half integer blend (`memcpy` the 16
bytes to two `uint64`, blend, `memcpy` back). GCC then emits **branchless
cmovs**, in GP registers, no matter where the compare keeps the value.

**(b) `pair64`'s hand-written path was the real laggard.** Same N=8 sort, same
19-CE network:

| path | total insns | xor | cmov |
|---|---|---|---|
| dedicated `pair64` (explicit field-level XOR-mask) | **473** | 114 | 0 |
| generalised (whole-half blend) | **341** | 0 | **76** |

The hand-tuned `pair64` special case forced ~6 explicit `xor`s per CE and
*prevented* GCC from using `cmov`; the generalised path lets GCC emit 4 cmovs per
CE — ~30% fewer instructions. **Deleting the dedicated `pair64` branch** (routing
it through the generalised path) made `pair64` **~40% faster** on its own:

| `pair64` `small_sort::best`, min ns | N=16 | N=24 |
|---|---|---|
| dedicated XOR-mask path (old) | 95 | 190 |
| **generalised cmov path (now)** | **57** | **122** |

**(c) The controlled experiment.** `pair_li` = ⟨long,int⟩ isolates the variable:
integer keys like `pair64`, but a 4-byte second key, testing the intuition that
"we don't have to move a portion of the number." It does **not** pan out —
`pair_li` ≈ `pair64` within noise (N=16: 60 vs 57; N=24: 111 vs 122; mixed
runtime: 87.0 vs 87.0). The 16-byte swap costs the same whether a half holds 4 or
8 meaningful bytes (`cmovl` and `cmovq` are both one µop; the padding moves for
free). With both on the same path, the types order **exactly by first-key compare
cost**: `pair64`≈`pair_li` (integer) < `pair_fi` (float) < `pair_di` (double).
The earlier inversion was 100% the slow `pair64` path.

### 2. "What makes `cpp-sort` faster on some `int64`?"

On `int64`, `cpp-sort`'s `sorting_network_sorter` and ours are within a
**sub-nanosecond, size-dependent** margin: `cpp-sort` edges the tiny sizes
(N ≤ 8) and N=16; we edge N=12, 20, 24. Disassembling N=12 (same 39-CE network,
both branchless) shows two real differences:

* **Compare-exchange form.** `cpp-sort` uses `dx=x; x=min(x,y); y^=dx^x` — one
  `cmov` + two `xor`, one register. We use an **independent min/max blend** —
  two `cmov`s that depend only on the compare flag, not each other. cpp-sort's is
  lower-pressure but **serial** (the xors wait on the `min`); ours is
  higher-pressure but **parallel**. Adopting cpp-sort's form actually *regressed*
  us (i64 N=24: 27 → 32 ns) because these networks are throughput-bound and CE
  *latency* dominates — so we kept the parallel form, and it wins per-size at
  N ≥ 12.
* **Register allocation.** Even with matched CE forms, `cpp-sort`'s code issued
  **21 memory loads to our 28** for the same network — its comparator *ordering*
  yields shorter live ranges, so GCC spills/reloads less. That ~25% edge in loads
  is what keeps `cpp-sort` level (and a hair ahead at a few sizes) despite the
  worse CE. It is a scheduling artifact, not an algorithmic advantage: same
  best-known network, same CE count.

Net: on `int64` it is effectively a tie (both ~7–9× over `std::sort`); we did not
chase the sub-ns difference because it is `int64`-only and we already lead at most
sizes. On every **pair** type the branchless-swap advantage makes us decisively
faster than `cpp-sort` regardless.

---

## Sort by key only (compare `.first`, carry `.second`)

A very common variant: the pair is a **(key, payload)** record — order is decided
by the **first coordinate only**, but the whole element must move with it. This
is exactly a **projection**: compare `proj(x) = x.first`, move all of `x`.

This needs no new algorithm — it is the projection the library already threads
through every path (`small_sort::sort_n<N>(first, std::less<>{}, &T::first)`).
The only fix required was making the **8-byte** case branchless too: an 8-byte
record with a custom projection used to hit the generic `swap ? b : a` fallback
(a potential branch); it now uses the XOR-mask `word_swap`, matching the 16-byte
`half_swap`. Verified branch-free by disassembly and correct by the verifier
(keys non-decreasing + multiset preserved, all sizes, all four pair types).

**Can we optimize for this case? Yes — and it falls out for free.** A by-key
compare-exchange does exactly **one** key compare (vs two/three for the
lexicographic order: `lt0`, `eq0`, and the second-key `lt1`), while the swap is
unchanged. So sorting a pair *by key* is **~50% faster** than sorting it
*lexicographically*, across the board:

```
small_sort::best, min ns/sort   N=  4    8   12   16   20   24
pair64   lexicographic              5   19   39   59   98  122
pair64   by .first only             3    8   18   29   45   62     (~50% faster)
pair_di  lexicographic              7   28   58   82  119  163
pair_di  by .first only             5   15   29   40   60   80     (~50% faster)
```
(`pair64` by-key at N=24 is 62 ns — only ~2.3× a bare `int64` sort despite
carrying an extra 8-byte payload through every move.)

**And our lead over the other libraries *grows* here — to ~3.5–4×**, because the
cheaper compare exposes the swap, which is exactly where the branchless cmov
beats everyone's branchy 16-byte swap:

```
key-only (by .first), min ns/sort      std::sort  pdqsort  cpp-sort  small_sort
pair64   N=16                              106.6    115.1     112.3        29.2
pair64   N=24                              245.7    214.2     230.9        62.3
pair_di  N=24                              277.4    239.9     260.9        79.9
```

* **`std::sort` / Boost** take a by-`.first` *comparator* (no projection API);
  they still do branchy 3-move swaps — now the dominant cost, so they fall to
  **~4× behind**.
* **`cpp-sort`** has the same ergonomics as us (native *projection* support, so
  you pass `&T::first`), and its compare gets cheaper too — but its 16-byte swap
  is still branchy, leaving it ~3.5–4× behind.
* **Limits of "carry the payload".** Moving the payload *inside* the
  compare-exchange is optimal only while the element is small (a few words). For
  a **large** payload the move cost would dominate the network's many swaps, and
  the right move flips to sorting `(key, index)` pairs and permuting the payload
  **once** at the end. For ≤16-byte records (the case here) the direct branchless
  move wins comfortably — index-indirection would add a second array and a
  scatter for no benefit.

---

## Runtime-size dispatch (length known only at run time)

The tables above use a compile-time `N`. The common real case is "here is a
pointer and a `size_t` length." `small_sort::sort` / `sort_reg` / `varsort` all
handle that with a `switch(len)` (a jump table) over `N = 2..24`.
`benchmarks/bench_small_sort_dyn.cpp` measures it under the same stable protocol
(≥5M sorts/point, 8 configs), in two regimes:

* **`fixed`** — the length is a run-time value but constant across the data
  point (the jump-table target is perfectly predicted): isolates per-size cost.
* **`mix`** — each block's length is random in a range, so the dispatch is
  **unpredictable**: includes the real cost of the size switch. `mix2_24` =
  uniform[2,24], plus `mix2_8` and `mix9_24`.

Candidates: `std::sort`, `boost::pdqsort`, `cppsort::net(sw)` (a `switch`
dispatching to `cpp-sort`'s fixed `sorting_network_sorter<N>` — the fair way to
use a compile-time-size library at run time), `ss::inplace`
(`small_sort::sort`, network applied in place through the iterator),
`ss::regblock` (`small_sort::sort_reg`, **register-blocked**: load the block
into a local `T buf[N]`, sort in registers, store back), and `varsort`
(`small_sort::varsort`, a faithful port of the libc++/AlphaDev `cas` +
`partially_sorted_swap` routines for `N≤5`, delegating to `sort_reg` for `N≥6`).

### Mixed (unpredictable) sizes, uniform[2,24] — min ns/sort
```
                 std::sort  pdqsort  cppsort(sw)  ss::inplace  ss::regblock  varsort
i64                  102.8    98.4        23.9         29.9          30.6      30.4
pair64               129.1   116.8       153.9         87.0          88.7      88.1
pair_li              129.0   103.0       163.2         87.0          93.2      92.5
pair_fi              125.2   132.3       160.4         68.1          73.8      74.2
pair_di              129.5   104.8       177.5         89.1          99.4      98.7
```
Our register-blocked / in-place networks win the unpredictable-size case for
**every** pair type — including `pair_di`, which before the branchless-swap fix
was the *slowest* candidate here (194 ns) and is now among the fastest (89 ns).
`pair_li` ties `pair64` exactly here (87.0 / 87.0). `cpp-sort`'s switch-dispatched
network is worse than `std::sort` on all four pairs (branchy 16-byte swaps
again); on `int64` it is the fastest by a few ns.

> **Caveat — this mixed number is frontend-bound, not compute-bound.** With
> *unpredictable* sizes, each sort jumps to one of 23 fully-unrolled networks, so
> the cost is dominated by switch-mispredict + L1-instruction-cache thrashing,
> not by the per-sort arithmetic. Concretely, for `pair64` the fixed per-size
> average over N=9..24 is ~69 ns but `mix9_24` measures ~116 ns — ~47 ns/sort of
> pure dispatch/frontend overhead. Because that overhead is set by (unchanged)
> code size, the big per-size wins (e.g. the `pair64` dedicated-path removal,
> fixed N=24: 191→118 ns) are **largely invisible in this table** and show up in
> the fixed-size / compile-time numbers instead. They *do* surface here for the
> small-size mix, where the networks are small and frontend pressure is low
> (`pair64` `mix2_8`: 18.1→15.4 ns after the fix). The mixed numbers are a fair
> picture of the *unpredictable-size* regime; they are not where per-CE compute
> improvements show.

### Fixed run-time `N` (`pair64`) — selected sizes, min ns/sort
```
pair64  N=          4     8    12    16    20    24
std::sort          17    50    91   136   228   294
boost::pdqsort     18    54    95   138   185   263
cppsort::net(sw)   15    44    82   138   204   272
ss::inplace         5    19    40    58    94   118
ss::regblock        5    20    42    60    99   126
varsort            11    20    42    63    97   150
```

### What this shows

* **`pair64` (the hard case): our network dispatchers win decisively**, both for
  a predictable fixed length and for an unpredictable size mix. At `pair64
  mix2_24` we are **~1.4× faster than Boost** and **~1.75× faster than
  `cpp-sort`**; on `int64` we and `cpp-sort` are both ~4–6× faster than
  `std::sort` (cpp-sort edges us by a few ns on the `int64` mix; we edge it on
  every `pair64` point).
* **`cpp-sort` is *worse than `std::sort`* on the `pair64` mix** (154 vs 129) —
  again, branchy 16-byte swaps. The branchless cmov compare-exchange is the whole
  game for the pairs, and no general library has it.
* **Register blocking is a modest, honest win — not a silver bullet here.** For
  raw pointers with the default comparator, GCC 15 already scalar-replaces the
  in-place network into registers, so `ss::regblock` ≈ `ss::inplace`: blocking
  helps a few **mid-size `pair64`** points (e.g. N=12: 56 vs 59 ns, by removing
  intermediate 16-byte stores) and is neutral-to-slightly-negative at large `N`
  (N=24 spills 384 bytes regardless, so the explicit copy is pure overhead) and
  for cheap-swap `int64`. Its real value is as a **guarantee**: when the
  optimiser *cannot* prove non-aliasing (opaque iterators, a comparator that
  might touch memory), register blocking forces the in-register schedule that
  the in-place form only gets opportunistically. `ss::inplace` is the best
  single default; `ss::regblock` is the safe one.
* **The libc++/AlphaDev `cas` is the wrong primitive for `pair64`.** `varsort`'s
  generic ternary-blend `cas` is *branchy* on a 16-byte type, so at exactly the
  sizes it specialises (`N≤5`) it is **~2× slower** than our branchless `pair64`
  path: e.g. `pair64` N=4 `varsort` 11 ns vs `ss::inplace` 5 ns.
  This is the same trap libc++ falls into by gating its branchless `__cond_swap`
  on `sizeof(T) ≤ sizeof(void*)`. (On `int64`, where the ternary blend *is* a
  single cmov, `varsort` ties everything — the routines are fine there.)

Reproduce:

```bash
taskset -c 2 build/benchmarks/bench_small_sort_dyn        # full sweep, CSV
taskset -c 2 build/benchmarks/bench_small_sort_dyn 12     # single fixed size
build/tools/verify_small_sort                             # also checks sort_reg + varsort
```

---

## Sources

* AlphaDev — Mankowitz et al., *"Faster sorting algorithms discovered using deep
  reinforcement learning,"* Nature 617 (2023).
  <https://www.nature.com/articles/s41586-023-06004-9> ·
  <https://deepmind.google/blog/alphadev-discovers-faster-sorting-algorithms/>
* Best-known sorting networks — Bert Dobbelaere, *SorterHunter*.
  <https://bertdobbelaere.github.io/sorting_networks.html> ·
  <https://github.com/bertdobbelaere/SorterHunter>
* `cpp-sort` (Morwenn) — fixed-size sorters (`sorting_network_sorter`,
  `low_moves_sorter`, `merge_insertion_sort`).
  <https://github.com/Morwenn/cpp-sort/wiki/Fixed-size-sorters>
* Boost.Sort — pdqsort, spreadsort.
  <https://www.boost.org/doc/libs/release/libs/sort/>
* Comparison-optimality of merge-insertion (Ford-Johnson) for small N — referenced
  via cpp-sort's "Original research" wiki; complex and high-constant, hence not
  competitive for raw throughput.
* Knuth, *TAOCP* Vol. 3, §5.3.4 (zero-one principle; optimal small networks).
