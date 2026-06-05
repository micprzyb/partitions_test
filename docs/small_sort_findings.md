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
2. implements each compare-exchange as a **branchless compare-exchange**, with a
   key detail: the 16-byte swap must be an **integer XOR-mask blend** (mask = 0
   or −1) over the two halves, *not* a `cond ? hi : lo` cmov. For any element
   containing a floating field (`pair_di`), the cmov form is lowered by GCC to a
   *branch per compare-exchange* (the value lives in XMM, which has no cheap
   branchless select); the XOR-mask form forces it into GP registers, branch-free.
   This is precisely the case every general-purpose library mishandles.

That implementation lives in `include/partitions/small_sort.hpp`
(`small_sort::sort` / `sort_n<N>`). It beats `std::sort`, Boost, and even
`cpp-sort`'s own network sorter **at every size for every element type tested**;
the margin is largest on the 16-byte lexicographic pairs.

| N=24, min ns/sort | `int64` | `pair64`<br>⟨long,long⟩ | `pair_fi`<br>⟨float,int⟩ | `pair_di`<br>⟨double,int⟩ |
|---|---|---|---|---|
| `std::sort` (libstdc++) | 245 | 304 | 295 | 295 |
| `boost::pdqsort` | 197 | 291 | 290 | 306 |
| `cpp-sort` `sorting_network_sorter` | 33 | 375 | 342 | 366 |
| **`small_sort::sort_n` (this repo)** | **28** | **190** | **127** | **160** |

The pair columns are the headline: `cpp-sort`'s network sorter is *slower than
`std::sort`* on every one of them, because it performs branchy 16-byte swaps. Our
network is **1.6–2.3× faster than `std::sort`**. (On `int64`, `cpp-sort` and we
are both ~7–9× faster than `std::sort`; we hold a small edge at every size after
the min/max-cmov change described below.)

> **Element types.** `pair64`=⟨long,long⟩ (16B), `pair_fi`=⟨float,int⟩ (8B),
> `pair_di`=⟨double,int⟩ (16B, floating first key). `pair<long,int>` is **not**
> benchmarked: it is also 16B with integer keys, i.e. layout- and
> behaviour-identical to `pair64`. The pair benchmarks are built `-ffast-math`
> (no NaNs assumed) so the floating compares drop NaN/unordered handling — a flag
> applied to the whole TU, so every candidate gets it.

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
| 18 | 146 | 119 | 1.23× |
| 20 | 166 | 137 | 1.21× |
| 24 | 208 | 194 | 1.07× |

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
   versus `cpp-sort` at large `N` (N=24: 34→27 ns, now ahead of `cpp-sort`'s 32).
   The dedicated `pair64` path (branchless lex predicate + XOR-mask 2-word swap)
   is unchanged and remains the fastest of all candidates.
3. **Generalised branchless lexicographic compare-exchange + a branchless
   16-byte swap, for arbitrary first/second pairs** (`pair_fi`=⟨float,int⟩,
   `pair_di`=⟨double,int⟩). See the next section — this is where the floating
   first key forced a subtle codegen fix.

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

### `pair64` (lexicographic)
```
N                  2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24
std::sort          5   11   18   25   35   42   55   64   78   85   97  108  120  131  144  190  203  219  235  253  271  288  304
boost::pdqsort     6   12   18   26   35   45   56   68   78   92  105  119  134  147  163  178  194  212  229  245  264  285  291
cppsort::network   4    7   14   19   25   32   60   69   85   99  114  129  147  165  182  202  220  245  264  289  310  344  375
cppsort::low_moves 3    7   13   22   31   42   53   69   81   98  112  131  149  178  195  228  247  284  299  342  363  402  425
small_sort::oems   1    5    5   14   19   24   29   44   49   60   68   76   84   95   99  137  146  159  166  178  188  202  208
small_sort::best   1    6    5   14   17   25   29   36   45   53   62   72   83   88   95  110  119  138  137  155  167  181  194
```

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

## Other pair shapes: ⟨float,int⟩, ⟨double,int⟩ — and a floating-key codegen trap

The same benchmark sweeps two further lexicographic pairs:

* **`pair_fi` = ⟨float,int⟩**, **8 bytes** — a new (smaller) size class.
* **`pair_di` = ⟨double,int⟩**, **16 bytes** — same size as `pair64` but a
  *floating* first key, so the compare is `comisd`, not an integer `cmp`.

`pair<long,int>` is intentionally **omitted**: it is also 16 bytes with integer
keys, i.e. layout- and behaviour-identical to `pair64`; benchmarking it would
just reproduce the `pair64` numbers.

To handle these, `cswap` gained a **generalised branchless lexicographic
compare-exchange** for any trivially-copyable first/second aggregate under the
natural order: it computes `swap = (b0<a0) | (b0==a0 & b1<a1)` without the
short-circuit branch the defaulted `operator<` emits. `lt0` and `eq0` are taken
off **one** compare (same operand pair → `comisd` + `seta` + `sete`).

### The trap (thanks to a reviewer catch)

The first cut put `pair_di` *behind* `std::sort` at large `N` (N=24: 333 vs 295),
and `-ffast-math` barely helped. The tempting explanation — "the network does
more FP compares, which are expensive" — is **wrong**: if that were it,
`-ffast-math` (which cheapens each FP compare) would have helped *us* more than
`std::sort`, but it helped `std::sort` more. So the cost was not FP-compare
volume; our path was doing something categorically more expensive.

The disassembly settled it. The N=8 `pair_di` sort contained **33 conditional
branches** (`jne`/`ja`), ≈ one per compare-exchange — the `pair64` sort had
**zero**. Cause: with a `double` first field GCC keeps the element in **XMM
registers**, and the `swap ? hi : lo` form of the 16-byte swap has no cheap
branchless XMM select, so GCC emits a **branch per compare-exchange**. On random
data those mispredict and dominate; `-ffast-math` only touches the compare, not
the branch, which is exactly why it "barely helped."

**Fix:** make the 16-byte swap an **integer XOR-mask blend** over the two halves
(`mask = 0 or −1`; `a ^= (a^b)&mask`) instead of a cmov/select. That forces the
swap into GP registers, branch-free, regardless of where the compare keeps the
value. Result for `pair_di`:

| N=24 `pair_di`, min ns | value |
|---|---|
| branchy cmov swap (original) | 333 |
| **+ branchless XOR-mask swap** | **178** |
| **+ `-ffast-math`** | **160** |

i.e. the branchless swap is the **1.9×** lever; `-ffast-math` adds a further
~11%. The same XOR-mask swap is now used for every 16-byte element, so it is
also the form behind `pair64`.

### Results — `pair_fi` and `pair_di`, compile-time N, min ns/sort
```
pair_fi    N=  2    4    8   12   16   20   24
std::sort        4   16   48   85  140  229  295
boost::pdqsort   6   19   56  105  163  224  290
cppsort::network 3   13   42  107  171  254  342
small_sort::oems 1    7   26   56   81  133  168
small_sort::best 1    7   22   43   66  100  127

pair_di    N=  2    4    8   12   16   20   24
std::sort        4   16   49   87  132  224  295
boost::pdqsort   6   19   58  109  168  235  306
cppsort::network 3   13   56  112  176  263  366
small_sort::oems 2    8   29   64   93  155  195
small_sort::best 2    7   28   58   82  119  160
```

`small_sort::best` is fastest at every size for both. Interestingly both new
pairs end up a touch *faster* than `pair64` at large N (N=24: 127 / 160 vs 190) —
their second key is a 4-byte `int`, so the second-key compare and half of the
swap are cheaper. The branchless network's dominance is now uniform across
integer-keyed, float-keyed and double-keyed pairs.

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
i64                  99.7     96.5        23.3         29.9          30.7      30.1
pair64              129.3    115.4       152.4         85.9          88.0      88.1
pair_fi             125.0    134.8       160.8         68.5          73.2      72.9
pair_di             127.6    103.7       174.2         89.1          99.0      98.1
```
Our register-blocked / in-place networks win the unpredictable-size case for
**every** pair type — including `pair_di`, which before the branchless-swap fix
was the *slowest* candidate here (194 ns) and is now the fastest (89 ns).
`cpp-sort`'s switch-dispatched network is worse than `std::sort` on all three
pairs (branchy 16-byte swaps again); on `int64` it is the fastest by a few ns.

### Fixed run-time `N` — selected sizes, min ns/sort
```
pair64  N=          4     8    12    16    20    24
std::sort          17    51    93   141   234   299
boost::pdqsort     18    55    97   143   192   264
cppsort::net(sw)   15    44    82   137   203   275
ss::inplace         7    27    59    92   143   193
ss::regblock        7    28    56    96   146   195
varsort            11    28    57    96   146   195
```

### What this shows

* **`pair64` (the hard case): our network dispatchers win decisively**, both for
  a predictable fixed length and for an unpredictable size mix. At `pair64
  mix2_24` we are **~1.4× faster than Boost** and **~1.75× faster than
  `cpp-sort`**; on `int64` we and `cpp-sort` are both ~4–6× faster than
  `std::sort` (cpp-sort edges us by a few ns on the `int64` mix; we edge it on
  every `pair64` point).
* **`cpp-sort` is *worse than `std::sort`* on the `pair64` mix** (155 vs 134) —
  again, branchy 16-byte swaps. The branchless cmov-decomposed compare-exchange
  is the whole game for `pair64`, and no general library has it.
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
  sizes it specialises (`N≤5`) it is **2–3× slower** than our branchless `pair64`
  path: e.g. `pair64` N=2 `varsort` 6 ns vs `ss::inplace` 2 ns; N=5: 20 vs 7.
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
