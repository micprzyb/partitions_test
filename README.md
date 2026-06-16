# partitions_test

[![CI](https://github.com/micprzyb/partitions_test/actions/workflows/ci.yml/badge.svg)](https://github.com/micprzyb/partitions_test/actions/workflows/ci.yml)

A C++23 test bench for **custom, single-threaded partition functions** —
checking both **correctness** and **performance**, measuring **pivot quality**
independently of any partition algorithm — and a study of what can be built
from partitioning *alone*: pure-partition quicksorts (no insertion-sort leaf),
reversed partitions at zero extra cost, branchless rank-split networks
("halvers"), and selection-style routines. Every study is measured; the
findings live in `docs/`.

A *partition* here reorders a block so that every element strictly smaller than
a pivot precedes every element greater-than-or-equal to it:

```
[ x : x < pivot ]  [ x : x >= pivot ]
                 ^ partition point (returned)
```

It is parameterised by a **comparison** and a **projection** (à la
`std::ranges`), works from an **iterator + sentinel**, and supports four forms:

|              | pivot given as a **key value** (need not be present) | pivot given as a **position** (an iterator into the block) |
|--------------|------------------------------------------------------|------------------------------------------------------------|
| **forward** (`<` left)  | `partition_by_key`          | `partition_by_position`         |
| **reverse** (`>=` left) | `reverse_partition_by_key`  | `reverse_partition_by_position` |

## The one thing you implement

Every algorithm — and every algorithm *you* want to test — models
`PivotPartitioner` (`concepts.hpp`): a single **partition-around-a-pivot**
threading the comparator and projection directly:

```cpp
struct my_partition {
    static constexpr const char* name = "my_partition";

    template <std::random_access_iterator I, std::sentinel_for<I> S,
              class K, class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        // reorder [first,last) so elements with comp(proj(x), pivot) come
        // first; return the partition point.
    }
};
```

The pivot is taken **by value**: a partition permutes the block while comparing
against the pivot, so a reference pivot forces the compiler to assume aliasing
and reload it every comparison; by value it stays in a register (verified by
disassembly). During a partition `comp` is only ever evaluated against the one
fixed `pivot`, so it acts as a unary predicate `below(x) = comp(proj(x), pivot)`
— strict-weak ordering matters for pivot *selection* and sorting, not for the
partition step itself. That is what lets the library *derive* all four forms
above from this one primitive (reverse just negates the comparator; key/position
just decide how the pivot key is obtained).

Add your type to `default_partitioners()` in
`include/partitions/partitions.hpp` and it is automatically covered by:

* the full correctness sweep (every type × distribution × form × size),
* the throughput benchmark,
* and any report that iterates the algorithm registry.

The same registry pattern applies to **pivot strategies**
(`default_pivots()`) and **input distributions** (`default_distributions()`).

### Optional: exploiting the pivot position (sentinels / guards)

Some algorithms are faster when handed the pivot **position** rather than just
its value — e.g. they move the pivot to an end and use it as a sentinel to drop
bound checks from the scan. An algorithm opts into this by additionally
providing

```cpp
template <std::random_access_iterator I, std::sentinel_for<I> S, class Comp, class Proj>
I at(I first, S last, I pivot, Comp comp, Proj proj) const;   // [first,m) < pivot, [m,last) >= pivot
```

`partition_by_position` dispatches to `at` when present (and reuses it on
reverse iterators for `reverse_partition_by_position`), otherwise it falls back
to the key path. Key pivots never use `at`, since a key may be absent from the
block and so cannot serve as a sentinel. `algo::hoare_guarded` and
`algo::boost_block` are worked examples.

## Repository layout

```
include/partitions/      header-only library
  types.hpp              element types (i32, i64, pair64, pair_fi, pair_di,
                         pair_li, keyed) + projections
  concepts.hpp           PivotPartitioner — the extension point
  algorithms.hpp         std_partition, lomuto, lomuto_branchless, hoare,
                         hoare_guarded, block, boost_block, fulcrum,
                         sized (size-dispatching)
  partition_api.hpp      forward/reverse × key/position adapters (+ at dispatch)
  partition_with_pivot.hpp  convention-agnostic glue (position OR value pivot)
  pivot.hpp              position pivots (first/middle/last, median_of_{3,5},
                         ninther, medians-of-medians family, random),
                         value pivots (pseudo-medians, midpoints),
                         reordering variants (*_inplace)
  distributions.hpp      the input generators ("difficult cases")
  statistics.hpp         pivot-balance measurement + aggregation
  partitions.hpp         umbrella header + the three registries
                         --- built on the primitive: ---
  small_sort.hpp         branchless sorting networks, n <= 24
  small_halve.hpp        branchless HALVERS (rank-split networks, n <= 24)
  small_halve_rev.hpp    descending halvers (reversed compare-exchange)
  small_merge.hpp        extend/merge a mostly-sorted small block
  reverse_partition.hpp  algo_rev::* — [>= | <] split at forward cost
  quicksort.hpp          pure-partition size-adaptive quicksort (ascending)
  quicksort_rev.hpp      its descending mirror
  quicksort_lr.hpp       non-recursive left-to-right quicksort, selection leaf
  move_low_half.hpp      move the bottom half of below-key elements to the end
  offset_partition.hpp   forward partition with a known all->= prefix
  offset_low_half.hpp    low-half selection with a known all->= prefix
tests/                   dependency-free test framework + 12 suites
benchmarks/              steady-clock harness + one benchmark per study (CSV)
tools/                   balance_report — pivot-quality statistics
                         verify_small_{sort,halve,halve_rev} — exhaustive 0/1
                         verification of every comparator network
docs/                    measured findings, one report per study
                         (difficult-cases.md explains the adversarial inputs)
cmake/                   warning flags & helpers
```

## Build & run

Requires CMake ≥ 3.20 and a C++23 compiler (developed on GCC 16).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

ctest --test-dir build --output-on-failure        # correctness (12 suites)
```

The core correctness sweep runs all registered partitioners ×
`{i32, i64, pair64, keyed}` × all 13 distributions × {forward, reverse} ×
{by-key, by-position} × sizes `0 … 4096`, verifying both the partition
postcondition and that the multiset of elements is preserved. The pivot-by-key
form is additionally tested with keys **absent** from the block (below-min and
above-max), exercising the empty-side boundaries. The remaining suites cover
the derived algorithms (quicksorts, reversed partitions, low-half/offset
routines, statistics, distributions).

### Benchmarks

There is one benchmark per study (`benchmarks/bench_*.cpp`), each paired with a
findings report in `docs/`. All print CSV — pipe to a file and plot. The core
ones:

```bash
build/benchmarks/bench_partition          # full sweep, sizes 4 … 2^22
build/benchmarks/bench_partition quick    # cap at 4096 for a fast look
build/benchmarks/bench_partition 65536    # cap at a chosen max size
build/benchmarks/bench_pivot              # pivot-selection cost + resulting balance
build/benchmarks/bench_quicksort          # pure-partition quicksort sweep
```

`bench_partition` measures the partition step **in isolation** (the pivot is
chosen once, outside the timed region). Small blocks (4–24) are **batched** so
many partitions are timed together and normalised to `ns_per_elem`; large
blocks (up to 2²²) are timed singly.

The two comparison benchmarks (`bench_small_sort`, `bench_small_sort_dyn`),
which pit our small-array sort against Boost.Sort and cpp-sort, are
**optional**: they need Boost (`libboost-dev`) and cpp-sort vendored at
`third_party/cpp-sort` (not committed — fetch with
`git clone https://github.com/Morwenn/cpp-sort third_party/cpp-sort`). If
either is missing those two targets are skipped and everything else builds.

### Pivot-quality statistics

```bash
build/tools/balance_report          # 200 trials per (distribution, size, strategy)
build/tools/balance_report 1000     # more trials
```

For each pivot strategy it reports the distribution of the **left-side
fraction** (elements strictly `< pivot`): `mean_frac` (0.5 is ideal),
`stddev`, `worst_side` (the worst `max(left,right)` split a recursive sort
would pay for), and `mean_equal` (share of keys equal to the pivot, which a
2-way split cannot balance). This is exactly "how balanced a partition would
this pivot give?", answered without running any partition.

## What the numbers show

Representative findings reproduced by this bench (your hardware will vary;
full data in `docs/`):

* **Branchless Lomuto** is ~3× faster than Lomuto/Hoare/block on small random
  `i32` blocks — the eliminated branch dominates there.
* **median_of_medians_5** gives by far the tightest balance (worst-side split
  near 0.69 vs **1.0** for first/middle/last), at an O(n) selection cost the
  other samplers avoid.
* **all_equal** drives every pivot's `mean_equal` to ~1.0 and imbalance to its
  maximum, illustrating why a 2-way split needs a three-way variant for
  duplicate-heavy data.
* A **halver** — a comparator network that only splits a small block by rank
  (bottom half ≤ top half, each half unordered) — needs ~20–33% fewer
  compare-exchanges than the best known sorting network of the same size, and
  finishing a quicksort's smallest blocks by recursive halving beats recursing
  the partition to size 1 by ~20–35% total sort time
  (`docs/pure_quicksort.md`).
* A **reversed** partition (`>=` left, `<` right) costs exactly the same as a
  forward one when done by exchanging the roles of the two scans — not by
  negating the comparator or reverse iteration
  (`docs/reverse_partition_report.md`). This makes "move the smallest elements
  to the end" free of any post-partition move pass (`docs/move_low_half.md`).

## Extending

* **New partition algorithm** → add a `PivotPartitioner` to `algorithms.hpp`
  and list it in `default_partitioners()`. Optionally add an `at(...)` member
  for a position-aware fast path (see "exploiting the pivot position" above).
* **New pivot strategy** (e.g. your own median scheme) → add a function object
  to `pivot.hpp` and list it in `default_pivots()`; it is benchmarked and
  balance-reported automatically. A strategy may return **either** a position
  (an iterator into the block) **or** a value (`pivot::value_pivot<K>{key}`, for
  a synthetic or untracked pivot such as `(min+max)/2` that need not occur in
  the block). The harness is agnostic: `pivot::pivot_key_of` reads the key from
  either, and `partition_with_pivot` partitions by position or by key
  accordingly. Set `static constexpr bool reorders = true;` if it mutates the
  block.
* **New input** → add a generator to `distributions.hpp` and list it in
  `default_distributions()`.
* **New sorting/halving network** → add the comparator list to
  `small_sort.hpp` / `small_halve.hpp` *and* prove it in the corresponding
  `tools/verify_small_*` tool — every network in the tree is verified by
  exhaustive 0/1 enumeration (all 2^N inputs), not trusted.
* **Multi-threaded partitions** are explicitly out of scope for now; the
  contract and harness assume single-threaded execution.

See `docs/difficult-cases.md` for the rationale and references behind the
adversarial inputs.

## References

* M. D. McIlroy, *A Killer Adversary for Quicksort*, SP&E 29(4), 1999.
* D. Musser, *Introspective Sorting and Selection Algorithms*, SP&E 27(8), 1997.
* Edelkamp & Weiss, *BlockQuicksort: Avoiding Branch Mispredictions in
  Quicksort*, ESA 2016.
* O. Peters, *Branchless Lomuto partitioning* and *Pattern-defeating
  Quicksort* (arXiv:2106.05123).
* B. Dobbelaere, *The smallest and fastest sorting networks for small numbers
  of inputs* (bertdobbelaere.github.io/sorting_networks.html).
* Mankowitz et al., *Faster sorting algorithms discovered using deep
  reinforcement learning* (AlphaDev), Nature 618, 2023.
