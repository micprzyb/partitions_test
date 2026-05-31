# partitions_test

[![CI](https://github.com/micprzyb/partitions_test/actions/workflows/ci.yml/badge.svg)](https://github.com/micprzyb/partitions_test/actions/workflows/ci.yml)

A C++23 test bench for **custom, single-threaded partition functions** —
checking both **correctness** and **performance**, and measuring **pivot
quality** independently of any partition algorithm.

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

Every algorithm — and every algorithm *you* want to test — is expressed as a
single **partition-by-predicate**:

```cpp
struct my_partition {
    static constexpr const char* name = "my_partition";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
    I operator()(I first, S last, Pred keep_left) const {
        // reorder [first,last) so keep_left(*x) elements come first;
        // return the partition point.
    }
};
```

From that primitive the library *derives* all four forms above (forward/reverse
just flip the predicate; key/position just decide how the pivot key is
obtained). Add your type to `default_partitioners()` in
`include/partitions/partitions.hpp` and it is automatically covered by:

* the full correctness sweep (every type × distribution × form × size),
* the throughput benchmark,
* and any report that iterates the algorithm registry.

The same registry pattern applies to **pivot strategies**
(`default_pivots()`) and **input distributions** (`default_distributions()`).

## Repository layout

```
include/partitions/      header-only library
  types.hpp              element types (i32, i64, pair64, keyed) + projections
  concepts.hpp           PredicatePartitioner — the extension point
  algorithms.hpp         std_partition, lomuto, lomuto_branchless, hoare, block
  partition_api.hpp      forward/reverse × key/position adapters
  pivot.hpp              first/middle/last, median_of_{3,5}, ninther,
                         median_of_medians_5, random
  distributions.hpp      the input generators ("difficult cases")
  statistics.hpp         pivot-balance measurement + aggregation
  partitions.hpp         umbrella header + the three registries
tests/                   dependency-free test framework + correctness suites
benchmarks/              steady-clock harness + partition / pivot benchmarks
tools/                   balance_report — pivot-quality statistics
docs/difficult-cases.md  why each adversarial input exists (with references)
cmake/                   warning flags & helpers
```

## Build & run

Requires CMake ≥ 3.20 and a C++23 compiler (developed on GCC 16).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

ctest --test-dir build --output-on-failure        # correctness (4 suites)
```

The correctness suite runs ~100k checks across
`{std_partition, lomuto, lomuto_branchless, hoare, block}` ×
`{i32, i64, pair64, keyed}` × all 13 distributions × {forward, reverse} ×
{by-key, by-position} × sizes `0 … 4096`, verifying both the partition
postcondition and that the multiset of elements is preserved. The pivot-by-key
form is additionally tested with keys **absent** from the block (below-min and
above-max), exercising the empty-side boundaries.

### Benchmarks

```bash
build/benchmarks/bench_partition          # full sweep, sizes 4 … 2^22 (CSV)
build/benchmarks/bench_partition quick    # cap at 4096 for a fast look
build/benchmarks/bench_partition 65536    # cap at a chosen max size
build/benchmarks/bench_pivot              # pivot-selection cost + resulting balance
```

`bench_partition` measures the partition step **in isolation** (the pivot is
chosen once, outside the timed region). Small blocks (4–24) are **batched** so
many partitions are timed together and normalised to `ns_per_elem`; large
blocks (up to 2²²) are timed singly. Output is CSV — pipe to a file and plot.

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

Representative findings reproduced by this bench (your hardware will vary):

* **Branchless Lomuto** is ~3× faster than Lomuto/Hoare/block on small random
  `i32` blocks — the eliminated branch dominates there.
* **median_of_medians_5** gives by far the tightest balance (worst-side split
  near 0.69 vs **1.0** for first/middle/last), at an O(n) selection cost the
  other samplers avoid.
* **all_equal** drives every pivot's `mean_equal` to ~1.0 and imbalance to its
  maximum, illustrating why a 2-way split needs a three-way variant for
  duplicate-heavy data.

## Extending

* **New partition algorithm** → add a `PredicatePartitioner` to `algorithms.hpp`
  and list it in `default_partitioners()`.
* **New pivot strategy** (e.g. your own median scheme) → add a function object
  to `pivot.hpp` and list it in `default_pivots()`; it is benchmarked and
  balance-reported automatically.
* **New input** → add a generator to `distributions.hpp` and list it in
  `default_distributions()`.
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
