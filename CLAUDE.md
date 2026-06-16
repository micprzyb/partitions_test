# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure          # all 12 test suites
ctest --test-dir build -R test_partition_correctness -V   # one suite

build/benchmarks/bench_partition quick              # bench (sizes capped at 4096)
build/benchmarks/bench_pivot
build/tools/balance_report 200                      # pivot-balance stats
```

There is one benchmark per study (`benchmarks/bench_*.cpp`, ~19 of them, all CSV to stdout); each pairs with a findings report in `docs/`. The two comparison benchmarks (`bench_small_sort`, `bench_small_sort_dyn`) are **optional**: they need Boost (`libboost-dev`) and cpp-sort vendored at `third_party/cpp-sort` (not committed — `git clone https://github.com/Morwenn/cpp-sort third_party/cpp-sort`). When either is missing those two targets are silently skipped; everything else builds.

Requires CMake ≥ 3.20 and C++23 (developed on GCC 16; CI uses g++-14 and clang++-18 with `-DPARTITIONS_NATIVE=OFF`).

Tests are deliberately built at `-O1` (see `tests/CMakeLists.txt`) regardless of build type — they are a large templated matrix where `-O3` costs minutes of compile time for no correctness benefit. Do not "fix" this.

## Architecture

The library is header-only under `include/partitions/`. Everything composes through three plain-tuple **registries** in `partitions.hpp`:

- `default_partitioners()` — algorithms under test
- `default_pivots()` — pivot-selection strategies
- `default_distributions()` — adversarial inputs

Tests, benchmarks, and the balance report all iterate these tuples via `partitions::for_each`. **Adding an entry to a registry automatically enrolls it in every suite** — no other wiring is needed. This is the central design choice; preserve it when extending.

### The single extension point: PivotPartitioner

A partition algorithm implements **one** primitive (`concepts.hpp`), threading the comparator and projection directly (like the small sorters in `small_sort.hpp`):

```cpp
template <std::random_access_iterator I, std::sentinel_for<I> S,
          class K, class Comp, class Proj>
I operator()(I first, S last, K pivot, Comp comp, Proj proj) const;
```

It puts elements with `comp(proj(x), pivot)` first and returns the boundary. A partition only ever evaluates `comp` against the single fixed `pivot`, so `comp` acts as a unary predicate `below(x) = comp(proj(x), pivot)` — strict-weak ordering is irrelevant for the partition step itself.

The pivot is passed **by value**, not `const K&` or `K&&`: a partition permutes the block while comparing against the pivot, so a *reference* pivot makes the compiler assume aliasing and reload it from memory every comparison; by value it stays in a register (verified by disassembly — both `const K&` and `K&&` reload, by value does not). Keys here are small (≤16 bytes), so the copy is free.

From that, `partition_api.hpp` derives all four user-facing forms (forward/reverse × by-key/by-position) by varying only the comparator: forward passes `comp`, reverse passes the negated `ge(a,b) = !comp(a,b)` (which flips which side each element lands on). Don't add separate code paths for the four forms — they collapse to one call.

### Position-aware fast path (`at`)

An algorithm may optionally provide a position-aware overload:

```cpp
template <...> I at(I first, S last, I pivot, Comp comp, Proj proj) const;
```

`partition_by_position` dispatches to `at` when present (used by `algo::hoare_guarded` to drop bound checks via a sentinel). `reverse_partition_by_position` reuses the same `at` on `reverse_iterator`s and maps the boundary back with `.base()`. **Key pivots never use `at`** — a key may be absent from the block and cannot serve as a sentinel.

### Pivot strategies: position OR value

A pivot strategy returns *either*:
- an iterator into the block (a position), or
- `pivot::value_pivot<K>{key}` — a synthetic key that need not occur in the block (e.g. `(min+max)/2`).

`partition_with_pivot` (in `partition_with_pivot.hpp`) is the convention-agnostic glue that dispatches via `pivot::is_value_pivot_v<R>`. A strategy that mutates the block declares `static constexpr bool reorders = true;` (e.g. `median_of_3_inplace`).

### Partition contract

Every form returns iterator `m` such that `[first, m) < pivot` and `[m, last) >= pivot`. `m` is the **boundary index**, not the resting place of any pivot element; this is what lets the by-key form work even when no element equals the pivot (`m == first` means empty left, `m == last` means empty right). Comparisons go through a single `Comp` (a strict-weak-ordering "less"); `>=` is expressed as `!(x < pivot)`.

### Beyond the registry core: the derived algorithm families

The registries cover the partition/pivot/distribution matrix. On top of that primitive sit several self-contained modules, each with its own `test_*.cpp`, `bench_*.cpp`, and `docs/*.md` findings report:

- `quicksort.hpp` — pure-partition, size-adaptive ascending quicksort (no insertion-sort leaf; smallest blocks finish by recursive halving). `quicksort_rev.hpp` is its descending mirror; `quicksort_lr.hpp` is a non-recursive left-to-right variant with a selection-sort leaf.
- `reverse_partition.hpp` (`algo_rev::*`) — partitioners that put `>= pivot` LEFT and `< pivot` RIGHT at identical cost to forward.
- `small_sort.hpp` / `small_halve.hpp` / `small_halve_rev.hpp` / `small_merge.hpp` — branchless comparator networks for n ≤ 24. A *halver* splits by rank (bottom half ≤ top half, halves unordered) in ~20–33% fewer compare-exchanges than a full sorting network.
- `move_low_half.hpp`, `offset_partition.hpp`, `offset_low_half.hpp` — selection-style routines (move/collect the bottom half of below-key elements; partition with a known all-`>=` prefix), built on the reversed family.

### Two ways to reverse — don't mix them up

- **Derived API forms** (`partition_api.hpp`): reversing is just passing the negated comparator `ge(a,b) = !comp(a,b)` — sound because a partition uses `comp` only as a unary predicate.
- **Dedicated reversed partitioners** (`reverse_partition.hpp`): keep the *same* comparator and exchange the roles of the two scans/fills — identical op count to forward, which is what makes `move_low_half` free of its move epilogue.
- **Comparator networks** (`small_halve_rev.hpp`): negating the comparator is **unsound** (the 0/1-principle proofs need a strict order; `ge` is reflexive). Reverse the compare-exchange *direction* instead (`cswap_rev` puts the larger element at the lower index).

### Networks are proved, not trusted

Every sorting/halving network is verified by exhaustive 0/1 enumeration (all 2^N inputs) in `tools/verify_small_sort`, `verify_small_halve`, `verify_small_halve_rev`. A new or edited network entry must pass the corresponding verifier — run it, don't reason about it.

### Test framework

`tests/framework.hpp` is a tiny in-tree framework (no external deps). Tests use `TEST_CASE("name") { CHECK(...); REQUIRE(...); }` and link against `framework_main.cpp`. Each `test_*.cpp` is its own CTest target.

## Style

`.clang-format` is Google base, 4-space indent, 88-column limit, left pointer alignment. CI runs clang-format-18 in advisory mode (won't fail the build, but keep diffs clean).
