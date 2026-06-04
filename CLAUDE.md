# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure          # all 5 test suites
ctest --test-dir build -R test_partition_correctness -V   # one suite

build/benchmarks/bench_partition quick              # bench (sizes capped at 4096)
build/benchmarks/bench_pivot
build/tools/balance_report 200                      # pivot-balance stats
```

Requires CMake ≥ 3.20 and C++23 (developed on GCC 16; CI uses g++-14 and clang++-18 with `-DPARTITIONS_NATIVE=OFF`).

Tests are deliberately built at `-O1` (see `tests/CMakeLists.txt`) regardless of build type — they are a large templated matrix where `-O3` costs minutes of compile time for no correctness benefit. Do not "fix" this.

## Architecture

The library is header-only under `include/partitions/`. Everything composes through three plain-tuple **registries** in `partitions.hpp`:

- `default_partitioners()` — algorithms under test
- `default_pivots()` — pivot-selection strategies
- `default_distributions()` — adversarial inputs

Tests, benchmarks, and the balance report all iterate these tuples via `partitions::for_each`. **Adding an entry to a registry automatically enrolls it in every suite** — no other wiring is needed. This is the central design choice; preserve it when extending.

### The single extension point: PredicatePartitioner

A partition algorithm implements **one** primitive (`concepts.hpp`):

```cpp
template <std::random_access_iterator I, std::sentinel_for<I> S, class Pred>
I operator()(I first, S last, Pred keep_left) const;
```

From that, `partition_api.hpp` derives all four user-facing forms (forward/reverse × by-key/by-position) by varying only the predicate. Don't add separate code paths for the four forms — they collapse to one predicate call.

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

### Test framework

`tests/framework.hpp` is a tiny in-tree framework (no external deps). Tests use `TEST_CASE("name") { CHECK(...); REQUIRE(...); }` and link against `framework_main.cpp`. Each `test_*.cpp` is its own CTest target.

## Style

`.clang-format` is Google base, 4-space indent, 88-column limit, left pointer alignment. CI runs clang-format-18 in advisory mode (won't fail the build, but keep diffs clean).
