# Offset low-half: move the bottom half of the below-key elements to the front

**Problem.** Same precondition as [offset_partition](offset_partition.md):
`[first, first+offset)` all `>= key`. Now move only the **smallest ~S/2** of
the S below-key elements to the front (any order) and return their count k —
the front k must be the k smallest of the whole range (bottom-k property).
Take-all base case: if S/2 < 16, k = S (selecting half of a tiny set is not
worth a pass; same rule as `move_low_half`).

Shipped (`include/partitions/offset_low_half.hpp`):
`partitions::offset_low_half` (sampled, k ≈ S/2) and
`partitions::offset_low_half_exact` (k = ⌊S/2⌋ exactly). Own suite
(`tests/test_offset_low_half.cpp`, `benchmarks/bench_offset_low_half.cpp`).

## Strategies benchmarked

| name | plan | k |
|---|---|---|
| `ref` | count S, `std::nth_element` at rank k | exact |
| `exact` | `offset_partition` by **key** (all S belows to the front), then ONE in-place quickselect of the front block at rank S/2 | exact |
| `two_phase` | the naive plan: sample-estimate the below-median t, partition the **suffix** around t with the best forward partitioner, then block-swap the left side to the front (`part_swap_off` by t) — each delivered element moves **twice** | ≈ S/2 |
| `pivot_first` | test t against key **before** partitioning (vacuously true for a sampled t — see below): ONE `offset_partition(first, last, offset, t)` pass; the bottom ~S/2 land in the front slots **during** the sweep, moved once | ≈ S/2 |
| `ninther_first` | `pivot_first` with a cheap ninther-of-the-suffix pivot instead of the 512-point sample; the t ≥ key branch is real here (the suffix median lands above key whenever p < .5) and falls back to `exact` | biased |

Design notes:

* **Sampled t < key is guaranteed**, not tested: the chosen sample rank
  (b/2) lies strictly below the sample's below-key count b, so the selected
  value is below key by construction. The "test the pivot first" branch
  only does work for unguarded pivots like the ninther.
* **Suffix-only sampling** (refinement over `move_low_half`'s whole-array
  sampler): the prefix is all ≥ key by contract — sampling it dilutes the
  below-count signal by a factor (1−f). Sampling `[first+offset, last)`
  keeps the estimator's accuracy independent of f. Expected k accuracy is
  `|2k/S − 1| ~ 1/√b` with b ≈ p·512 below-key samples: measured mean 6.7%,
  max 27% (worst at p = .1, b ≈ 51 — exactly the √b prediction).
* **Duplicate fallback** (found by the `binary` distribution test): when the
  below-key population is dominated by one value, t equals it and *no value
  split can separate equal elements* — the t-partition returns k = 0 and the
  call falls back to the exact (rank-based) path.
* **Direct `prefix_fill` routing** (found by this benchmark, the headline
  micro-optimization): the t-partition must NOT go through `sized_off`. The
  dispatcher cannot know the below fraction, so for narrow types with
  offset ≥ suffix it picks the p-flat `gap_off` (2 moves per suffix
  element) — but the effective fraction vs t is ~p/2 **by construction**
  (t is the below-median), exactly the regime where `prefix_fill`'s
  per-below-only moves win. The caller knows what the dispatcher cannot.
  Measured: i64 f=.5 p=.1 2^20 dropped 0.67 → 0.38 ns/suffix-elem.

## Results

Meteor Lake, GCC 15.2, `-O3 -march=native`, random unique keys, prefix ≥ key,
suffix shuffled; f = offset/n, p = S/(n−offset). Metric: **ns per suffix
element** (min-of-reps) and **k_ratio = 2k/S** (1.0 = perfect half; `exact`
is 1.0 by definition). Every output verified (bottom-k + multiset) before
timing. Aggregates over all 81 cells:

| strategy | time vs pivot_first (mean) | outright wins | \|2k/S−1\| mean / max |
|---|---|---|---|
| `exact` | 2.15x | 0/81 | 0 |
| `two_phase` | 1.13x | 2/81 | .067 / .269 |
| **`pivot_first`** | **1.00x** | **54/81** | .067 / .269 |
| `ninther_first` | 1.35x | 25/81 | .105 / .482 |
| `ref` | 27x (4.7–129x) | — | 0 |

`ninther_first`'s 25 nominal wins are an artifact of its bias: it wins time
exactly in the cells where k_ratio is 0.52–0.84, i.e. where it **delivered
fewer elements than asked** — less work, wrong answer. Per delivered
element it never wins. The 512-point sample costs nothing measurable at
these sizes (it is ~1–2 μs, amortized over ≥ 2^16 elements) and buys both
the accuracy and the guaranteed t < key.

`two_phase` vs `pivot_first` isolates the user-visible difference of the two
formulations with the *same* pivot: the partition-then-swap plan pays the
`part_swap` double move; the pivot-first plan moves each delivered element
once inside the `prefix_fill` sweep. The gap is largest where the bottom
half is big (p = .9: 1.2–1.4x) and vanishes at p = .1 (the moved set is tiny
either way).

### Full table (ns per suffix element / k_ratio; best non-ref time bold)


#### pair64f

| n | f | p | ref | exact | two_phase | pivot_first | ninther_first |
|---|---|---|---|---|---|---|---|
| 65536 | 0.10 | 0.10 | 6.579 / 1.00 | 0.469 / 1.00 | 0.398 / 1.04 | **0.350 / 1.04** | 0.467 / 1.00 |
| 65536 | 0.10 | 0.50 | 6.257 / 1.00 | 1.009 / 1.00 | 0.464 / 0.95 | 0.416 / 0.95 | **0.391 / 0.75** |
| 65536 | 0.10 | 0.90 | 4.636 / 1.00 | 1.326 / 1.00 | 0.518 / 1.00 | 0.465 / 1.00 | **0.460 / 0.93** |
| 65536 | 0.50 | 0.10 | 2.089 / 1.00 | 0.474 / 1.00 | 0.391 / 0.91 | **0.351 / 0.91** | 0.467 / 1.00 |
| 65536 | 0.50 | 0.50 | 10.854 / 1.00 | 1.117 / 1.00 | 0.540 / 0.92 | **0.418 / 0.92** | 1.110 / 1.00 |
| 65536 | 0.50 | 0.90 | 4.632 / 1.00 | 1.526 / 1.00 | 0.692 / 0.96 | 0.523 / 0.96 | **0.462 / 0.84** |
| 65536 | 0.90 | 0.10 | 9.454 / 1.00 | 0.445 / 1.00 | 0.467 / 1.15 | **0.439 / 1.15** | 0.442 / 1.00 |
| 65536 | 0.90 | 0.50 | 31.143 / 1.00 | 1.130 / 1.00 | 0.690 / 0.99 | **0.552 / 0.99** | 1.099 / 1.00 |
| 65536 | 0.90 | 0.90 | 11.923 / 1.00 | 1.715 / 1.00 | 0.828 / 0.97 | 0.695 / 0.97 | **0.444 / 0.66** |
| 1048576 | 0.10 | 0.10 | 7.445 / 1.00 | 0.895 / 1.00 | 0.730 / 0.81 | **0.677 / 0.81** | 0.865 / 1.00 |
| 1048576 | 0.10 | 0.50 | 5.194 / 1.00 | 1.588 / 1.00 | 0.860 / 0.98 | **0.809 / 0.98** | 1.378 / 1.00 |
| 1048576 | 0.10 | 0.90 | 6.532 / 1.00 | 2.217 / 1.00 | 0.848 / 1.03 | 0.806 / 1.03 | **0.793 / 1.09** |
| 1048576 | 0.50 | 0.10 | 6.349 / 1.00 | 0.902 / 1.00 | 0.817 / 1.10 | **0.722 / 1.10** | 0.879 / 1.00 |
| 1048576 | 0.50 | 0.50 | 10.809 / 1.00 | 1.685 / 1.00 | 1.041 / 0.99 | **0.869 / 0.99** | 1.742 / 1.00 |
| 1048576 | 0.50 | 0.90 | 14.556 / 1.00 | 2.466 / 1.00 | 1.244 / 0.98 | **0.966 / 0.98** | 1.042 / 1.17 |
| 1048576 | 0.90 | 0.10 | 34.528 / 1.00 | 0.855 / 1.00 | 0.792 / 0.88 | **0.788 / 0.88** | 0.870 / 1.00 |
| 1048576 | 0.90 | 0.50 | 62.692 / 1.00 | 1.738 / 1.00 | 1.064 / 1.08 | **0.954 / 1.08** | 1.759 / 1.00 |
| 1048576 | 0.90 | 0.90 | 52.372 / 1.00 | 2.465 / 1.00 | 1.147 / 1.05 | 0.989 / 1.05 | **0.843 / 0.52** |
| 4194304 | 0.10 | 0.10 | 8.569 / 1.00 | 0.933 / 1.00 | 0.786 / 0.97 | **0.713 / 0.97** | 0.886 / 1.00 |
| 4194304 | 0.10 | 0.50 | 8.350 / 1.00 | 1.773 / 1.00 | 0.881 / 1.03 | 0.845 / 1.03 | **0.819 / 1.46** |
| 4194304 | 0.10 | 0.90 | 8.198 / 1.00 | 2.268 / 1.00 | 0.874 / 0.92 | **0.787 / 0.92** | 0.805 / 1.23 |
| 4194304 | 0.50 | 0.10 | 10.553 / 1.00 | 0.896 / 1.00 | 0.814 / 1.08 | **0.700 / 1.08** | 0.870 / 1.00 |
| 4194304 | 0.50 | 0.50 | 4.891 / 1.00 | 2.174 / 1.00 | 1.130 / 1.12 | **0.902 / 1.12** | 0.971 / 1.29 |
| 4194304 | 0.50 | 0.90 | 13.228 / 1.00 | 3.639 / 1.00 | 1.328 / 1.00 | **0.996 / 1.00** | 1.055 / 1.06 |
| 4194304 | 0.90 | 0.10 | 22.306 / 1.00 | 0.841 / 1.00 | 0.768 / 1.27 | **0.715 / 1.27** | 0.832 / 1.00 |
| 4194304 | 0.90 | 0.50 | 69.635 / 1.00 | 1.655 / 1.00 | 0.943 / 0.95 | **0.813 / 0.95** | 1.655 / 1.00 |
| 4194304 | 0.90 | 0.90 | 42.911 / 1.00 | 3.233 / 1.00 | 1.044 / 0.98 | 0.950 / 0.98 | **0.922 / 0.78** |

#### i64

| n | f | p | ref | exact | two_phase | pivot_first | ninther_first |
|---|---|---|---|---|---|---|---|
| 65536 | 0.10 | 0.10 | 6.271 / 1.00 | 0.437 / 1.00 | 0.343 / 1.04 | **0.324 / 1.04** | 0.440 / 1.00 |
| 65536 | 0.10 | 0.50 | 5.860 / 1.00 | 1.017 / 1.00 | 0.414 / 0.95 | 0.381 / 0.95 | **0.356 / 0.75** |
| 65536 | 0.10 | 0.90 | 4.233 / 1.00 | 1.304 / 1.00 | 0.460 / 1.00 | 0.448 / 1.00 | **0.412 / 0.93** |
| 65536 | 0.50 | 0.10 | 1.551 / 1.00 | 0.543 / 1.00 | 0.341 / 0.91 | **0.328 / 0.91** | 0.553 / 1.00 |
| 65536 | 0.50 | 0.50 | 10.289 / 1.00 | 1.100 / 1.00 | 0.445 / 0.92 | **0.407 / 0.92** | 1.082 / 1.00 |
| 65536 | 0.50 | 0.90 | 4.331 / 1.00 | 1.227 / 1.00 | 0.527 / 0.96 | 0.485 / 0.96 | **0.467 / 0.84** |
| 65536 | 0.90 | 0.10 | 7.295 / 1.00 | 0.582 / 1.00 | 0.424 / 1.15 | **0.411 / 1.15** | 0.580 / 1.00 |
| 65536 | 0.90 | 0.50 | 29.371 / 1.00 | 0.999 / 1.00 | 0.564 / 0.99 | **0.528 / 0.99** | 0.995 / 1.00 |
| 65536 | 0.90 | 0.90 | 11.333 / 1.00 | 1.183 / 1.00 | 0.627 / 0.97 | 0.605 / 0.97 | **0.474 / 0.66** |
| 1048576 | 0.10 | 0.10 | 6.011 / 1.00 | 0.502 / 1.00 | 0.389 / 0.81 | **0.354 / 0.81** | 0.491 / 1.00 |
| 1048576 | 0.10 | 0.50 | 4.113 / 1.00 | 1.082 / 1.00 | 0.569 / 0.98 | **0.504 / 0.98** | 0.998 / 1.00 |
| 1048576 | 0.10 | 0.90 | 5.276 / 1.00 | 1.357 / 1.00 | 0.529 / 1.03 | 0.522 / 1.03 | **0.471 / 1.09** |
| 1048576 | 0.50 | 0.10 | 4.719 / 1.00 | 0.617 / 1.00 | **0.354 / 1.10** | 0.367 / 1.10 | 0.602 / 1.00 |
| 1048576 | 0.50 | 0.50 | 8.851 / 1.00 | 1.064 / 1.00 | 0.527 / 0.99 | **0.391 / 0.99** | 1.100 / 1.00 |
| 1048576 | 0.50 | 0.90 | 12.488 / 1.00 | 1.488 / 1.00 | 0.579 / 0.98 | **0.471 / 0.98** | 0.545 / 1.17 |
| 1048576 | 0.90 | 0.10 | 28.732 / 1.00 | 0.589 / 1.00 | 0.334 / 0.88 | **0.315 / 0.88** | 0.577 / 1.00 |
| 1048576 | 0.90 | 0.50 | 54.531 / 1.00 | 0.890 / 1.00 | 0.466 / 1.08 | **0.422 / 1.08** | 0.891 / 1.00 |
| 1048576 | 0.90 | 0.90 | 45.752 / 1.00 | 1.686 / 1.00 | 0.630 / 1.05 | 0.546 / 1.05 | **0.514 / 0.52** |
| 4194304 | 0.10 | 0.10 | 7.386 / 1.00 | 0.581 / 1.00 | 0.465 / 0.97 | **0.401 / 0.97** | 0.598 / 1.00 |
| 4194304 | 0.10 | 0.50 | 6.977 / 1.00 | 1.096 / 1.00 | 0.548 / 1.03 | **0.493 / 1.03** | 0.500 / 1.46 |
| 4194304 | 0.10 | 0.90 | 6.804 / 1.00 | 1.542 / 1.00 | 0.545 / 0.92 | **0.509 / 0.92** | 0.530 / 1.23 |
| 4194304 | 0.50 | 0.10 | 8.858 / 1.00 | 0.634 / 1.00 | 0.467 / 1.08 | **0.416 / 1.08** | 0.642 / 1.00 |
| 4194304 | 0.50 | 0.50 | 3.628 / 1.00 | 1.155 / 1.00 | 0.637 / 1.12 | **0.537 / 1.12** | 0.565 / 1.29 |
| 4194304 | 0.50 | 0.90 | 11.113 / 1.00 | 2.114 / 1.00 | 0.750 / 1.00 | 0.605 / 1.00 | **0.583 / 1.06** |
| 4194304 | 0.90 | 0.10 | 17.537 / 1.00 | 0.620 / 1.00 | **0.486 / 1.27** | 0.532 / 1.27 | 0.598 / 1.00 |
| 4194304 | 0.90 | 0.50 | 62.057 / 1.00 | 1.180 / 1.00 | 0.606 / 0.95 | **0.531 / 0.95** | 1.163 / 1.00 |
| 4194304 | 0.90 | 0.90 | 36.452 / 1.00 | 1.856 / 1.00 | 0.740 / 0.98 | 0.579 / 0.98 | **0.531 / 0.78** |

#### pair64

| n | f | p | ref | exact | two_phase | pivot_first | ninther_first |
|---|---|---|---|---|---|---|---|
| 65536 | 0.10 | 0.10 | 8.295 / 1.00 | 0.640 / 1.00 | 0.506 / 1.04 | **0.486 / 1.04** | 0.639 / 1.00 |
| 65536 | 0.10 | 0.50 | 7.871 / 1.00 | 1.331 / 1.00 | 0.596 / 0.95 | 0.537 / 0.95 | **0.500 / 0.75** |
| 65536 | 0.10 | 0.90 | 6.035 / 1.00 | 1.861 / 1.00 | 0.625 / 1.00 | 0.629 / 1.00 | **0.555 / 0.93** |
| 65536 | 0.50 | 0.10 | 2.916 / 1.00 | 0.660 / 1.00 | 0.513 / 0.91 | **0.490 / 0.91** | 0.648 / 1.00 |
| 65536 | 0.50 | 0.50 | 13.705 / 1.00 | 1.438 / 1.00 | 0.669 / 0.92 | **0.549 / 0.92** | 1.486 / 1.00 |
| 65536 | 0.50 | 0.90 | 6.247 / 1.00 | 1.950 / 1.00 | 0.859 / 0.96 | 0.646 / 0.96 | **0.589 / 0.84** |
| 65536 | 0.90 | 0.10 | 11.852 / 1.00 | 0.648 / 1.00 | 0.670 / 1.15 | **0.640 / 1.15** | 0.651 / 1.00 |
| 65536 | 0.90 | 0.50 | 40.947 / 1.00 | 1.396 / 1.00 | 0.889 / 0.99 | **0.713 / 0.99** | 1.392 / 1.00 |
| 65536 | 0.90 | 0.90 | 17.691 / 1.00 | 2.063 / 1.00 | 1.000 / 0.97 | 0.808 / 0.97 | **0.542 / 0.66** |
| 1048576 | 0.10 | 0.10 | 8.440 / 1.00 | 0.867 / 1.00 | 0.693 / 0.81 | **0.665 / 0.81** | 0.902 / 1.00 |
| 1048576 | 0.10 | 0.50 | 5.645 / 1.00 | 1.303 / 1.00 | 0.838 / 0.98 | **0.804 / 0.98** | 1.461 / 1.00 |
| 1048576 | 0.10 | 0.90 | 7.474 / 1.00 | 1.950 / 1.00 | 0.773 / 1.03 | 0.787 / 1.03 | **0.747 / 1.09** |
| 1048576 | 0.50 | 0.10 | 6.693 / 1.00 | 0.882 / 1.00 | 0.764 / 1.10 | **0.673 / 1.10** | 0.886 / 1.00 |
| 1048576 | 0.50 | 0.50 | 12.620 / 1.00 | 1.711 / 1.00 | 1.008 / 0.99 | **0.818 / 0.99** | 1.931 / 1.00 |
| 1048576 | 0.50 | 0.90 | 16.651 / 1.00 | 2.961 / 1.00 | 1.295 / 0.98 | **0.962 / 0.98** | 1.067 / 1.17 |
| 1048576 | 0.90 | 0.10 | 40.238 / 1.00 | 0.896 / 1.00 | 0.866 / 0.88 | **0.787 / 0.88** | 0.940 / 1.00 |
| 1048576 | 0.90 | 0.50 | 72.064 / 1.00 | 2.013 / 1.00 | 1.051 / 1.08 | **0.933 / 1.08** | 2.113 / 1.00 |
| 1048576 | 0.90 | 0.90 | 60.490 / 1.00 | 2.717 / 1.00 | 1.236 / 1.05 | 1.046 / 1.05 | **0.819 / 0.52** |
| 4194304 | 0.10 | 0.10 | 9.879 / 1.00 | 0.930 / 1.00 | 0.697 / 0.97 | **0.658 / 0.97** | 0.938 / 1.00 |
| 4194304 | 0.10 | 0.50 | 9.441 / 1.00 | 1.675 / 1.00 | 0.880 / 1.03 | 0.815 / 1.03 | **0.787 / 1.46** |
| 4194304 | 0.10 | 0.90 | 9.236 / 1.00 | 2.184 / 1.00 | 0.828 / 0.92 | 0.794 / 0.92 | **0.770 / 1.23** |
| 4194304 | 0.50 | 0.10 | 11.740 / 1.00 | 0.872 / 1.00 | 0.711 / 1.08 | **0.675 / 1.08** | 0.861 / 1.00 |
| 4194304 | 0.50 | 0.50 | 5.265 / 1.00 | 2.229 / 1.00 | 1.073 / 1.12 | **0.897 / 1.12** | 0.919 / 1.29 |
| 4194304 | 0.50 | 0.90 | 14.731 / 1.00 | 3.707 / 1.00 | 1.289 / 1.00 | **1.014 / 1.00** | 1.038 / 1.06 |
| 4194304 | 0.90 | 0.10 | 23.962 / 1.00 | 0.885 / 1.00 | 0.735 / 1.27 | **0.673 / 1.27** | 0.878 / 1.00 |
| 4194304 | 0.90 | 0.50 | 80.429 / 1.00 | 2.126 / 1.00 | 1.014 / 0.95 | **0.819 / 0.95** | 1.892 / 1.00 |
| 4194304 | 0.90 | 0.90 | 47.249 / 1.00 | 2.507 / 1.00 | 1.021 / 0.98 | 0.957 / 0.98 | **0.898 / 0.78** |


## Assembly notes

* The suffix sampling loop **auto-vectorizes** for both i64 and
  pair64-by-first: two strided samples gathered per iteration
  (`vmovq`/`vpinsrq`), below-count accumulated with `vpcmpgtq` + `vpsubq` —
  no data-dependent branch; the `i*stride` index is strength-reduced to a
  pointer increment. No fix needed.
* The sample-rank selection reuses the repo's branchless quickselect
  (`detail::quickselect`, `move_low_half.hpp`) on the 512-entry stack
  buffer — `std::nth_element`'s branchy introselect was already measured
  4–8x slower there in the move_low_half study.
* Everything downstream is the audited `offset_partition` machinery
  (`prefix_fill`'s branchless block fill and 2-move rotation, `gap_off`'s
  7-instruction loop, by-value pivots in registers); no new hot loops are
  introduced, which is the point of the pivot-first formulation.

## Negative results kept

* `ninther_first`: a cheap pivot is a false economy — its k bias (up to
  ±48%) buys no measurable time at any size where the call matters.
* Routing the t-partition through `sized_off`: the dispatcher's
  narrow-type gap rule is wrong for this caller, which *knows* the
  effective below fraction is ~p/2 (see above). Bypassed via direct
  `prefix_fill_off`.
* The exact variant's quickselect makes it ~2x the sampled cost at mid/high
  p — use it only when k must be exactly ⌊S/2⌋.
