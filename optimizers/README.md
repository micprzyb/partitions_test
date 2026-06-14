# optimizers/ — perfect-halver network optimisation

Tools that found the faster `nets::hN` halver networks in
`include/partitions/small_halve.hpp` (n=10..17, 24; see the `Plan*.md` files).
They search for a halver with **fewer comparators**, then pick the **fastest**
arrangement by direct benchmark. Raw per-block speed is the only objective.

## What a (perfect) halver is, and the validity test

`halve_n<N>` splits a block by median rank: the `k = ⌊N/2⌋` smallest land in
`[0,k)`, the rest in `[k,N)` (each half unordered). By the 0/1 principle a
comparator network is a valid split@k halver iff, for every 0/1 input, after the
network `max(bottom k) ≤ min(top k)` — i.e. no bottom wire is 1 while a top wire
is 0.

**Monotonicity reduction (key efficiency idea, borrowed from the SAT finders in
`python/`):** it suffices to check only the `C(N,k)` inputs with **exactly k
zeros**. Any other input reduces to a k-zero one (flip spare zeros up / spare
ones down) and comparator networks are monotone, so the k-zero inputs dominate.
That is ~6.3× fewer inputs at n=24 (`C(24,12)=2.70M` vs `2²⁴=16.8M`; measured
11.9 ms → 1.9 ms for the exhaustive sweep), and the gap grows with N.

## Programs

| file | what |
|------|------|
| `search_halve.cpp` | ILS search (greedy comparator deletion + perturb-reprune). **Exhaustive flat 2ⁿ** 0/1 check. Best for **N ≤ 17**. |
| `search_halve_cegar.cpp` | Same ILS, but the check is **CEGAR** (cheap witness pre-filter + full **k-zero** sweep) — small (≈L2-resident) and parallel-scalable. Use for **N ≥ 18** (works for any N). |
| `bench_pool.cpp` | Times candidate nets (compile-time constants) vs the baseline for `i64`, `pair64` (lex), `pair64f` (first-key). |
| `make_seeds.py` | Extracts seeds (current `hN` + sorter `nN`) from the headers → `/tmp/seeds_<N>.txt`. |
| `make_pool.py` | Merges the search outputs, keeps the floor, builds `/tmp/hpool.inc` for the bench. |
| `optimize.sh` | One-shot driver: seeds → parallel search → pool → benchmark. |

The search tools are standalone (`g++`, no deps). `bench_pool.cpp` needs the
library headers (`-Iinclude`). The benchmark deliberately compiles each candidate
as a compile-time constant network (a reference NTTP) so codegen matches
`halve_n` exactly — passing nets at runtime inflates times ~3×.

## Quick start

```bash
# One command: search n=14 on all cores, then benchmark the floor candidates.
optimizers/optimize.sh 14
#   args: optimize.sh <N> [jobs] [restarts] [iters] [poolcap]
```

Then paste the winning net into `nets::hN` in
`include/partitions/small_halve.hpp`, rebuild, and **verify** (the trusted,
independent exhaustive 2ⁿ check — never skip this):

```bash
cmake --build build -j --target verify_small_halve verify_small_halve_rev
build/tools/verify_small_halve        # forward halver, all 2^N inputs
build/tools/verify_small_halve_rev    # reverse/descending halver (same array)
ctest --test-dir build                # full suite
```

## Manual / large-N workflow (what was used for n=24)

```bash
python3 optimizers/make_seeds.py 24
g++ -O3 -march=native -std=c++23 optimizers/search_halve_cegar.cpp -o /tmp/shb

# Wave 1: many parallel searches from the header seeds.
for i in $(seq 0 29); do S=$(((i+1)*2417+5));
  taskset -c $((i%30)) /tmp/shb 24 $S 10 1500 40 >/dev/null 2>/tmp/w_$S.txt & done; wait

# Reseed a second wave from the current frontier to push the floor lower:
cat /tmp/h24_*_pool.inc | grep -oE '\{\{.*\}\}' | sort -u > /tmp/uniq_24.txt
python3 - <<'PY'
import re
floor=min(len(re.findall(r'\{\d+,\d+\}',l)) for l in open('/tmp/uniq_24.txt'))
open('/tmp/seeds_24.txt','w').write("\n".join(
    l.strip() for l in open('/tmp/uniq_24.txt')
    if len(re.findall(r'\{\d+,\d+\}',l))==floor)+"\n")
print("reseeded from floor",floor)
PY
rm -f /tmp/h24_*_pool.inc        # then run the parallel loop again...

# Benchmark the floor pool:
python3 optimizers/make_pool.py 24 80
g++ -O3 -march=native -std=c++23 -DHN=24 -DPOOL_INC='"/tmp/hpool.inc"' \
    -Iinclude optimizers/bench_pool.cpp -o /tmp/benchpool
taskset -c 4 /tmp/benchpool
```

Reseeding from the floor and re-running ("waves") is what drove n=24 from
88 → 84 → 83 → 82 comparators. Pin with `taskset` and take the min over reps;
treat absolute ns as ±5% on this box.

## Method notes & limits

- **Selection is by measurement, not by metric.** Among same-size nets, speed
  varies (register pressure, scheduling); comparator-layer *depth* barely matters
  in this throughput-bound regime. Do **not** "layer-group" the emission order —
  it inflates the live set and is ~10% slower on `pair64`. Keep the search order.
- **Floor ≠ proven optimum.** The ILS floor is the smallest *found*; it is not a
  proof of minimality. For small N the SAT finders in `python/` can *prove*
  optimal comparator counts (sound: same k-zero encoding), but they are
  depth-bounded SAT and don't scale past small/moderate N — see those files. The
  C++ heuristic here is the practical tool for large N.
- **Always re-verify** the committed net with `verify_small_halve{,_rev}`. The
  search tools are trusted only to *propose*; the repo's exhaustive verifier is
  the gate.
