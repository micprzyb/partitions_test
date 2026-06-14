#!/usr/bin/env bash
# optimize.sh <N> [jobs] [restarts] [iters] [poolcap]
#
# End-to-end perfect-halver optimisation for block size N:
#   1. extract seeds (current hN + sorter nN) from the headers       (make_seeds.py)
#   2. run `jobs` parallel ILS searches with distinct RNG seeds      (search_halve*)
#      - exhaustive flat check for N<=17, CEGAR+k-zero for N>=18
#   3. merge distinct minimal nets, build the candidate pool          (make_pool.py)
#   4. benchmark every floor candidate vs the baseline               (bench_pool.cpp)
#      for i64 / pair64 (lex) / pair64f (first-key); print the fastest.
#
# Then: paste the winning net into nets::hN, rebuild, and run
#   build/tools/verify_small_halve  and  build/tools/verify_small_halve_rev
# (the trusted exhaustive 2^N check) before committing.  A second wave can be run
# by reseeding /tmp/seeds_<N>.txt from the floor nets (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:?usage: optimize.sh N [jobs] [restarts] [iters] [poolcap]}
JOBS=${2:-$(nproc)}
RESTARTS=${3:-150}
ITERS=${4:-800}
CAP=${5:-120}
CXX=${CXX:-g++}
OPT="-O3 -march=native -std=c++23"

if [ "$N" -ge 18 ]; then SRC=optimizers/search_halve_cegar.cpp; else SRC=optimizers/search_halve.cpp; fi
echo "[1/4] seeds";   python3 optimizers/make_seeds.py "$N"
echo "[build] $SRC";  $CXX $OPT "$SRC" -o /tmp/sh_opt
echo "[2/4] $JOBS parallel searches (restarts=$RESTARTS iters=$ITERS)"
rm -f /tmp/h${N}_*_pool.inc
ncores=$(nproc)
for i in $(seq 0 $((JOBS-1))); do
  S=$(( (i+1)*7919 + 3 ))
  taskset -c $(( i % ncores )) /tmp/sh_opt "$N" "$S" "$RESTARTS" "$ITERS" "$CAP" >/dev/null 2>/tmp/opt_${N}_${S}.log &
done
wait
echo "[3/4] pool";   python3 optimizers/make_pool.py "$N" "$CAP"
echo "[4/4] benchmark"
$CXX $OPT -DHN="$N" -DPOOL_INC='"/tmp/hpool.inc"' -Iinclude optimizers/bench_pool.cpp -o /tmp/benchpool
taskset -c 0 /tmp/benchpool
echo
echo "Pick a winner above, paste into include/partitions/small_halve.hpp (nets::h$N),"
echo "then: cmake --build build -j && build/tools/verify_small_halve && build/tools/verify_small_halve_rev"
