#!/usr/bin/env bash
# bench_pool_par.sh <N> [cap] [chunks] [reps] [benchcore]
#
# Parallel-compile + serial-benchmark driver for optimizers/bench_pool.cpp.
#
# WHY: bench_pool.cpp bakes every candidate halver in as a compile-time template
# instantiation in ONE translation unit, so a ~1000-net pool takes ~9-10 min to
# compile and `make -j` cannot speed it up.  This script splits the candidate pool
# into <chunks> separate include files, compiles the <chunks> bench binaries IN
# PARALLEL pinned to the P-cores (compile is the bottleneck), then runs them
# SEQUENTIALLY on one quiet core (timing MUST be uncontended), and merges the
# results -> fastest per element type and by sum.  Cuts a 9.5-min compile to ~2 min.
#
# Pre-req: per-seed pool files /tmp/h<N>_*_pool.inc exist (from the search), so
# make_pool.py can build /tmp/hpool.inc (base = committed hN + the floor candidates).
#
# Examples:
#   optimizers/bench_pool_par.sh 24            # all floor candidates, 6 chunks, 1 rep
#   optimizers/bench_pool_par.sh 24 3000 6 9   # median of 9 reps (robust confirm)
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:?usage: bench_pool_par.sh N [cap] [chunks] [reps] [benchcore]}
CAP=${2:-3000}
CHUNKS=${3:-6}
REPS=${4:-1}
BCORE=${5:-4}
PCORES=(0 1 3 6 8 10)            # one logical CPU per physical P-core (no HT contention)
OPT="-O3 -march=native -std=c++23"

echo "[1/4] build pool: make_pool.py $N $CAP"
python3 optimizers/make_pool.py "$N" "$CAP"

echo "[2/4] split candidates into $CHUNKS chunks (round-robin, balanced)"
python3 - "$CHUNKS" <<'PY'
import re, sys
chunks = int(sys.argv[1])
base = None; cands = []
for l in open('/tmp/hpool.inc').read().splitlines():
    m = re.match(r'inline constexpr std::array<P,\d+>\s+(\w+)\s*=', l)
    if not m:
        continue
    if m.group(1) == 'base': base = l
    else: cands.append((m.group(1), l))
assert base is not None, "no 'base' line in /tmp/hpool.inc"
groups = [[] for _ in range(chunks)]
for i, c in enumerate(cands):
    groups[i % chunks].append(c)
for g, grp in enumerate(groups):
    with open(f'/tmp/hpool_c{g}.inc', 'w') as o:
        o.write(base + '\n')
        for _, line in grp: o.write(line + '\n')
        o.write('#define HN_POOL_N %d\n' % len(grp))
        o.write('#define HN_POOL_LIST ' + ' '.join('X(%s)' % n for n, _ in grp) + '\n')
print(f'  {len(cands)} candidates -> {chunks} chunks (~{len(cands)//max(chunks,1)} each)')
PY

echo "[3/4] compile $CHUNKS bench binaries IN PARALLEL on P-cores ${PCORES[*]}"
pids=()
for g in $(seq 0 $((CHUNKS-1))); do
  c=${PCORES[$((g % ${#PCORES[@]}))]}
  taskset -c "$c" g++ $OPT -DHN="$N" -DPOOL_INC="\"/tmp/hpool_c${g}.inc\"" \
      -Iinclude optimizers/bench_pool.cpp -o "/tmp/bpp_${N}_${g}" 2>"/tmp/bpp_${N}_${g}.cc.log" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "  a compile failed; see /tmp/bpp_${N}_*.cc.log"; exit 1; }
echo "  compiled $CHUNKS binaries"

echo "[4/4] benchmark SERIALLY on quiet core $BCORE (reps=$REPS), then merge"
: > "/tmp/bpp_${N}_all.txt"
for g in $(seq 0 $((CHUNKS-1))); do
  for _ in $(seq 1 "$REPS"); do
    taskset -c "$BCORE" "/tmp/bpp_${N}_${g}" >> "/tmp/bpp_${N}_all.txt" 2>&1
  done
done
python3 - "$N" <<'PY'
import re, sys, statistics
from collections import defaultdict
N = sys.argv[1]
d = defaultdict(lambda: {'i': [], 'p': [], 'pf': []})
for l in open(f'/tmp/bpp_{N}_all.txt'):
    m = re.match(r'(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+sum=', l)
    if not m: continue
    n = m.group(1)
    d[n]['i'].append(float(m.group(2))); d[n]['p'].append(float(m.group(3))); d[n]['pf'].append(float(m.group(4)))
rows = []
for n, v in d.items():
    mi, mp, mpf = statistics.median(v['i']), statistics.median(v['p']), statistics.median(v['pf'])
    rows.append((mi + mp + mpf, n, mi, mp, mpf))
rows.sort()
print(f"\n{'net':8}{'i64':>8}{'pair64':>9}{'p64f':>8}{'sum':>9}   (median over reps; base = committed h%s)" % N)
for s, n, mi, mp, mpf in rows[:20]:
    print(f"{n:8}{mi:8.4f}{mp:9.4f}{mpf:8.4f}{s:9.4f}")
bi = min(rows, key=lambda r: r[2]); bp = min(rows, key=lambda r: r[3]); bpf = min(rows, key=lambda r: r[4])
print(f"\nbest i64: {bi[1]} ({bi[2]:.4f}) | best pair64: {bp[1]} ({bp[3]:.4f}) | "
      f"best p64f: {bpf[1]} ({bpf[4]:.4f}) | best sum: {rows[0][1]} ({rows[0][0]:.4f})")
print("(pN names index into /tmp/hpool.inc; re-bench the top few with higher reps to confirm)")
PY
