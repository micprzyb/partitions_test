#!/usr/bin/env bash
# bench_cycles.sh <HN> <netA_file> <netB_file> [reps] [R] [benchcore]
#
# Reliable cycle-accurate comparison of TWO halver networks via perf HW counters.
# Reports cycles/element, IPC, and instructions/element for each of i64 / pair64
# (lex) / pair64f (first-key).  cycles & IPC are frequency-invariant and (because
# the kernels are branchless) data-independent, so this is far steadier than the
# ns timing in bench_pool.cpp -- use it to settle close calls and to read the
# *mechanism* (a low-IPC net is dependency-bound, not work-bound).
#
# netA_file / netB_file: a file whose content is the comma comparator list, e.g.
#   {4,6},{20,22},...   (one net; trailing comma/newline ok)
# Requires perf with cpu_core PMU readable (perf_event_paranoid low enough).
#
# Example:
#   printf '%s' "$(sed -n '1p' seeds/h24_80pool.txt)" > /tmp/a.txt   # some net
#   optimizers/bench_cycles.sh 24 /tmp/a.txt /tmp/b.txt 5
set -euo pipefail
cd "$(dirname "$0")/.."
HN=${1:?usage: bench_cycles.sh HN netA_file netB_file [reps] [R] [benchcore]}
FA=${2:?need netA file}
FB=${3:?need netB file}
REPS=${4:-5}
R=${5:-30000}
BCORE=${6:-4}
PC=(0 1 3 6 8 10)
OPT="-O3 -march=native -std=c++23"

A=$(grep -oE '\{[0-9]+,[0-9]+\}' "$FA" | paste -sd, -)
B=$(grep -oE '\{[0-9]+,[0-9]+\}' "$FB" | paste -sd, -)
SA=$(grep -oc '{' <<<"$A" || true); SA=$(grep -oE '\{[0-9]+,[0-9]+\}' "$FA" | wc -l)
SB=$(grep -oE '\{[0-9]+,[0-9]+\}' "$FB" | wc -l)
cat > /tmp/cyc_nets.inc <<EOF
inline constexpr std::array<P,$SA> A = {{$A}};
inline constexpr std::array<P,$SB> B = {{$B}};
EOF
echo "#define HN $HN" >> /tmp/cyc_nets.inc
echo "netA=$SA CE, netB=$SB CE, HN=$HN"

echo "compiling 6 binaries (A/B x i64,pair64,pair64f) in parallel..."
k=0
for net in A B; do for v in 0 1 2; do
  c=${PC[$((k % 6))]}; k=$((k + 1))
  taskset -c "$c" g++ $OPT -DNET=$net -DVARIANT=$v -DCYC_INC='"/tmp/cyc_nets.inc"' \
      -Iinclude optimizers/bench_cycles.cpp -o "/tmp/bc_${net}_$v" 2>"/tmp/bc_${net}_$v.log" &
done; done
wait
for net in A B; do for v in 0 1 2; do [ -x "/tmp/bc_${net}_$v" ] || { echo "compile failed $net $v:"; cat /tmp/bc_${net}_$v.log; exit 1; }; done; done

echo "perf-stat on core $BCORE (median of $REPS reps each)..."
for net in A B; do for v in 0 1 2; do
  : > "/tmp/bc_acc_${net}_$v.txt"
  for _ in $(seq 1 "$REPS"); do
    taskset -c "$BCORE" perf stat -x, -e cpu_core/cycles/,cpu_core/instructions/ \
        -o "/tmp/bc_po.csv" "/tmp/bc_${net}_$v" "$R" >/dev/null 2>&1 || true
    cyc=$(awk -F, '/cpu_core\/cycles\//{print $1}' /tmp/bc_po.csv)
    ins=$(awk -F, '/cpu_core\/instructions\//{print $1}' /tmp/bc_po.csv)
    echo "$cyc $ins" >> "/tmp/bc_acc_${net}_$v.txt"
  done
done; done

python3 - "$R" <<'PY'
import statistics, sys
R=int(sys.argv[1]); B=96;
names={0:'i64',1:'pair64',2:'pair64f'}
def med(net,v):
    rows=[l.split() for l in open(f'/tmp/bc_acc_{net}_{v}.txt') if l.split()]
    return statistics.median(float(r[0]) for r in rows), statistics.median(float(r[1]) for r in rows)
# read HN/ELEMS by re-deriving from one run's stdout? simpler: ELEMS=B*R*HN, get HN from inc
HN=int([l for l in open('/tmp/cyc_nets.inc') if 'define HN' in l][0].split()[-1])
ELEMS=B*R*HN
print(f"\n{'type':9}{'A cyc/el':>10}{'B cyc/el':>10}{'B-A %':>8}   {'A IPC':>7}{'B IPC':>7}   {'A ins/el':>9}{'B ins/el':>9}")
for v in [0,1,2]:
    ca,ia=med('A',v); cb,ib=med('B',v)
    print(f"{names[v]:9}{ca/ELEMS:10.3f}{cb/ELEMS:10.3f}{100*(cb-ca)/ca:+8.1f}   {ia/ca:7.3f}{ib/cb:7.3f}   {ia/ELEMS:9.3f}{ib/ELEMS:9.3f}")
print("\n(A = first net, B = second net; cycles/IPC are frequency-invariant; lower cyc/el = faster)")
PY
