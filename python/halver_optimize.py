#!/usr/bin/env python3
"""
halver_optimize.py — optimise a small halver network for a real CPU, not for the
proxy metrics (#comparators / depth).

Why the proxies are wrong (measured, see docs/pivot_and_halver_report.md):
  * #comparators / depth ignore register pressure, critical-path *latency* (which
    is per-comparator-cost dependent, i.e. differs for i64 vs 16-byte pairs),
    instruction scheduling, and the compiler's own (often counter-productive)
    auto-vectorisation.  Two nets with the same count/depth can differ >40%.
  * A WIRE PERMUTATION of a net preserves the DAG exactly (same count, depth and
    critical path) yet changes which physical registers the compiler uses and how
    it schedules -- so it changes speed for reasons no static count can see.  The
    only reliable way to choose among them is to MEASURE.

So this is an empirical autotuner.  For a target (n, split k, element type) it:
  1. takes one or more correct base halvers,
  2. generates candidates by (a) split-preserving wire permutations -- permute the
     bottom-k and top-k wires independently, which keeps it a correct split@k
     halver; (b) alternative topological *schedules* of the same comparators
     (changes register pressure); (c) any extra base nets you pass in,
  3. verifies every candidate is still a halver (exhaustive 0/1 enumeration),
  4. emits ONE C++ TU with every candidate as a scalar (no-tree-vectorize) applier
     -- matching how halve_n is really used (one block per call, not a vectorised
     batch) -- and also a "fused" variant that keeps extra values live across the
     loop to expose register pressure (the cost that bites when the net is inlined
     into a sort),
  5. compiles once with the real toolchain and times each candidate,
  6. reports the fastest arrangement per element type, next to the static metrics
     so you can see how poorly count/depth predict the winner.

Usage:  python3 halver_optimize.py            # optimises n=7 and n=8 defaults
        edit TARGETS below to add sizes / base nets / element types.
"""
import itertools, random, subprocess, sys, tempfile, os, re, math, argparse
from pathlib import Path
from collections import defaultdict
from statistics import median
sys.path.insert(0, str(Path(__file__).resolve().parent))
import halver_io as H  # parse halver_finder.py output, reflect, format

CXX = os.environ.get("CXX", "g++")
INCLUDE = str(Path(__file__).resolve().parent.parent / "include")

# ----- correctness: exhaustive 0/1 halver check (bit-parallel over all inputs) --
def is_halver(net, n, k):
    N = 1 << n
    full = (1 << N) - 1
    # w[i] = the 2^n-bit column holding input bit i across all 2^n inputs
    w = [0] * n
    for m in range(N):
        for i in range(n):
            if (m >> i) & 1:
                w[i] |= 1 << m
    for a, b in net:
        lo = w[a] & w[b]
        hi = w[a] | w[b]
        w[a], w[b] = lo, hi
    orbottom = 0
    for i in range(k):
        orbottom |= w[i]
    andtop = full
    for i in range(k, n):
        andtop &= w[i]
    # halver iff no bottom wire is 1 while some top wire is 0
    return (orbottom & ~andtop) == 0

def norm(c):
    a, b = c
    return (a, b) if a < b else (b, a)

# ----- static metrics -----------------------------------------------------------
def depth(net, n):
    t = [0] * n
    for a, b in net:
        d = max(t[a], t[b]) + 1
        t[a] = t[b] = d
    return max(t) if net else 0

def reg_pressure(net, n):
    """peak number of wires live (between first and last touch) under this order
    -- a proxy for register pressure / spill risk when fused."""
    first = {}; last = {}
    for idx, (a, b) in enumerate(net):
        for w in (a, b):
            first.setdefault(w, idx); last[w] = idx
    peak = 0
    for idx in range(len(net)):
        live = sum(1 for w in first if first[w] <= idx <= last[w])
        peak = max(peak, live)
    return peak

# ----- candidate generation -----------------------------------------------------
def permute(net, perm):
    return [norm((perm[a], perm[b])) for a, b in net]

def enumerate_split_perms(n, k, max_enum, seed):
    """Yield permutations mapping [0,k)->[0,k) and [k,n)->[k,n) -- the only wire
    relabelings that can keep a split@k halver.  ENUMERATED EXHAUSTIVELY (in a
    fixed, deterministic order) when the group size k!*(n-k)! <= max_enum, so the
    search is complete and the .pN labels are stable.  Only if the group is bigger
    than max_enum do we fall back to a reproducible random sample of size max_enum,
    using the EXPLICIT `seed` (never Python's per-process-salted hash())."""
    total = math.factorial(k) * math.factorial(n - k)
    if total <= max_enum:
        for b in itertools.permutations(range(k)):
            for t in itertools.permutations(range(k, n)):
                yield list(b) + list(t)
    else:
        rng = random.Random(seed)
        seen = set()
        while len(seen) < max_enum:
            b = list(range(k)); t = list(range(k, n))
            rng.shuffle(b); rng.shuffle(t)
            p = tuple(b + t)
            if p not in seen:
                seen.add(p); yield list(p)

def layer_schedule(net, n):
    """reorder comparators into topological layers (changes register pressure,
    keeps the DAG/result identical)."""
    t = [0] * n; tagged = []
    for idx, (a, b) in enumerate(net):
        L = max(t[a], t[b]) + 1; t[a] = t[b] = L
        tagged.append((L, idx, (a, b)))
    tagged.sort(key=lambda z: (z[0], z[1]))
    return [c for _, _, c in tagged]

def gen_candidates(name, base, n, k, perms="all", max_enum=20000, max_cands=256,
                   seed=0, schedules=False):
    """Distinct VALID halver arrangements of `base`, deduped by the exact ordered
    net (identical nets compile identically).  Deterministic.  Options:
      perms='all' -> try the split-preserving permutation group; 'none' -> base only.
      schedules=True -> also try the layer-ordered reschedule of the base."""
    cands = {}; seen = set()
    def add(label, net):
        key = tuple(net)
        if key not in seen and is_halver(net, n, k):
            seen.add(key); cands[label] = net; return True
        return False
    add(name, base)
    if schedules:
        add(f"{name}.layer", layer_schedule(base, n))
    if perms != "none":
        i = 0
        for p in enumerate_split_perms(n, k, max_enum, seed):
            if len(cands) >= max_cands:
                break
            if add(f"{name}.p{i}", permute(base, p)):
                i += 1
    return cands

# ----- C++ codegen + measure ----------------------------------------------------
CPP_HEAD = r'''
#include "partitions/partitions.hpp"
#include "partitions/small_sort.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <utility>
using namespace partitions;
using PR = std::pair<int,int>;
struct firstk { template<class P> auto operator()(const P&p)const{return p.first;} };
template<class T,std::size_t K,class Proj>
__attribute__((noinline,optimize("no-tree-vectorize")))
void plain(T* a,long batch,const std::array<PR,K>& net,Proj pr){auto c=std::less<>{};
  for(long b=0;b<batch;++b) small_sort::apply_network(a+b*N_,net,std::make_index_sequence<K>{},c,pr);
  asm volatile("":::"memory");}
template<class T,std::size_t K,class Proj>
__attribute__((noinline,optimize("no-tree-vectorize")))
void fused(T* a,long batch,const std::array<PR,K>& net,Proj pr,long* sink){auto c=std::less<>{};
  long s0=sink[0],s1=sink[1],s2=sink[2],s3=sink[3],s4=sink[4],s5=sink[5];
  for(long b=0;b<batch;++b){ small_sort::apply_network(a+b*N_,net,std::make_index_sequence<K>{},c,pr);
    long* q=(long*)(a+b*N_); s0^=q[0];s1+=q[1];s2^=q[2];s3+=q[0];s4^=q[1];s5+=q[2]; }
  sink[0]=s0;sink[1]=s1;sink[2]=s2;sink[3]=s3;sink[4]=s4;sink[5]=s5; asm volatile("":::"memory");}
template<class T,std::size_t K,class Proj,class F>
double tm(F f,std::vector<T>&m,long batch,const std::array<PR,K>&net,Proj pr){
  std::vector<T> w(m.size()); long sink[6]={0,0,0,0,0,0}; double best=1e18;
  for(int r=0;r<REPS_;++r){w=m;auto t0=std::chrono::steady_clock::now();f(w.data(),batch,net,pr,sink);
    auto t1=std::chrono::steady_clock::now();best=std::min(best,std::chrono::duration<double,std::nano>(t1-t0).count()/batch);}
  return best;}
'''

def emit_and_run(cands, n, k, types, reps=400, runs=1):
    """Measure each candidate by WALL-CLOCK ns per network application: min over
    `reps` in-process samples, then the binary is run `runs` times and we take the
    MEDIAN across runs (with min..max spread).  Returns (median_map, spread_map),
    each keyed by (label, tname, mode in {plain,fused})."""
    items = sorted(cands.items())
    src = CPP_HEAD.replace("N_", str(n)).replace("REPS_", str(reps))
    # net constants
    for label, net in items:
        arr = ",".join("{%d,%d}" % (a, b) for a, b in net)
        src += "constexpr std::array<PR,%d> NET_%s = {{%s}};\n" % (len(net), _id(label), arr)
    # wrappers so each (net) call site is distinct (no merging)
    src += "int main(){\n  long batch=(1<<16)/%d;\n" % n
    src += '  printf("label,type,mode,ns\\n");\n'
    for tname, ctype, proj in types:
        src += "  {\n    std::vector<%s> m(batch*%d); std::mt19937_64 rng(1);\n" % (ctype, n)
        src += "    std::uniform_int_distribution<long> d(0,1000000000);\n"
        if ctype == "pair64":
            src += "    for(auto&x:m)x=pair64{d(rng),d(rng)};\n"
        else:
            src += "    for(auto&x:m)x=(%s)d(rng);\n" % ctype
        for label, net in items:
            pl = "plain<%s,%d>" % (ctype, len(net))
            fu = "fused<%s,%d>" % (ctype, len(net))
            src += '    printf("%s,%s,plain,%%.4f\\n", tm(%s, m, batch, NET_%s, %s{}));\n' % (label, tname, pl_wrap(pl), _id(label), proj)
            src += '    printf("%s,%s,fused,%%.4f\\n", tm(%s, m, batch, NET_%s, %s{}));\n' % (label, tname, fu_wrap(fu), _id(label), proj)
        src += "  }\n"
    src += "  return 0;\n}\n"
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / "bench.cpp"; exe = Path(td) / "bench"
        cpp.write_text(src)
        cc = [CXX, "-std=c++23", "-O3", "-march=native", "-fno-tree-vectorize",
              "-I", INCLUDE, "-o", str(exe), str(cpp)]
        r = subprocess.run(cc, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[:4000]); raise SystemExit("compile failed")
        samples = defaultdict(list)
        for _ in range(runs):
            out = subprocess.run(["taskset", "-c", "2", str(exe)],
                                 capture_output=True, text=True).stdout
            for line in out.splitlines()[1:]:
                lab, tn, mode, ns = line.split(",")
                samples[(lab, tn, mode)].append(float(ns))
    res = {key: median(v) for key, v in samples.items()}
    spread = {key: (min(v), max(v)) for key, v in samples.items()}
    return res, spread

# the proj/projection is passed as the 5th arg to plain/fused via tm's F; wrap to
# pass the net+proj.  We just need lambdas binding the right template instances:
def pl_wrap(pl): return "[](auto*a,long b,const auto&net,auto pr,long*){%s(a,b,net,pr);}" % pl
def fu_wrap(fu): return "[](auto*a,long b,const auto&net,auto pr,long*s){%s(a,b,net,pr,s);}" % fu
def _id(label): return re.sub(r"[^A-Za-z0-9]", "_", label)

# ----- drive --------------------------------------------------------------------
TARGETS = [
    # name, base net, n, k
    ("h7",     [(0,4),(1,5),(2,6),(0,2),(1,3),(4,6),(2,4),(3,5),(0,1),(2,3),(4,5),(1,4)], 7, 3),
    ("h7_alt", [(0,6),(2,4),(1,3),(3,6),(4,5),(2,3),(0,1),(0,5),(3,4),(1,2),(2,3)],        7, 3),
    ("h8_old", [(0,2),(1,3),(5,7),(0,4),(1,5),(3,7),(0,1),(2,3),(4,5),(6,7),(2,4),(3,5),(1,4),(3,6),(3,4)], 8, 4),
    ("h8_new", [(0,7),(1,4),(2,5),(3,6),(0,2),(1,3),(4,6),(5,7),(0,6),(1,7),(2,4),(3,5),(2,5),(3,4)],       8, 4),
]
TYPES = [("i64","long","std::identity"), ("pair64","pair64","std::identity"),
         ("pair64f","pair64","firstk")]

def _cands_for(name, base, n, k, o):
    return gen_candidates(name, [norm(c) for c in base], n, k, perms=o.perms,
                          max_enum=o.max_enum, max_cands=o.max_cands,
                          seed=o.seed, schedules=o.schedules)

def _report(cands, n, k, o):
    res, spread = emit_and_run(cands, n, k, TYPES, reps=o.reps, runs=o.runs)
    print(f"\n==============  n={n}, split@{k}  |  {len(cands)} candidates, "
          f"metric=min over reps={o.reps}, median over runs={o.runs}  ==============")
    for tname, _, _ in TYPES:
        print(f"\n  {tname}:  median ns/network  [min..max across {o.runs} run(s)]")
        rows = []
        for label, net in cands.items():
            p = res[(label, tname, "plain")]; f = res[(label, tname, "fused")]
            sp = spread[(label, tname, "plain")]
            rows.append((p, f, sp, label, len(net), depth(net, n), reg_pressure(net, n)))
        rows.sort()
        for p, f, sp, label, c, dep, rp in rows[:o.top]:
            star = " <- WINNER" if label == rows[0][3] else ""
            print(f"    {label:<13} plain={p:6.3f} [{sp[0]:.3f}..{sp[1]:.3f}] "
                  f"fused={f:6.3f}  cmp={c} depth={dep} regpress={rp}{star}")
        win = cands[rows[0][3]]
        print("    " + H.format_cpp(win, f"h{n}") + f"  // fastest for {tname}")

def run_for(n, k, names, o):
    cands = {}
    for name, base, nn, kk in TARGETS:
        if nn == n and name in names:
            cands.update(_cands_for(name, base, n, k, o))
    _report(cands, n, k, o)

def run_bases(bases, o):
    groups = {}
    for name, net, n, k in bases:
        groups.setdefault((n, k), []).append((name, net))
    for (n, k), items in sorted(groups.items()):
        cands = {}
        for name, net in items:
            cands.update(_cands_for(name, net, n, k, o))
        _report(cands, n, k, o)

def load_finder_bases(paths, reflect_to_floor=True):
    """Parse halver_finder.py output file(s) into optimiser base nets.  For odd n
    the finder gives split@ceil; reflect to split@floor (what halve_n<n> uses)."""
    bases = []
    for p in paths:
        n, net = H.parse_finder_file_or_stdin(p)
        k = H.detect_split(net, n)
        if k is None:
            sys.exit(f"{p}: parsed net is not a halver")
        name = f"finder_n{n}"
        if reflect_to_floor and k != n // 2 and H.is_halver(H.reflect(net, n), n, n - k):
            net, k = H.reflect(net, n), n - k
            name += "_refl"
        bases.append((name, net, n, k))
    return bases

def main():
    ap = argparse.ArgumentParser(
        description="Empirically autotune a small halver network for THIS CPU/compiler. "
                    "Candidates are the split-preserving wire-permutation group of the base "
                    "net (enumerated exhaustively by default -- no hidden randomness). Each is "
                    "verified, then timed (wall-clock ns/network) and ranked.")
    ap.add_argument("files", nargs="*",
                    help="halver_finder.py output file(s) / '-' for stdin; none -> built-in h7/h8 demo")
    ap.add_argument("--reps", type=int, default=400,
                    help="in-process timing samples; the MIN is the per-run figure (default 400)")
    ap.add_argument("--runs", type=int, default=1,
                    help="re-run the compiled binary N times; report MEDIAN + min..max spread (default 1)")
    ap.add_argument("--perms", choices=["all", "none"], default="all",
                    help="'all' = the split-preserving permutation group (default); 'none' = base only")
    ap.add_argument("--max-enum", type=int, default=20000, dest="max_enum",
                    help="enumerate the group EXHAUSTIVELY up to this size; above it, a reproducible "
                         "random sample of this many is used with --seed (default 20000)")
    ap.add_argument("--max-cands", type=int, default=256, dest="max_cands",
                    help="cap on distinct candidates compiled/timed (default 256)")
    ap.add_argument("--seed", type=int, default=0,
                    help="explicit seed for the random sample fallback (huge groups only; default 0)")
    ap.add_argument("--schedules", action="store_true",
                    help="also try the layer-ordered reschedule of the base net")
    ap.add_argument("--top", type=int, default=6, help="rows printed per element type (default 6)")
    ap.add_argument("--no-reflect", action="store_true", dest="no_reflect",
                    help="don't auto-reflect odd-n finder nets to split@floor")
    o = ap.parse_args()
    if o.files:
        run_bases(load_finder_bases(o.files, reflect_to_floor=not o.no_reflect), o)
    else:
        run_for(7, 3, ["h7_alt", "h7"], o)
        run_for(8, 4, ["h8_new", "h8_old"], o)

if __name__ == "__main__":
    main()
