#!/usr/bin/env python3
"""make_pool.py <N> [cap]  -- build /tmp/hpool.inc for bench_pool.cpp.

Reads every distinct minimal net the search wrote to /tmp/h<N>_*_pool.inc, keeps
those at the smallest comparator count found (the "floor"), caps the list, and
emits a C++ include defining:
    cand::base                       (the current committed hN, the comparison)
    cand::p0 .. cand::pK             (the floor candidates)
    HN_POOL_N / HN_POOL_LIST         (X-macro list the bench iterates)

The baseline hN is read from include/partitions/small_halve.hpp so the bench
always compares against what is actually committed.
"""
import re, sys, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HALVE = ROOT / "include/partitions/small_halve.hpp"

def baseline(n):
    txt = HALVE.read_text()
    m = re.search(r'std::array<P,\s*(\d+)>\s+h%d\s*=\s*\{\{(.*?)\}\}\s*;' % n, txt, re.S)
    if not m:
        sys.exit(f"no h{n} baseline in {HALVE}")
    sz = int(m.group(1))
    body = "".join(m.group(2).split())
    return sz, body

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: make_pool.py <N> [cap]")
    n = int(sys.argv[1])
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    # gather distinct nets from all per-seed pool files
    seen, nets = set(), []
    for f in glob.glob(f"/tmp/h{n}_*_pool.inc"):
        for line in open(f):
            comps = re.findall(r'\{(\d+),(\d+)\}', line)
            if not comps:
                continue
            body = ",".join("{%s,%s}" % (a, b) for a, b in comps)
            key = (len(comps), body)
            if key not in seen:
                seen.add(key); nets.append((len(comps), body))
    if not nets:
        sys.exit(f"no /tmp/h{n}_*_pool.inc files (run the search first)")
    floor = min(c for c, _ in nets)
    pool = [b for c, b in nets if c == floor][:cap]
    bsz, bbody = baseline(n)
    with open("/tmp/hpool.inc", "w") as out:
        out.write("inline constexpr std::array<P,%d> base = {{%s}};\n" % (bsz, bbody))
        names = []
        for i, b in enumerate(pool):
            out.write("inline constexpr std::array<P,%d> p%d = {{%s}};\n" % (floor, i, b))
            names.append("p%d" % i)
        out.write("#define HN_POOL_N %d\n" % len(names))
        out.write("#define HN_POOL_LIST " + " ".join("X(%s)" % x for x in names) + "\n")
    print(f"/tmp/hpool.inc: baseline h{n}={bsz} CE, {len(pool)} candidates @ floor={floor} CE")

if __name__ == "__main__":
    main()
