#!/usr/bin/env python3
"""make_seeds.py <N>  -- write /tmp/seeds_<N>.txt for the search tools.

Seeds = the current committed halver hN (from include/partitions/small_halve.hpp)
plus the best-known sorter nN (from include/partitions/small_sort.hpp) for extra
diversity.  Each seed is one line "{a,b},{c,d},...".  The search tools tolerate
extra braces, so the raw array body is fine.

Both seeds are valid split@floor(N/2) networks (a sorter is trivially a halver),
so greedy deletion can prune either down toward the halver minimum.
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HALVE = ROOT / "include/partitions/small_halve.hpp"
SORT = ROOT / "include/partitions/small_sort.hpp"

def extract(path, name):
    """Return the comparator-body string for `inline constexpr std::array<P,..> name = {{...}}`."""
    txt = path.read_text()
    m = re.search(r'std::array<[^>]*>\s+' + re.escape(name) + r'\s*=\s*\{\{(.*?)\}\}\s*;',
                  txt, re.S)
    if not m:
        return None
    body = "".join(m.group(1).split())          # strip whitespace/newlines
    pairs = re.findall(r'\{(\d+),(\d+)\}', body)
    return ",".join("{%s,%s}" % (a, b) for a, b in pairs)

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: make_seeds.py <N>")
    n = int(sys.argv[1])
    seeds = []
    h = extract(HALVE, f"h{n}")
    if h:
        seeds.append(h)
    s = extract(SORT, f"n{n}")
    if s:
        seeds.append(s)
    if not seeds:
        sys.exit(f"no h{n} in {HALVE} nor n{n} in {SORT}")
    out = Path(f"/tmp/seeds_{n}.txt")
    out.write_text("\n".join(seeds) + "\n")
    ces = [len(re.findall(r'\{\d+,\d+\}', x)) for x in seeds]
    print(f"wrote {out}: {len(seeds)} seed net(s), comparator counts {ces}")

if __name__ == "__main__":
    main()
