#!/usr/bin/env python3
"""merge_reseed.py <n> <frontier_cap>  -- between search waves.

1. Merge the nets the latest wave wrote (/tmp/h<n>_*_pool.inc) into the campaign's
   accumulated distinct set (campaign_state/n<n>/uniq.txt).
2. Recompute the floor (smallest comparator count seen).
3. Write the NEXT wave's seed file /tmp/seeds_<n>.txt =
       the fixed base seeds (/tmp/seeds_base_<n>.txt: header hN + sorter + transfer)
     + a random sample (<= frontier_cap) of the current floor nets,
   so successive waves keep climbing down from the frontier while never losing the
   good starting points.
Prints: floor=<F> floornets=<C> distinct=<D>
"""
import re, sys, glob, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "optimizers/campaign_state"

def nets_in(path):
    out = []
    for line in open(path):
        comps = re.findall(r'\{(\d+),(\d+)\}', line)
        if comps:
            out.append((len(comps), ",".join("{%s,%s}" % (a, b) for a, b in comps)))
    return out

def main():
    n = int(sys.argv[1]); cap = int(sys.argv[2])
    d = STATE / f"n{n}"; d.mkdir(parents=True, exist_ok=True)
    uniq = d / "uniq.txt"
    seen = {}
    if uniq.exists():
        for c, b in nets_in(uniq):
            seen[b] = c
    for f in glob.glob(f"/tmp/h{n}_*_pool.inc"):
        for c, b in nets_in(f):
            seen[b] = c
    if not seen:
        print("floor=0 floornets=0 distinct=0"); return
    with uniq.open("w") as o:
        for b in seen:
            o.write("{{" + b + "}}\n")
    floor = min(seen.values())
    floornets = [b for b, c in seen.items() if c == floor]
    # keep best.txt current so reduce_seed.py (transfer to n-1) uses the search's
    # best even before finalize.py runs.
    (d / "best.txt").write_text("{{" + floornets[0] + "}}\n")
    base = []
    bp = Path(f"/tmp/seeds_base_{n}.txt")
    if bp.exists():
        base = [l.strip() for l in bp.read_text().splitlines() if l.strip()]
    rng = random.Random()
    sample = floornets[:] ; rng.shuffle(sample); sample = sample[:cap]
    Path(f"/tmp/seeds_{n}.txt").write_text("\n".join(base + sample) + "\n")
    print(f"floor={floor} floornets={len(floornets)} distinct={len(seen)}")

if __name__ == "__main__":
    main()
