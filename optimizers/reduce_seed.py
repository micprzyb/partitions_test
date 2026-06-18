#!/usr/bin/env python3
"""reduce_seed.py <n>  -- derive an n-input halver seed from the best (n+1) net.

Transfer bound (the user's observation that big-n results bound small-n): an
(n+1)-input split@k1 halver, with the top wire's input forced to +infinity,
leaves every comparator that touches the top wire (always the high index) a no-op
on the real values.  Dropping those comparators yields a valid split@k1 halver on
the remaining n wires with FEWER comparators -- both an upper bound and a strong
seed for n.

  * n even : k1 = floor((n+1)/2) = n/2 = k2  -> use the reduced net directly.
  * n odd  : k1 = (n+1)/2 = k2+1            -> reflect (i -> n-1-i) to split@k2.

Source net: campaign_state/n<n+1>/best.txt if present, else the committed h(n+1)
from include/partitions/small_halve.hpp.  Appends one line to /tmp/seeds_<n>.txt.
"""
import re, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HALVE = ROOT / "include/partitions/small_halve.hpp"
STATE = ROOT / os.environ.get("STATE", "optimizers/campaign_state")

def parse_net(s):
    return [(int(a), int(b)) for a, b in re.findall(r'\{(\d+),(\d+)\}', s)]

def source_net(np1):
    best = STATE / f"n{np1}" / "best.txt"
    if best.exists():
        net = parse_net(best.read_text())
        if net:
            return net, f"campaign best n={np1}"
    txt = HALVE.read_text()
    m = re.search(r'std::array<P,\s*\d+>\s+h%d\s*=\s*\{\{(.*?)\}\}\s*;' % np1, txt, re.S)
    if m:
        return parse_net(m.group(1)), f"header h{np1}"
    return None, None

def main():
    n = int(sys.argv[1])
    np1 = n + 1
    net, src = source_net(np1)
    if not net:
        print(f"reduce_seed: no source for n+1={np1}; skipping")
        return
    # drop comparators touching the top wire `n` (always the high index b==n)
    reduced = [(a, b) for (a, b) in net if b != n and a != n]
    if n % 2 == 1:  # reflect i -> n-1-i to convert split@(k+1) into split@k
        reduced = [tuple(sorted((n - 1 - a, n - 1 - b))) for (a, b) in reduced]
    body = ",".join("{%d,%d}" % (a, b) for a, b in reduced)
    seeds = Path(f"/tmp/seeds_{n}.txt")
    with seeds.open("a") as f:
        f.write(body + "\n")
    print(f"reduce_seed n={n}: from {src} ({len(net)} CE) -> seed {len(reduced)} CE "
          f"(transfer upper bound for n={n})")

if __name__ == "__main__":
    main()
