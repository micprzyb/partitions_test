#!/usr/bin/env python3
"""
halver_reflect.py — convert a halver_finder.py network to the OTHER split point by
reflecting the wires (i -> (M-1)-i).

Reflection maps a split@k halver to a split@(M-k) halver (the bottom/top groups
swap roles).  This is exactly what is needed for ODD-size nets: halver_finder.py
emits an M-input halver at split@ceil(M/2), but `halve_n<M>` returns first + M/2
(= floor), so it needs the split@floor(M/2) version.

For an EVEN n, the (n+1)-input finder net is split@(1 + n/2) = split@ceil; this
script reflects it to split@(n/2) = split@floor, the form `halve_n<n+1>` uses.
(For an even M the finder split already equals floor = ceil, so nothing changes.)

Usage:
    python3 halver_finder.py 7 5 | python3 halver_reflect.py
    python3 halver_reflect.py finder_out.txt [--name h7_alt]
"""
import argparse
import sys

import halver_io as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="-", help="finder output file (or stdin)")
    ap.add_argument("--name", default=None, help="C++ array name (default h<M>)")
    args = ap.parse_args()

    n, net = H.parse_finder_file_or_stdin(args.file)
    k_in = H.detect_split(net, n)
    if k_in is None:
        sys.exit(f"ERROR: input is not a halver for any split (n={n})")

    refl = H.reflect(net, n)
    k_out = n - k_in
    if not H.is_halver(refl, n, k_out):
        sys.exit("ERROR: reflected net failed verification (should never happen)")

    name = args.name or f"h{n}"
    print(f"# input : M={n}, {len(net)} comparators, split@{k_in} "
          f"({'ceil' if k_in == H.finder_split(n) else 'floor' if k_in == n // 2 else 'other'})")
    print(f"# output: reflected i->{n-1}-i  ->  split@{k_out} "
          f"({'floor (halve_n<%d> wants this)' % n if k_out == n // 2 else 'ceil'})  VERIFIED")
    print(H.format_layers(refl, n))
    print(H.format_cpp(refl, name))


if __name__ == "__main__":
    main()
