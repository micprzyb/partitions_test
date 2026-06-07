#!/usr/bin/env python3
"""
halver_to_cpp.py — turn halver_finder.py output into the array literal used in
include/partitions/small_halve.hpp:

    inline constexpr std::array<P, K> hN = {{{a,b},{a,b},...}};

By default it verifies the net as a halver and names it h<n>.  Use --reflect to
emit the split@(M-k) (reflected) form instead — handy for odd-size nets where
halve_n<M> wants split@floor(M/2) but the finder gives split@ceil.

Usage:
    python3 halver_finder.py 8 4 | python3 halver_to_cpp.py
    python3 halver_to_cpp.py finder_out.txt --name h8_new
    python3 halver_finder.py 7 5 | python3 halver_to_cpp.py --reflect --name h7_alt
"""
import argparse
import sys

import halver_io as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="-", help="finder output file (or stdin)")
    ap.add_argument("--name", default=None, help="array name (default h<M>)")
    ap.add_argument("--reflect", action="store_true",
                    help="emit the reflected (split@(M-k)) form")
    args = ap.parse_args()

    n, net = H.parse_finder_file_or_stdin(args.file)
    k = H.detect_split(net, n)
    if k is None:
        sys.exit(f"ERROR: parsed net is not a halver for any split (n={n})")
    if args.reflect:
        net = H.reflect(net, n)
        k = n - k
        if not H.is_halver(net, n, k):
            sys.exit("ERROR: reflected net failed verification")

    name = args.name or f"h{n}"
    sys.stderr.write(f"[ok] n={n}, {len(net)} comparators, verified halver @ split {k}\n")
    print(H.format_cpp(net, name))


if __name__ == "__main__":
    main()
