#!/usr/bin/env python3
"""
halver_cpp_to_layers.py — the inverse of halver_to_cpp.py.

Read a network in the small_halve.hpp array form, e.g.

    inline constexpr std::array<P, 14> h8_new = {{{0,7},{1,4},{2,5},{3,6},
        {0,2},{1,3},{4,6},{5,7},{0,6},{1,7},{2,4},{3,5},{2,5},{3,4}}};

(the full line, just the `{{...}}` body, or a bare list of `{a,b}` pairs) and print
it as topological layers in the SAME format halver_finder.py emits:

    n=8, depth=4, #comparators = 14
    Layer 0: [(0, 7), (1, 4), (2, 5), (3, 6)]
    Layer 1: [(0, 2), (1, 3), (4, 6), (5, 7)]
    ...

The output is round-trippable: it can be piped straight into halver_reflect.py /
halver_optimize.py / halver_to_cpp.py.

Usage:
    python3 halver_cpp_to_layers.py small_halve_h8.txt
    grep 'h8_new =' include/partitions/small_halve.hpp | python3 halver_cpp_to_layers.py
    python3 halver_cpp_to_layers.py file.txt --n 8 --check-split 4
"""
import argparse
import sys

import halver_io as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="-", help="file with the C++ array (or stdin)")
    ap.add_argument("--n", type=int, default=None,
                    help="number of wires (default: inferred as max index + 1)")
    ap.add_argument("--detect-split", action="store_true",
                    help="also detect & report which split the net is a halver for "
                         "(0/1 enumeration; cost grows as 2^n)")
    ap.add_argument("--check-split", type=int, default=None,
                    help="verify the net is a halver at this split point (0/1 enumeration)")
    args = ap.parse_args()

    n, net = H.parse_cpp_file_or_stdin(args.file)
    if args.n is not None:
        n = args.n

    depth = len(H.to_layers(net, n))
    # Core task = layout only: never verify by default (verification is 2^n work,
    # impractical for n=24).  Verify only when explicitly asked.
    if args.check_split is not None:
        ok = H.is_halver(net, n, args.check_split)
        sys.stderr.write(f"[{'ok' if ok else 'FAIL'}] n={n}, {len(net)} comparators, "
                         f"halver @ split {args.check_split}: {ok}\n")
    elif args.detect_split:
        k = H.detect_split(net, n)
        sys.stderr.write(f"[info] n={n}, {len(net)} comparators, depth={depth}, "
                         f"halver @ split {k}\n")
    else:
        sys.stderr.write(f"[info] n={n}, {len(net)} comparators, depth={depth} "
                         f"(use --detect-split to verify)\n")

    # match halver_finder.py's textual layout (header line + Layer lines)
    print(f"n={n}, depth={depth}, #comparators = {len(net)}")
    print(H.format_layers(net, n))


if __name__ == "__main__":
    main()
