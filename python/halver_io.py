#!/usr/bin/env python3
"""
Shared helpers for the halver tooling: parse halver_finder.py output, verify the
halver property, reflect wires, and emit the small_halve.hpp array literal.

halver_finder.py prints (for `python3 halver_finder.py <n> <d>`):

    n=8, depth=4, minimal #comparators = 14
    Layer 0: [(0, 7), (1, 4), (2, 5), (3, 6)]
    Layer 1: [(0, 2), (1, 3), (4, 6), (5, 7)]
    ...

and its halver puts the k = (n+1)//2 smallest into the bottom (split@k, k=ceil(n/2)).
"""
import re
import sys


def parse_finder(text):
    """Return (n, net) from halver_finder.py output text.  net = list of (a,b)."""
    n = None
    for line in text.splitlines():
        m = re.search(r'\bn\s*=\s*(\d+)', line)
        if m and 'depth' in line:
            n = int(m.group(1))
    net = []
    for line in text.splitlines():
        m = re.match(r'\s*Layer\s+\d+\s*:\s*\[(.*)\]\s*$', line)
        if m:
            for a, b in re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', m.group(1)):
                net.append(norm((int(a), int(b))))
    if not net:
        raise ValueError("no 'Layer N: [...]' lines found in input")
    if n is None:
        n = 1 + max(max(a, b) for a, b in net)
    return n, net


def parse_finder_file_or_stdin(path=None):
    text = sys.stdin.read() if path in (None, "-") else open(path).read()
    return parse_finder(text)


def parse_cpp_array(text):
    """Parse a small_halve.hpp-style array literal into (n, net).  Accepts the full
    `inline constexpr std::array<P, K> name = {{{a,b},...}};` line, just the
    `{{a,b},...}` body, or a bare list of {a,b} pairs -- it simply collects every
    `{a,b}` comparator in order (the `<P, K>` size and outer braces are ignored).
    n is inferred as max wire index + 1."""
    pairs = re.findall(r'\{\s*(\d+)\s*,\s*(\d+)\s*\}', text)
    if not pairs:
        raise ValueError("no {a,b} comparators found in input")
    net = [norm((int(a), int(b))) for a, b in pairs]
    n = 1 + max(max(a, b) for a, b in net)
    return n, net


def parse_cpp_file_or_stdin(path=None):
    text = sys.stdin.read() if path in (None, "-") else open(path).read()
    return parse_cpp_array(text)


def norm(c):
    a, b = c
    return (a, b) if a < b else (b, a)


def finder_split(n):
    """The split halver_finder.py targets: k = ceil(n/2) smallest in the bottom."""
    return (n + 1) // 2


def _make_wires(n):
    """w[i] = the 2^n-bit column holding input bit i across all 2^n inputs, built by
    bit-DOUBLING (O(n log 2^n) big-int ops) instead of an O(2^n * n) per-input loop.
    w[i] is the periodic pattern: 2^i zeros, 2^i ones, repeated."""
    N = 1 << n
    full = (1 << N) - 1
    w = []
    for i in range(n):
        block = 1 << i
        x = ((1 << block) - 1) << block       # one period: 2^i zeros then 2^i ones
        period = block * 2
        while period < N:                      # tile across all N bits by doubling
            x |= x << period
            period <<= 1
        w.append(x & full)
    return w


def _apply(net, wires):
    """Run the comparator network on the input columns (min->low index)."""
    w = list(wires)
    for a, b in net:
        lo = w[a] & w[b]
        w[a], w[b] = lo, w[a] | w[b]
    return w


def _split_ok(w, n, k):
    full = (1 << (1 << n)) - 1
    orbottom = 0
    for i in range(k):
        orbottom |= w[i]
    andtop = full
    for i in range(k, n):
        andtop &= w[i]
    return (orbottom & ~andtop) == 0


def is_halver(net, n, k):
    """Exhaustive 0/1 proof: for every input, max(bottom k) <= min(top n-k)."""
    return _split_ok(_apply(net, _make_wires(n)), n, k)


def detect_split(net, n):
    """Return the split k (1..n-1) this net is a halver for, or None.  Builds the
    columns and applies the net ONCE, then tests each k (cheap)."""
    w = _apply(net, _make_wires(n))
    for k in range(1, n):
        if _split_ok(w, n, k):
            return k
    return None


def reflect(net, n):
    """Reflect wires i -> (n-1)-i.  Maps a split@k halver to a split@(n-k) halver
    (verified empirically and provable from the 0-1 principle)."""
    return [norm((n - 1 - b, n - 1 - a)) for a, b in net]


def to_layers(net, n):
    """Group comparators into greedy topological layers (display only)."""
    t = [0] * n
    layers = {}
    for a, b in net:
        L = max(t[a], t[b]) + 1
        t[a] = t[b] = L
        layers.setdefault(L, []).append((a, b))
    return [layers[L] for L in sorted(layers)]


def format_layers(net, n):
    out = []
    for i, layer in enumerate(to_layers(net, n)):
        out.append(f"Layer {i}: {layer}")
    return "\n".join(out)


def format_cpp(net, name):
    """Emit the small_halve.hpp array literal:
       inline constexpr std::array<P, K> name = {{{a,b},...}};"""
    body = ",".join("{%d,%d}" % (a, b) for a, b in net)
    return "inline constexpr std::array<P, %d> %s = {{%s}};" % (len(net), name, body)
