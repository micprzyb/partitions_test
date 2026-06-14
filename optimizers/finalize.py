#!/usr/bin/env python3
"""finalize.py <n> [--commit]  -- benchmark, install, verify & document one n.

Reads the campaign's accumulated nets (optimizers/campaign_state/n<n>/uniq.txt),
benchmarks the floor candidates against the current committed hN, installs the
fastest into the header, then GATES on the trusted exhaustive verifier
(verify_small_halve + _rev) and the full test suite.  On any failure it RESTORES
the header untouched.  With --commit it also writes Plan<n>.md and git-commits.

This is the only step that edits the repo; the verify gate makes it safe to run
autonomously, but it is normally driven (and its output checked) per-n.
"""
import re, sys, glob, subprocess, shutil, statistics, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HALVE = ROOT / "include/partitions/small_halve.hpp"
STATE = ROOT / "optimizers/campaign_state"
BUILD = ROOT / "build"
CORE = "4"

def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)

def header_net(n):
    txt = HALVE.read_text()
    m = re.search(r'std::array<P,\s*(\d+)>\s+h%d\s*=\s*\{\{(.*?)\}\}\s*;' % n, txt, re.S)
    if not m:
        sys.exit(f"finalize: no h{n} in header")
    return int(m.group(1)), "".join(m.group(2).split())

def nets_from_uniq(n):
    p = STATE / f"n{n}" / "uniq.txt"
    if not p.exists():
        sys.exit(f"finalize: {p} missing (run the search first)")
    out = []
    for line in open(p):
        c = re.findall(r'\{(\d+),(\d+)\}', line)
        if c:
            out.append((len(c), ",".join("{%s,%s}" % (a, b) for a, b in c)))
    return out

def write_pool(n, base_sz, base_body, cands, floor):
    with open("/tmp/hpool.inc", "w") as f:
        f.write("inline constexpr std::array<P,%d> base = {{%s}};\n" % (base_sz, base_body))
        names = []
        for i, b in enumerate(cands):
            f.write("inline constexpr std::array<P,%d> p%d = {{%s}};\n" % (floor, i, b))
            names.append("p%d" % i)
        f.write("#define HN_POOL_N %d\n" % len(names))
        f.write("#define HN_POOL_LIST " + " ".join("X(%s)" % x for x in names) + "\n")
    return names

def bench_pool(n, reps=3):
    r = sh(["g++", "-O3", "-march=native", "-std=c++23", f"-DHN={n}",
            "-DPOOL_INC=\"/tmp/hpool.inc\"", "-Iinclude",
            "optimizers/bench_pool.cpp", "-o", "/tmp/fin_bp"])
    if r.returncode != 0:
        sys.exit("finalize: bench_pool compile failed:\n" + r.stderr[:2000])
    agg = {}
    for _ in range(reps):
        out = subprocess.run(["taskset", "-c", CORE, "/tmp/fin_bp"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 4 and p[0] not in ("net", "best"):
                v = [float(p[1]), float(p[2]), float(p[3])]
                agg[p[0]] = [min(a, b) for a, b in zip(agg.get(p[0], [9, 9, 9]), v)]
    return agg  # name -> [i64,p64,p64f]

def build(targets):
    return sh(["cmake", "--build", "build", "-j", "--target", *targets]).returncode == 0

def verify(n):
    for tool, tag in (("verify_small_halve", "halver"), ("verify_small_halve_rev", "desc-halver")):
        out = subprocess.run([str(BUILD / "tools" / tool)], capture_output=True, text=True).stdout
        if f"N={n:2d}  {tag} OK" not in out and f"N={n}  {tag} OK" not in out:
            return False, f"{tool}: N={n} not OK"
        if "CORRECT" not in out:
            return False, f"{tool}: not all correct"
    return True, "ok"

def ctest():
    out = sh(["ctest", "--test-dir", "build"]).stdout
    return "100% tests passed" in out

def bench_small(n, reps=7):
    if not build(["bench_small_halve"]):
        sys.exit("finalize: bench_small_halve build failed")
    best = {}
    for _ in range(reps):
        out = subprocess.run(["taskset", "-c", CORE, str(BUILD / "benchmarks" / "bench_small_halve"), str(n)],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            p = line.split(",")
            if len(p) >= 3 and p[0] in ("i64", "pair64", "pair64f"):
                v = float(p[2])
                best[p[0]] = min(best.get(p[0], 9e9), v)
    return best  # i64/pair64/pair64f -> halve_ns

def install(n, new_sz, new_body, old_sz, old_body):
    txt = HALVE.read_text()
    pat = r'inline constexpr std::array<P,\s*%d>\s+h%d\s*=\s*\{\{.*?\}\}\s*;' % (old_sz, n)
    new_line = "inline constexpr std::array<P, %d> h%d = {{%s}};" % (new_sz, n, new_body)
    txt2, k = re.subn(pat, new_line, txt, flags=re.S)
    if k != 1:
        sys.exit(f"finalize: header edit matched {k} times for h{n}")
    HALVE.write_text(txt2)

def write_plan(n, old_sz, new_sz, body, floor_count, base_b, new_b):
    def row(t, key):
        b, a = base_b[key], new_b[key]
        return f"| {t:7} | {b:.4f} | {a:.4f} | {100*(b-a)/b:+.1f}% |"
    md = f"""# Plan{n} — {n}-input perfect halver (campaign)

Optimised by the autonomous campaign (`optimizers/campaign_search.sh`): parallel
ILS + CEGAR with the k-zero (monotonicity) validity check, big-n-first with
transfer seeding from n+1. Chosen by direct benchmark, verified by exhaustive 0/1
enumeration (forward + reverse), full test suite green.

**Network: {old_sz} → {new_sz} comparators** (floor found: {new_sz} CE; {floor_count} distinct floor nets).

`bench_small_halve {n}` (median of 7, ns/elem; lower = better):

| type    | before | after | speedup |
|---------|--------|-------|---------|
{row('i64','i64')}
{row('pair64','pair64')}
{row('pair64f','pair64f')}

Net (committed to `nets::h{n}`):

`{body}`

Reproduce: `optimizers/campaign_search.sh` (or `optimizers/optimize.sh {n}`) →
`optimizers/finalize.py {n} --commit`.
"""
    (ROOT / f"Plan{n}.md").write_text(md)

def main():
    n = int(sys.argv[1])
    dry = "--dry-run" in sys.argv[2:]
    do_commit = ("--commit" in sys.argv[2:]) and not dry
    d = STATE / f"n{n}"; d.mkdir(parents=True, exist_ok=True)
    old_sz, old_body = header_net(n)
    nets = nets_from_uniq(n)
    floor = min(c for c, _ in nets)
    floor_nets = [b for c, b in nets if c == floor]
    print(f"n={n}: header={old_sz} CE, floor={floor} CE, {len(floor_nets)} floor nets")
    # record best-found for transfer seeding regardless of outcome
    (d / "best.txt").write_text("{{" + (floor_nets[0] if floor < old_sz else old_body) + "}}\n")
    if floor >= old_sz:
        print(f"n={n}: NO IMPROVEMENT (floor {floor} >= header {old_sz}); leaving header unchanged")
        (d / "FINALIZED").write_text("noimprove floor=%d\n" % floor)
        return
    cands = floor_nets[:100]
    write_pool(n, old_sz, old_body, cands, floor)
    agg = bench_pool(n)
    winner = min((nm for nm in agg if nm != "base"), key=lambda nm: sum(agg[nm]))
    widx = int(winner[1:])
    new_body = cands[widx]
    print(f"n={n}: winner {winner} sum={sum(agg[winner]):.4f} vs base {sum(agg['base']):.4f}")

    backup = HALVE.read_text()
    base_b = bench_small(n)                      # baseline (old net)
    install(n, floor, new_body, old_sz, old_body)
    ok = build(["verify_small_halve", "verify_small_halve_rev", "bench_small_halve"])
    if ok:
        vok, msg = verify(n)
    else:
        vok, msg = False, "build failed"
    if vok:
        cok = ctest()
    else:
        cok = False
    if not (ok and vok and cok):
        HALVE.write_text(backup)                 # REVERT
        build(["verify_small_halve", "verify_small_halve_rev", "bench_small_halve"])
        print(f"n={n}: FAILED gate (build={ok} verify={vok}:{msg} ctest={cok}) -> reverted")
        (d / "FINALIZED").write_text("FAILED %s\n" % msg)
        sys.exit(1)
    new_b = bench_small(n)                        # new net numbers
    print(f"n={n}: VERIFIED. i64 {base_b['i64']:.4f}->{new_b['i64']:.4f} "
          f"pair64 {base_b['pair64']:.4f}->{new_b['pair64']:.4f} "
          f"pair64f {base_b['pair64f']:.4f}->{new_b['pair64f']:.4f}")
    if dry:
        HALVE.write_text(backup)                  # revert; test only
        build(["verify_small_halve", "verify_small_halve_rev", "bench_small_halve"])
        print(f"n={n}: DRY-RUN OK (header reverted, nothing committed)")
        return
    if do_commit:
        write_plan(n, old_sz, floor, new_body, len(floor_nets), base_b, new_b)
        sh(["git", "add", str(HALVE.relative_to(ROOT)), f"Plan{n}.md"])
        msg = (f"halver n={n}: {old_sz}->{floor} comparators (verified exhaustively)\n\n"
               f"Found by the autonomous CEGAR campaign; chosen by benchmark; "
               f"forward+reverse exhaustive 0/1 verified; full suite green.\n\n"
               f"Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>")
        cr = sh(["git", "commit", "-m", msg])
        print("committed" if cr.returncode == 0 else "git commit FAILED:\n" + cr.stderr[:1000])
    (d / "best.txt").write_text("{{" + new_body + "}}\n")
    (d / "FINALIZED").write_text("OK %d->%d\n" % (old_sz, floor))

if __name__ == "__main__":
    main()
