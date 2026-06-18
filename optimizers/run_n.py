#!/usr/bin/env python3
"""run_n.py <n> <budget_s> <jobs> <core0> <long_restarts> <long_iters> <capK>

Event-driven parallel search for one n (the idea: long per-process runs, but the
instant ANY process improves the global bound, restart the laggards from it while
letting the finders keep their momentum).

  * Launch `jobs` long-running search_halve_cegar processes (each does
    long_restarts x long_iters ILS, flushing its best to its slot file the moment
    it improves -- atomically, so reads are clean).
  * Poll every few seconds:
      - relaunch any process that finished its long budget (fresh RNG seed, current
        frontier) -- keeps stuck processes exploring with diversity;
      - if the global min over all slot files DROPS: accumulate + reseed the
        frontier (merge_reseed, frontier-focused), then KILL+restart only the
        processes whose slot is still above the new min, and LEAVE the finders
        (those already at the new min) running.
  * Also merge periodically so the accumulated pool (uniq.txt, read by finalize)
    stays current, and stop at the time budget.

This self-adapts: improvements fire frequent resyncs while the floor drops fast;
when stuck, processes run their full long budget undisturbed (deep search).
"""
import re, sys, glob, os, time, random, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEARCH = "/tmp/sh_campaign"   # built by campaign_search.sh

def slot_path(n, i): return f"/tmp/h{n}_slot{i}_pool.inc"

def file_min(path):
    try:
        best = 10**9
        for line in open(path):
            c = re.findall(r'\{(\d+),(\d+)\}', line)
            if c:
                best = min(best, len(c))
        return best
    except OSError:
        return 10**9

def seeds_min(n):
    return file_min(f"/tmp/seeds_{n}.txt")

def main():
    n = int(sys.argv[1]); budget = float(sys.argv[2]); jobs = int(sys.argv[3])
    core0 = int(sys.argv[4]); lr = sys.argv[5]; li = sys.argv[6]; capK = sys.argv[7]
    ncores = os.cpu_count() or 32

    # clean any stale pool files for this n
    for f in glob.glob(f"/tmp/h{n}_*_pool.inc"):
        try: os.remove(f)
        except OSError: pass

    state = os.environ.get("STATE", "optimizers/campaign_state")
    errdir = (ROOT / state / f"n{n}"); errdir.mkdir(parents=True, exist_ok=True)
    errlog = open(errdir / "run_n.err", "a")
    seedctr = [1000]
    def launch(i):
        seedctr[0] += 1
        try: os.remove(slot_path(n, i))   # clear so its min reads as +inf until it writes
        except OSError: pass
        core = core0 + (i % ncores)
        return subprocess.Popen(
            ["taskset", "-c", str(core), SEARCH, str(n), str(seedctr[0]), str(lr),
             str(li), str(capK), slot_path(n, i)],
            stdout=subprocess.DEVNULL, stderr=errlog)

    procs = [launch(i) for i in range(jobs)]
    best = seeds_min(n)
    print(f"run_n n={n}: {jobs} jobs, budget {int(budget)}s, start bound={best}", flush=True)

    deadline = time.time() + budget
    last_merge = 0.0
    while time.time() < deadline:
        time.sleep(8)
        # relaunch finished/crashed processes (long budget elapsed -> fresh RNG)
        for i in range(jobs):
            if procs[i].poll() is not None:
                procs[i] = launch(i)
        smin = [file_min(slot_path(n, i)) for i in range(jobs)]
        gmin = min(smin)
        now = time.time()
        if gmin < best:
            subprocess.run(["python3", "optimizers/merge_reseed.py", str(n), "40"],
                           cwd=ROOT, capture_output=True)
            best = gmin
            last_merge = now
            finders = [i for i in range(jobs) if smin[i] <= gmin]
            for i in range(jobs):
                if smin[i] > gmin:           # laggard -> restart from the new bound
                    procs[i].kill(); procs[i].wait()
                    procs[i] = launch(i)
            print(f"run_n n={n}: improved bound -> {best} "
                  f"({len(finders)} finder(s) kept, {jobs-len(finders)} restarted)", flush=True)
        elif now - last_merge > 120:          # keep accumulated pool fresh when stuck
            subprocess.run(["python3", "optimizers/merge_reseed.py", str(n), "40"],
                           cwd=ROOT, capture_output=True)
            last_merge = now

    for p in procs:
        p.kill()
        try: p.wait(timeout=5)
        except Exception: pass
    subprocess.run(["python3", "optimizers/merge_reseed.py", str(n), "40"],
                   cwd=ROOT, capture_output=True)
    print(f"run_n n={n}: done, final bound={best}", flush=True)

if __name__ == "__main__":
    main()
