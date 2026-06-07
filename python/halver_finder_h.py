#!/usr/bin/env python3
"""
Parallel Portfolio CEGAR Hunter for Halver Networks
Architecture: Fast Python Verification + CaDiCaL 3.0.0 Multi-Core Portfolio
"""

import sys
import subprocess
import tempfile
import os
import time
from itertools import combinations
from pysat.card import CardEnc, EncType

def run_cadical_portfolio(num_vars: int, clauses: list, cores: int, seed_offset: int, time_limit: int = 15):
    """
    Spins up multiple CaDiCaL instances with different seeds.
    Returns the model of the FIRST one to finish, and immediately kills the rest.
    """
    fd, cnf_path = tempfile.mkstemp(suffix=".cnf", text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        
        processes = []
        for i in range(cores):
            # Give each core a unique seed based on the current attempt offset
            current_seed = seed_offset + i
            cmd = ["cadical", cnf_path, "--quiet", f"--seed={current_seed}"]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append((p, current_seed))
            
        winner_rc = None
        winner_stdout = None
        winning_seed = None
        
        start_time = time.time()
        # Polling loop: Wait for ANY process to return 10 (SAT) or 20 (UNSAT)
        while True:
            for p, seed in processes:
                rc = p.poll()
                if rc is not None:
                    if rc in (10, 20):
                        winner_rc = rc
                        winner_stdout, _ = p.communicate()
                        winning_seed = seed
                        break
            
            if winner_rc is not None:
                break
                
            if time.time() - start_time > time_limit:
                break # Timeout hit, trigger restart
                
            time.sleep(0.05) # Prevent Python from eating CPU during poll

        # CRITICAL: Terminate all surviving background processes
        for p, _ in processes:
            if p.poll() is None:
                p.kill()
                p.wait()

        if winner_rc == 20:
            return "UNSAT", winning_seed
        elif winner_rc == 10:
            model = []
            for line in winner_stdout.splitlines():
                if line.startswith('v '):
                    tokens = line.split()[1:]
                    for t in tokens:
                        if t == '0': break
                        model.append(int(t))
            return model, winning_seed
        else:
            return "TIMEOUT", None
            
    finally:
        if os.path.exists(cnf_path):
            os.remove(cnf_path)

def fast_python_verify(n: int, k: int, network: list, all_subsets: list) -> list:
    """Simulates the candidate network against all possible inputs."""
    failed_inputs = []
    for S in all_subsets:
        w = [0 if idx in S else 1 for idx in range(n)]
        for layer in network:
            for p, q in layer:
                if w[p] > w[q]:
                    w[p], w[q] = w[q], w[p]
        if not all(w[idx] == 0 for idx in range(k)):
            failed_inputs.append(S)
            if len(failed_inputs) >= 5: # Limit batch size to prevent SAT bloat
                break
    return failed_inputs
def hunt_halver(n: int, d: int, s: int, cores: int, base_timeout: int):
    k = (n + 1) // 2
    all_subsets = list(combinations(range(n), k))
    num_inputs = len(all_subsets)
    
    print(f"[{cores}-CORE HUNTER] Target: n={n}, d={d}, EXACT comparators={s}")
    print(f"[INFO] Verification space: {num_inputs} inputs")
    print(f"[INFO] Base timeout: {base_timeout}s (Will adapt dynamically)")

    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    num_pairs = len(pairs)
    comp_base = 1
    v_base = comp_base + d * num_pairs

    def get_comp_var(l, pair_idx): return comp_base + l * num_pairs + pair_idx
    def get_ce_var(ce_idx, lvl, w): return v_base + ce_idx * (d + 1) * n + lvl * n + w

    base_hard_clauses = []
    for l in range(1, d + 1):
        layer_idx = l - 1
        for w in range(n):
            incident = [get_comp_var(layer_idx, pidx)
                        for pidx, (p, q) in enumerate(pairs) if p == w or q == w]
            for a in range(len(incident)):
                for b in range(a + 1, len(incident)):
                    base_hard_clauses.append([-incident[a], -incident[b]])

    monitored_inputs = []
    iteration = 1
    seed_offset = 1000
    consecutive_timeouts = 0 # Track consecutive failures
    
    while True:
        current_clauses = list(base_hard_clauses)
        
        for ce_idx, S in enumerate(monitored_inputs):
            S_set = set(S)
            for w in range(n):
                v0 = get_ce_var(ce_idx, 0, w)
                current_clauses.append([-v0] if w in S_set else [v0])
            for w in range(k):
                current_clauses.append([-get_ce_var(ce_idx, d, w)])
                
            for l in range(1, d + 1):
                layer_idx, old_lvl = l - 1, l - 1
                for pidx, (p, q) in enumerate(pairs):
                    x = get_comp_var(layer_idx, pidx)
                    vp_old, vq_old = get_ce_var(ce_idx, old_lvl, p), get_ce_var(ce_idx, old_lvl, q)
                    vp_new, vq_new = get_ce_var(ce_idx, l, p), get_ce_var(ce_idx, l, q)
                    current_clauses.extend([
                        [-x, -vp_new, vp_old], [-x, -vp_new, vq_old], [-x, -vp_old, -vq_old, vp_new],
                        [-x, -vp_old, vq_new], [-x, -vq_old, vq_new], [-x, -vq_new, vp_old, vq_old]
                    ])
                    
                for w in range(n):
                    incident = [get_comp_var(layer_idx, pidx) for pidx, (pp, qq) in enumerate(pairs) if pp == w or qq == w]
                    v_new, v_old = get_ce_var(ce_idx, l, w), get_ce_var(ce_idx, old_lvl, w)
                    if not incident:
                        current_clauses.extend([[-v_new, v_old], [-v_old, v_new]])
                    else:
                        current_clauses.extend([[-v_new, v_old] + incident, [v_new, -v_old] + incident])

        max_var_id = v_base - 1 if not monitored_inputs else get_ce_var(len(monitored_inputs) - 1, d, n - 1)
            
        all_comp_lits = [get_comp_var(l, pidx) for l in range(d) for pidx in range(num_pairs)]
        card_cnf = CardEnc.atmost(all_comp_lits, bound=s, top_id=max_var_id, encoding=EncType.seqcounter)
        current_clauses.extend(card_cnf.clauses)
        
        total_max_var = max_var_id
        if card_cnf.clauses:
            total_max_var = max(total_max_var, max(max(abs(lit) for lit in clause) for clause in card_cnf.clauses))
            
        # UPGRADE 1: Adaptive Timeout (Base timeout + 0.5s for every tracked edge case)
        dynamic_timeout = base_timeout + int(len(monitored_inputs) * 0.5)

        print(f"\n--- CEGAR Iter {iteration} | Edge Cases: {len(monitored_inputs)} | Timeout: {dynamic_timeout}s ---")
        print(f"Launching {cores} CaDiCaL workers (Seeds {seed_offset} to {seed_offset + cores - 1})")
        
        result, winning_seed = run_cadical_portfolio(total_max_var, current_clauses, cores, seed_offset, time_limit=dynamic_timeout)
        
        if result == "TIMEOUT":
            consecutive_timeouts += 1
            print(f"  [!] Workers hit {dynamic_timeout}s timeout. (Strike {consecutive_timeouts}/3)")
            
            # UPGRADE 2: CEGAR Amnesia
            if consecutive_timeouts >= 3 and len(monitored_inputs) > 10:
                prune_amount = len(monitored_inputs) // 4
                print(f"  [!] TRAP DETECTED. Forgetting the oldest {prune_amount} edge cases to un-stick the solver.")
                monitored_inputs = monitored_inputs[prune_amount:]
                consecutive_timeouts = 0 # Reset strikes after pruning
                
            seed_offset += cores
            continue 
            
        elif result == "UNSAT":
            print(f"  [X] UNSAT proved by seed {winning_seed}!")
            print(f"  [RESULT] A halver of exactly size {s} is MATHEMATICALLY IMPOSSIBLE for n={n}, d={d}.")
            return
            
        else:
            consecutive_timeouts = 0 # Reset strikes on success
            print(f"  [+] SAT found rapidly by seed {winning_seed}!")
            model = result
            
            model_dict = {abs(lit): (lit > 0) for lit in model}
            network = []
            for l in range(d):
                layer = []
                for pidx, (p, q) in enumerate(pairs):
                    if model_dict.get(get_comp_var(l, pidx), False):
                        layer.append((p, q))
                network.append(layer)
                
            failed = fast_python_verify(n, k, network, all_subsets)
            
            if not failed:
                print("\n" + "="*70)
                print(f"HEURISTIC SUCCESS! VALID NETWORK FOUND.")
                print("="*70)
                for l, layer in enumerate(network):
                    print(f"Layer {l}: {layer}")
                return
                
            # UPGRADE 3: Strict Bottlenecking (Only allow max 2 new edge cases per round to slow bloat)
            added = 0
            for f in failed:
                if f not in monitored_inputs:
                    monitored_inputs.append(f)
                    added += 1
                    if added >= 2: 
                        break
                        
            print(f"  [-] Verification failed on {len(failed)} subsets. Added {added} to monitored list.")
            iteration += 1
            seed_offset += cores


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 halver_portfolio_hunter.py <n> <d> <s> <cores> <timeout>")
        print("Example: python3 halver_portfolio_hunter.py 16 6 45 8 20")
        sys.exit(1)
        
    try:
        subprocess.run(["cadical", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("[ERROR] 'cadical' binary not found in system PATH.")
        sys.exit(1)

    n = int(sys.argv[1])
    d = int(sys.argv[2])
    s = int(sys.argv[3])
    cores = int(sys.argv[4])
    timeout = int(sys.argv[5])
    
    hunt_halver(n, d, s, cores, timeout)
