#!/usr/bin/env python3
"""
Optimal Perfect Halver Network Generator
Architecture: CEGAR (Counter-Example Guided Abstraction Refinement) + CaDiCaL 3.0.0 Binary Bypass
"""

import sys
import subprocess
import tempfile
import os
from itertools import combinations
from pysat.card import CardEnc, EncType

def run_cadical_binary(num_vars: int, clauses: list) -> list:
    """Invokes the CaDiCaL system binary via DIMACS file."""
    fd, cnf_path = tempfile.mkstemp(suffix=".cnf", text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        
        cmd = ["cadical", cnf_path, "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 20:
            return None # UNSAT
        if result.returncode == 10:
            model = []
            for line in result.stdout.splitlines():
                if line.startswith('v '):
                    tokens = line.split()[1:]
                    for t in tokens:
                        if t == '0': break
                        model.append(int(t))
            return model
        raise RuntimeError(f"CaDiCaL failed. Exit code: {result.returncode}\n{result.stderr}")
    finally:
        if os.path.exists(cnf_path):
            os.remove(cnf_path)

def fast_python_verify(n: int, k: int, network: list, all_subsets: list) -> list:
    """
    Simulates the candidate network against all possible inputs in raw Python.
    Returns a list of Counter-Examples (inputs that failed to halve).
    """
    failed_inputs = []
    for S in all_subsets:
        # Create input array: 0 if index is in the LO subset, 1 otherwise
        w = [0 if idx in S else 1 for idx in range(n)]
        
        # Pass through the network layers
        for layer in network:
            for p, q in layer:
                if w[p] > w[q]:
                    w[p], w[q] = w[q], w[p] # Swap: smaller value goes left (p)
                    
        # Check correctness: The first k wires must all be 0
        if not all(w[idx] == 0 for idx in range(k)):
            failed_inputs.append(S)
            
            # Optimization: Only gather a small batch of counter-examples per iteration
            # Adding too many at once bloats the SAT solver unnecessarily.
            if len(failed_inputs) >= 5:
                break
                
    return failed_inputs

def optimal_perfect_halver_cegar(n: int, d: int):
    k = (n + 1) // 2
    m = n - k

    all_subsets = list(combinations(range(n), k))
    print(f"[INFO] Target: Perfect Halver (n={n}, d={d}, LO={k})")
    print(f"[INFO] Total verification space: {len(all_subsets)} inputs")

    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    num_pairs = len(pairs)

    comp_base = 1
    v_base = comp_base + d * num_pairs

    def get_comp_var(l: int, pair_idx: int) -> int:
        return comp_base + l * num_pairs + pair_idx

    # Dynamic Variable Allocator: Only generate variables for tracked counter-examples
    def get_ce_var(ce_idx: int, lvl: int, w: int) -> int:
        return v_base + ce_idx * (d + 1) * n + lvl * n + w

    lower_bound_s = max(k, m)
    upper_bound_s = d * (n // 2)

    # Base Structural Clauses (At most 1 comparator per wire per layer)
    base_hard_clauses = []
    for l in range(1, d + 1):
        layer_idx = l - 1
        for w in range(n):
            incident = [get_comp_var(layer_idx, pidx)
                        for pidx, (p, q) in enumerate(pairs) if p == w or q == w]
            for a in range(len(incident)):
                for b in range(a + 1, len(incident)):
                    base_hard_clauses.append([-incident[a], -incident[b]])

    # The CEGAR state: persists across different sizes (s)
    monitored_inputs = []

    for s in range(lower_bound_s, upper_bound_s + 1):
        print(f"\n[SEARCH] Exploring size s = {s} ...")
        iteration = 1
        
        while True: # The Feedback Loop
            current_clauses = list(base_hard_clauses)
            
            # Generate clause logic strictly for our monitored counter-examples
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

            # Dynamically calculate the highest variable ID used
            max_var_id = v_base - 1 if not monitored_inputs else get_ce_var(len(monitored_inputs) - 1, d, n - 1)
                
            # Add cardinality logic for size 's'
            all_comp_lits = [get_comp_var(l, pidx) for l in range(d) for pidx in range(num_pairs)]
            card_cnf = CardEnc.atmost(all_comp_lits, bound=s, top_id=max_var_id, encoding=EncType.seqcounter)
            current_clauses.extend(card_cnf.clauses)
            
            # Ensure CaDiCaL header gets the absolute highest variable from the cardinality logic
            total_max_var = max_var_id
            if card_cnf.clauses:
                total_max_var = max(total_max_var, max(max(abs(lit) for lit in clause) for clause in card_cnf.clauses))
                
            # Invoke CaDiCaL
            model = run_cadical_binary(num_vars=total_max_var, clauses=current_clauses)
            
            if model is None:
                print(f"  -> Iteration {iteration}: UNSAT. No perfect halver of size {s} exists.")
                break # Move to the next s
                
            # Extract candidate network
            model_dict = {abs(lit): (lit > 0) for lit in model}
            network = []
            for l in range(d):
                layer = []
                for pidx, (p, q) in enumerate(pairs):
                    if model_dict.get(get_comp_var(l, pidx), False):
                        layer.append((p, q))
                network.append(layer)
                
            # Verify the candidate against ALL inputs in Python
            failed = fast_python_verify(n, k, network, all_subsets)
            
            if not failed:
                print(f"  -> Iteration {iteration}: VERIFIED! Network handles all {len(all_subsets)} inputs perfectly.")
                print("\n" + "="*70)
                print(f"OPTIMAL PERFECT HALVER FOUND (CEGAR + CaDiCaL 3.0.0)")
                print(f"n={n}, depth={d}, minimal #comparators = {s}")
                print(f"Total edge-cases needed to prove: {len(monitored_inputs)}")
                print("="*70)
                for l, layer in enumerate(network):
                    print(f"Layer {l}: {layer}")
                return s, network
                
            # Refine the abstraction by adding new Counter-Examples
            added = 0
            for f in failed:
                if f not in monitored_inputs:
                    monitored_inputs.append(f)
                    added += 1
                    
            print(f"  -> Iteration {iteration}: Candidate failed on {len(failed)} tested subsets. Added {added} to monitored list (Total: {len(monitored_inputs)}).")
            iteration += 1

    print(f"[RESULT] No perfect halver of depth {d} exists for n={n}.")
    return None, None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 optimal_halver_cegar.py <n> <d>")
        sys.exit(1)
    
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    optimal_perfect_halver_cegar(n, d)
