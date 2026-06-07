#!/usr/bin/env python3
"""
Optimal Perfect Halver Network Generator via SAT
Optimized for CaDiCaL 3.0.0 (System Binary Bypass via DIMACS CNF)
"""

import sys
import subprocess
import tempfile
import os
from itertools import combinations
from pysat.card import CardEnc, EncType

def run_cadical_binary(num_vars: int, clauses: list) -> list:
    """
    Writes clauses to a DIMACS CNF file, invokes the system CaDiCaL binary,
    and parses the output. Returns the model (list of ints) if SAT, or None if UNSAT.
    """
    # Create a temporary file for the DIMACS format
    fd, cnf_path = tempfile.mkstemp(suffix=".cnf", text=True)
    
    try:
        with os.fdopen(fd, 'w') as f:
            # DIMACS Header: p cnf <vars> <clauses>
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            # Write all clauses rapidly
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        
        # Invoke CaDiCaL 3.0.0
        # --quiet minimizes IO blocking. It still prints 's' and 'v' lines.
        cmd = ["cadical", cnf_path, "--quiet"]
        
        # check=False because CaDiCaL returns 10 (SAT) or 20 (UNSAT), which are "errors" to subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Exit code 20 means UNSAT
        if result.returncode == 20:
            return None
        
        # Exit code 10 means SAT
        if result.returncode == 10:
            model = []
            # Parse the model from lines starting with 'v'
            for line in result.stdout.splitlines():
                if line.startswith('v '):
                    # Extract variables, ignore the 'v' and the trailing '0'
                    tokens = line.split()[1:]
                    for t in tokens:
                        if t == '0':
                            break
                        model.append(int(t))
            return model
            
        # If it returns something else, something broke (e.g., binary not found)
        raise RuntimeError(f"CaDiCaL failed with exit code {result.returncode}.\nStderr: {result.stderr}")

    finally:
        # Always clean up the temp file
        if os.path.exists(cnf_path):
            os.remove(cnf_path)


def optimal_perfect_halver(n: int, d: int):
    k = (n + 1) // 2
    m = n - k

    subsets = list(combinations(range(n), k))
    num_inputs = len(subsets)
    print(f"[INFO] Targeting System Binary: CaDiCaL 3.0.0")
    print(f"[INFO] n={n}, d={d}, k={k} (LO), m={m} (HI), #0-1-inputs={num_inputs}")

    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    num_pairs = len(pairs)

    comp_base = 1
    v_base = comp_base + d * num_pairs

    def get_comp_var(l: int, pair_idx: int) -> int:
        return comp_base + l * num_pairs + pair_idx

    def get_value_var(i: int, lvl: int, w: int) -> int:
        return v_base + i * (d + 1) * n + lvl * n + w

    # Calculate max variable ID for CardEnc and DIMACS header
    max_var_id = get_value_var(num_inputs - 1, d, n - 1)

    lower_bound_s = max(k, m)
    upper_bound_s = d * (n // 2)
    print(f"[INFO] Searching s ∈ [{lower_bound_s}, {upper_bound_s}] ...")

    min_s = None
    best_layers = None
    hard_clauses = []

    # 1. Initial values
    for i, S in enumerate(subsets):
        S_set = set(S)
        for w in range(n):
            v0 = get_value_var(i, 0, w)
            hard_clauses.append([-v0] if w in S_set else [v0])

    # 2. Final correctness
    for i in range(num_inputs):
        for w in range(k):
            hard_clauses.append([-get_value_var(i, d, w)])

    # 3. Structural constraints
    for l in range(1, d + 1):
        layer_idx = l - 1
        for w in range(n):
            incident = [get_comp_var(layer_idx, pidx)
                        for pidx, (p, q) in enumerate(pairs) if p == w or q == w]
            for a in range(len(incident)):
                for b in range(a + 1, len(incident)):
                    hard_clauses.append([-incident[a], -incident[b]])

    # 4. Propagation & Copy-when-idle
    for i, S in enumerate(subsets):
        for l in range(1, d + 1):
            old_lvl = l - 1
            layer_idx = l - 1 

            for pidx, (p, q) in enumerate(pairs):
                x = get_comp_var(layer_idx, pidx)
                vp_old = get_value_var(i, old_lvl, p)
                vq_old = get_value_var(i, old_lvl, q)
                vp_new = get_value_var(i, l, p)
                vq_new = get_value_var(i, l, q)

                hard_clauses.extend([
                    [-x, -vp_new, vp_old],
                    [-x, -vp_new, vq_old],
                    [-x, -vp_old, -vq_old, vp_new],
                    [-x, -vp_old, vq_new],
                    [-x, -vq_old, vq_new],
                    [-x, -vq_new, vp_old, vq_old]
                ])

            for w in range(n):
                incident = [get_comp_var(layer_idx, pidx)
                            for pidx, (pp, qq) in enumerate(pairs) if pp == w or qq == w]
                v_new = get_value_var(i, l, w)
                v_old = get_value_var(i, old_lvl, w)
                if not incident:
                    hard_clauses.extend([[-v_new, v_old], [-v_old, v_new]])
                    continue
                hard_clauses.extend([
                    [-v_new, v_old] + incident,
                    [v_new, -v_old] + incident
                ])

    print(f"[INFO] Hard clauses generated: {len(hard_clauses):,} (Ready for export)")

    # 5. Search for minimal s
    all_comp_lits = [get_comp_var(l, pidx) for l in range(d) for pidx in range(num_pairs)]

    for s in range(lower_bound_s, upper_bound_s + 1):
        print(f"[SEARCH] Trying s = {s} ...", end=" ", flush=True)

        # Generate cardinality clauses for this specific 's'
        card_cnf = CardEnc.atmost(all_comp_lits, bound=s, top_id=max_var_id, encoding=EncType.seqcounter)
        
        # Combine hard clauses with cardinality clauses
        total_clauses = hard_clauses + card_cnf.clauses
        
        # The true max variable ID might now be higher due to CardEnc's auxiliary variables
        current_max_var = max_var_id
        if card_cnf.clauses:
            current_max_var = max(current_max_var, max(max(abs(lit) for lit in clause) for clause in card_cnf.clauses))

        # Invoke the System CaDiCaL Binary
        model = run_cadical_binary(num_vars=current_max_var, clauses=total_clauses)

        if model is not None:
            print("SAT -> optimal!")
            min_s = s
            
            # Map the model to a dictionary for easy O(1) lookups
            # CaDiCaL returns a 1-based list of literals where negative means false.
            model_dict = {abs(lit): (lit > 0) for lit in model}

            best_layers = []
            for l in range(d):
                layer = []
                for pidx, (p, q) in enumerate(pairs):
                    var = get_comp_var(l, pidx)
                    # Check if the comparator variable is true in the model
                    if model_dict.get(var, False): 
                        layer.append((p, q))
                best_layers.append(layer)
            break
        else:
            print("UNSAT")

    if min_s is None:
        print(f"[RESULT] No perfect halver of depth {d} exists for n={n}.")
        return None, None

    # Output
    print("\n" + "="*70)
    print(f"OPTIMAL PERFECT HALVER FOUND (Powered by CaDiCaL 3.0.0)")
    print(f"n={n}, depth={d}, minimal #comparators = {min_s}")
    print("="*70)
    for l, layer in enumerate(best_layers):
        print(f"Layer {l}: {layer}")
    return min_s, best_layers


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 optimal_halver_cadical.py <n> <d>")
        sys.exit(1)
    
    # Quick sanity check to ensure the cadical binary is installed
    try:
        subprocess.run(["cadical", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("[ERROR] 'cadical' binary not found in system PATH.")
        print("Please ensure CaDiCaL 3.0.0 is installed and accessible.")
        sys.exit(1)

    n = int(sys.argv[1])
    d = int(sys.argv[2])
    optimal_perfect_halver(n, d)
