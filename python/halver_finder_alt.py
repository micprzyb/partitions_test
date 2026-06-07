#!/usr/bin/env python3
"""
Optimal Perfect Halver Network Generator via SAT
Fixed: Variable ID collision, duplicate structural clauses, and removed dead symmetry block.
"""

import sys
from itertools import combinations
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver, Glucose3
# Try to explicitly import the versioned CaDiCaL your python-sat supports
try:
    from pysat.solvers import Cadical300 # Change to Cadical195 if that's what Step 2 output
    def create_solver():
        return Cadical300()
    SOLVER_NAME = "CaDiCaL (v1.5.3)"
except ImportError:
    def create_solver():
        return Glucose3()
    SOLVER_NAME = "Glucose3"# Try to use CaDiCaL via the generic Solver factory, fall back to Glucose3

print(f"[INFO] Using solver: {SOLVER_NAME}")

def optimal_perfect_halver(n: int, d: int):
    k = (n + 1) // 2
    m = n - k

    subsets = list(combinations(range(n), k))
    num_inputs = len(subsets)
    print(f"[INFO] n={n}, d={d}, k={k} (LO), m={m} (HI), #0-1-inputs={num_inputs}")

    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    num_pairs = len(pairs)

    comp_base = 1
    v_base = comp_base + d * num_pairs

    def get_comp_var(l: int, pair_idx: int) -> int:
        return comp_base + l * num_pairs + pair_idx

    def get_value_var(i: int, lvl: int, w: int) -> int:
        return v_base + i * (d + 1) * n + lvl * n + w

    # CRITICAL FIX: Calculate the highest variable ID used in our manual logic
    # so we can explicitly tell PySAT where to start creating auxiliary variables.
    max_var_id = get_value_var(num_inputs - 1, d, n - 1)

    lower_bound_s = max(k, m)
    upper_bound_s = d * (n // 2)
    print(f"[INFO] Searching s ∈ [{lower_bound_s}, {upper_bound_s}] ...")

    min_s = None
    best_layers = None

    hard_clauses = []

    # 1. Initial values (layer 0)
    for i, S in enumerate(subsets):
        S_set = set(S)
        for w in range(n):
            v0 = get_value_var(i, 0, w)
            hard_clauses.append([-v0] if w in S_set else [v0])

    # 2. Final correctness: wires 0..k-1 must be 0 after last layer
    for i in range(num_inputs):
        for w in range(k):
            hard_clauses.append([-get_value_var(i, d, w)])

    # 3. Structural constraints (Pairwise at-most-1 per wire)
    # CRITICAL FIX: Moved this OUTSIDE the input scenarios loop to prevent duplicate clauses
    for l in range(1, d + 1):
        layer_idx = l - 1
        for w in range(n):
            incident = [get_comp_var(layer_idx, pidx)
                        for pidx, (p, q) in enumerate(pairs) if p == w or q == w]
            for a in range(len(incident)):
                for b in range(a + 1, len(incident)):
                    hard_clauses.append([-incident[a], -incident[b]])

    # 4. Propagation & Copy-when-idle (Input dependent)
    for i, S in enumerate(subsets):
        for l in range(1, d + 1):
            old_lvl = l - 1
            layer_idx = l - 1 

            # Propagation: min on p, max on q when comparator active
            for pidx, (p, q) in enumerate(pairs):
                x = get_comp_var(layer_idx, pidx)
                vp_old = get_value_var(i, old_lvl, p)
                vq_old = get_value_var(i, old_lvl, q)
                vp_new = get_value_var(i, l, p)
                vq_new = get_value_var(i, l, q)

                # MIN on p
                hard_clauses.extend([
                    [-x, -vp_new, vp_old],
                    [-x, -vp_new, vq_old],
                    [-x, -vp_old, -vq_old, vp_new]
                ])
                # MAX on q
                hard_clauses.extend([
                    [-x, -vp_old, vq_new],
                    [-x, -vq_old, vq_new],
                    [-x, -vq_new, vp_old, vq_old]
                ])

            # Copy-when-idle
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

    print(f"[INFO] Hard clauses generated: {len(hard_clauses):,} (clean structure)")

    # 5. Search for minimal s
    for s in range(lower_bound_s, upper_bound_s + 1):
        print(f"[SEARCH] Trying s = {s} ...", end=" ", flush=True)

        solver = create_solver()
        for clause in hard_clauses:
            solver.add_clause(clause)

        all_comp_lits = [get_comp_var(l, pidx) for l in range(d) for pidx in range(num_pairs)]
        
        # CRITICAL FIX: Pass top_id to prevent CardEnc from overwriting our logic variables
        cnf = CardEnc.atmost(all_comp_lits, bound=s, top_id=max_var_id, encoding=EncType.seqcounter)
        for clause in cnf.clauses:
            solver.add_clause(clause)

        if solver.solve():
            print("SAT -> optimal!")
            min_s = s
            model = solver.get_model()

            best_layers = []
            for l in range(d):
                layer = []
                for pidx, (p, q) in enumerate(pairs):
                    var = get_comp_var(l, pidx)
                    # Use absolute value logic just in case solver returns negative literal correctly
                    if model[var - 1] > 0: 
                        layer.append((p, q))
                best_layers.append(layer)
            solver.delete()
            break
        else:
            print("UNSAT")
            solver.delete()

    if min_s is None:
        print(f"[RESULT] No perfect halver of depth {d} exists for n={n}.")
        return None, None

    # Output
    print("\n" + "="*70)
    print(f"OPTIMAL PERFECT HALVER FOUND")
    print(f"n={n}, depth={d}, minimal #comparators = {min_s}")
    print("="*70)
    for l, layer in enumerate(best_layers):
        print(f"Layer {l}: {layer}")
    print("(Each layer contains disjoint comparators (p < q) that route min -> p, max -> q.)")
    return min_s, best_layers


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 optimal_halver_pysat.py <n> <d>")
        sys.exit(1)
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    optimal_perfect_halver(n, d)
