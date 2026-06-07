from pysat.card import CardEnc
lits = [1, 2, 3, 4]
clauses = CardEnc.atmost(lits, bound=2, encoding=0)
print("✅ Cardinality encoding works (", len(clauses), "clauses generated)")
