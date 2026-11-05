"""N-1 example: rank every single-branch outage of IEEE 14-bus by the
voltage-stability margin that remains.

Run:  python examples/n1_ranking.py
"""

from pathlib import Path

from cpflow import case_to_ppc, n1_margin_ranking

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

ppc, net = case_to_ppc("case14")
df = n1_margin_ranking(ppc)

print("IEEE 14-bus N-1 margin ranking (worst first):")
print(df.to_string(index=False))
df.to_csv(OUT / "case14_n1_ranking.csv", index=False)
print(f"\nSaved {OUT / 'case14_n1_ranking.csv'}")
