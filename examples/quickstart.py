"""Quickstart: trace the P-V curve of IEEE 30-bus, find its voltage
stability margin and weakest buses, and save the curve + trace.

Run:  python examples/quickstart.py
"""

from pathlib import Path

from cpflow import (
    case_to_ppc,
    modal_weakest,
    run_cpf,
    weakest_buses,
)
from cpflow.plotting import plot_pv_curve

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

ppc, net = case_to_ppc("case30")
res = run_cpf(ppc)

print(f"IEEE 30-bus: lambda_max = {res.lambda_max:.4f} "
      f"({len(res.lam)} points, {len(res.events)} PV->PQ switches)")
for ev in res.events:
    print(f"  switch: bus {int(res.bus_index[ev['bus']]) + 1} at "
          f"lambda={ev['lambda']:.3f} (Q limit {ev['q_limit_mw']:.0f} MVAr)")

print("\nWeakest buses by |dV/dlambda| near the nose:")
print(weakest_buses(res, topn=5).to_string(index=False))

mw = modal_weakest(
    ppc, res.V[res.nose_index], topn=5, bus_types=res.bus_types_at_nose
)
print(f"\nModal analysis (smallest eigenvalue "
      f"{mw.attrs['smallest_eigenvalue']:.4f}):")
print(mw.to_string(index=False))

plot_pv_curve(res, path=str(OUT / "case30_pv_curve.png"))
lam, v = res.pv_curve()
import pandas as pd

pd.DataFrame({"lambda": res.lam, "v_monitored": v}).to_csv(
    OUT / "case30_trace.csv", index=False
)
print(f"\nSaved {OUT / 'case30_pv_curve.png'}")
print(f"Saved {OUT / 'case30_trace.csv'}")
