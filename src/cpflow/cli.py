"""Command-line interface: ``cpflow <case> [--bus N] [--n1] ...``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .analysis import modal_weakest, n1_margin_ranking, weakest_buses
from .cpf import run_cpf
from .networks import case_to_ppc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cpflow",
        description="Continuation power flow: voltage-stability margin "
        "(P-V curve, lambda_max, weakest bus, N-1 margin ranking) for "
        "pandapower networks.",
    )
    p.add_argument("case", help="built-in case, e.g. case9, case14, case30")
    p.add_argument("--bus", type=int, default=None,
                   help="scale only this bus's load (1-based IEEE bus number)")
    p.add_argument("--n1", action="store_true",
                   help="also rank every single-branch outage by lambda_max")
    p.add_argument("--topn", type=int, default=5,
                   help="rows shown for weakest-bus / N-1 rankings")
    p.add_argument("--plot", default=None, help="save the P-V curve to this PNG path")
    p.add_argument("--csv", default=None, help="save the P-V trace to this CSV path")
    p.add_argument("--step0", type=float, default=0.05)
    p.add_argument("--lambda-target", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ppc, net = case_to_ppc(args.case)
    load_bus = None
    if args.bus is not None:
        idx = np.flatnonzero(ppc["bus"][:, 0] == args.bus - 1)
        if len(idx) == 0:
            print(f"bus {args.bus} not in case {args.case}", file=sys.stderr)
            return 2
        load_bus = int(idx[0])
    res = run_cpf(ppc, load_bus=load_bus, step0=args.step0,
                  lambda_target=args.lambda_target)
    print(f"case {args.case}: lambda_max = {res.lambda_max:.4f} "
          f"(nose {'found' if res.nose_found else 'not found'}, "
          f"{len(res.lam)} points, status={res.status})")
    if res.status in ("islanded", "infeasible_base"):
        print("no P-V curve: the case is degenerate at base load "
              f"(status={res.status}); see the README honesty notes.")
        return 0
    print(f"monitored bus: {int(res.bus_index[res.monitored_bus]) + 1}, "
          f"V at nose: {abs(res.V[res.nose_index, res.monitored_bus]):.4f} p.u.")
    if res.events:
        for ev in res.events:
            print(f"  event: PV->PQ at bus {int(res.bus_index[ev['bus']]) + 1} (lambda={ev['lambda']:.3f}, "
                  f"Q limit {ev['q_limit_mw']:.1f} MVAr)")
    print(f"\nweakest buses by |dV/dlambda| near the nose:")
    print(weakest_buses(res, topn=args.topn).to_string(index=False))
    print(f"\nmodal analysis at the nose (smallest eigenvalue "
          f"{modal_weakest(ppc, res.V[res.nose_index], topn=1, bus_types=res.bus_types_at_nose).attrs.get('smallest_eigenvalue', float('nan')):.4f}):")
    print(modal_weakest(ppc, res.V[res.nose_index], topn=args.topn,
                        bus_types=res.bus_types_at_nose).to_string(index=False))
    if args.csv:
        import pandas as pd

        lam, v = res.pv_curve()
        pd.DataFrame({"lambda": res.lam,
                      f"v_bus_{int(res.bus_index[res.monitored_bus]) + 1}": v}).to_csv(args.csv, index=False)
        print(f"\ntrace saved to {args.csv}")
    if args.plot:
        from .plotting import plot_pv_curve

        plot_pv_curve(res, path=args.plot)
        print(f"plot saved to {args.plot}")
    if args.n1:
        print("\nN-1 margin ranking (worst first):")
        df = n1_margin_ranking(ppc, topn=args.topn, step0=args.step0)
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
