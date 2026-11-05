"""Analysis built on top of the CPF trace.

* Weakest-bus ranking: buses whose voltage collapses fastest as the load
  grows (``dV/dlambda`` near the nose), cross-checked against modal
  analysis (Gao, Morison & Kundur, 1992) -- the reduced-Jacobian smallest
  eigenvalue and its participation factors.
* N-1 margin ranking: re-run the CPF with each in-service branch out of
  service and rank the contingencies by the remaining voltage-stability
  margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandapower.pypower.idx_brch import BR_STATUS, F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_I, PQ, PV, REF

from .cpf import CPFResult, run_cpf
from .powerflow import build_ybus, power_flow_jacobian


def weakest_buses(result: CPFResult, topn: int = 5) -> pd.DataFrame:
    """Rank PQ buses by |dV/dlambda| over the last upper-branch steps.

    The derivative is a secant over the final steps before the nose, where
    the collapse is fastest.  Returns a DataFrame with the bus number
    (1-based, IEEE style), voltage at the nose and the collapse rate.
    """
    nose = result.nose_index if result.nose_index >= 0 else len(result.lam) - 1
    i0 = max(0, nose - 3)
    i1 = nose
    dlam = result.lam[i1] - result.lam[i0]
    # Rank PQ buses at the nose only: PV buses hold their setpoint, so
    # their dV/dlambda is ~0 and they cannot be the collapsing bus.
    pq_mask = None
    if result.bus_types_at_nose is not None:
        pq_mask = result.bus_types_at_nose == int(PQ)
    rows = []
    for b in range(result.V.shape[1]):
        if pq_mask is not None and not pq_mask[b]:
            continue
        dv = abs(result.V[i1, b]) - abs(result.V[i0, b])
        rows.append(
            {
                "bus": int(result.bus_index[b]) + 1,
                "v_nose": float(abs(result.V[i1, b])),
                "dv_dlam": float(dv / dlam) if dlam != 0 else float("nan"),
            }
        )
    df = pd.DataFrame(rows).sort_values("dv_dlam").reset_index(drop=True)
    return df.head(topn)


def modal_weakest(
    ppc: dict, V: np.ndarray, topn: int = 5, bus_types: np.ndarray | None = None
) -> pd.DataFrame:
    """Modal-analysis weakest buses (Gao, Morison & Kundur, 1992).

    Builds the reduced Jacobian ``Jr = J22 - J21 J11^-1 J12`` at a solved
    voltage profile, takes its smallest eigenvalue (closest to zero = the
    mode driving the collapse) and ranks buses (1-based, IEEE style) by
    participation factor ``|xi * eta|`` of that mode.  ``bus_types``
    overrides the bus-type column (pass ``CPFResult.bus_types_at_nose``
    so PV -> PQ switches that happened before the nose are reflected in
    the Jacobian).
    """
    bus = ppc["bus"].copy()
    if bus_types is not None:
        bus[:, 1] = bus_types
    Ybus, _, _ = build_ybus(ppc)
    ref = np.flatnonzero(bus[:, 1] == REF)
    pv = np.flatnonzero(bus[:, 1] == PV)
    pq = np.flatnonzero(bus[:, 1] == PQ)
    J, _, _, _, _ = power_flow_jacobian(Ybus, V, pv, pq, ref)
    n_va = len(pv) + len(pq)
    J11 = J[:n_va, :n_va]
    J12 = J[:n_va, n_va:]
    J21 = J[n_va:, :n_va]
    J22 = J[n_va:, n_va:]
    Jr = J22 - J21 @ np.linalg.solve(J11, J12)
    eigvals, eigvecs = np.linalg.eig(Jr)
    i_min = int(np.argmin(np.abs(eigvals)))
    lam_min = float(eigvals[i_min].real)
    xi = eigvecs[:, i_min].real
    # Left eigenvector: solve Jr^T eta = lam eta.
    _, eigvecs_left = np.linalg.eig(Jr.T)
    eta = eigvecs_left[:, i_min].real
    part = np.abs(xi * eta)
    part = part / part.sum()
    order = np.argsort(-part)
    rows = [
        {
            "bus": int(bus[pq[j], BUS_I]) + 1,
            "participation": float(part[j]),
            "v": float(abs(V[pq[j]])),
        }
        for j in order[:topn]
    ]
    df = pd.DataFrame(rows)
    df.attrs["smallest_eigenvalue"] = lam_min
    return df


def n1_margin_ranking(
    ppc: dict,
    topn: int | None = None,
    step0: float = 0.05,
    tol: float = 1e-10,
    **cpf_kwargs,
) -> pd.DataFrame:
    """Rank every single-branch outage by remaining voltage margin.

    Returns a DataFrame with one row per in-service branch (plus the base
    case as the first row): branch endpoints (1-based bus numbers),
    ``lambda_max`` with that branch out, and the margin lost versus the
    base case.  Outages that make the base case itself unsolvable are
    reported with ``lambda_max = 0`` and ``status = 'infeasible'``;
    outages whose base case violates Qmax so badly that the switched
    base case does not solve are reported with ``status =
    'infeasible_base'``; outages that island a bus are reported with
    ``lambda_max = NaN`` and ``status = 'islanded'`` (their margin is
    undefined on the full network -- the islanded bus makes the Jacobian
    singular -- so they sort first as needing attention).
    """
    base = run_cpf(ppc, step0=step0, tol=tol, **cpf_kwargs)
    rows = [
        {
            "branch": -1,
            "from_bus": -1,
            "to_bus": -1,
            "lambda_max": float(base.lambda_max),
            "margin_lost": 0.0,
            "status": base.status,
        }
    ]
    branch = ppc["branch"]
    for bi in range(len(branch)):
        if branch[bi, BR_STATUS] <= 0:
            continue
        trial = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
        trial["branch"][bi, BR_STATUS] = 0
        try:
            res = run_cpf(trial, step0=step0, tol=tol, **cpf_kwargs)
            lam = float(res.lambda_max)
            st = res.status
        except RuntimeError:
            lam, st = 0.0, "infeasible"
        rows.append(
            {
                "branch": int(bi),
                "from_bus": int(trial["bus"][int(branch[bi, F_BUS]), BUS_I]) + 1,
                "to_bus": int(trial["bus"][int(branch[bi, T_BUS]), BUS_I]) + 1,
                "lambda_max": lam,
                "margin_lost": float(base.lambda_max - lam),
                "status": st,
            }
        )
    df = pd.DataFrame(rows).sort_values(
        "lambda_max", na_position="first"
    ).reset_index(drop=True)
    return df if topn is None else df.head(topn)
