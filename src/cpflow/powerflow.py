"""Power-flow layer for cpflow.

Everything here works on a MATPOWER-style ``ppc`` dict (as produced by
``pandapower.converter.pypower.to_ppc``) so the continuation power flow can
control the Newton iteration itself: PV -> PQ switching on reactive-power
limits, warm starts from the previous continuation point, and an augmented
Jacobian for the predictor/corrector steps.

The Newton solver is written in polar coordinates and uses pandapower's
vendored pypower derivatives (``dSbus_dV``) for the exact Jacobian, so the
base-case solution agrees with ``pandapower.runpp`` to solver tolerance.
"""

from __future__ import annotations

import numpy as np
from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.idx_bus import BUS_I, PD, PQ, PV, QD, REF, VA, VM
from pandapower.pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)
from pandapower.pypower.makeYbus import makeYbus

# Bus types (MATPOWER convention).
PQ_ = int(PQ)
PV_ = int(PV)
REF_ = int(REF)


class PowerFlowError(RuntimeError):
    """Raised when the Newton power flow fails to converge."""


def bus_type_lists(bus: np.ndarray, gen: np.ndarray):
    """Return (ref, pv, pq) bus index arrays (0-based, internal ordering).

    ``bustypes`` from pypower returns the index sets used by MATPOWER; we
    re-derive them here from the bus type column so callers can pass a
    *modified* type column (e.g. after a PV -> PQ switch) without touching
    the generator table.
    """
    ref = np.flatnonzero(bus[:, 1] == REF_)
    pv = np.flatnonzero(bus[:, 1] == PV_)
    pq = np.flatnonzero(bus[:, 1] == PQ_)
    return ref, pv, pq


def specified_injection(ppc: dict) -> np.ndarray:
    """Net specified complex injection S = (Pg - Pd) + j(Qg - Qd), in p.u.

    Generator rows with ``GEN_STATUS == 0`` are ignored.  For PV buses the
    Qg entry is only a starting value: the Newton solver never enforces the
    Q row of a PV bus, it lets Qg float.
    """
    bus = ppc["bus"]
    gen = ppc["gen"]
    base_mva = ppc["baseMVA"]
    s = -(bus[:, PD] + 1j * bus[:, QD]) / base_mva
    online = gen[:, GEN_STATUS] > 0
    for g in gen[online]:
        i = int(g[GEN_BUS])
        s[i] += (g[PG] + 1j * g[QG]) / base_mva
    return s


def build_ybus(ppc: dict):
    """Return (Ybus, Yf, Yt) for the in-service branches of ``ppc``.

    pandapower's branch matrix has 26 columns (asymmetric R/X/G/B live in
    columns 21-25) while ``to_ppc`` emits 22; the missing columns are
    zero for symmetric branches, so pad with zeros before calling the
    vendored pypower ``makeYbus``.
    """
    branch = ppc["branch"]
    if branch.shape[1] < 26:
        branch = np.pad(branch, ((0, 0), (0, 26 - branch.shape[1])))
    return makeYbus(ppc["baseMVA"], ppc["bus"], branch)


def power_flow_jacobian(Ybus, V, pv, pq, ref):
    """Polar Newton Jacobian of the mismatch F = S_spec - S_calc.

    Rows: P mismatches at PV+PQ buses, then Q mismatches at PQ buses.
    Columns: voltage angles at non-slack buses, then voltage magnitudes
    at PQ buses.  Since F = S_spec - S_calc, dF/dx = -dS_calc/dx.
    Returns (J, row_p, row_q, col_va, col_vm).
    """
    # NOTE: pandapower's vendored dSbus_dV returns (dS_dVm, dS_dVa) --
    # the opposite order of MATPOWER's documented (dS_dVa, dS_dVm).
    # Verified against finite differences (2026-11-05).
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
    if hasattr(dS_dVa, "toarray"):  # vendored pypower returns sparse matrices
        dS_dVa = dS_dVa.toarray()
        dS_dVm = dS_dVm.toarray()
    row_p = np.concatenate([pv, pq])
    row_q = pq
    col_va = np.concatenate([pv, pq])
    col_vm = pq
    J = np.zeros((len(row_p) + len(row_q), len(col_va) + len(col_vm)))
    J[: len(row_p), : len(col_va)] = -dS_dVa[np.ix_(row_p, col_va)].real
    J[: len(row_p), len(col_va) :] = -dS_dVm[np.ix_(row_p, col_vm)].real
    J[len(row_p) :, : len(col_va)] = -dS_dVa[np.ix_(row_q, col_va)].imag
    J[len(row_p) :, len(col_va) :] = -dS_dVm[np.ix_(row_q, col_vm)].imag
    return J, row_p, row_q, col_va, col_vm


def mismatch(Ybus, V, s_spec, pv, pq):
    """Complex power mismatch F = S_spec - S_calc, split into P/Q rows."""
    s_calc = V * np.conj(Ybus @ V)
    f = s_spec - s_calc
    row_p = np.concatenate([pv, pq])
    f_p = f[row_p].real
    f_q = f[pq].imag
    return np.concatenate([f_p, f_q]), f


def solve_power_flow(
    ppc: dict,
    V0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> tuple[np.ndarray, bool, int, float]:
    """Newton-Raphson power flow on a ppc dict.

    Returns ``(V, converged, iterations, max_mismatch)`` where ``V`` is the
    complex bus-voltage vector in p.u.  The slack angle is held at its
    starting value; PV magnitudes are held at the generator ``VG`` setpoint
    (taken from the bus ``VM`` column, which ``to_ppc`` already fills with
    the setpoint).
    """
    bus = ppc["bus"]
    gen = ppc["gen"]
    Ybus, _, _ = build_ybus(ppc)
    s_spec = specified_injection(ppc)
    ref, pv, pq = bus_type_lists(bus, gen)
    if V0 is None:
        V = bus[:, VM].astype(float) * np.exp(1j * np.radians(bus[:, VA]))
    else:
        V = np.array(V0, dtype=complex)
    # Enforce PV setpoints / slack angle on the starting point.
    for g in gen[gen[:, GEN_STATUS] > 0]:
        i = int(g[GEN_BUS])
        if bus[i, 1] == PV_:
            V[i] = g[VG] * np.exp(1j * np.angle(V[i]))
    max_mis = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        F, _ = mismatch(Ybus, V, s_spec, pv, pq)
        max_mis = float(np.max(np.abs(F))) if F.size else 0.0
        if max_mis < tol:
            return V, True, it - 1, max_mis
        J, row_p, row_q, col_va, col_vm = power_flow_jacobian(Ybus, V, pv, pq, ref)
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -F, rcond=None)[0]
        dVa = np.zeros(len(V))
        dVm = np.zeros(len(V))
        dVa[col_va] = dx[: len(col_va)]
        dVm[col_vm] = dx[len(col_va) :]
        Va = np.angle(V) + dVa
        Vm = np.abs(V) + dVm
        V = Vm * np.exp(1j * Va)
    F, _ = mismatch(Ybus, V, s_spec, pv, pq)
    max_mis = float(np.max(np.abs(F))) if F.size else 0.0
    return V, max_mis < tol, max_iter, max_mis


def generator_reactive_power(Ybus, V, ppc: dict) -> dict[int, float]:
    """Actual reactive output (MW, on system base) of each online generator.

    Computed from the solved voltages, so it reflects what a PV bus really
    has to inject to hold its setpoint -- the quantity checked against
    Qmax/Qmin for PV -> PQ switching.
    """
    bus = ppc["bus"]
    gen = ppc["gen"]
    base_mva = ppc["baseMVA"]
    s_calc = V * np.conj(Ybus @ V)
    qg = {}
    for gi, g in enumerate(gen):
        if g[GEN_STATUS] <= 0:
            continue
        i = int(g[GEN_BUS])
        # Q injected by the generator = Q flowing into the network + load Q.
        qg[gi] = float((s_calc[i].imag + bus[i, QD] / base_mva) * base_mva)
    return qg


def reactive_limit_violations(
    ppc: dict, qg: dict[int, float], enforce_qmin: bool = False
) -> list[tuple[int, float, float]]:
    """PV generators violating their Q limits.

    Returns a list of ``(gen_row, q_actual_MW, limit_MW)`` sorted by
    violation size, largest first.  A generator with ``Qmax <= Qmin``
    (both zero in many case files) is treated as unlimited.

    Only the upper limit (``Qmax``, the field-heating limit) is enforced
    by default.  The lower limit is off because standard case files
    carry ``Qmin = 0`` placeholders that the base case itself violates
    (e.g. IEEE 39-bus: the generator at bus 36 absorbs 1.4 MVAr at the
    base case while its ``Qmin`` reads 0), so enforcing it makes the
    margin depend on a data artefact and on the step grid (for case39
    the switched margin moves between 0.28 and 0.32 depending on when
    that 1.4 MVAr violation is caught).  Pass ``enforce_qmin=True`` when
    the case file carries deliberate, verified lower limits.
    """
    bus = ppc["bus"]
    gen = ppc["gen"]
    out = []
    for gi, q in qg.items():
        g = gen[gi]
        i = int(g[GEN_BUS])
        if bus[i, 1] != PV_:
            continue
        qmax, qmin = float(g[QMAX]), float(g[QMIN])
        if qmax <= qmin:
            continue
        if q > qmax:
            out.append((gi, q, qmax))
        elif enforce_qmin and q < qmin:
            out.append((gi, q, qmin))
    out.sort(key=lambda t: abs(t[1] - t[2]), reverse=True)
    return out


def check_ppc(ppc: dict) -> None:
    """Fail fast on ppc dicts that cannot support a power flow."""
    if "bus" not in ppc or "branch" not in ppc or "gen" not in ppc:
        raise ValueError("ppc dict needs 'bus', 'branch' and 'gen' arrays")
    bus = ppc["bus"]
    if not np.any(bus[:, 1] == REF_):
        raise ValueError("no slack (REF) bus in ppc")
    if not np.any(bus[:, 1] == PQ_):
        raise ValueError("no PQ bus in ppc; CPF needs at least one load bus")
