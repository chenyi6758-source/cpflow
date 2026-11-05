"""Continuation power flow (CPF) -- predictor/corrector method.

Implements the classical continuation power flow of Ajjarapu & Christy
(IEEE Trans. Power Systems, 1992): a tangent predictor followed by a
Newton corrector on an augmented system, with the continuation parameter
chosen at each step as the state variable (or lambda) with the largest
tangent component, so the trace passes smoothly through the nose point
(saddle-node bifurcation) of the P-V curve.

Conventions
-----------
* ``lam`` scales the loads: ``Pd(lam) = Pd0 * (1 + lam * w)`` and likewise
  ``Qd`` (constant power factor), where ``w`` is the per-bus load-direction
  weight (default 1 on every bus with load).
* The extra generation is shared among the online non-slack generators in
  proportion to their ``Pmax`` (or user weights); a generator that hits
  ``Pmax`` is clamped and its share redistributed to the remaining headroom.
  The slack absorbs the rest (losses included).
* PV buses that hit a reactive-power limit are switched to PQ with Q fixed
  at the violated limit (one-way switching, events recorded).
* ``lam`` is dimensionless: ``lam = 1`` means the directed load has doubled.
  ``lambda_max`` is therefore the voltage-stability margin in units of the
  base directed load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pandapower.pypower.idx_brch import BR_STATUS, F_BUS, T_BUS
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

from .powerflow import (
    PQ_ as PQ_T,
    PV_ as PV_T,
    REF_ as REF_T,
    build_ybus,
    check_ppc,
    generator_reactive_power,
    reactive_limit_violations,
    solve_power_flow,
)


@dataclass
class CPFResult:
    """Trace of one continuation power flow run."""

    lam: np.ndarray
    V: np.ndarray  # (n_steps, n_bus) complex voltages, p.u.
    nose_index: int  # index of the nose point, -1 if not reached
    lambda_max: float
    converged: bool
    status: str
    events: list = field(default_factory=list)
    monitored_bus: int = -1
    bus_index: np.ndarray = field(default_factory=lambda: np.array([]))
    bus_types_at_nose: np.ndarray | None = None

    @property
    def nose_found(self) -> bool:
        return self.nose_index >= 0

    def pv_curve(self, bus: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(lambda, |V|) for one bus (default: the monitored bus)."""
        b = self.monitored_bus if bus is None else bus
        return self.lam, np.abs(self.V[:, b])

    def voltage_at(self, bus: int) -> np.ndarray:
        return np.abs(self.V[:, bus])


class _ContinuationState:
    """Mutable per-run state: scaled injections and bus types."""

    def __init__(self, ppc: dict, load_w: np.ndarray, gen_weight: np.ndarray):
        self.ppc = ppc
        self.bus = ppc["bus"]
        self.gen = ppc["gen"]
        self.base_mva = float(ppc["baseMVA"])
        self.pd0 = self.bus[:, PD].copy()  # MW
        self.qd0 = self.bus[:, QD].copy()  # MVAr
        self.load_w = load_w
        self.total_load_dir = float(np.sum(self.load_w * self.pd0))
        self.pg0 = self.gen[:, PG].copy()
        self.gen_weight = gen_weight
        self.online = self.gen[:, GEN_STATUS] > 0
        self.ref_buses = np.flatnonzero(self.bus[:, 1] == REF_T)
        self.ref_bus = int(self.ref_buses[0])
        # Participating generators: online, non-slack.
        self.participants = [
            gi
            for gi in range(len(self.gen))
            if self.online[gi] and int(self.gen[gi, GEN_BUS]) != self.ref_bus
        ]
        # Direction of travel in lambda (+1 on the upper branch, -1 on
        # the lower branch).  The generation shares are piecewise linear
        # in lambda with kinks at Pmax/Pmin, so dS/dlambda must be the
        # one-sided derivative in the direction of travel: at the base
        # case a generator sitting exactly at Pmin (e.g. IEEE 14-bus
        # gens 2-4 at Pg = 0) has a forward derivative but a zero
        # backward one, and a central difference averages the two and
        # corrupts the tangent (the branch-18 outage failed its first
        # step for exactly this reason).
        self.lam_dir = 1.0
        # bus -> fixed Q injection (MW) for buses switched PV -> PQ.
        self.q_fixed: dict[int, float] = {}
        self.Ybus, _, _ = build_ybus(ppc)

    # -- injections as a function of lambda --------------------------------
    def gen_shares(self, lam: float) -> np.ndarray:
        """Desired Pg (MW) of every generator row at loading ``lam``.

        Extra load (``lam > 0``) is shared among the online non-slack
        generators in proportion to Pmax, clamped at Pmax with
        redistribution; a load *decrease* (``lam < 0``, which the
        derivative at the base case and the lower branch both need) is
        shared the same way, clamped at Pmin.  The slack absorbs the rest.
        """
        pg = self.pg0.copy()
        dP = lam * self.total_load_dir
        if dP == 0.0 or not self.participants:
            return pg
        remaining = dP
        active = list(self.participants)
        # Redistribute until every participant is within [Pmin, Pmax].
        for _ in range(len(active) + 1):
            if not active or remaining == 0:
                break
            wsum = float(sum(self.gen_weight[gi] for gi in active))
            if wsum <= 0:
                break
            if remaining > 0:
                cap = self.gen[:, PMAX]
                capped = [
                    gi for gi in active
                    if pg[gi] + remaining * self.gen_weight[gi] / wsum > cap[gi]
                ]
            else:
                cap = self.gen[:, PMIN]
                capped = [
                    gi for gi in active
                    if pg[gi] + remaining * self.gen_weight[gi] / wsum < cap[gi]
                ]
            if not capped:
                for gi in active:
                    pg[gi] += remaining * self.gen_weight[gi] / wsum
                remaining = 0.0
                break
            for gi in capped:
                headroom = max(cap[gi] - pg[gi], 0.0) if remaining > 0 \
                    else max(pg[gi] - cap[gi], 0.0)
                pg[gi] += headroom if remaining > 0 else -headroom
                remaining += -headroom if remaining > 0 else headroom
            active = [gi for gi in active if gi not in capped]
        return pg

    def specified_injection(self, lam: float) -> np.ndarray:
        """Net specified injection S (p.u.) at loading ``lam``."""
        pd = self.pd0 * (1.0 + lam * self.load_w)
        qd = self.qd0 * (1.0 + lam * self.load_w)
        s = -(pd + 1j * qd) / self.base_mva
        pg = self.gen_shares(lam)
        for gi in range(len(self.gen)):
            if not self.online[gi]:
                continue
            i = int(self.gen[gi, GEN_BUS])
            qg = self.gen[gi, QG]
            if i in self.q_fixed:
                qg = self.q_fixed[i]
            s[i] += (pg[gi] + 1j * qg) / self.base_mva
        return s

    def dS_dlam(self, lam: float) -> np.ndarray:
        """Derivative of the specified injection w.r.t. lambda (p.u.).

        One-sided in the direction of travel (``lam_dir``): the
        generation shares are piecewise linear in lambda, so a central
        difference is wrong at a Pmax/Pmin kink.
        """
        ds = -(self.load_w * self.pd0 + 1j * self.load_w * self.qd0) / self.base_mva
        eps = 1e-6
        d = self.lam_dir
        dpg = (self.gen_shares(lam + d * eps) - self.gen_shares(lam)) / (d * eps)
        for gi in range(len(self.gen)):
            if not self.online[gi]:
                continue
            i = int(self.gen[gi, GEN_BUS])
            ds[i] += dpg[gi] / self.base_mva
        return ds

    # -- bus types ----------------------------------------------------------
    def types(self):
        ref = np.flatnonzero(self.bus[:, 1] == REF_T)
        pv = np.flatnonzero(self.bus[:, 1] == PV_T)
        pq = np.flatnonzero(self.bus[:, 1] == PQ_T)
        return ref, pv, pq

    def switch_pv_to_pq(self, gen_row: int, limit_mw: float) -> int:
        """Switch a PV generator's bus to PQ with Q fixed at ``limit_mw``."""
        i = int(self.gen[gen_row, GEN_BUS])
        self.bus[i, 1] = PQ_T
        self.q_fixed[i] = limit_mw
        return i


def _mismatch_and_jacobian(state: _ContinuationState, V: np.ndarray, lam: float):
    """Mismatch vector F, Jacobian J and dF/dlambda at a point."""
    from .powerflow import mismatch, power_flow_jacobian

    ref, pv, pq = state.types()
    s_spec = state.specified_injection(lam)
    F, _ = mismatch(state.Ybus, V, s_spec, pv, pq)
    J, _, _, _, _ = power_flow_jacobian(state.Ybus, V, pv, pq, ref)
    # dF/dlambda = dS_spec/dlambda (S_calc does not depend on lambda).
    ds = state.dS_dlam(lam)
    row_p = np.concatenate([pv, pq])
    flam = np.concatenate([ds[row_p].real, ds[pq].imag])
    return F, J, flam, ref, pv, pq


def _z_to_V(z: np.ndarray, state: _ContinuationState, V_ref: np.ndarray) -> np.ndarray:
    """Unpack the state vector z = [Va(non-slack), Vm(PQ), lam] to voltages."""
    ref, pv, pq = state.types()
    col_va = np.concatenate([pv, pq])
    Va = np.angle(V_ref).copy()
    Vm = np.abs(V_ref).copy()
    Va[col_va] = z[: len(col_va)]
    Vm[pq] = z[len(col_va) : len(col_va) + len(pq)]
    return Vm * np.exp(1j * Va)


def _V_to_z(V: np.ndarray, lam: float, state: _ContinuationState) -> np.ndarray:
    ref, pv, pq = state.types()
    col_va = np.concatenate([pv, pq])
    return np.concatenate([np.angle(V)[col_va], np.abs(V)[pq], [lam]])


def _tangent(state, V, lam, t_prev):
    """Tangent vector at a converged point.

    The continuation parameter is the component with the largest |tangent|
    among the *monotone* candidates: lambda and the voltage magnitudes of
    PQ buses.  Generator-bus angles are deliberately excluded: an angle
    can turn around along the curve (e.g. IEEE 39-bus bus 30 peaks at
    lambda = 0.18 and then falls), and fixing such a component past its
    turning point has no solution, which stalls the corrector.  PQ voltage
    magnitudes fall monotonically from base case to collapse, so they are
    safe through the nose.  The tangent is oriented by continuity: its
    dot product with the previous step's displacement must stay positive,
    so the trace keeps moving forward along the curve instead of turning
    back at the nose point.
    """
    F, J, flam, ref, pv, pq = _mismatch_and_jacobian(state, V, lam)
    n = J.shape[0] + 1
    n_va = len(pv) + len(pq)
    candidates = list(range(n_va, n - 1)) + [n - 1]  # PQ Vm..., lambda
    if t_prev is not None and len(t_prev) == n:
        k = max(candidates, key=lambda j: abs(t_prev[j]))
    else:
        k = n - 1  # lambda
    A = np.zeros((n, n))
    A[: n - 1, : n - 1] = J
    A[: n - 1, n - 1] = flam
    A[n - 1, k] = 1.0
    rhs = np.zeros(n)
    rhs[n - 1] = 1.0
    try:
        t = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        t = np.linalg.lstsq(A, rhs, rcond=None)[0]
    # Orient the tangent by continuity over the candidate components
    # (lambda + PQ voltage magnitudes) only: the full-vector dot product
    # was tried first and is wrong here, because the 38 angle components
    # can outweigh the continuation component and flip the tangent
    # backwards mid-curve (IEEE 39-bus stalled at lambda = 0.18 that
    # way), and orienting by the single continuation component's sign
    # fails right after a PV -> PQ switch, when the largest displacement
    # can sit on a bus whose voltage rises with loading.  Angles are
    # excluded from the orientation test because they are non-monotone
    # along the curve; lambda and PQ voltages are the variables that
    # parameterise the P-V curve.
    if t_prev is not None and len(t_prev) == n:
        cands = list(range(n_va, n - 1)) + [n - 1]
        if float(np.dot(t[cands], t_prev[cands])) < 0:
            t = -t
    # Normalise so the largest component is 1: the continuation step then
    # bounds the predictor move in *every* component (an unnormalised
    # tangent lets a 0.1 rad angle step hide behind a small lambda step).
    m = float(np.max(np.abs(t)))
    if m > 0:
        t = t / m
    return t, k


def _corrector(state, z_pred, k, V_ref, tol=1e-10, max_iter=25):
    """Newton corrector on the augmented system [F; z_k - z_pred_k] = 0.

    Plain Newton with a backtracking line search on the infinity norm of
    the augmented residual: the first step fraction in {1, 1/2, 1/4, ...}
    that reduces the residual is taken.  This breaks the Newton 2-cycles
    that a fixed predictor step can otherwise fall into near a generator
    Pmax kink.  A fixed proportional damping was tried first and did not
    work (it scales the step without checking the residual, so the cycle
    persists); the line search checks the residual, so it cannot cycle.
    A corrector that cannot reduce the residual fails, and the caller
    halves the continuation step and retries.
    """
    z = z_pred.copy()
    n = len(z)

    def _g(zz):
        V = _z_to_V(zz, state, V_ref)
        F, J, flam, ref, pv, pq = _mismatch_and_jacobian(state, V, zz[-1])
        return np.concatenate([F, [zz[k] - z_pred[k]]]), F, J, flam

    for _ in range(max_iter):
        g, F, J, flam = _g(z)
        merit = float(np.max(np.abs(g)))
        if merit < tol:
            return z, True, merit
        if not np.isfinite(merit) or merit > 1e6:
            break
        A = np.zeros((n, n))
        A[: n - 1, : n - 1] = J
        A[: n - 1, n - 1] = flam
        A[n - 1, k] = 1.0
        try:
            dz = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(A, -g, rcond=None)[0]
        stepped = False
        alpha = 1.0
        for _ls in range(6):
            z_try = z + alpha * dz
            g_try, _, _, _ = _g(z_try)
            if float(np.max(np.abs(g_try))) < merit:
                z = z_try
                stepped = True
                break
            alpha *= 0.5
        if not stepped:
            break
    g, _, _, _ = _g(z)
    return z, False, float(np.max(np.abs(g)))


def _feasible_at_lambda(
    ppc_base: dict,
    load_w: np.ndarray,
    gen_w: np.ndarray,
    q_fixed: dict[int, tuple[int, float]],
    V_start: np.ndarray,
    lam: float,
    tol: float = 1e-9,
) -> bool:
    """Is there a power-flow solution at loading ``lam``?

    Plain Newton power flow from ``V_start`` on a ppc copy whose
    injections are scaled to ``lam`` (same generation participation and
    PV -> PQ switching state as the snapshot ``ppc_base`` carries).
    ``q_fixed`` maps switched-bus index -> (gen row, Q limit in MW).
    Used to bisect the nose point to high precision.
    """
    ppc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc_base.items()}
    state = _ContinuationState(ppc, load_w, gen_w)
    ppc["bus"][:, PD] = state.pd0 * (1.0 + lam * load_w)
    ppc["bus"][:, QD] = state.qd0 * (1.0 + lam * load_w)
    ppc["gen"][:, PG] = state.gen_shares(lam)
    for bus_i, (gi, lim) in q_fixed.items():
        ppc["gen"][gi, QG] = lim
    _, ok, _, _ = solve_power_flow(ppc, V0=V_start, tol=tol, max_iter=30)
    return bool(ok)


def run_cpf(
    ppc: dict,
    load_bus: int | None = None,
    load_weight: dict[int, float] | None = None,
    gen_weight: dict[int, float] | None = None,
    step0: float = 0.05,
    step_min: float = 1e-4,
    step_max: float = 0.1,
    max_dlam: float = 0.05,
    tol: float = 1e-10,
    max_steps: int = 400,
    max_lower_steps: int = 80,
    v_min: float = 0.5,
    lower_lam_drop: float = 0.35,
    lambda_target: float | None = None,
    monitored_bus: int | None = None,
    enable_switching: bool = True,
    enforce_qmin: bool = False,
    refine_nose: bool = True,
) -> CPFResult:
    """Run a continuation power flow on a ppc dict.

    Parameters
    ----------
    ppc : dict
        MATPOWER-style case dict (``pandapower.converter.pypower.to_ppc``).
    load_bus : int, optional
        If given, only this bus's load is scaled (single-bus P-V curve);
        otherwise all loads scale uniformly.
    load_weight : dict, optional
        Per-bus load-direction weights (bus index -> weight); overrides
        ``load_bus`` / uniform scaling.
    gen_weight : dict, optional
        Per-generator participation weights (gen row -> weight); default
        is proportional to ``Pmax``.
    step0, step_min, step_max : float
        Initial / minimum / maximum continuation step (in units of the
        continuation variable).
    max_dlam : float
        Maximum lambda change per step; keeps the nose region resolved
        even when the continuation parameter is a voltage.
    tol : float
        Newton tolerance on the power mismatch (p.u.).
    max_steps : int
        Maximum continuation steps (upper + lower branch).
    max_lower_steps : int
        Maximum steps traced past the nose point.
    v_min : float
        Stop tracing the lower branch once the monitored bus voltage falls
        below this (p.u.).
    lower_lam_drop : float
        Stop tracing the lower branch once lambda has fallen this far
        below lambda_max (the P-V curve is a closed loop; tracing it back
        to zero load adds nothing to the margin story).
    lambda_target : float, optional
        Stop early once ``lam`` reaches this value (margin beyond the
        target is not needed).
    monitored_bus : int, optional
        Bus whose voltage defines the P-V curve / stopping rule.  Default:
        the PQ bus with the largest |dV/dlambda| at the base case.
    enable_switching : bool
        If True (default), PV buses hitting a reactive-power limit are
        switched to PQ.  Disable only for cross-checks against a plain
        Newton continuation.
    enforce_qmin : bool
        If True, also switch on lower-limit (Qmin) violations.  Default
        False: standard case files carry ``Qmin = 0`` placeholders that
        the base case itself violates, so enforcing Qmin makes the margin
        depend on a data artefact; see
        :func:`cpflow.powerflow.reactive_limit_violations`.
    refine_nose : bool
        If True (default), bisect the nose point with plain Newton
        feasibility tests to pin ``lambda_max`` to ~1e-4.
    """
    check_ppc(ppc)
    ppc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}

    nb = len(ppc["bus"])
    # --- load direction ----------------------------------------------------
    load_w = np.zeros(nb)
    if load_weight is not None:
        for b, w in load_weight.items():
            load_w[int(b)] = float(w)
    elif load_bus is not None:
        load_w[int(load_bus)] = 1.0
    else:
        load_w = np.where(ppc["bus"][:, PD] > 0, 1.0, 0.0)
    if not np.any(load_w):
        raise ValueError("load direction is zero everywhere: nothing to scale")
    # --- generation participation -------------------------------------------
    gen_w = np.zeros(len(ppc["gen"]))
    if gen_weight is not None:
        for gi, w in gen_weight.items():
            gen_w[int(gi)] = float(w)
    else:
        gen_w = np.where(ppc["gen"][:, PMAX] > 0, ppc["gen"][:, PMAX], 0.0)

    state = _ContinuationState(ppc, load_w, gen_w)
    events: list = []

    # --- base case -----------------------------------------------------------
    V0, ok, _, _ = solve_power_flow(ppc, tol=tol)
    if not ok:
        raise RuntimeError(
            "base-case power flow did not converge; "
            "CPF needs a solved starting point"
        )
    # Islanded buses (degree 0 after an outage) make the Jacobian
    # structurally singular: their voltage is a free variable, so no
    # tangent exists.  E.g. IEEE 14-bus branch 7-8 is the only connection
    # of bus 8; removing it islands a generator that produces nothing.
    # Report this explicitly instead of letting the corrector fail.
    branch = ppc["branch"]
    in_service = branch[:, BR_STATUS] > 0
    deg = np.zeros(len(ppc["bus"]), dtype=int)
    for f_, t_ in branch[in_service][:, [F_BUS, T_BUS]].astype(int):
        deg[f_] += 1
        deg[t_] += 1
    if np.any(deg == 0):
        return CPFResult(
            lam=np.array([0.0]),
            V=np.array([V0]),
            nose_index=-1,
            lambda_max=float("nan"),
            converged=False,
            status="islanded",
            events=events,
            monitored_bus=-1,
            bus_index=ppc["bus"][:, BUS_I].astype(int),
            bus_types_at_nose=None,
        )
    # Base-case Q-limit handling: a severe outage can leave the base case
    # itself violating Qmax (e.g. IEEE 14-bus with line 1-2 out: the
    # generator at bus 2 must inject 83.7 MVAr against a 50 MVAr limit).
    # Switch those violators at lambda = 0 and re-solve; if the re-solve
    # fails, the case is infeasible at base load and the margin is zero.
    if enable_switching:
        for _ in range(10):
            qg0 = generator_reactive_power(state.Ybus, V0, state.ppc)
            viol0 = reactive_limit_violations(
                state.ppc, qg0, enforce_qmin=enforce_qmin
            )
            if not viol0:
                break
            gi, _q, lim = viol0[0]
            bus_i = state.switch_pv_to_pq(gi, lim)
            events.append(
                {
                    "step": 0,
                    "kind": "PV->PQ",
                    "gen_row": int(gi),
                    "bus": int(bus_i),
                    "lambda": 0.0,
                    "q_limit_mw": float(lim),
                }
            )
            ppc_sw = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv)
                      for kk, vv in ppc.items()}
            ppc_sw["bus"][:, 1] = state.bus[:, 1]
            for b, l in state.q_fixed.items():
                grow = next(
                    g for g in range(len(ppc_sw["gen"]))
                    if int(ppc_sw["gen"][g, GEN_BUS]) == b
                )
                ppc_sw["gen"][grow, QG] = l
            V0b, okb, _, _ = solve_power_flow(ppc_sw, V0=V0, tol=tol)
            if not okb:
                return CPFResult(
                    lam=np.array([0.0]),
                    V=np.array([V0]),
                    nose_index=-1,
                    lambda_max=0.0,
                    converged=False,
                    status="infeasible_base",
                    events=events,
                    monitored_bus=-1,
                    bus_index=ppc["bus"][:, BUS_I].astype(int),
                    bus_types_at_nose=None,
                )
            V0 = V0b
    # Monitored bus: largest |dV/dlambda| at the base case.
    if monitored_bus is None:
        t0, _ = _tangent(state, V0, 0.0, None)
        _, _, pq0 = state.types()
        dV = np.abs(t0[len(t0) - 1 - len(pq0) : -1]) if len(pq0) else np.array([])
        monitored_bus = int(pq0[int(np.argmax(dV))]) if len(pq0) else -1
    lam = 0.0
    lams = [0.0]
    Vs = [V0.copy()]
    # Bus-type / Q-limit snapshot per recorded point, so the nose point
    # (the max-lambda sample, on the upper branch) can be recovered
    # exactly for the modal analysis and the nose refinement.
    types_trace = [state.bus[:, 1].copy()]
    qfixed_trace: list[dict[int, float]] = [dict(state.q_fixed)]
    step = step0
    # Direction reference for the tangent: displacement of the last step,
    # always rebuilt in the *current* state-vector layout (a PV -> PQ
    # switch adds one component, so the previous tangent may not fit).
    t_prev: np.ndarray | None = None
    nose_index = -1
    nose_detected = False
    lam_max = 0.0
    lower_steps = 0
    status = "max_steps"
    converged = True
    nose_snapshot: dict | None = None
    bus_types_at_nose: np.ndarray | None = None

    for _ in range(max_steps):
        if lambda_target is not None and lam >= lambda_target:
            status = "lambda_target"
            break
        # --- one step: predictor + corrector + switching, with retries -----
        z_base = _V_to_z(Vs[-1], lams[-1], state)
        t, k = _tangent(state, Vs[-1], lams[-1], t_prev)
        if t[-1] < 0 and not nose_detected:
            nose_detected = True
            lam_max = float(np.max(lams))
            # Past the nose lambda decreases: the one-sided dS/dlambda
            # must follow the direction of travel.
            state.lam_dir = -1.0
        if nose_detected:
            lower_steps += 1
            if lower_steps > max_lower_steps:
                status = "lower_branch_done"
                break
            if abs(Vs[-1][monitored_bus]) < v_min:
                status = "v_min"
                break
            if lam_max - lams[-1] > lower_lam_drop or lams[-1] < 0:
                status = "lower_branch_done"
                break
        s = step
        accepted = False
        for _attempt in range(14):
            # Cap the lambda change of this step so the nose stays resolved.
            s_eff = s
            if abs(t[-1]) > 1e-12:
                s_eff = min(s, max_dlam / abs(t[-1]))
            z_pred = z_base + s_eff * t
            z_new, ok_c, _ = _corrector(state, z_pred, k, Vs[-1], tol=tol)
            if not ok_c:
                s *= 0.5
                if s < step_min:
                    break
                z_base = _V_to_z(Vs[-1], lams[-1], state)
                t, k = _tangent(state, Vs[-1], lams[-1], t_prev)
                continue
            V_new = _z_to_V(z_new, state, Vs[-1])
            lam_new = float(z_new[-1])
            # --- PV -> PQ switching (re-solve at the same lambda) -----------
            # Switches are staged in this attempt's local lists and only
            # committed when the step is accepted: a switch detected at a
            # larger lambda must NOT survive a retry at a smaller lambda,
            # where the limit may not bind yet (forcing Q to the limit
            # there distorts the system and can throw the trace onto the
            # lower branch).
            attempt_events: list = []
            attempt_switches: list[tuple[int, int]] = []
            switch_failed = False
            if enable_switching:
                for _sw in range(5):
                    qg = generator_reactive_power(state.Ybus, V_new, state.ppc)
                    viol = reactive_limit_violations(
                        state.ppc, qg, enforce_qmin=enforce_qmin
                    )
                    if not viol:
                        break
                    gi, _q, lim = viol[0]
                    bus_i = state.switch_pv_to_pq(gi, lim)
                    attempt_switches.append((gi, bus_i))
                    attempt_events.append(
                        {
                            "step": len(lams),
                            "kind": "PV->PQ",
                            "gen_row": int(gi),
                            "bus": int(bus_i),
                            "lambda": lam_new,
                            "q_limit_mw": float(lim),
                        }
                    )
                    # Re-solve this point at fixed lambda under the new
                    # types.  The state vector gains one component (the
                    # switched bus voltage magnitude), so rebuild it.
                    z_adj = _V_to_z(V_new, lam_new, state)
                    z_adj, ok_c, _ = _corrector(
                        state, z_adj, len(z_adj) - 1, V_new, tol=tol
                    )
                    if not ok_c:
                        switch_failed = True
                        break
                    V_new = _z_to_V(z_adj, state, V_new)
                    lam_new = float(z_adj[-1])
                    z_new = z_adj
            if switch_failed:
                # Roll back this attempt's switches before retrying.
                for gi, bus_i in attempt_switches:
                    state.bus[bus_i, 1] = PV_T
                    state.q_fixed.pop(bus_i, None)
                s *= 0.5
                if s < step_min:
                    break
                # The switch stays out (it is only re-applied if the
                # retried step still violates the limit); retry the step
                # from the last converged point.
                z_base = _V_to_z(Vs[-1], lams[-1], state)
                t, k = _tangent(state, Vs[-1], lams[-1], t_prev)
                continue
            events.extend(attempt_events)
            accepted = True
            break
        if not accepted:
            status = "step_min"
            converged = False
            break
        # Displacement of this step, in the current layout: the continuity
        # reference for the next tangent.
        t_prev = z_new - _V_to_z(Vs[-1], lams[-1], state)
        lam = lam_new
        lams.append(lam)
        Vs.append(V_new)
        types_trace.append(state.bus[:, 1].copy())
        qfixed_trace.append(dict(state.q_fixed))
        lam_max = max(lam_max, lam)
        # --- adaptive step ----------------------------------------------------
        step = min(s * 1.3, step_max)
    else:
        status = "max_steps"

    # The nose *sample* is the max-lambda point (upper branch).  The flip
    # is only detected once the trace is already on the lower branch, so
    # using the flip point itself would put the modal analysis and the
    # refinement on the wrong side of the bifurcation.
    if nose_detected or lam_max > 0:
        nose_index = int(np.argmax(lams))
        lam_max = float(lams[nose_index])
        bus_types_at_nose = types_trace[nose_index]
        ppc_nose = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in ppc.items()}
        ppc_nose["bus"][:, 1] = bus_types_at_nose
        nose_snapshot = {
            "ppc": ppc_nose,
            "q_fixed": {b: (next(gi for gi, g in enumerate(state.gen)
                                 if int(g[GEN_BUS]) == b), lim)
                        for b, lim in qfixed_trace[nose_index].items()},
            "V": Vs[nose_index].copy(),
        }

    # --- refine the nose by bisection ---------------------------------------
    # lam_max from the trace is the largest *sampled* lambda; the true nose
    # lies between it and the next (infeasible) lambda.  Bisect with plain
    # Newton feasibility tests from the nose voltages.
    if refine_nose and nose_snapshot is not None and nose_index >= 0:
        lam_lo = float(lam_max)
        gap = 0.02
        if nose_index > 0:
            gap = max(0.01, abs(lams[nose_index] - lams[nose_index - 1]))
        lam_hi = lam_lo + gap
        snap = nose_snapshot
        feasible = lambda L: _feasible_at_lambda(
            snap["ppc"], load_w, gen_w, snap["q_fixed"], snap["V"], L
        )
        while feasible(lam_hi) and lam_hi - lam_lo < 0.5:
            lam_hi = lam_lo + (lam_hi - lam_lo) * 2
        if not feasible(lam_hi):
            for _ in range(25):
                mid = 0.5 * (lam_lo + lam_hi)
                if feasible(mid):
                    lam_lo = mid
                else:
                    lam_hi = mid
                if lam_hi - lam_lo < 1e-4:
                    break
            lam_max = float(lam_lo)

    return CPFResult(
        lam=np.array(lams),
        V=np.array(Vs),
        nose_index=nose_index,
        lambda_max=float(lam_max),
        converged=converged,
        status=status,
        events=events,
        monitored_bus=int(monitored_bus),
        bus_index=ppc["bus"][:, BUS_I].astype(int),
        bus_types_at_nose=bus_types_at_nose,
    )


def brute_force_lambda_max(
    ppc: dict,
    load_bus: int | None = None,
    step: float = 0.01,
    max_lam: float = 5.0,
    tol: float = 1e-10,
) -> float:
    """Independent cross-check of ``lambda_max``.

    Plain Newton power flow at lambda = 0, step, 2*step, ... until it stops
    converging.  Uses no predictor/corrector machinery, so agreement with
    :func:`run_cpf` (with switching disabled, i.e. the same equations)
    validates the continuation implementation rather than the test.
    Returns the last lambda that converged.
    """
    ppc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
    nb = len(ppc["bus"])
    load_w = np.zeros(nb)
    if load_bus is not None:
        load_w[int(load_bus)] = 1.0
    else:
        load_w = np.where(ppc["bus"][:, PD] > 0, 1.0, 0.0)
    gen_w = np.where(ppc["gen"][:, PMAX] > 0, ppc["gen"][:, PMAX], 0.0)
    state = _ContinuationState(ppc, load_w, gen_w)
    lam = 0.0
    V = None
    last_ok = 0.0
    while lam <= max_lam:
        ppc_try = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
        ppc_try["bus"][:, PD] = ppc["bus"][:, PD] * (1.0 + lam * load_w)
        ppc_try["bus"][:, QD] = ppc["bus"][:, QD] * (1.0 + lam * load_w)
        ppc_try["gen"][:, PG] = state.gen_shares(lam)
        V, ok, _, _ = solve_power_flow(ppc_try, V0=V, tol=tol)
        if not ok:
            break
        last_ok = lam
        lam += step
    return last_ok
