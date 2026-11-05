"""Tests for the Newton power flow (cpflow.powerflow)."""

import numpy as np
import pandapower as pp
import pytest

from cpflow.networks import case_to_ppc
from cpflow.powerflow import (
    build_ybus,
    mismatch,
    power_flow_jacobian,
    solve_power_flow,
)


@pytest.mark.parametrize("case", ["case9", "case14", "case30"])
def test_base_pf_matches_runpp(case):
    """Our Newton solver reproduces pandapower's runpp voltages."""
    ppc, net = case_to_ppc(case)
    pp.runpp(net)
    V, ok, _, _ = solve_power_flow(ppc)
    assert ok
    v_ref = net.res_bus.vm_pu.values
    # pandapower bus order == ppc bus order for these case networks.
    assert np.max(np.abs(np.abs(V) - v_ref)) < 1e-6
    # Angles agree up to the slack reference (pandapower reports degrees).
    va_ref = np.deg2rad(net.res_bus.va_degree.values)
    assert np.max(np.abs(np.angle(V) - va_ref)) < 1e-6


def test_mismatch_zero_at_solution():
    ppc, net = case_to_ppc("case14")
    V, ok, _, _ = solve_power_flow(ppc)
    assert ok
    Ybus, _, _ = build_ybus(ppc)
    bus = ppc["bus"]
    ref = np.flatnonzero(bus[:, 1] == 3)
    pv = np.flatnonzero(bus[:, 1] == 2)
    pq = np.flatnonzero(bus[:, 1] == 1)
    from cpflow.cpf import _ContinuationState

    load_w = np.where(bus[:, 2] > 0, 1.0, 0.0)
    gen_w = np.where(ppc["gen"][:, 8] > 0, ppc["gen"][:, 8], 0.0)
    state = _ContinuationState(ppc, load_w, gen_w)
    F, _ = mismatch(Ybus, V, state.specified_injection(0.0), pv, pq)
    assert np.max(np.abs(F)) < 1e-9


def test_jacobian_finite_difference():
    """The analytic Jacobian matches central finite differences."""
    ppc, net = case_to_ppc("case9")
    V, ok, _, _ = solve_power_flow(ppc)
    assert ok
    Ybus, _, _ = build_ybus(ppc)
    bus = ppc["bus"]
    ref = np.flatnonzero(bus[:, 1] == 3)
    pv = np.flatnonzero(bus[:, 1] == 2)
    pq = np.flatnonzero(bus[:, 1] == 1)
    J, row_p, row_q, col_va, col_vm = power_flow_jacobian(Ybus, V, pv, pq, ref)
    from cpflow.cpf import _ContinuationState

    load_w = np.where(bus[:, 2] > 0, 1.0, 0.0)
    gen_w = np.where(ppc["gen"][:, 8] > 0, ppc["gen"][:, 8], 0.0)
    state = _ContinuationState(ppc, load_w, gen_w)
    x0 = np.concatenate([np.angle(V)[col_va], np.abs(V)[col_vm]])

    def V_from_x(x):
        Vv = V.copy()
        Vv[col_va] = np.abs(Vv[col_va]) * np.exp(1j * x[: len(col_va)])
        Vv[col_vm] = x[len(col_va) :] * np.exp(1j * np.angle(Vv[col_vm]))
        return Vv

    h = 1e-6
    for j in [0, len(col_va) // 2, len(col_va), len(x0) - 1]:
        e = np.zeros(len(x0))
        e[j] = h
        fp, _ = mismatch(Ybus, V_from_x(x0 + e), state.specified_injection(0.0), pv, pq)
        fm, _ = mismatch(Ybus, V_from_x(x0 - e), state.specified_injection(0.0), pv, pq)
        fd = (fp - fm) / (2 * h)
        assert np.max(np.abs(J[:, j] - fd)) < 1e-5
