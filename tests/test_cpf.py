"""Tests for the continuation power flow (cpflow.cpf)."""

import numpy as np
import pytest

from cpflow import brute_force_lambda_max, case_to_ppc, run_cpf

# Expected nose margins from the validated runs (2026-11-05); the ranges
# are deliberately wider than the numerical spread so the test checks the
# physics, not the last digit.
NOSE_RANGES = {
    "case9": (1.60, 1.70),
    "case14": (1.30, 1.35),
    "case30": (1.60, 1.65),
    "case39": (0.30, 0.34),
    "case57": (0.42, 0.45),
    "case118": (1.50, 1.60),
}


@pytest.mark.parametrize("case,lo,hi", [(c, *r) for c, r in NOSE_RANGES.items()])
def test_cpf_finds_nose(case, lo, hi):
    ppc, net = case_to_ppc(case)
    res = run_cpf(ppc)
    assert res.nose_found
    assert lo <= res.lambda_max <= hi
    assert len(res.lam) > 5


@pytest.mark.parametrize("case", ["case9", "case14", "case30"])
def test_no_switch_matches_brute_force(case):
    """Without Q-limit switching, CPF must agree with plain Newton."""
    ppc, net = case_to_ppc(case)
    res = run_cpf(ppc, enable_switching=False)
    bf = brute_force_lambda_max(ppc, step=0.02)
    assert abs(res.lambda_max - bf) <= 0.05


def test_switching_reduces_margin_case14():
    """Q limits bind on case14: the switched margin is far below the
    unswitched one (1.33 vs 3.99)."""
    ppc, net = case_to_ppc("case14")
    sw = run_cpf(ppc)
    nosw = run_cpf(ppc, enable_switching=False)
    assert sw.lambda_max < nosw.lambda_max
    assert sw.events  # at least one PV -> PQ switch happened
    assert all(ev["lambda"] < sw.lambda_max for ev in sw.events)


def test_pv_curve_shape():
    """Voltages fall along the upper branch; lambda falls again on the
    lower branch while the system-wide minimum voltage keeps falling
    (individual buses can recover on the lower branch as the collapse
    localises, so the invariant is the minimum over all buses)."""
    ppc, net = case_to_ppc("case30")
    res = run_cpf(ppc)
    i = res.nose_index
    v = np.abs(res.V[:, res.monitored_bus])
    assert v[i] < v[0]  # voltage dropped by the nose
    assert res.lam[-1] < res.lambda_max  # lower branch: lambda recedes
    assert np.min(np.abs(res.V[-1])) < np.min(np.abs(res.V[i]))


def test_lambda_target_stops_early():
    ppc, net = case_to_ppc("case14")
    res = run_cpf(ppc, lambda_target=0.5)
    assert res.status == "lambda_target"
    assert res.lam[-1] <= 0.55
    assert res.lambda_max < 1.3304  # never reached the nose


def test_single_bus_direction():
    """Scaling only one bus's load still traces a full P-V curve."""
    ppc, net = case_to_ppc("case14")
    res = run_cpf(ppc, load_bus=13)  # 0-based row of IEEE bus 14
    assert res.nose_found
    assert res.lambda_max > 0


def test_islanded_outage():
    """Removing IEEE 14-bus branch 7-8 islands bus 8: the Jacobian is
    structurally singular, so the run reports 'islanded', not a crash."""
    ppc, net = case_to_ppc("case14")
    trial = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
    trial["branch"][18, 10] = 0
    res = run_cpf(trial)
    assert res.status == "islanded"
    assert np.isnan(res.lambda_max)


def test_infeasible_base_outage():
    """Removing IEEE 14-bus branch 1-2 leaves the base case itself
    violating Qmax beyond repair: margin zero, status 'infeasible_base'."""
    ppc, net = case_to_ppc("case14")
    trial = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc.items()}
    trial["branch"][0, 10] = 0
    res = run_cpf(trial)
    assert res.status == "infeasible_base"
    assert res.lambda_max == 0.0
