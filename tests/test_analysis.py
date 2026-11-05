"""Tests for weakest-bus / modal analysis and the N-1 ranking."""

import numpy as np

from cpflow import (
    case_to_ppc,
    modal_weakest,
    n1_margin_ranking,
    run_cpf,
    weakest_buses,
)


def test_weakest_and_modal_agree_case14():
    """Two independent methods name the same collapsing bus on case14."""
    ppc, net = case_to_ppc("case14")
    res = run_cpf(ppc)
    wb = weakest_buses(res, topn=3)
    mw = modal_weakest(
        ppc, res.V[res.nose_index], topn=3, bus_types=res.bus_types_at_nose
    )
    assert wb["bus"].iloc[0] == mw["bus"].iloc[0] == 14
    assert len(set(wb["bus"]) & set(mw["bus"])) >= 2
    assert abs(mw.attrs["smallest_eigenvalue"]) < 0.05


def test_n1_ranking_case14():
    ppc, net = case_to_ppc("case14")
    df = n1_margin_ranking(ppc)
    assert len(df) == 21  # 20 branches + base case
    base = df[df["branch"] == -1].iloc[0]
    assert abs(base["lambda_max"] - 1.3304) < 0.01
    # Islanding outage sorts first with NaN margin.
    assert df.iloc[0]["status"] == "islanded"
    assert df.iloc[0]["branch"] == 18
    # The next-worst outage is the branch 1-2 base-infeasible one.
    assert df.iloc[1]["status"] == "infeasible_base"
    # Every other outage keeps a non-negative margin below the base one.
    rest = df.iloc[2:]
    assert (rest["lambda_max"] >= 0).all()
    assert rest["lambda_max"].max() <= base["lambda_max"] + 1e-9
    # The worst ordinary outage (branch 2-3) loses about half the margin.
    assert rest["lambda_max"].min() < 0.70
