"""Built-in test networks and ppc conversion helpers."""

from __future__ import annotations

import pandapower.networks as _nw
from pandapower.converter.pypower import to_ppc

#: Case networks shipped with pandapower (MATPOWER cases).
CASE_NETWORKS = {
    "case9": _nw.case9,
    "case14": _nw.case14,
    "case30": _nw.case30,
    "case39": _nw.case39,
    "case57": _nw.case57,
    "case118": _nw.case118,
    "case33bw": _nw.case33bw,
}


def load_case(name: str):
    """Load a built-in pandapower case network by name."""
    key = name.lower().replace("-", "").replace("_", "")
    for cand, fn in CASE_NETWORKS.items():
        if cand.replace("_", "") == key:
            return fn()
    raise ValueError(
        f"unknown case {name!r}; available: {sorted(CASE_NETWORKS)}"
    )


def net_to_ppc(net) -> dict:
    """Convert a pandapower network to a MATPOWER-style ppc dict.

    ``to_ppc`` initialises voltages from ``res_bus`` by default, so an
    unsolved network is solved first (flat start is the fallback).
    """
    import pandapower as pp

    if net.res_bus.empty:
        try:
            pp.runpp(net)
        except Exception:
            return to_ppc(net, init="flat")
    return to_ppc(net)


def case_to_ppc(name: str) -> tuple[dict, object]:
    """Load a built-in case and return ``(ppc, net)``."""
    net = load_case(name)
    return net_to_ppc(net), net
