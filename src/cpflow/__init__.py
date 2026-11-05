"""cpflow -- continuation power flow for pandapower networks.

Trace P-V curves through the nose point and read off the voltage-stability
margin (``lambda_max``), the weakest buses, and the N-1 margin ranking.
"""

from .analysis import modal_weakest, n1_margin_ranking, weakest_buses
from .cpf import CPFResult, brute_force_lambda_max, run_cpf
from .networks import case_to_ppc, load_case, net_to_ppc
from .powerflow import solve_power_flow

__version__ = "0.1.0"

__all__ = [
    "CPFResult",
    "brute_force_lambda_max",
    "case_to_ppc",
    "load_case",
    "modal_weakest",
    "n1_margin_ranking",
    "net_to_ppc",
    "run_cpf",
    "solve_power_flow",
    "weakest_buses",
]
