"""P-V curve plotting (matplotlib is an optional dependency)."""

from __future__ import annotations

import numpy as np

from .cpf import CPFResult


def plot_pv_curve(
    result: CPFResult,
    buses: list[int] | None = None,
    path: str | None = None,
    ax=None,
    show_nose: bool = True,
):
    """Plot P-V curves (lambda vs |V|) for the given buses.

    ``buses`` are ppc bus indices (0-based internal order); if omitted, the
    monitored bus plus the four fastest-collapsing buses are shown.
    Returns the matplotlib Axes.
    """
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if buses is None:
        from .analysis import weakest_buses

        buses = [result.monitored_bus] + [
            int(b) for b in weakest_buses(result, topn=4)["bus"].tolist()
        ]
        # weakest_buses reports 1-based bus numbers; map them back to
        # row indices.
        idx_of = {int(n) + 1: i for i, n in enumerate(result.bus_index)}
        buses = [idx_of.get(b, b) for b in buses]
        buses = sorted(set(buses))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for b in buses:
        lam, v = result.pv_curve(b)
        ax.plot(lam, v, label=f"bus {int(result.bus_index[b]) + 1}")
    if show_nose and result.nose_found:
        ax.axvline(result.lambda_max, color="k", ls="--", lw=1)
        ax.plot(
            [result.lambda_max],
            [abs(result.V[result.nose_index, result.monitored_bus])],
            "ko",
            label=f"nose  λmax={result.lambda_max:.3f}",
        )
    ax.set_xlabel("loading factor λ  (λ = 1 doubles the directed load)")
    ax.set_ylabel("voltage magnitude (p.u.)")
    ax.set_title("P-V curves — continuation power flow")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    if path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(path, dpi=150)
    return ax
