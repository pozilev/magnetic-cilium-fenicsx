from __future__ import annotations

from .specs import PlotSpec


def mechanics_plot_specs() -> list[PlotSpec]:
    return [
        PlotSpec(
            name="deformed_geometry",
            filename="mechanics_deformed_geometry.png",
            kind="mechanics",
            required_fields=("max_top_u_x_m", "reaction_force_x_N"),
            description="Deformed cilium shape under prescribed displacement.",
        ),
        PlotSpec(
            name="J_detF",
            filename="mechanics_J_detF.png",
            kind="mechanics",
            required_fields=("J_min", "J_max"),
            description="det(F) quality field.",
        ),
        PlotSpec(
            name="von_mises",
            filename="mechanics_von_mises.png",
            kind="mechanics",
            required_fields=("von_mises_max_Pa",),
            description="von Mises stress distribution.",
        ),
    ]
