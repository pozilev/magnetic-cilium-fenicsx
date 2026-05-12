from __future__ import annotations

from .specs import PlotSpec


def geometry_plot_specs() -> list[PlotSpec]:
    return [
        PlotSpec(
            name="initial_geometry",
            filename="geometry_initial.png",
            kind="geometry",
            required_fields=("D", "L1", "L2", "substrate_radius", "substrate_thickness"),
            description="Initial cilium, substrate and Hall sensor layout.",
        ),
        PlotSpec(
            name="sensor_layout",
            filename="sensor_layout.png",
            kind="geometry",
            required_fields=("sensor_x_m", "sensor_y_m", "sensor_z_m"),
            description="Hall sensor position relative to the cilium axis.",
        ),
    ]
