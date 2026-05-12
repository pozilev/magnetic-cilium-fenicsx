from __future__ import annotations

from .specs import PlotSpec


def magnetics_plot_specs() -> list[PlotSpec]:
    return [
        PlotSpec(
            name="B_initial_vs_deformed",
            filename="magnetics_B_initial_vs_deformed.png",
            kind="magnetics",
            required_fields=("B0_sensor_norm_uT", "B1_sensor_norm_uT", "dB_sensor_norm_uT"),
            description="Initial and deformed magnetic response at Hall sensor.",
        ),
        PlotSpec(
            name="delta_B_components",
            filename="magnetics_delta_B_components.png",
            kind="magnetics",
            required_fields=("dB_sensor_x_uT", "dB_sensor_y_uT", "dB_sensor_z_uT"),
            description="Delta-B component comparison.",
        ),
    ]
