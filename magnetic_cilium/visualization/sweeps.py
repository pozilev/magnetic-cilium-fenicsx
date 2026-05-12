from __future__ import annotations

from .specs import PlotSpec


def sweep_plot_specs() -> list[PlotSpec]:
    return [
        PlotSpec(
            name="sensor_position_sweep",
            filename="sweep_sensor_position.png",
            kind="sweep",
            required_fields=("sensor_x_over_R", "sensor_z_over_R", "dB_sensor_norm_uT"),
            description="Magnetic signal as a function of Hall sensor position.",
        ),
        PlotSpec(
            name="Br_scaling",
            filename="sweep_Br_scaling.png",
            kind="sweep",
            required_fields=("Br_magnetic_T", "dB_sensor_norm_uT"),
            description="Magnetic signal scaling with remanence.",
        ),
    ]
