from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from typing import Any, Mapping

from magnetic_cilium._compat import legacy_attr
from magnetic_cilium.config.schema import SimulationConfig


def simulation_config_to_model_params(
    config: SimulationConfig,
    *,
    overrides: Mapping[str, Any] | None = None,
):
    """Convert the validated architecture config to the legacy solver params.

    The numerical backend still consumes ``ModelParams``. Keeping this adapter
    explicit gives the new architecture one place where units, defaults and
    legacy names are reconciled.
    """
    model_params_cls = legacy_attr("params", "ModelParams")
    data = {
        "D": config.geometry.D,
        "L1": config.geometry.L1,
        "L2": config.geometry.L2,
        "substrate_radius": config.geometry.substrate_radius,
        "substrate_thickness": config.geometry.substrate_thickness,
        "E_pdms": config.materials.E_pdms,
        "nu_pdms": config.materials.nu_pdms,
        "E_magnetic": config.materials.E_magnetic,
        "nu_magnetic": config.materials.nu_magnetic,
        "h_cilium": config.mesh.h_cilium,
        "h_substrate": config.mesh.h_substrate,
        "element_degree": config.mesh.element_degree,
        "delta_x": config.mechanics.delta_x,
        "n_steps": config.mechanics.n_steps,
        "Br_magnetic": config.magnetics.Br_magnetic,
        "sensor_x": config.magnetics.sensor_x,
        "sensor_y": config.magnetics.sensor_y,
        "sensor_z": config.magnetics.sensor_z,
        "rotate_magnetization": config.magnetics.rotate_magnetization,
        "outdir": config.output.outdir or config.output.output_dir or "runs",
    }
    if overrides:
        data.update(dict(overrides))
    return model_params_cls(**data)


def simulation_config_to_legacy_namespace(
    config: SimulationConfig,
    *,
    mode: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Namespace:
    """Build an argparse-like object for legacy pipeline entry points."""
    params = simulation_config_to_model_params(config)
    data = asdict(params)
    data.update(
        {
            "mode": mode or config.run.mode,
            "config": config.run.config_path,
            "outdir": config.output.outdir or config.output.output_dir,
            "restart_dir": config.output.restart_dir,
            "results_write_mode": config.output.results_write_mode,
            "experiment_id": config.output.experiment_id,
            "master_csv_path": config.output.master_csv_path,
            "local_summary_path": config.output.local_summary_path,
            "Br": config.magnetics.Br_magnetic,
            "sensor_x": config.magnetics.sensor_x,
            "sensor_y": config.magnetics.sensor_y,
            "sensor_z": config.magnetics.sensor_z,
            "no_rotate_magnetization": not config.magnetics.rotate_magnetization,
            "target_sensitivity_uT_per_uN": config.magnetics.target_sensitivity_uT_per_uN,
            "air_radius_factor": config.magnetics.air_radius_factor,
            "air_below_factor": config.magnetics.air_below_factor,
            "air_above_factor": config.magnetics.air_above_factor,
            "h_air": config.mesh.h_air,
            "h_air_near": config.mesh.h_air_near or config.mesh.h_air,
            "h_air_far": config.mesh.h_air_far or 3.0 * config.mesh.h_air,
            "near_radius_factor": config.magnetics.near_radius_factor,
            "near_source_padding_factor": config.magnetics.near_source_padding_factor,
            "near_sensor_padding_factor": config.magnetics.near_sensor_padding_factor,
            "magnetic_boundary": config.magnetics.magnetic_boundary,
            "sensor_average": config.magnetics.sensor_average,
            "sensor_average_radius": config.magnetics.sensor_average_radius,
            "sensor_average_n": config.magnetics.sensor_average_n,
            "projection_mode": config.magnetics.projection_mode,
            "max_air_cells": config.mesh.max_air_cells,
            "allow_large_air_mesh": config.mesh.allow_large_air_mesh,
            "input_csv": config.raw.get("input_csv"),
            "interpolation_output_dir": config.output.output_dir,
            "model_type": config.raw.get("model_type"),
            "mode_filter": config.raw.get("mode_filter"),
            "x_column": config.raw.get("x_column", "sensor_x_over_R"),
            "z_column": config.raw.get("z_column", "sensor_z_over_R"),
            "target_column": config.raw.get("target_column", "abs_dBx_uT"),
            "reliable_only": config.raw.get("reliable_only", True),
        }
    )
    if overrides:
        data.update(dict(overrides))
    return Namespace(**data)
