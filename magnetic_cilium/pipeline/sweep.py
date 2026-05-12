from __future__ import annotations

from magnetic_cilium.config.loader import load_simulation_config


def run_magnetics_fem_validation_from_config(*args, **kwargs):
    from magnetic_cilium.pipeline.execution import (
        run_magnetics_fem_validation_from_config as _run_magnetics_fem_validation_from_config,
    )

    return _run_magnetics_fem_validation_from_config(*args, **kwargs)


def run_validation_study(*args, **kwargs):
    from magnetic_cilium.pipeline.execution import run_validation_study as _run_validation_study

    return _run_validation_study(*args, **kwargs)


def build_sweep_plan(config_or_path):
    config = load_simulation_config(config_or_path) if isinstance(config_or_path, str) else config_or_path
    groups = []
    for group_name, group in config.sweep.raw.items():
        cases = group.get("cases", []) if isinstance(group, dict) else []
        groups.append({"name": group_name, "case_count": len(cases)})
    return {
        "mode": config.run.mode,
        "enabled": config.sweep.enabled,
        "restart_dir": config.output.restart_dir,
        "output_dir": config.output.output_dir or config.output.outdir,
        "groups": groups,
    }
