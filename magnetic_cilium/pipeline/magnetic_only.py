from __future__ import annotations

from magnetic_cilium._compat import legacy_attr
from magnetic_cilium.config.adapters import simulation_config_to_legacy_namespace
from magnetic_cilium.config.loader import load_simulation_config
from magnetic_cilium.io.restart import ensure_restart_dir
from magnetic_cilium.magnetics.state import MagneticResult


def run_magnetics_from_restart(*args, **kwargs):
    return legacy_attr("pipeline", "run_magnetics_from_restart")(*args, **kwargs)


def run_magnetic_only_validation(*args, **kwargs):
    return legacy_attr("pipeline", "run_magnetic_only_validation")(*args, **kwargs)


def run_magnetic_only_pipeline(config_or_path, *, dry_run: bool = False):
    config = load_simulation_config(config_or_path) if isinstance(config_or_path, str) else config_or_path
    if not config.output.restart_dir:
        raise ValueError("magnetic-only pipeline requires output.restart_dir")
    restart_status = ensure_restart_dir(config.output.restart_dir)
    args = simulation_config_to_legacy_namespace(config, mode="magnetics")
    if dry_run:
        return {
            "mode": "magnetics",
            "restart": restart_status.to_dict(),
            "legacy_args": vars(args),
        }
    raw_results = legacy_attr("pipeline", "run_magnetics_from_restart")(args)
    if not raw_results:
        return None
    return MagneticResult.from_legacy(raw_results[0], model_type="dipole")
