from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def run_magnetics_fem_validation_from_config(*args, **kwargs):
    return legacy_attr("pipeline", "run_magnetics_fem_validation_from_config")(*args, **kwargs)


def run_validation_study(*args, **kwargs):
    return legacy_attr("pipeline", "run_validation_study")(*args, **kwargs)
