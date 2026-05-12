from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def run_magnetics_from_restart(*args, **kwargs):
    return legacy_attr("pipeline", "run_magnetics_from_restart")(*args, **kwargs)


def run_magnetic_only_validation(*args, **kwargs):
    return legacy_attr("pipeline", "run_magnetic_only_validation")(*args, **kwargs)
