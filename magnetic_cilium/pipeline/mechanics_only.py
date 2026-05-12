from __future__ import annotations

from magnetic_cilium._compat import legacy_attr
from magnetic_cilium.pipeline.full import run_full_pipeline


def run_mechanics_case(*args, **kwargs):
    return legacy_attr("pipeline", "run_mechanics_case")(*args, **kwargs)


def run_mechanics_pipeline(config_or_path, **kwargs):
    return run_full_pipeline(config_or_path, **kwargs)
