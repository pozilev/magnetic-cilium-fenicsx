from __future__ import annotations

from magnetic_cilium.pipeline.full import run_full_pipeline


def run_mechanics_case(*args, **kwargs):
    from magnetic_cilium.pipeline.execution import run_mechanics_case as _run_mechanics_case

    return _run_mechanics_case(*args, **kwargs)


def run_mechanics_pipeline(config_or_path, **kwargs):
    return run_full_pipeline(config_or_path, **kwargs)
