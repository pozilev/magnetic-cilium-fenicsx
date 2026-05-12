from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def run_mechanics_case(*args, **kwargs):
    return legacy_attr("pipeline", "run_mechanics_case")(*args, **kwargs)
