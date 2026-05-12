from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def run_full_case(*args, **kwargs):
    return legacy_attr("pipeline", "run_full_case")(*args, **kwargs)
