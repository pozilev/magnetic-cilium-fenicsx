from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def run_magnetic_interpolation(*args, **kwargs):
    return legacy_attr("magnetic_interpolation", "run_magnetic_interpolation")(*args, **kwargs)
