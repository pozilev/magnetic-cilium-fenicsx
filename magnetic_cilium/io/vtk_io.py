from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def save_mechanics_vtk(*args, **kwargs):
    return legacy_attr("mechanics_model", "save_results")(*args, **kwargs)
