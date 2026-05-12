from __future__ import annotations


def save_mechanics_vtk(*args, **kwargs):
    from magnetic_cilium.mechanics.backend import save_results

    return save_results(*args, **kwargs)
