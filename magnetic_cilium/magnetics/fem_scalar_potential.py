from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def solve_magnetostatic_scalar_potential(*args, **kwargs):
    return legacy_attr("magnetics_fem", "solve_magnetostatic_scalar_potential")(*args, **kwargs)


def compute_magnetostatic_fem_diagnostics(*args, **kwargs):
    return legacy_attr("magnetics_fem", "compute_magnetostatic_fem_diagnostics")(*args, **kwargs)


def gradient_at_point(*args, **kwargs):
    return legacy_attr("magnetics_fem", "gradient_at_point")(*args, **kwargs)
