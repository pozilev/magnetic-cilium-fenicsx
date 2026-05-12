from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def solve_mechanics(domain, E, nu, params):
    return legacy_attr("mechanics_model", "solve_hyperelasticity_displacement_control_3d")(domain, E, nu, params)


def solve_hyperelasticity_displacement_control_3d(domain, E, nu, params):
    return solve_mechanics(domain, E, nu, params)
