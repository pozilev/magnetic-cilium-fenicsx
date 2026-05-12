from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def compute_J_field(domain, u):
    return legacy_attr("mechanics_model", "compute_J_field")(domain, u)


def compute_von_mises_field(domain, u, E, nu):
    return legacy_attr("mechanics_model", "compute_von_mises_field")(domain, u, E, nu)


def compute_reaction_force_x(domain, u, E, nu, control_facets):
    return legacy_attr("mechanics_model", "compute_reaction_force_x")(domain, u, E, nu, control_facets)


def compute_material_volumes(domain, material):
    return legacy_attr("mechanics_model", "compute_material_volumes")(domain, material)
