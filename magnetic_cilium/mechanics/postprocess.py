from __future__ import annotations


def compute_J_field(domain, u):
    from magnetic_cilium.mechanics.backend import compute_J_field as _compute_J_field

    return _compute_J_field(domain, u)


def compute_von_mises_field(domain, u, E, nu):
    from magnetic_cilium.mechanics.backend import compute_von_mises_field as _compute_von_mises_field

    return _compute_von_mises_field(domain, u, E, nu)


def compute_reaction_force_x(domain, u, E, nu, control_facets):
    from magnetic_cilium.mechanics.backend import compute_reaction_force_x as _compute_reaction_force_x

    return _compute_reaction_force_x(domain, u, E, nu, control_facets)


def compute_material_volumes(domain, material):
    from magnetic_cilium.mechanics.backend import compute_material_volumes as _compute_material_volumes

    return _compute_material_volumes(domain, material)
