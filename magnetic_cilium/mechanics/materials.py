from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def lame_parameters(E, nu):
    return legacy_attr("mechanics_model", "lame_parameters")(E, nu)


def make_material_field(domain, params):
    return legacy_attr("mechanics_model", "make_material_field")(domain, params)
