from __future__ import annotations


def lame_parameters(E, nu):
    from magnetic_cilium.mechanics.backend import lame_parameters as _lame_parameters

    return _lame_parameters(E, nu)


def make_material_field(domain, params):
    from magnetic_cilium.mechanics.backend import make_material_field as _make_material_field

    return _make_material_field(domain, params)
