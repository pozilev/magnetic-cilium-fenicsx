from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def compute_B_from_magnetic_layer_dipoles(*args, **kwargs):
    return legacy_attr("magnetics_dipoles", "compute_B_from_magnetic_layer_dipoles")(*args, **kwargs)


def compute_magnetic_dipole_diagnostics(*args, **kwargs):
    return legacy_attr("magnetics_dipoles", "compute_magnetic_dipole_diagnostics")(*args, **kwargs)


def dipole_field_at_point(*args, **kwargs):
    return legacy_attr("magnetics_dipoles", "dipole_field_at_point")(*args, **kwargs)
