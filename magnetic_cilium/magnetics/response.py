from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def compute_dipole_response(*args, **kwargs):
    return legacy_attr("magnetics_dipoles", "compute_magnetic_dipole_diagnostics")(*args, **kwargs)


def compute_fem_response(*args, **kwargs):
    return legacy_attr("magnetics_fem", "compute_magnetostatic_fem_diagnostics")(*args, **kwargs)
