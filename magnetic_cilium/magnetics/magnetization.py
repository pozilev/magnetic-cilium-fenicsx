from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def rotation_from_deformation_gradient(F):
    return legacy_attr("magnetics_dipoles", "rotation_from_deformation_gradient")(F)


def compute_cell_deformation_gradient_P1(X, u_values):
    return legacy_attr("magnetics_dipoles", "compute_cell_deformation_gradient_P1")(X, u_values)
