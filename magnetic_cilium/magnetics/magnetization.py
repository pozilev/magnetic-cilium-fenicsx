from __future__ import annotations

def rotation_from_deformation_gradient(F):
    from magnetic_cilium.magnetics.dipole import rotation_from_deformation_gradient as _rotation_from_deformation_gradient

    return _rotation_from_deformation_gradient(F)


def compute_cell_deformation_gradient_P1(X, u_values):
    from magnetic_cilium.magnetics.dipole import compute_cell_deformation_gradient_P1 as _compute_cell_deformation_gradient_P1

    return _compute_cell_deformation_gradient_P1(X, u_values)
