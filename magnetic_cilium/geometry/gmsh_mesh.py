from __future__ import annotations


def build_mechanics_mesh(params):
    from magnetic_cilium.mechanics.backend import build_gmsh_mesh_3d

    return build_gmsh_mesh_3d(params)


def create_dolfinx_mesh(points, cells):
    from magnetic_cilium.mechanics.backend import create_dolfinx_mesh_3d

    return create_dolfinx_mesh_3d(points, cells)


def build_air_box_mesh(*args, **kwargs):
    from magnetic_cilium.magnetics.fem_scalar_potential import build_air_box_mesh as _build_air_box_mesh

    return _build_air_box_mesh(*args, **kwargs)
