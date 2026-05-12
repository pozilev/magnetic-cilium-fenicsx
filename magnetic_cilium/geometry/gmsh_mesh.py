from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def build_mechanics_mesh(params):
    return legacy_attr("mechanics_model", "build_gmsh_mesh_3d")(params)


def create_dolfinx_mesh(points, cells):
    return legacy_attr("mechanics_model", "create_dolfinx_mesh_3d")(points, cells)


def build_air_box_mesh(*args, **kwargs):
    return legacy_attr("magnetics_fem", "build_air_box_mesh")(*args, **kwargs)
