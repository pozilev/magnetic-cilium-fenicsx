from __future__ import annotations

def mechanical_magnetic_tets(*args, **kwargs):
    from magnetic_cilium.magnetics.fem_scalar_potential import mechanical_magnetic_tets as _mechanical_magnetic_tets

    return _mechanical_magnetic_tets(*args, **kwargs)


def build_source_fields(*args, **kwargs):
    from magnetic_cilium.magnetics.fem_scalar_potential import build_source_fields as _build_source_fields

    return _build_source_fields(*args, **kwargs)


def build_source_fields_cell_center(*args, **kwargs):
    from magnetic_cilium.magnetics.fem_scalar_potential import build_source_fields_cell_center as _build_source_fields_cell_center

    return _build_source_fields_cell_center(*args, **kwargs)
