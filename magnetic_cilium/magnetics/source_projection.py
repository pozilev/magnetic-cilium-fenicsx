from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def mechanical_magnetic_tets(*args, **kwargs):
    return legacy_attr("magnetics_fem", "mechanical_magnetic_tets")(*args, **kwargs)


def build_source_fields(*args, **kwargs):
    return legacy_attr("magnetics_fem", "build_source_fields")(*args, **kwargs)


def build_source_fields_cell_center(*args, **kwargs):
    return legacy_attr("magnetics_fem", "build_source_fields_cell_center")(*args, **kwargs)
