from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def make_material_field(domain, params):
    return legacy_attr("mechanics_model", "make_material_field")(domain, params)


def make_material_field_from_cell_ids(domain, material_cell_ids):
    return legacy_attr("mechanics_model", "make_material_field_from_cell_ids")(domain, material_cell_ids)


def extract_material_cell_ids(material):
    return legacy_attr("mechanics_model", "extract_material_cell_ids")(material)
