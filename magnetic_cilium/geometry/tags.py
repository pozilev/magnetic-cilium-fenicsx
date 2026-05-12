from __future__ import annotations


def make_material_field(domain, params):
    from magnetic_cilium.mechanics.backend import make_material_field as _make_material_field

    return _make_material_field(domain, params)


def make_material_field_from_cell_ids(domain, material_cell_ids):
    from magnetic_cilium.mechanics.backend import make_material_field_from_cell_ids as _make_material_field_from_cell_ids

    return _make_material_field_from_cell_ids(domain, material_cell_ids)


def extract_material_cell_ids(material):
    from magnetic_cilium.mechanics.backend import extract_material_cell_ids as _extract_material_cell_ids

    return _extract_material_cell_ids(material)
