from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def append_master_results(*args, **kwargs):
    return legacy_attr("magnetic_results", "append_master_results")(*args, **kwargs)


def write_summary(*args, **kwargs):
    return legacy_attr("pipeline", "write_summary")(*args, **kwargs)


def write_magnetic_fem_summary(*args, **kwargs):
    return legacy_attr("pipeline", "write_magnetic_fem_summary")(*args, **kwargs)
