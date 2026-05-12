from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def append_master_result(*args, **kwargs):
    return legacy_attr("magnetic_results", "append_master_result")(*args, **kwargs)


def append_master_results(*args, **kwargs):
    return legacy_attr("magnetic_results", "append_master_results")(*args, **kwargs)


def build_magnetic_master_row(*args, **kwargs):
    return legacy_attr("magnetic_results", "build_magnetic_master_row")(*args, **kwargs)
