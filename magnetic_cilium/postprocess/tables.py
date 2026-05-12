from __future__ import annotations


def append_master_results(*args, **kwargs):
    from magnetic_cilium.io.master_table import append_master_results as _append_master_results

    return _append_master_results(*args, **kwargs)


def write_summary(*args, **kwargs):
    from magnetic_cilium.pipeline.execution import write_summary as _write_summary

    return _write_summary(*args, **kwargs)


def write_magnetic_fem_summary(*args, **kwargs):
    from magnetic_cilium.pipeline.execution import write_magnetic_fem_summary as _write_magnetic_fem_summary

    return _write_magnetic_fem_summary(*args, **kwargs)
