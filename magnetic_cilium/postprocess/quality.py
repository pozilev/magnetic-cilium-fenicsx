from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def compute_quality_flags(*args, **kwargs):
    return legacy_attr("magnetic_results", "compute_quality_flags")(*args, **kwargs)


def validate_result_quality(*args, **kwargs):
    return legacy_attr("mechanics_model", "validate_result_quality")(*args, **kwargs)
