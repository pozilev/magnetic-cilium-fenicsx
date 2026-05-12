from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def validate_result_quality(*args, **kwargs):
    return legacy_attr("mechanics_model", "validate_result_quality")(*args, **kwargs)


def validate_magnetics_fem_validation_config(*args, **kwargs):
    return legacy_attr("pipeline", "validate_magnetics_fem_validation_config")(*args, **kwargs)
