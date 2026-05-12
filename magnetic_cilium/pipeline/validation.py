from __future__ import annotations

from magnetic_cilium._compat import legacy_attr
from magnetic_cilium.postprocess.quality import evaluate_quality


def validate_result_quality(*args, **kwargs):
    return legacy_attr("mechanics_model", "validate_result_quality")(*args, **kwargs)


def validate_magnetics_fem_validation_config(*args, **kwargs):
    return legacy_attr("pipeline", "validate_magnetics_fem_validation_config")(*args, **kwargs)


def validate_result_contract(result, *, model_type: str = "unknown"):
    return evaluate_quality(result, model_type=model_type)
