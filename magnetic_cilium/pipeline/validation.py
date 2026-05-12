from __future__ import annotations

from magnetic_cilium.postprocess.quality import evaluate_quality


def validate_result_quality(*args, **kwargs):
    from magnetic_cilium.mechanics.backend import validate_result_quality as _validate_result_quality

    return _validate_result_quality(*args, **kwargs)


def validate_magnetics_fem_validation_config(*args, **kwargs):
    from magnetic_cilium.pipeline.execution import (
        validate_magnetics_fem_validation_config as _validate_magnetics_fem_validation_config,
    )

    return _validate_magnetics_fem_validation_config(*args, **kwargs)


def validate_result_contract(result, *, model_type: str = "unknown"):
    return evaluate_quality(result, model_type=model_type)
