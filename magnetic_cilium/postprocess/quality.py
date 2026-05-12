from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from magnetic_cilium._compat import legacy_attr


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    flags: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_quality(
    result: Mapping[str, Any],
    *,
    model_type: str = "unknown",
    max_air_cells: int | None = None,
) -> QualityReport:
    reasons: list[str] = []
    flags: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    def metric(name: str, *aliases: str) -> float | None:
        value = _first_number(result, name, *aliases)
        if value is not None:
            metrics[name] = value
        return value

    j_min = metric("J_min")
    j_max = metric("J_max")
    vm_max = metric("von_mises_max_Pa")
    dB_norm = metric("dB_sensor_norm_uT")
    air_cells = metric("air_cells", "Ntet")

    flags["finite_J"] = _finite_or_absent(j_min) and _finite_or_absent(j_max)
    flags["positive_J"] = j_min is None or j_min > 0.0
    flags["finite_von_mises"] = _finite_or_absent(vm_max)
    flags["finite_delta_B"] = _finite_or_absent(dB_norm)
    flags["nonnegative_delta_B"] = dB_norm is None or dB_norm >= 0.0
    flags["air_mesh_within_limit"] = max_air_cells is None or air_cells is None or air_cells <= max_air_cells

    if not flags["finite_J"]:
        reasons.append("nonfinite_J")
    if not flags["positive_J"]:
        reasons.append("nonpositive_J")
    if not flags["finite_von_mises"]:
        reasons.append("nonfinite_von_mises")
    if not flags["finite_delta_B"]:
        reasons.append("nonfinite_delta_B")
    if not flags["nonnegative_delta_B"]:
        reasons.append("negative_delta_B_norm")
    if not flags["air_mesh_within_limit"]:
        reasons.append("air_mesh_too_large")

    if model_type == "fem":
        source_ok = _as_bool(result.get("source_projection_ok"), default=False)
        flags["source_projection_ok"] = source_ok
        if not source_ok:
            reasons.append("source_projection_failed")

    return QualityReport(ok=not reasons, flags=flags, metrics=metrics, reasons=reasons)


def compute_quality_flags(*args, **kwargs):
    return legacy_attr("magnetic_results", "compute_quality_flags")(*args, **kwargs)


def validate_result_quality(*args, **kwargs):
    return legacy_attr("mechanics_model", "validate_result_quality")(*args, **kwargs)


def _first_number(result: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = result.get(key)
        if value in ("", None):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan
    return None


def _finite_or_absent(value: float | None) -> bool:
    return value is None or math.isfinite(value)


def _as_bool(value: Any, *, default: bool) -> bool:
    if value in ("", None):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
