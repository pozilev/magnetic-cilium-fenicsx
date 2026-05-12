from __future__ import annotations

import csv
import hashlib
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np


log = logging.getLogger("magnetic_cilium_3d")


MASTER_COLUMNS = [
    "run_id", "experiment_id", "timestamp", "mode", "model_type", "config_name", "config_hash",
    "restart_path", "params_path", "mechanics_result_path", "git_commit",
    "Br_T", "M_magnitude_A_per_m", "magnetization_mode", "rotate_magnetization",
    "sensor_x_m", "sensor_y_m", "sensor_z_m", "sensor_depth_mm",
    "sensor_inside_air_box", "sensor_margin_to_air_boundary_m",
    "sensor_x_over_R", "sensor_y_over_R", "sensor_z_over_R",
    "sensor_average", "sensor_average_radius_m", "sensor_average_n",
    "sensor_average_points_requested", "sensor_average_points_used", "sensor_area_effective_m2",
    "magnetic_boundary", "Rair_m", "Rair_over_R", "air_below_m", "air_below_over_R",
    "air_above_m", "air_above_over_R", "h_air_m", "hnear_m", "hfar_m",
    "Rnear_m", "Rnear_over_R", "Rnear_over_Rair", "near_radius_saturates_air_box",
    "Ntet", "max_air_cells", "laptop_safe",
    "projection_mode", "source_volume_ref_m3", "source_volume_0_m3", "source_volume_1_m3",
    "epsV0_percent", "epsV1_percent", "source_cells_0", "source_cells_1", "source_projection_ok",
    "B0x_T", "B0y_T", "B0z_T", "B1x_T", "B1y_T", "B1z_T",
    "dBx_T", "dBy_T", "dBz_T", "norm_dB_T",
    "dBx_uT", "dBy_uT", "dBz_uT", "norm_dB_uT",
    "abs_dBx_uT", "abs_dBy_uT", "abs_dBz_uT",
    "reaction_force_x_N", "reaction_force_x_uN",
    "Sx_uT_per_uN", "Sy_uT_per_uN", "Sz_uT_per_uN", "Snorm_uT_per_uN",
    "target_sensitivity_uT_per_uN", "target_dB_uT", "target_ratio_norm",
    "required_Br_for_target_norm_T", "required_Br_ratio_norm",
    "solver_success", "reliable", "fem_result_reliable_for_comparison",
    "boundary_comparison_ok", "br_linearity_case_ok", "sensor_position_case_ok",
    "under_cilium_sensor_case_ok", "quality_reason",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{utc_timestamp()}_{uuid.uuid4().hex[:8]}"


def resolve_experiment_id(mode: str, requested: str | None, write_mode: str) -> str:
    if requested:
        return str(requested)
    if write_mode == "debug":
        return "debug"
    return f"{mode}_{utc_timestamp()}"


def config_hash(config_path: str | None) -> str:
    if not config_path or not os.path.exists(config_path):
        return ""
    h = hashlib.sha256()
    with open(config_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def result_value(result: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in result and result[key] not in (None, ""):
            return result[key]
    return default


def as_float_or_none(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_f):
        return None
    return value_f


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def compute_quality_flags(result: Dict[str, Any], model_type: str, max_air_cells: int | None = None) -> Dict[str, Any]:
    """Compute stable quality flags for master magnetic results."""
    solver_success = as_bool(result.get("solver_success", True), True)
    Ntet = as_float_or_none(result_value(result, "air_cells", "Ntet"))
    max_cells = as_float_or_none(max_air_cells)
    if max_cells is None:
        max_cells = as_float_or_none(result_value(result, "max_air_cells"))
    laptop_safe = True if Ntet is None or max_cells is None else Ntet <= max_cells

    epsV0 = as_float_or_none(result_value(result, "epsV0_percent", "source_volume_error_initial_percent"))
    epsV1 = as_float_or_none(result_value(result, "epsV1_percent", "source_volume_error_deformed_percent"))
    if model_type == "fem":
        source_projection_ok = as_bool(result.get("source_projection_ok"), False)
    else:
        source_projection_ok = True

    if not solver_success:
        reason = "solver_failed"
    elif not laptop_safe:
        reason = "Ntet_too_large"
    elif model_type == "fem" and epsV0 is None:
        reason = "missing_required_fields"
    elif model_type == "fem" and epsV1 is None:
        reason = "missing_required_fields"
    elif epsV0 is not None and epsV0 >= 5.0:
        reason = "epsV0_too_large"
    elif epsV1 is not None and epsV1 >= 5.0:
        reason = "epsV1_too_large"
    elif not source_projection_ok:
        reason = "source_projection_failed"
    else:
        reason = "ok"

    reliable = solver_success and laptop_safe and source_projection_ok and reason == "ok"
    fem_reliable = (
        reliable and model_type == "fem" and epsV0 is not None and epsV1 is not None
        and epsV0 < 5.0 and epsV1 < 5.0
    )
    under_cilium_sensor_case_ok = False
    try:
        under_cilium_sensor_case_ok = (
            abs(float(result_value(result, "sensor_x_m", default=1.0))) <= 1.0e-12
            and abs(float(result_value(result, "sensor_y_m", default=1.0))) <= 1.0e-12
            and abs(float(result_value(result, "sensor_z_m", default=0.0)) + 1.0e-3) <= 1.0e-12
            and as_bool(result.get("sensor_average"), False)
            and reliable
        )
    except (TypeError, ValueError):
        under_cilium_sensor_case_ok = False
    return {
        "solver_success": solver_success,
        "laptop_safe": laptop_safe,
        "source_projection_ok": source_projection_ok,
        "reliable": reliable,
        "fem_result_reliable_for_comparison": fem_reliable if model_type == "fem" else "",
        "boundary_comparison_ok": fem_reliable if model_type == "fem" else "",
        "br_linearity_case_ok": reliable,
        "sensor_position_case_ok": reliable,
        "under_cilium_sensor_case_ok": under_cilium_sensor_case_ok,
        "quality_reason": reason,
    }


def build_magnetic_master_row(
    result: Dict[str, Any],
    *,
    mode: str,
    model_type: str,
    experiment_id: str,
    config_path: str | None,
    restart_dir: str | None,
    max_air_cells: int | None = None,
) -> Dict[str, Any]:
    """Convert a local magnetic result dictionary into one stable master CSV row."""
    mu0 = 4.0 * np.pi * 1e-7
    Br = result_value(result, "Br_magnetic_T", "Br_T")
    Br_float = as_float_or_none(Br)
    M_magnitude = result_value(result, "M_magnetic_A_per_m", "M_magnitude_A_per_m")
    if M_magnitude == "" and Br_float is not None:
        M_magnitude = Br_float / mu0

    max_cells = max_air_cells if max_air_cells is not None else result_value(result, "max_air_cells")
    quality = compute_quality_flags(result, model_type, max_cells)

    restart_path = restart_dir or result_value(result, "restart_dir", "restart_file")
    params_path = os.path.join(restart_path, "params.json") if restart_path else ""
    mechanics_path = os.path.join(restart_path, "mechanics_result.json") if restart_path else ""

    def uT_to_T(key: str) -> Any:
        value = as_float_or_none(result.get(key))
        return value * 1e-6 if value is not None else ""

    Br_req_norm = result_value(result, "required_Br_for_target_norm_T")
    Br_req_norm_float = as_float_or_none(Br_req_norm)
    Br_ratio_norm = result_value(result, "required_Br_ratio_norm")
    if Br_ratio_norm == "" and Br_req_norm_float is not None and Br_float is not None and abs(Br_float) > 1e-30:
        Br_ratio_norm = Br_req_norm_float / Br_float

    row = {
        "run_id": make_run_id(model_type),
        "experiment_id": experiment_id,
        "timestamp": utc_timestamp(),
        "mode": mode,
        "model_type": model_type,
        "config_name": os.path.basename(config_path) if config_path else "",
        "config_hash": config_hash(config_path),
        "restart_path": restart_path or "",
        "params_path": params_path,
        "mechanics_result_path": mechanics_path,
        "git_commit": git_commit(),
        "Br_T": Br,
        "M_magnitude_A_per_m": M_magnitude,
        "magnetization_mode": "rotated" if as_bool(result.get("rotate_magnetization"), True) else "fixed",
        "rotate_magnetization": result_value(result, "rotate_magnetization", default=True),
        "sensor_x_m": result_value(result, "sensor_x_m"),
        "sensor_y_m": result_value(result, "sensor_y_m"),
        "sensor_z_m": result_value(result, "sensor_z_m"),
        "sensor_depth_mm": result_value(result, "sensor_depth_mm"),
        "sensor_inside_air_box": result_value(result, "sensor_inside_air_box"),
        "sensor_margin_to_air_boundary_m": result_value(result, "sensor_margin_to_air_boundary_m"),
        "sensor_x_over_R": result_value(result, "sensor_x_over_R", "sensor_x_over_r"),
        "sensor_y_over_R": result_value(result, "sensor_y_over_R", "sensor_y_over_r"),
        "sensor_z_over_R": result_value(result, "sensor_z_over_R", "sensor_z_over_r"),
        "sensor_average": result_value(result, "sensor_average", default=False),
        "sensor_average_radius_m": result_value(result, "sensor_average_radius_m"),
        "sensor_average_n": result_value(result, "sensor_average_n"),
        "sensor_average_points_requested": result_value(result, "sensor_average_points_requested"),
        "sensor_average_points_used": result_value(result, "sensor_average_points_used"),
        "sensor_area_effective_m2": result_value(result, "sensor_area_effective_m2"),
        "magnetic_boundary": result_value(result, "magnetic_boundary"),
        "Rair_m": result_value(result, "air_radius_m", "Rair_m"),
        "Rair_over_R": result_value(result, "air_radius_factor", "Rair_over_R"),
        "air_below_m": result_value(result, "air_below_m"),
        "air_below_over_R": result_value(result, "air_below_factor", "air_below_over_R"),
        "air_above_m": result_value(result, "air_above_m"),
        "air_above_over_R": result_value(result, "air_above_factor", "air_above_over_R"),
        "h_air_m": result_value(result, "h_air_m"),
        "hnear_m": result_value(result, "h_air_near_m", "hnear_m"),
        "hfar_m": result_value(result, "h_air_far_m", "hfar_m"),
        "Rnear_m": result_value(result, "near_radius_m", "Rnear_m"),
        "Rnear_over_R": result_value(result, "near_radius_over_R", "Rnear_over_R"),
        "Rnear_over_Rair": result_value(result, "near_radius_to_air_radius", "Rnear_over_Rair"),
        "near_radius_saturates_air_box": result_value(result, "near_radius_saturates_air_box"),
        "Ntet": result_value(result, "air_cells", "Ntet"),
        "max_air_cells": max_cells,
        "laptop_safe": quality["laptop_safe"],
        "projection_mode": result_value(result, "projection_mode", default="cell_center" if model_type == "fem" else ""),
        "source_volume_ref_m3": result_value(result, "reference_magnetic_volume_m3", "source_volume_ref_m3"),
        "source_volume_0_m3": result_value(result, "source_volume_initial_m3", "source_volume_0_m3"),
        "source_volume_1_m3": result_value(result, "source_volume_deformed_m3", "source_volume_1_m3"),
        "epsV0_percent": result_value(result, "epsV0_percent", "source_volume_error_initial_percent"),
        "epsV1_percent": result_value(result, "epsV1_percent", "source_volume_error_deformed_percent"),
        "source_cells_0": result_value(result, "source_cells_initial", "initial_source_cells"),
        "source_cells_1": result_value(result, "source_cells_deformed", "deformed_source_cells"),
        "source_projection_ok": quality["source_projection_ok"],
        "B0x_T": result_value(result, "B0_sensor_x_T", default=uT_to_T("B0_sensor_x_uT")),
        "B0y_T": result_value(result, "B0_sensor_y_T", default=uT_to_T("B0_sensor_y_uT")),
        "B0z_T": result_value(result, "B0_sensor_z_T", default=uT_to_T("B0_sensor_z_uT")),
        "B1x_T": result_value(result, "B1_sensor_x_T", default=uT_to_T("B1_sensor_x_uT")),
        "B1y_T": result_value(result, "B1_sensor_y_T", default=uT_to_T("B1_sensor_y_uT")),
        "B1z_T": result_value(result, "B1_sensor_z_T", default=uT_to_T("B1_sensor_z_uT")),
        "dBx_T": result_value(result, "dB_sensor_x_T", "dBx_T", default=uT_to_T("dB_sensor_x_uT")),
        "dBy_T": result_value(result, "dB_sensor_y_T", "dBy_T", default=uT_to_T("dB_sensor_y_uT")),
        "dBz_T": result_value(result, "dB_sensor_z_T", "dBz_T", default=uT_to_T("dB_sensor_z_uT")),
        "norm_dB_T": result_value(result, "dB_sensor_norm_T", "deltaB_norm_T", default=uT_to_T("dB_sensor_norm_uT")),
        "dBx_uT": result_value(result, "dB_sensor_x_uT"),
        "dBy_uT": result_value(result, "dB_sensor_y_uT"),
        "dBz_uT": result_value(result, "dB_sensor_z_uT"),
        "norm_dB_uT": result_value(result, "dB_sensor_norm_uT"),
        "abs_dBx_uT": result_value(result, "abs_dB_sensor_x_uT", "abs_dB_x_uT", default=abs(as_float_or_none(result_value(result, "dB_sensor_x_uT")) or 0.0)),
        "abs_dBy_uT": result_value(result, "abs_dB_sensor_y_uT", default=abs(as_float_or_none(result_value(result, "dB_sensor_y_uT")) or 0.0)),
        "abs_dBz_uT": result_value(result, "abs_dB_sensor_z_uT", "abs_dB_z_uT", default=abs(as_float_or_none(result_value(result, "dB_sensor_z_uT")) or 0.0)),
        "reaction_force_x_N": result_value(result, "reaction_force_x_N"),
        "reaction_force_x_uN": result_value(result, "reaction_force_x_uN"),
        "Sx_uT_per_uN": result_value(result, "sensitivity_x_uT_per_uN"),
        "Sy_uT_per_uN": result_value(result, "sensitivity_y_uT_per_uN"),
        "Sz_uT_per_uN": result_value(result, "sensitivity_z_uT_per_uN"),
        "Snorm_uT_per_uN": result_value(result, "sensitivity_norm_uT_per_uN"),
        "target_sensitivity_uT_per_uN": result_value(result, "target_sensitivity_uT_per_uN"),
        "target_dB_uT": result_value(result, "target_dB_uT"),
        "target_ratio_norm": result_value(result, "target_ratio_norm"),
        "required_Br_for_target_norm_T": Br_req_norm,
        "required_Br_ratio_norm": Br_ratio_norm,
    }
    row.update(quality)
    return {key: row.get(key, "") for key in MASTER_COLUMNS}


def append_master_result(row: Dict[str, Any], master_csv_path: str) -> None:
    """Append one computed magnetic result to the master CSV."""
    os.makedirs(os.path.dirname(master_csv_path) or ".", exist_ok=True)
    if os.path.exists(master_csv_path) and os.path.getsize(master_csv_path) > 0:
        with open(master_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_columns = reader.fieldnames or []
            if existing_columns != MASTER_COLUMNS:
                old_rows = list(reader)
        if os.path.exists(master_csv_path) and existing_columns != MASTER_COLUMNS:
            with open(master_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow({key: old_row.get(key, "") for key in MASTER_COLUMNS})
            log.info("master magnetic CSV schema upgraded: %s", master_csv_path)
    write_header = not os.path.exists(master_csv_path) or os.path.getsize(master_csv_path) == 0
    with open(master_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in MASTER_COLUMNS})


def append_master_results(
    results: Iterable[Dict[str, Any]],
    *,
    master_csv_path: str,
    mode: str,
    model_type: str,
    experiment_id: str,
    config_path: str | None,
    restart_dir: str | None,
    max_air_cells: int | None = None,
) -> List[Dict[str, Any]]:
    rows = [
        build_magnetic_master_row(
            result,
            mode=mode,
            model_type=model_type,
            experiment_id=experiment_id,
            config_path=config_path,
            restart_dir=restart_dir,
            max_air_cells=max_air_cells,
        )
        for result in results
    ]
    for row in rows:
        append_master_result(row, master_csv_path)
    log.info("master magnetic CSV updated: %s (%d rows appended)", master_csv_path, len(rows))
    return rows


def sync_master_parquet(master_csv_path: str, parquet_path: str | None = None) -> Dict[str, Any]:
    """Mirror master.csv to parquet when optional table dependencies exist."""
    target = parquet_path or os.path.splitext(master_csv_path)[0] + ".parquet"
    if not os.path.exists(master_csv_path):
        return {"ok": False, "path": target, "reason": "missing_master_csv"}
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError:
        return {"ok": False, "path": target, "reason": "pandas_not_installed"}
    try:
        frame = pd.read_csv(master_csv_path)
        frame.to_parquet(target, index=False)
    except Exception as exc:
        return {"ok": False, "path": target, "reason": str(exc)}
    return {"ok": True, "path": target, "reason": "ok"}
