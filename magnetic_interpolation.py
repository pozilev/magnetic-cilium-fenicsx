from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def _write_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _filter_rows(rows: List[Dict[str, str]], args) -> List[Dict[str, str]]:
    filtered = rows
    if args.experiment_id:
        filtered = [r for r in filtered if r.get("experiment_id") == args.experiment_id]
    if args.model_type:
        filtered = [r for r in filtered if r.get("model_type") == args.model_type]
    if args.mode_filter:
        filtered = [r for r in filtered if r.get("mode") == args.mode_filter]
    if _as_bool(args.reliable_only):
        filtered = [r for r in filtered if _as_bool(r.get("reliable"))]
    return filtered


def _best_existing(points: List[Dict[str, Any]], target_column: str) -> Dict[str, Any] | None:
    if not points:
        return None
    return max(points, key=lambda r: float(r[target_column]))


def _unique_sorted(values: List[float]) -> List[float]:
    return sorted(set(float(v) for v in values))


def _min_spacing(values: List[float], fallback: float) -> float:
    if len(values) < 2:
        return fallback
    diffs = [b - a for a, b in zip(values[:-1], values[1:]) if b > a]
    return min(diffs) if diffs else fallback


def _is_close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _recommend_next_points(best: Dict[str, Any] | None, xs: List[float], zs: List[float], x_column: str, z_column: str) -> List[Dict[str, Any]]:
    if best is None or len(xs) == 0 or len(zs) == 0:
        return []
    bx = float(best[x_column])
    bz = float(best[z_column])
    unique_x = _unique_sorted(xs)
    unique_z = _unique_sorted(zs)
    dx = _min_spacing(unique_x, 0.5)
    dz = _min_spacing(unique_z, 0.5)
    proposals: List[Dict[str, Any]] = []

    if _is_close(bx, min(unique_x)):
        proposals.append({x_column: bx - dx, z_column: bz, "reason": "boundary_max_expand_x", "priority": 1})
    if _is_close(bx, max(unique_x)):
        proposals.append({x_column: bx + dx, z_column: bz, "reason": "boundary_max_expand_x", "priority": 1})
    if _is_close(bz, min(unique_z)):
        proposals.append({x_column: bx, z_column: bz - dz, "reason": "boundary_max_expand_z", "priority": 1})
    if _is_close(bz, max(unique_z)):
        proposals.append({x_column: bx, z_column: bz + dz, "reason": "boundary_max_expand_z", "priority": 1})

    for sx, sz in ((-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5), (-0.5, -0.5), (0.5, 0.5)):
        proposals.append({
            x_column: bx + sx * dx,
            z_column: bz + sz * dz,
            "reason": "local_refinement",
            "priority": 2,
        })
    return proposals


def _try_interpolate(points: List[Dict[str, Any]], x_column: str, z_column: str, target_column: str, warnings: List[str]):
    xs = [float(p[x_column]) for p in points]
    zs = [float(p[z_column]) for p in points]
    ys = [float(p[target_column]) for p in points]
    if len(points) < 3:
        warnings.append("not_enough_points_for_interpolation")
        return [], None

    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    xi = [xmin + (xmax - xmin) * i / 50.0 for i in range(51)]
    zi = [zmin + (zmax - zmin) * i / 50.0 for i in range(51)]
    grid_values = None
    X = None
    Z = None
    method = "none"

    try:
        import numpy as np
        from scipy.interpolate import griddata
        X, Z = np.meshgrid(np.asarray(xi), np.asarray(zi), indexing="xy")
        xs_np = np.asarray(xs, dtype=np.float64)
        zs_np = np.asarray(zs, dtype=np.float64)
        ys_np = np.asarray(ys, dtype=np.float64)
        grid_values = griddata(np.column_stack([xs_np, zs_np]), ys_np, (X, Z), method="linear")
        if np.all(np.isnan(grid_values)):
            grid_values = griddata(np.column_stack([xs_np, zs_np]), ys_np, (X, Z), method="nearest")
        method = "scipy.griddata"
    except Exception as exc:
        warnings.append(f"scipy_interpolation_unavailable:{exc}")

    if grid_values is None or X is None or Z is None:
        return [], None

    rows = []
    for x, z, value in zip(X.ravel(), Z.ravel(), grid_values.ravel()):
        rows.append({
            x_column: float(x),
            z_column: float(z),
            f"interpolated_{target_column}": "" if math.isnan(float(value)) else float(value),
            "method": method,
        })

    import numpy as np
    finite = np.isfinite(grid_values)
    best_interp = None
    if np.any(finite):
        idx = int(np.nanargmax(grid_values))
        best_interp = {
            x_column: float(X.ravel()[idx]),
            z_column: float(Z.ravel()[idx]),
            target_column: float(grid_values.ravel()[idx]),
            "method": method,
        }
    return rows, best_interp


def run_magnetic_interpolation(args) -> Dict[str, Any]:
    """Read master CSV, interpolate an existing magnetic sweep, and propose next points."""
    warnings: List[str] = []
    input_csv = args.input_csv
    output_dir = args.interpolation_output_dir
    os.makedirs(output_dir, exist_ok=True)

    report: Dict[str, Any] = {
        "input_csv": input_csv,
        "experiment_id": args.experiment_id,
        "model_type": args.model_type,
        "mode_filter": args.mode_filter,
        "reliable_only": _as_bool(args.reliable_only),
        "target_column": args.target_column,
        "x_column": args.x_column,
        "z_column": args.z_column,
        "warnings": warnings,
    }

    try:
        rows = _load_rows(input_csv)
    except Exception as exc:
        warnings.append(f"failed_to_read_input_csv:{exc}")
        rows = []
    report["input_rows"] = len(rows)

    required = [args.x_column, args.z_column, args.target_column]
    missing = [col for col in required if rows and col not in rows[0]]
    if missing:
        warnings.append(f"missing_columns:{','.join(missing)}")

    filtered = _filter_rows(rows, args) if not missing else []
    points = []
    for row in filtered:
        x = _as_float(row.get(args.x_column))
        z = _as_float(row.get(args.z_column))
        target = _as_float(row.get(args.target_column))
        if x is None or z is None or target is None:
            continue
        points.append({args.x_column: x, args.z_column: z, args.target_column: target})
    report["filtered_rows"] = len(points)

    best_existing = _best_existing(points, args.target_column)
    report["best_existing_point"] = best_existing

    grid_rows, best_interpolated = _try_interpolate(points, args.x_column, args.z_column, args.target_column, warnings)
    report["best_interpolated_point"] = best_interpolated

    if grid_rows:
        _write_csv(
            os.path.join(output_dir, "interpolation_grid.csv"),
            grid_rows,
            [args.x_column, args.z_column, f"interpolated_{args.target_column}", "method"],
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            xs = np.asarray([float(r[args.x_column]) for r in grid_rows], dtype=np.float64)
            zs = np.asarray([float(r[args.z_column]) for r in grid_rows], dtype=np.float64)
            vals = np.asarray([
                np.nan if r[f"interpolated_{args.target_column}"] == "" else float(r[f"interpolated_{args.target_column}"])
                for r in grid_rows
            ])
            plt.figure(figsize=(6, 4))
            sc = plt.scatter(xs, zs, c=vals)
            plt.colorbar(sc, label=args.target_column)
            plt.xlabel(args.x_column)
            plt.ylabel(args.z_column)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"heatmap_{args.target_column}.png"), dpi=160)
            plt.close()
        except Exception as exc:
            warnings.append(f"matplotlib_heatmap_unavailable:{exc}")
    else:
        warnings.append("fallback_no_interpolation_grid")

    xs = [p[args.x_column] for p in points]
    zs = [p[args.z_column] for p in points]
    next_points = _recommend_next_points(best_existing, xs, zs, args.x_column, args.z_column)
    if best_interpolated is not None:
        next_points.insert(0, {
            args.x_column: best_interpolated[args.x_column],
            args.z_column: best_interpolated[args.z_column],
            "reason": "near_interpolated_max",
            "priority": 0,
        })
    elif best_existing is not None:
        next_points.insert(0, {
            args.x_column: best_existing[args.x_column],
            args.z_column: best_existing[args.z_column],
            "reason": "near_best_existing_point",
            "priority": 0,
        })

    _write_csv(
        os.path.join(output_dir, "next_points.csv"),
        next_points,
        [args.x_column, args.z_column, "reason", "priority"],
    )
    report["recommended_next_points_count"] = len(next_points)

    with open(os.path.join(output_dir, "interpolation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
