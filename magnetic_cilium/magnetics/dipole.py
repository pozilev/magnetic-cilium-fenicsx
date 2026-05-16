import logging
from dataclasses import replace
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from mpi4py import MPI

from magnetic_cilium.config.params import ModelParams
from magnetic_cilium.magnetics.sensor import make_sensor_sample_points, sensor_average_requested_points, sensor_effective_area


log = logging.getLogger("magnetic_cilium_3d")

GLOBAL_EZ = np.array([0.0, 0.0, 1.0], dtype=np.float64)
GLOBAL_EX = np.array([1.0, 0.0, 0.0], dtype=np.float64)
SENSOR_NORMAL = GLOBAL_EZ.copy()

MAGNETIZATION_MODELS = {
    "fixed_global",
    "rotate_with_material",
    "prescribed_theta_mu",
    "follow_factor",
}
DIPOLE_MODES = {
    "tetrahedral",
    "tip_single_dipole",
    "n_point_dipoles",
}


def tetra_volume_from_points(p: np.ndarray) -> float:
    return abs(
        np.linalg.det(
            np.vstack([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
        )
    ) / 6.0


def rotation_from_deformation_gradient(F: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def compute_cell_deformation_gradient_P1(X: np.ndarray, u_values: np.ndarray) -> np.ndarray:
    A = np.vstack([X[1] - X[0], X[2] - X[0], X[3] - X[0]]).T
    B = np.vstack([u_values[1] - u_values[0], u_values[2] - u_values[0], u_values[3] - u_values[0]]).T
    grad_u = B @ np.linalg.inv(A)
    return np.eye(3) + grad_u


def unit_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-30:
        return np.array(GLOBAL_EZ if fallback is None else fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def angle_to_global_z(vector: np.ndarray) -> float:
    direction = unit_vector(vector)
    return float(np.arccos(np.clip(np.dot(direction, GLOBAL_EZ), -1.0, 1.0)))


def angle_to_sensor_normal(vector: np.ndarray) -> float:
    direction = unit_vector(vector)
    normal = unit_vector(SENSOR_NORMAL)
    return float(np.arccos(np.clip(np.dot(direction, normal), -1.0, 1.0)))


def direction_from_theta_and_geometry(theta_rad: float, geometry_axis: np.ndarray) -> np.ndarray:
    """Rotate M0 || ez by theta in the local geometric tilt plane."""
    axis = unit_vector(geometry_axis)
    transverse = axis - np.dot(axis, GLOBAL_EZ) * GLOBAL_EZ
    transverse = unit_vector(transverse, fallback=GLOBAL_EX)
    return unit_vector(np.cos(theta_rad) * GLOBAL_EZ + np.sin(theta_rad) * transverse)


def resolve_magnetization_model(params: ModelParams) -> str:
    """Resolve the explicit model while preserving the legacy boolean contract."""
    model = getattr(params, "magnetization_model", None)
    if model is None or str(model).strip() == "":
        return "rotate_with_material" if bool(getattr(params, "rotate_magnetization", True)) else "fixed_global"

    value = str(model).strip()
    aliases = {
        "fixed": "fixed_global",
        "global": "fixed_global",
        "fixed_global_z": "fixed_global",
        "rotated": "rotate_with_material",
        "rotate": "rotate_with_material",
        "material": "rotate_with_material",
        "prescribed": "prescribed_theta_mu",
        "theta_mu": "prescribed_theta_mu",
        "follow": "follow_factor",
    }
    value = aliases.get(value, value)
    if value not in MAGNETIZATION_MODELS:
        raise ValueError(f"Unknown magnetization_model={model!r}; expected one of {sorted(MAGNETIZATION_MODELS)}.")
    return value


def resolve_dipole_mode(params: ModelParams) -> str:
    mode = str(getattr(params, "dipole_mode", "tetrahedral") or "tetrahedral").strip()
    aliases = {
        "volume": "tetrahedral",
        "tetra": "tetrahedral",
        "tetrahedral_dipoles": "tetrahedral",
        "single_tip": "tip_single_dipole",
        "tip": "tip_single_dipole",
        "points": "n_point_dipoles",
        "n_points": "n_point_dipoles",
    }
    mode = aliases.get(mode, mode)
    if mode not in DIPOLE_MODES:
        raise ValueError(f"Unknown dipole_mode={mode!r}; expected one of {sorted(DIPOLE_MODES)}.")
    return mode


def magnetization_model_params(params: ModelParams, model: str) -> ModelParams:
    model = resolve_magnetization_model(replace(params, magnetization_model=model))
    return replace(
        params,
        magnetization_model=model,
        rotate_magnetization=(model == "rotate_with_material"),
    )


def normalize_magnetization_model_list(
    requested: Iterable[str] | str | None,
    params: ModelParams,
) -> list[str]:
    if requested is None or requested == "":
        current = resolve_magnetization_model(params)
        third = current if current in {"prescribed_theta_mu", "follow_factor"} else "follow_factor"
        requested_values: Sequence[str] = ("fixed_global", "rotate_with_material", third)
    elif isinstance(requested, str):
        requested_values = [part.strip() for part in requested.replace(",", " ").split()]
    else:
        requested_values = [str(value).strip() for value in requested]
    if not requested_values:
        current = resolve_magnetization_model(params)
        third = current if current in {"prescribed_theta_mu", "follow_factor"} else "follow_factor"
        requested_values = ("fixed_global", "rotate_with_material", third)

    models: list[str] = []
    for value in requested_values:
        if not value:
            continue
        model = resolve_magnetization_model(replace(params, magnetization_model=value))
        if model not in models:
            models.append(model)
    return models


def magnetization_direction_from_deformation_gradient(
    F: np.ndarray | None,
    params: ModelParams,
    *,
    deformed: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return M direction and local geometric axis, both derived from M0 || ez."""
    if not deformed:
        return GLOBAL_EZ.copy(), GLOBAL_EZ.copy(), 0.0, 0.0

    F_eff = np.eye(3, dtype=np.float64) if F is None else np.asarray(F, dtype=np.float64)
    geometry_axis = unit_vector(F_eff @ GLOBAL_EZ)
    geometry_theta = angle_to_global_z(geometry_axis)
    model = resolve_magnetization_model(params)

    if model == "fixed_global":
        M_dir = GLOBAL_EZ.copy()
    elif model == "rotate_with_material":
        M_dir = unit_vector(rotation_from_deformation_gradient(F_eff) @ GLOBAL_EZ)
    elif model == "prescribed_theta_mu":
        theta_mu = float(getattr(params, "theta_mu_rad", 0.0) or 0.0)
        M_dir = direction_from_theta_and_geometry(theta_mu, geometry_axis)
    elif model == "follow_factor":
        alpha = float(getattr(params, "follow_factor_alpha", 1.0) or 0.0)
        theta_mu = alpha * geometry_theta
        M_dir = direction_from_theta_and_geometry(theta_mu, geometry_axis)
    else:
        raise ValueError(f"Unknown magnetization model: {model!r}")

    return M_dir, geometry_axis, angle_to_global_z(M_dir), geometry_theta


def magnetization_vector_from_deformation_gradient(
    F: np.ndarray | None,
    params: ModelParams,
    *,
    deformed: bool = True,
) -> np.ndarray:
    mu0 = 4.0 * np.pi * 1e-7
    M_dir, _, _, _ = magnetization_direction_from_deformation_gradient(F, params, deformed=deformed)
    return M_dir * (params.Br_magnetic / mu0)


def dipole_field_at_point(sensor_point: np.ndarray, centroid: np.ndarray, m_cell: np.ndarray) -> np.ndarray:
    mu0 = 4.0 * np.pi * 1e-7
    Rvec = sensor_point - centroid
    Rnorm = float(np.linalg.norm(Rvec))
    if Rnorm < 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    R2 = Rnorm * Rnorm
    R3 = R2 * Rnorm
    R5 = R3 * R2
    return (mu0 / (4.0 * np.pi)) * (3.0 * Rvec * np.dot(m_cell, Rvec) / R5 - m_cell / R3)


def _empty_cell_data() -> dict[str, Any]:
    return {
        "centers": np.zeros((0, 3), dtype=np.float64),
        "volumes": np.zeros(0, dtype=np.float64),
        "moments": np.zeros((0, 3), dtype=np.float64),
        "M_dirs": np.zeros((0, 3), dtype=np.float64),
        "geometry_axes": np.zeros((0, 3), dtype=np.float64),
        "theta_mu_rad": np.zeros(0, dtype=np.float64),
        "theta_geometry_rad": np.zeros(0, dtype=np.float64),
        "ref_z": np.zeros(0, dtype=np.float64),
        "ref_vertices": np.zeros((0, 4, 3), dtype=np.float64),
        "current_vertices": np.zeros((0, 4, 3), dtype=np.float64),
        "source_cells": 0,
        "source_volume": 0.0,
    }


def _magnetic_cell_data(domain, u_vertices: np.ndarray, material, params: ModelParams, deformed: bool) -> dict[str, Any]:
    mu0 = 4.0 * np.pi * 1e-7
    M_abs = params.Br_magnetic / mu0

    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cell_ids = np.arange(num_cells, dtype=np.int32)

    domain.topology.create_connectivity(tdim, 0)
    c_to_v = domain.topology.connectivity(tdim, 0)
    X_vertices = domain.geometry.x[:, :3].copy()

    if u_vertices.shape[0] != X_vertices.shape[0]:
        raise RuntimeError(f"u_vertices length mismatch: got {u_vertices.shape[0]}, expected {X_vertices.shape[0]}.")

    Q = material.function_space
    data = _empty_cell_data()
    centers: list[np.ndarray] = []
    volumes: list[float] = []
    moments: list[np.ndarray] = []
    M_dirs: list[np.ndarray] = []
    geometry_axes: list[np.ndarray] = []
    theta_mu_values: list[float] = []
    theta_geometry_values: list[float] = []
    ref_z_values: list[float] = []
    ref_vertices: list[np.ndarray] = []
    current_vertices: list[np.ndarray] = []

    source_cells = 0
    source_volume = 0.0
    for c in cell_ids:
        dof = Q.dofmap.cell_dofs(int(c))[0]
        mat_id = int(round(float(material.x.array[dof])))
        if mat_id != 2:
            continue

        vertex_ids = c_to_v.links(int(c))
        X = X_vertices[vertex_ids]
        if X.shape[0] != 4:
            continue
        uX = u_vertices[vertex_ids]

        if deformed:
            x_cell = X + uX
            V_cell = tetra_volume_from_points(x_cell)
            F = compute_cell_deformation_gradient_P1(X, uX)
        else:
            x_cell = X
            V_cell = tetra_volume_from_points(X)
            F = None

        if V_cell <= 0.0:
            continue

        M_dir, geometry_axis, theta_mu, theta_geometry = magnetization_direction_from_deformation_gradient(
            F, params, deformed=deformed
        )
        centers.append(np.mean(x_cell, axis=0))
        volumes.append(V_cell)
        moments.append(M_dir * M_abs * V_cell)
        M_dirs.append(M_dir)
        geometry_axes.append(geometry_axis)
        theta_mu_values.append(theta_mu)
        theta_geometry_values.append(theta_geometry)
        ref_z_values.append(float(np.mean(X[:, 2])))
        ref_vertices.append(X)
        current_vertices.append(x_cell)
        source_cells += 1
        source_volume += V_cell

    if not centers:
        return data

    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "volumes": np.asarray(volumes, dtype=np.float64),
        "moments": np.asarray(moments, dtype=np.float64),
        "M_dirs": np.asarray(M_dirs, dtype=np.float64),
        "geometry_axes": np.asarray(geometry_axes, dtype=np.float64),
        "theta_mu_rad": np.asarray(theta_mu_values, dtype=np.float64),
        "theta_geometry_rad": np.asarray(theta_geometry_values, dtype=np.float64),
        "ref_z": np.asarray(ref_z_values, dtype=np.float64),
        "ref_vertices": np.asarray(ref_vertices, dtype=np.float64),
        "current_vertices": np.asarray(current_vertices, dtype=np.float64),
        "source_cells": source_cells,
        "source_volume": source_volume,
    }


def _empty_dipole_records() -> dict[str, np.ndarray]:
    return {
        "centers": np.zeros((0, 3), dtype=np.float64),
        "volumes": np.zeros(0, dtype=np.float64),
        "moments": np.zeros((0, 3), dtype=np.float64),
        "M_dirs": np.zeros((0, 3), dtype=np.float64),
        "geometry_axes": np.zeros((0, 3), dtype=np.float64),
        "theta_mu_rad": np.zeros(0, dtype=np.float64),
        "theta_geometry_rad": np.zeros(0, dtype=np.float64),
    }


def _subset_cell_data_as_records(cell_data: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "centers": cell_data["centers"],
        "volumes": cell_data["volumes"],
        "moments": cell_data["moments"],
        "M_dirs": cell_data["M_dirs"],
        "geometry_axes": cell_data["geometry_axes"],
        "theta_mu_rad": cell_data["theta_mu_rad"],
        "theta_geometry_rad": cell_data["theta_geometry_rad"],
    }


def _tip_point_from_cell_data(cell_data: dict[str, Any]) -> np.ndarray:
    ref_vertices = cell_data["ref_vertices"].reshape((-1, 3))
    current_vertices = cell_data["current_vertices"].reshape((-1, 3))
    if ref_vertices.size == 0:
        return np.zeros(3, dtype=np.float64)
    z_max = float(np.max(ref_vertices[:, 2]))
    z_span = float(np.max(ref_vertices[:, 2]) - np.min(ref_vertices[:, 2]))
    tol = max(1.0e-12, 1.0e-6 * max(z_span, 1.0e-12))
    mask = np.abs(ref_vertices[:, 2] - z_max) <= tol
    if not np.any(mask):
        mask[np.argmax(ref_vertices[:, 2])] = True
    return np.mean(current_vertices[mask], axis=0)


def _aggregate_records(
    centers: list[np.ndarray],
    volumes: list[float],
    moments: list[np.ndarray],
    geometry_axes: list[np.ndarray],
) -> dict[str, np.ndarray]:
    if not centers:
        return _empty_dipole_records()
    volumes_arr = np.asarray(volumes, dtype=np.float64)
    moments_arr = np.asarray(moments, dtype=np.float64)
    M_dirs = np.asarray([unit_vector(m) for m in moments_arr], dtype=np.float64)
    geometry_axes_arr = np.asarray([unit_vector(axis) for axis in geometry_axes], dtype=np.float64)
    theta_mu = np.asarray([angle_to_global_z(direction) for direction in M_dirs], dtype=np.float64)
    theta_geometry = np.asarray([angle_to_global_z(axis) for axis in geometry_axes_arr], dtype=np.float64)
    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "volumes": volumes_arr,
        "moments": moments_arr,
        "M_dirs": M_dirs,
        "geometry_axes": geometry_axes_arr,
        "theta_mu_rad": theta_mu,
        "theta_geometry_rad": theta_geometry,
    }


def _records_for_dipole_mode(cell_data: dict[str, Any], params: ModelParams) -> dict[str, np.ndarray]:
    mode = resolve_dipole_mode(params)
    if mode == "tetrahedral":
        return _subset_cell_data_as_records(cell_data)
    if cell_data["centers"].shape[0] == 0:
        return _empty_dipole_records()

    if mode == "tip_single_dipole":
        total_volume = float(np.sum(cell_data["volumes"]))
        total_moment = np.sum(cell_data["moments"], axis=0)
        axis = np.average(cell_data["geometry_axes"], axis=0, weights=cell_data["volumes"])
        return _aggregate_records(
            [_tip_point_from_cell_data(cell_data)],
            [total_volume],
            [total_moment],
            [axis],
        )

    n_points = max(1, int(getattr(params, "n_point_dipoles", 8) or 1))
    ref_z = cell_data["ref_z"]
    z_min = float(np.min(ref_z))
    z_max = float(np.max(ref_z))
    if abs(z_max - z_min) <= 1.0e-30:
        bins = np.zeros_like(ref_z, dtype=np.int64)
        n_bins = 1
    else:
        edges = np.linspace(z_min, z_max, n_points + 1)
        bins = np.searchsorted(edges, ref_z, side="right") - 1
        bins = np.clip(bins, 0, n_points - 1)
        n_bins = n_points

    centers: list[np.ndarray] = []
    volumes: list[float] = []
    moments: list[np.ndarray] = []
    geometry_axes: list[np.ndarray] = []
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if not np.any(mask):
            continue
        weights = cell_data["volumes"][mask]
        volume = float(np.sum(weights))
        centers.append(np.average(cell_data["centers"][mask], axis=0, weights=weights))
        volumes.append(volume)
        moments.append(np.sum(cell_data["moments"][mask], axis=0))
        geometry_axes.append(np.average(cell_data["geometry_axes"][mask], axis=0, weights=weights))
    return _aggregate_records(centers, volumes, moments, geometry_axes)


def _allreduce_vector(comm, value: np.ndarray, op=MPI.SUM) -> np.ndarray:
    send = np.asarray(value, dtype=np.float64)
    recv = np.zeros_like(send)
    comm.Allreduce(send, recv, op=op)
    return recv


def _distributed_scalar_stats(comm, values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    local_count = int(values.size)
    local_sum = float(np.sum(values)) if local_count else 0.0
    local_min = float(np.min(values)) if local_count else np.inf
    local_max = float(np.max(values)) if local_count else -np.inf

    count = int(comm.allreduce(local_count, op=MPI.SUM))
    if count == 0:
        return np.nan, np.nan, np.nan
    total = float(comm.allreduce(local_sum, op=MPI.SUM))
    min_value = float(comm.allreduce(local_min, op=MPI.MIN))
    max_value = float(comm.allreduce(local_max, op=MPI.MAX))
    return min_value, total / count, max_value


def _weighted_centroid(comm, centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        local_weight = 0.0
        local_sum = np.zeros(3, dtype=np.float64)
    else:
        local_weight = float(np.sum(weights))
        local_sum = np.sum(centers * weights[:, None], axis=0)
    total_weight = float(comm.allreduce(local_weight, op=MPI.SUM))
    total_sum = _allreduce_vector(comm, local_sum, op=MPI.SUM)
    if total_weight <= 0.0:
        return np.full(3, np.nan, dtype=np.float64)
    return total_sum / total_weight


def _geometry_bounds(comm, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        local_min = np.full(3, np.inf, dtype=np.float64)
        local_max = np.full(3, -np.inf, dtype=np.float64)
    else:
        flat = points.reshape((-1, 3))
        local_min = np.min(flat, axis=0)
        local_max = np.max(flat, axis=0)
    return _allreduce_vector(comm, local_min, op=MPI.MIN), _allreduce_vector(comm, local_max, op=MPI.MAX)


def _direction_diagnostics(
    comm,
    records: dict[str, np.ndarray],
    cell_data: dict[str, Any],
    used_mask: np.ndarray,
    *,
    M_abs: float,
) -> dict[str, Any]:
    used_count = int(np.sum(used_mask))
    global_used_count = int(comm.allreduce(used_count, op=MPI.SUM))
    source_cells = int(comm.allreduce(int(cell_data["source_cells"]), op=MPI.SUM))
    source_volume = float(comm.allreduce(float(cell_data["source_volume"]), op=MPI.SUM))

    centers = records["centers"][used_mask]
    volumes = records["volumes"][used_mask]
    moments = records["moments"][used_mask]
    M_dirs = records["M_dirs"][used_mask]

    local_volume = float(np.sum(volumes)) if volumes.size else 0.0
    dipole_volume = float(comm.allreduce(local_volume, op=MPI.SUM))
    local_moment_sum = np.sum(moments, axis=0) if moments.size else np.zeros(3, dtype=np.float64)
    moment_sum = _allreduce_vector(comm, local_moment_sum, op=MPI.SUM)

    local_unit_sum = np.sum(M_dirs * volumes[:, None], axis=0) if M_dirs.size else np.zeros(3, dtype=np.float64)
    unit_sum = _allreduce_vector(comm, local_unit_sum, op=MPI.SUM)
    mean_unit = unit_sum / dipole_volume if dipole_volume > 0.0 else np.full(3, np.nan, dtype=np.float64)

    theta_mu_deg = np.rad2deg(records["theta_mu_rad"][used_mask])
    theta_mu_to_normal_deg = np.rad2deg(
        np.asarray([angle_to_sensor_normal(direction) for direction in M_dirs], dtype=np.float64)
    )
    theta_geometry_deg = np.rad2deg(records["theta_geometry_rad"][used_mask])
    theta_diff_deg = theta_mu_deg - theta_geometry_deg
    theta_mu_min, theta_mu_mean, theta_mu_max = _distributed_scalar_stats(comm, theta_mu_deg)
    normal_min, normal_mean, normal_max = _distributed_scalar_stats(comm, theta_mu_to_normal_deg)
    theta_min, theta_mean, theta_max = _distributed_scalar_stats(comm, theta_geometry_deg)
    diff_min, diff_mean, diff_max = _distributed_scalar_stats(comm, theta_diff_deg)

    moment_center = _weighted_centroid(comm, centers, volumes)
    segment_centroid = _weighted_centroid(comm, cell_data["centers"], cell_data["volumes"])
    segment_min, segment_max = _geometry_bounds(comm, cell_data["current_vertices"])

    return {
        "magnetic_source_cells": source_cells,
        "magnetic_source_volume_m3": source_volume,
        "magnetic_dipole_points": global_used_count,
        "magnetic_dipole_points_requested": int(records["centers"].shape[0]),
        "magnetic_dipole_volume_m3": dipole_volume,
        "magnetization_mean_unit_x": float(mean_unit[0]),
        "magnetization_mean_unit_y": float(mean_unit[1]),
        "magnetization_mean_unit_z": float(mean_unit[2]),
        "magnetization_mean_unit_norm": float(np.linalg.norm(mean_unit)) if np.all(np.isfinite(mean_unit)) else np.nan,
        "sensor_normal_unit_x": float(SENSOR_NORMAL[0]),
        "sensor_normal_unit_y": float(SENSOR_NORMAL[1]),
        "sensor_normal_unit_z": float(SENSOR_NORMAL[2]),
        "dipole_magnetization_angle_to_normal_min_deg": normal_min,
        "dipole_magnetization_angle_to_normal_mean_deg": normal_mean,
        "dipole_magnetization_angle_to_normal_max_deg": normal_max,
        "theta_mu_to_global_z_min_deg": theta_mu_min,
        "theta_mu_to_global_z_mean_deg": theta_mu_mean,
        "theta_mu_to_global_z_max_deg": theta_mu_max,
        "geometry_theta_to_global_z_min_deg": theta_min,
        "geometry_theta_to_global_z_mean_deg": theta_mean,
        "geometry_theta_to_global_z_max_deg": theta_max,
        "theta_mu_minus_geometry_theta_min_deg": diff_min,
        "theta_mu_minus_geometry_theta_mean_deg": diff_mean,
        "theta_mu_minus_geometry_theta_max_deg": diff_max,
        "magnetization_sum_moment_x_A_m2": float(moment_sum[0]),
        "magnetization_sum_moment_y_A_m2": float(moment_sum[1]),
        "magnetization_sum_moment_z_A_m2": float(moment_sum[2]),
        "magnetization_sum_moment_norm_A_m2": float(np.linalg.norm(moment_sum)),
        "magnetization_moment_center_x_m": float(moment_center[0]),
        "magnetization_moment_center_y_m": float(moment_center[1]),
        "magnetization_moment_center_z_m": float(moment_center[2]),
        "magnetic_segment_centroid_x_m": float(segment_centroid[0]),
        "magnetic_segment_centroid_y_m": float(segment_centroid[1]),
        "magnetic_segment_centroid_z_m": float(segment_centroid[2]),
        "magnetic_segment_min_x_m": float(segment_min[0]),
        "magnetic_segment_min_y_m": float(segment_min[1]),
        "magnetic_segment_min_z_m": float(segment_min[2]),
        "magnetic_segment_max_x_m": float(segment_max[0]),
        "magnetic_segment_max_y_m": float(segment_max[1]),
        "magnetic_segment_max_z_m": float(segment_max[2]),
        "magnetization_mean_magnitude_A_per_m": M_abs * float(np.linalg.norm(mean_unit))
        if np.all(np.isfinite(mean_unit)) else np.nan,
    }


def compute_B_from_magnetic_layer_dipoles(
        domain,
        u_vertices: np.ndarray,
        material,
        params: ModelParams,
        deformed: bool,
        sensor_average: bool = False,
        sensor_average_radius: float = 25e-6,
        sensor_average_n: int = 5,
    ):
    mu0 = 4.0 * np.pi * 1e-7
    M0_abs = params.Br_magnetic / mu0
    sensor_point = np.array([params.sensor_x, params.sensor_y, params.sensor_z], dtype=np.float64)
    sensor_points = make_sensor_sample_points(sensor_point, sensor_average, sensor_average_radius, sensor_average_n)

    magnetization_model = resolve_magnetization_model(params)
    dipole_mode = resolve_dipole_mode(params)
    cell_data = _magnetic_cell_data(domain, u_vertices, material, params, deformed=deformed)
    records = _records_for_dipole_mode(cell_data, params)
    B_total = np.zeros((sensor_points.shape[0], 3), dtype=np.float64)
    skipped_near_cells = 0
    min_R = np.inf
    eps = 1e-12
    used_mask = np.zeros(records["centers"].shape[0], dtype=bool)

    for idx, centroid in enumerate(records["centers"]):
        Rnorm_min_cell = float(np.min(np.linalg.norm(sensor_points - centroid, axis=1)))
        min_R = min(min_R, Rnorm_min_cell)
        if Rnorm_min_cell < eps:
            skipped_near_cells += 1
            continue

        for i, point in enumerate(sensor_points):
            B_total[i, :] += dipole_field_at_point(point, centroid, records["moments"][idx])
        used_mask[idx] = True

    B_global_samples = np.zeros_like(B_total)
    domain.comm.Allreduce(B_total, B_global_samples, op=MPI.SUM)
    B_global = np.mean(B_global_samples, axis=0)

    direction_diag = _direction_diagnostics(domain.comm, records, cell_data, used_mask, M_abs=M0_abs)
    magnetic_dipole_cells = (
        direction_diag["magnetic_dipole_points"]
        if dipole_mode == "tetrahedral" else direction_diag["magnetic_source_cells"]
    )
    diagnostics = {
        "sensor_x_m": sensor_point[0],
        "sensor_y_m": sensor_point[1],
        "sensor_z_m": sensor_point[2],
        "Br_magnetic_T": params.Br_magnetic,
        "M_magnetic_A_per_m": M0_abs,
        "magnetization_model": magnetization_model,
        "dipole_mode": dipole_mode,
        "theta_mu_rad": float(getattr(params, "theta_mu_rad", 0.0) or 0.0),
        "theta_mu_deg": float(np.rad2deg(float(getattr(params, "theta_mu_rad", 0.0) or 0.0))),
        "follow_factor_alpha": float(getattr(params, "follow_factor_alpha", 1.0) or 0.0),
        "n_point_dipoles": int(getattr(params, "n_point_dipoles", 8) or 8),
        "sensor_average": bool(sensor_average),
        "sensor_average_radius_m": float(sensor_average_radius),
        "sensor_average_n": int(sensor_average_n),
        "sensor_average_points_requested": sensor_average_requested_points(bool(sensor_average), int(sensor_average_n)),
        "sensor_average_points_used": int(sensor_points.shape[0]),
        "sensor_area_effective_m2": sensor_effective_area(bool(sensor_average), float(sensor_average_radius)),
        "magnetic_dipole_cells": magnetic_dipole_cells,
        "magnetic_min_distance_to_sensor_m": domain.comm.allreduce(min_R, op=MPI.MIN),
        "magnetic_skipped_near_cells": domain.comm.allreduce(skipped_near_cells, op=MPI.SUM),
    }
    diagnostics.update(direction_diag)
    return B_global, diagnostics


def compute_magnetic_dipole_diagnostics(
        domain,
        u_vertices: np.ndarray,
        material,
        params: ModelParams,
        sensor_average: bool = False,
        sensor_average_radius: float = 25e-6,
        sensor_average_n: int = 5,
    ) -> Dict:
    B0, diag0 = compute_B_from_magnetic_layer_dipoles(
        domain, u_vertices, material, params, deformed=False,
        sensor_average=sensor_average,
        sensor_average_radius=sensor_average_radius,
        sensor_average_n=sensor_average_n,
    )
    B1, diag1 = compute_B_from_magnetic_layer_dipoles(
        domain, u_vertices, material, params, deformed=True,
        sensor_average=sensor_average,
        sensor_average_radius=sensor_average_radius,
        sensor_average_n=sensor_average_n,
    )
    dB = B1 - B0

    diagnostics = dict(diag1)
    diagnostics.update(
        {
            "rotate_magnetization": resolve_magnetization_model(params) == "rotate_with_material",
            "legacy_rotate_magnetization_input": bool(getattr(params, "rotate_magnetization", True)),
            "B0_sensor_x_T": B0[0],
            "B0_sensor_y_T": B0[1],
            "B0_sensor_z_T": B0[2],
            "B0_sensor_norm_T": float(np.linalg.norm(B0)),
            "B1_sensor_x_T": B1[0],
            "B1_sensor_y_T": B1[1],
            "B1_sensor_z_T": B1[2],
            "B1_sensor_norm_T": float(np.linalg.norm(B1)),
            "dB_sensor_x_T": dB[0],
            "dB_sensor_y_T": dB[1],
            "dB_sensor_z_T": dB[2],
            "dB_sensor_norm_T": float(np.linalg.norm(dB)),
            "B0_sensor_x_uT": B0[0] * 1e6,
            "B0_sensor_y_uT": B0[1] * 1e6,
            "B0_sensor_z_uT": B0[2] * 1e6,
            "B0_sensor_norm_uT": float(np.linalg.norm(B0)) * 1e6,
            "B1_sensor_x_uT": B1[0] * 1e6,
            "B1_sensor_y_uT": B1[1] * 1e6,
            "B1_sensor_z_uT": B1[2] * 1e6,
            "B1_sensor_norm_uT": float(np.linalg.norm(B1)) * 1e6,
            "dB_sensor_x_uT": dB[0] * 1e6,
            "dB_sensor_y_uT": dB[1] * 1e6,
            "dB_sensor_z_uT": dB[2] * 1e6,
            "dB_sensor_norm_uT": float(np.linalg.norm(dB)) * 1e6,
            "abs_dB_sensor_x_uT": abs(dB[0] * 1e6),
            "abs_dB_sensor_y_uT": abs(dB[1] * 1e6),
            "abs_dB_sensor_z_uT": abs(dB[2] * 1e6),
            "dominant_component": (
                "x" if abs(dB[0]) >= abs(dB[1]) and abs(dB[0]) >= abs(dB[2])
                else "y" if abs(dB[1]) >= abs(dB[2]) else "z"
            ),
            "magnetic_initial_volume_m3": diag0["magnetic_dipole_volume_m3"],
            "magnetic_deformed_volume_m3": diag1["magnetic_dipole_volume_m3"],
        }
    )
    return diagnostics


def compute_magnetization_model_comparison(
        domain,
        u_vertices: np.ndarray,
        material,
        params: ModelParams,
        models: Iterable[str] | str | None = None,
        sensor_average: bool = False,
        sensor_average_radius: float = 25e-6,
        sensor_average_n: int = 5,
    ) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for model in normalize_magnetization_model_list(models, params):
        model_params = magnetization_model_params(params, model)
        result = compute_magnetic_dipole_diagnostics(
            domain,
            u_vertices,
            material,
            model_params,
            sensor_average=sensor_average,
            sensor_average_radius=sensor_average_radius,
            sensor_average_n=sensor_average_n,
        )
        result["magnetization_model"] = model
        results.append(result)
    return results


def log_magnetic_diagnostics(magnetic_diag: Dict) -> None:
    log.info(
        "magnetics: Br = %.3e T, M = %.3e A/m, magnetization_model=%s, dipole_mode=%s",
        magnetic_diag["Br_magnetic_T"],
        magnetic_diag["M_magnetic_A_per_m"],
        magnetic_diag.get("magnetization_model", ""),
        magnetic_diag.get("dipole_mode", "tetrahedral"),
    )
    log.info(
        "magnetics: source cells = %d, dipole points = %d, deformed volume = %.9e m^3",
        magnetic_diag.get("magnetic_source_cells", magnetic_diag.get("magnetic_dipole_cells", 0)),
        magnetic_diag.get("magnetic_dipole_points", magnetic_diag.get("magnetic_dipole_cells", 0)),
        magnetic_diag["magnetic_deformed_volume_m3"],
    )
    log.info("magnetics: min distance from sensor to dipole centroid = %.9e m, skipped near cells = %d", magnetic_diag["magnetic_min_distance_to_sensor_m"], magnetic_diag["magnetic_skipped_near_cells"])
    log.info(
        "magnetics: mean M unit = [%.6g, %.6g, %.6g], sum(m) = [%.6e, %.6e, %.6e] A m^2",
        magnetic_diag.get("magnetization_mean_unit_x", float("nan")),
        magnetic_diag.get("magnetization_mean_unit_y", float("nan")),
        magnetic_diag.get("magnetization_mean_unit_z", float("nan")),
        magnetic_diag.get("magnetization_sum_moment_x_A_m2", float("nan")),
        magnetic_diag.get("magnetization_sum_moment_y_A_m2", float("nan")),
        magnetic_diag.get("magnetization_sum_moment_z_A_m2", float("nan")),
    )
    log.info(
        "magnetics: theta_mu=[%.3f, %.3f, %.3f] deg, theta_mu_to_normal=[%.3f, %.3f, %.3f] deg, theta_geom=[%.3f, %.3f, %.3f] deg, theta_mu-theta=[%.3f, %.3f, %.3f] deg",
        magnetic_diag.get("theta_mu_to_global_z_min_deg", float("nan")),
        magnetic_diag.get("theta_mu_to_global_z_mean_deg", float("nan")),
        magnetic_diag.get("theta_mu_to_global_z_max_deg", float("nan")),
        magnetic_diag.get("dipole_magnetization_angle_to_normal_min_deg", float("nan")),
        magnetic_diag.get("dipole_magnetization_angle_to_normal_mean_deg", float("nan")),
        magnetic_diag.get("dipole_magnetization_angle_to_normal_max_deg", float("nan")),
        magnetic_diag.get("geometry_theta_to_global_z_min_deg", float("nan")),
        magnetic_diag.get("geometry_theta_to_global_z_mean_deg", float("nan")),
        magnetic_diag.get("geometry_theta_to_global_z_max_deg", float("nan")),
        magnetic_diag.get("theta_mu_minus_geometry_theta_min_deg", float("nan")),
        magnetic_diag.get("theta_mu_minus_geometry_theta_mean_deg", float("nan")),
        magnetic_diag.get("theta_mu_minus_geometry_theta_max_deg", float("nan")),
    )
    log.info("magnetics: B0_sensor = [%.6e, %.6e, %.6e] T, |B0| = %.6e T", magnetic_diag["B0_sensor_x_T"], magnetic_diag["B0_sensor_y_T"], magnetic_diag["B0_sensor_z_T"], magnetic_diag["B0_sensor_norm_T"])
    log.info("magnetics: B1_sensor = [%.6e, %.6e, %.6e] T, |B1| = %.6e T", magnetic_diag["B1_sensor_x_T"], magnetic_diag["B1_sensor_y_T"], magnetic_diag["B1_sensor_z_T"], magnetic_diag["B1_sensor_norm_T"])
    log.info("magnetics: dB_sensor = [%.6e, %.6e, %.6e] T, |dB| = %.6e T", magnetic_diag["dB_sensor_x_T"], magnetic_diag["dB_sensor_y_T"], magnetic_diag["dB_sensor_z_T"], magnetic_diag["dB_sensor_norm_T"])
    log.info("magnetics: dB_sensor = [%.3f, %.3f, %.3f] uT, |dB| = %.3f uT", magnetic_diag["dB_sensor_x_uT"], magnetic_diag["dB_sensor_y_uT"], magnetic_diag["dB_sensor_z_uT"], magnetic_diag["dB_sensor_norm_uT"])
