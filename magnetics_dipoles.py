import logging
from typing import Dict

import numpy as np
from mpi4py import MPI

from params import ModelParams
from sensor_sampling import make_sensor_sample_points


log = logging.getLogger("magnetic_cilium_3d")


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
    M0_ref = np.array([0.0, 0.0, M0_abs], dtype=np.float64)
    sensor_point = np.array([params.sensor_x, params.sensor_y, params.sensor_z], dtype=np.float64)
    sensor_points = make_sensor_sample_points(sensor_point, sensor_average, sensor_average_radius, sensor_average_n)

    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cell_ids = np.arange(num_cells, dtype=np.int32)

    domain.topology.create_connectivity(tdim, 0)
    c_to_v = domain.topology.connectivity(tdim, 0)
    X_vertices = domain.geometry.x[:, :3].copy()

    if u_vertices.shape[0] != X_vertices.shape[0]:
        raise RuntimeError(f"u_vertices length mismatch: got {u_vertices.shape[0]}, expected {X_vertices.shape[0]}.")

    Q = material.function_space
    B_total = np.zeros((sensor_points.shape[0], 3), dtype=np.float64)
    magnetic_cells = 0
    magnetic_volume = 0.0
    skipped_near_cells = 0
    min_R = np.inf
    eps = 1e-12

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
            if params.rotate_magnetization:
                F = compute_cell_deformation_gradient_P1(X, uX)
                M_cell = rotation_from_deformation_gradient(F) @ M0_ref
            else:
                M_cell = M0_ref.copy()
        else:
            x_cell = X
            V_cell = tetra_volume_from_points(X)
            M_cell = M0_ref.copy()

        centroid = np.mean(x_cell, axis=0)
        m_cell = M_cell * V_cell

        Rnorm_min_cell = float(np.min(np.linalg.norm(sensor_points - centroid, axis=1)))
        min_R = min(min_R, Rnorm_min_cell)
        if Rnorm_min_cell < eps:
            skipped_near_cells += 1
            continue

        for i, point in enumerate(sensor_points):
            B_total[i, :] += dipole_field_at_point(point, centroid, m_cell)
        magnetic_cells += 1
        magnetic_volume += V_cell

    B_global_samples = np.zeros_like(B_total)
    domain.comm.Allreduce(B_total, B_global_samples, op=MPI.SUM)
    B_global = np.mean(B_global_samples, axis=0)

    diagnostics = {
        "sensor_x_m": sensor_point[0],
        "sensor_y_m": sensor_point[1],
        "sensor_z_m": sensor_point[2],
        "Br_magnetic_T": params.Br_magnetic,
        "M_magnetic_A_per_m": M0_abs,
        "sensor_average": bool(sensor_average),
        "sensor_average_radius_m": float(sensor_average_radius),
        "sensor_average_n": int(sensor_average_n),
        "sensor_average_points_used": int(sensor_points.shape[0]),
        "magnetic_dipole_cells": domain.comm.allreduce(magnetic_cells, op=MPI.SUM),
        "magnetic_dipole_volume_m3": domain.comm.allreduce(magnetic_volume, op=MPI.SUM),
        "magnetic_min_distance_to_sensor_m": domain.comm.allreduce(min_R, op=MPI.MIN),
        "magnetic_skipped_near_cells": domain.comm.allreduce(skipped_near_cells, op=MPI.SUM),
    }
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
            "rotate_magnetization": params.rotate_magnetization,
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
            "magnetic_initial_volume_m3": diag0["magnetic_dipole_volume_m3"],
            "magnetic_deformed_volume_m3": diag1["magnetic_dipole_volume_m3"],
        }
    )
    return diagnostics


def log_magnetic_diagnostics(magnetic_diag: Dict) -> None:
    log.info("magnetics: Br = %.3e T, M = %.3e A/m", magnetic_diag["Br_magnetic_T"], magnetic_diag["M_magnetic_A_per_m"])
    log.info("magnetics: magnetic dipole cells = %d, deformed volume = %.9e m^3", magnetic_diag["magnetic_dipole_cells"], magnetic_diag["magnetic_deformed_volume_m3"])
    log.info("magnetics: min distance from sensor to dipole centroid = %.9e m, skipped near cells = %d", magnetic_diag["magnetic_min_distance_to_sensor_m"], magnetic_diag["magnetic_skipped_near_cells"])
    log.info("magnetics: B0_sensor = [%.6e, %.6e, %.6e] T, |B0| = %.6e T", magnetic_diag["B0_sensor_x_T"], magnetic_diag["B0_sensor_y_T"], magnetic_diag["B0_sensor_z_T"], magnetic_diag["B0_sensor_norm_T"])
    log.info("magnetics: B1_sensor = [%.6e, %.6e, %.6e] T, |B1| = %.6e T", magnetic_diag["B1_sensor_x_T"], magnetic_diag["B1_sensor_y_T"], magnetic_diag["B1_sensor_z_T"], magnetic_diag["B1_sensor_norm_T"])
    log.info("magnetics: dB_sensor = [%.6e, %.6e, %.6e] T, |dB| = %.6e T", magnetic_diag["dB_sensor_x_T"], magnetic_diag["dB_sensor_y_T"], magnetic_diag["dB_sensor_z_T"], magnetic_diag["dB_sensor_norm_T"])
    log.info("magnetics: dB_sensor = [%.3f, %.3f, %.3f] uT, |dB| = %.3f uT", magnetic_diag["dB_sensor_x_uT"], magnetic_diag["dB_sensor_y_uT"], magnetic_diag["dB_sensor_z_uT"], magnetic_diag["dB_sensor_norm_uT"])
