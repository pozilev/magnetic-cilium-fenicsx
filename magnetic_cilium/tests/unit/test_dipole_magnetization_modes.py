from __future__ import annotations

import unittest

import numpy as np

from magnetic_cilium.config.params import ModelParams
from magnetic_cilium.magnetics.dipole import (
    compute_magnetic_dipole_diagnostics,
    compute_magnetization_model_comparison,
    normalize_magnetization_model_list,
)


class FakeComm:
    def allreduce(self, value, op=None):
        return value

    def Allreduce(self, send, recv, op=None):
        np.copyto(recv, send)


class FakeIndexMap:
    def __init__(self, size_local: int):
        self.size_local = size_local


class FakeConnectivity:
    def __init__(self, cells: np.ndarray):
        self._cells = cells

    def links(self, cell: int):
        return self._cells[cell]


class FakeTopology:
    dim = 3

    def __init__(self, cells: np.ndarray):
        self._cells = cells

    def index_map(self, dim: int):
        if dim == 3:
            return FakeIndexMap(self._cells.shape[0])
        return FakeIndexMap(int(np.max(self._cells)) + 1)

    def create_connectivity(self, dim0: int, dim1: int) -> None:
        return None

    def connectivity(self, dim0: int, dim1: int):
        return FakeConnectivity(self._cells)


class FakeGeometry:
    def __init__(self, points: np.ndarray):
        self.x = points


class FakeDomain:
    def __init__(self, points: np.ndarray, cells: np.ndarray):
        self.geometry = FakeGeometry(points)
        self.topology = FakeTopology(cells)
        self.comm = FakeComm()


class FakeDofMap:
    def cell_dofs(self, cell: int):
        return np.array([cell], dtype=np.int32)


class FakeFunctionSpace:
    def __init__(self):
        self.dofmap = FakeDofMap()


class FakeX:
    def __init__(self, values: np.ndarray):
        self.array = values


class FakeMaterial:
    def __init__(self, material_ids: np.ndarray):
        self.function_space = FakeFunctionSpace()
        self.x = FakeX(material_ids.astype(float))


def single_tet_domain():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0e-3, 0.0, 0.0],
            [0.0, 1.0e-3, 0.0],
            [0.0, 0.0, 1.0e-3],
        ],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return FakeDomain(points, cells), FakeMaterial(np.array([2]))


def rotated_displacement(points: np.ndarray, theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    return points @ R.T - points


def symmetric_vertical_domain():
    centers = [
        np.array([1.0e-3, 0.0, 0.0]),
        np.array([-1.0e-3, 0.0, 0.0]),
        np.array([0.0, 1.0e-3, 0.0]),
        np.array([0.0, -1.0e-3, 0.0]),
    ]
    offsets = 1.0e-4 * np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
        dtype=np.float64,
    )
    points = []
    cells = []
    for center in centers:
        start = len(points)
        points.extend(center + offsets)
        cells.append([start, start + 1, start + 2, start + 3])
    return (
        FakeDomain(np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int64)),
        FakeMaterial(np.full(len(cells), 2)),
    )


class DipoleMagnetizationModeTests(unittest.TestCase):
    def assert_fields_close(self, left: dict, right: dict) -> None:
        keys = [
            "B0_sensor_x_T",
            "B0_sensor_y_T",
            "B0_sensor_z_T",
            "B1_sensor_x_T",
            "B1_sensor_y_T",
            "B1_sensor_z_T",
            "dB_sensor_x_T",
            "dB_sensor_y_T",
            "dB_sensor_z_T",
        ]
        for key in keys:
            self.assertAlmostEqual(float(left[key]), float(right[key]), places=18, msg=key)

    def test_fixed_global_matches_legacy_no_rotate(self):
        domain, material = single_tet_domain()
        u_vertices = rotated_displacement(domain.geometry.x, theta=0.31)
        legacy = ModelParams(
            Br_magnetic=0.1,
            sensor_x=2.0e-3,
            sensor_y=1.0e-3,
            sensor_z=3.0e-3,
            rotate_magnetization=False,
        )
        explicit = ModelParams(
            Br_magnetic=0.1,
            sensor_x=2.0e-3,
            sensor_y=1.0e-3,
            sensor_z=3.0e-3,
            rotate_magnetization=True,
            magnetization_model="fixed_global",
        )
        self.assert_fields_close(
            compute_magnetic_dipole_diagnostics(domain, u_vertices, material, legacy),
            compute_magnetic_dipole_diagnostics(domain, u_vertices, material, explicit),
        )

    def test_rotate_with_material_matches_legacy_rotate(self):
        domain, material = single_tet_domain()
        u_vertices = rotated_displacement(domain.geometry.x, theta=0.31)
        legacy = ModelParams(
            Br_magnetic=0.1,
            sensor_x=2.0e-3,
            sensor_y=1.0e-3,
            sensor_z=3.0e-3,
            rotate_magnetization=True,
        )
        explicit = ModelParams(
            Br_magnetic=0.1,
            sensor_x=2.0e-3,
            sensor_y=1.0e-3,
            sensor_z=3.0e-3,
            rotate_magnetization=False,
            magnetization_model="rotate_with_material",
        )
        self.assert_fields_close(
            compute_magnetic_dipole_diagnostics(domain, u_vertices, material, legacy),
            compute_magnetic_dipole_diagnostics(domain, u_vertices, material, explicit),
        )

    def test_vertical_symmetric_cilium_has_small_transverse_field(self):
        domain, material = symmetric_vertical_domain()
        u_vertices = np.zeros_like(domain.geometry.x)
        params = ModelParams(
            Br_magnetic=0.1,
            sensor_x=0.0,
            sensor_y=0.0,
            sensor_z=5.0e-3,
            rotate_magnetization=False,
        )
        result = compute_magnetic_dipole_diagnostics(domain, u_vertices, material, params)
        transverse = float(np.hypot(result["B0_sensor_x_T"], result["B0_sensor_y_T"]))
        bz = abs(float(result["B0_sensor_z_T"]))
        self.assertGreater(bz, 0.0)
        self.assertLess(transverse, 1.0e-12 * bz + 1.0e-24)

    def test_magnetization_angle_to_sensor_normal_is_reported(self):
        domain, material = single_tet_domain()
        theta = 0.31
        u_vertices = rotated_displacement(domain.geometry.x, theta=theta)
        params = ModelParams(
            Br_magnetic=0.1,
            sensor_x=2.0e-3,
            sensor_y=1.0e-3,
            sensor_z=3.0e-3,
            magnetization_model="rotate_with_material",
        )
        result = compute_magnetic_dipole_diagnostics(domain, u_vertices, material, params)
        self.assertIn("dipole_magnetization_angle_to_normal_mean_deg", result)
        self.assertAlmostEqual(
            result["dipole_magnetization_angle_to_normal_mean_deg"],
            result["theta_mu_to_global_z_mean_deg"],
            places=12,
        )
        self.assertAlmostEqual(
            result["dipole_magnetization_angle_to_normal_mean_deg"],
            np.rad2deg(theta),
            places=12,
        )

    def test_empty_comparison_model_list_defaults_to_three_models(self):
        params = ModelParams(rotate_magnetization=True)
        self.assertEqual(
            normalize_magnetization_model_list((), params),
            ["fixed_global", "rotate_with_material", "follow_factor"],
        )

    def test_comparison_smoke_returns_model_rows(self):
        domain, material = single_tet_domain()
        u_vertices = rotated_displacement(domain.geometry.x, theta=0.2)
        params = ModelParams(
            Br_magnetic=0.1,
            sensor_x=2.0e-3,
            sensor_y=1.0e-3,
            sensor_z=3.0e-3,
        )
        rows = compute_magnetization_model_comparison(domain, u_vertices, material, params)
        self.assertEqual(
            [row["magnetization_model"] for row in rows],
            ["fixed_global", "rotate_with_material", "follow_factor"],
        )
        self.assertTrue(all("dB_sensor_norm_uT" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
