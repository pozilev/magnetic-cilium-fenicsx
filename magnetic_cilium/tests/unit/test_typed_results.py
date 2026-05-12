from __future__ import annotations

import unittest

from magnetic_cilium.magnetics.state import MagneticResult
from magnetic_cilium.mechanics.state import MechanicsResult
from magnetic_cilium.postprocess.quality import evaluate_quality


class TypedResultTests(unittest.TestCase):
    def test_mechanics_result_extracts_public_values(self) -> None:
        result = MechanicsResult.from_result_dict(
            {
                "run": 1,
                "study": "mechanics",
                "reaction_force_x_uN": 60.0,
                "_domain": object(),
                "_material": object(),
                "_u_vertices": object(),
            },
            run_id="unit_run",
        )
        self.assertEqual(result.run_id, "unit_run")
        self.assertEqual(result.reaction_force_x_uN, 60.0)
        self.assertNotIn("_domain", result.values)
        self.assertIsNotNone(result.state)

    def test_magnetic_result_extracts_sensor_vectors(self) -> None:
        result = MagneticResult.from_result_dict(
            {
                "sensor_x_m": 0.0,
                "sensor_y_m": 0.0,
                "sensor_z_m": -50e-6,
                "B0_sensor_x_uT": 1.0,
                "B0_sensor_y_uT": 2.0,
                "B0_sensor_z_uT": 3.0,
                "B1_sensor_x_uT": 2.0,
                "B1_sensor_y_uT": 3.0,
                "B1_sensor_z_uT": 4.0,
                "dB_sensor_x_uT": 1.0,
                "dB_sensor_y_uT": 1.0,
                "dB_sensor_z_uT": 1.0,
                "dB_sensor_norm_uT": 1.732,
            },
            model_type="dipole",
            run_id="unit_run",
        )
        self.assertEqual(result.model_type, "dipole")
        self.assertAlmostEqual(result.sensor_response.delta_B.x_T, 1.0e-6)
        self.assertAlmostEqual(result.delta_B_norm_uT, 1.732)

    def test_quality_report_flags_nonpositive_J(self) -> None:
        report = evaluate_quality({"J_min": -0.1, "J_max": 1.2, "dB_sensor_norm_uT": 5.0})
        self.assertFalse(report.ok)
        self.assertIn("nonpositive_J", report.reasons)


if __name__ == "__main__":
    unittest.main()
