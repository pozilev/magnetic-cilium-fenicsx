from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from magnetic_cilium.visualization.report import (
    component_value_uT,
    discover_result_files,
    generate_visual_report,
    magnetic_components_uT,
    reaction_force_uN,
    select_baseline_magnetic_row,
)


HAS_PLOTTING = importlib.util.find_spec("matplotlib") is not None and importlib.util.find_spec("numpy") is not None


class VisualReportTests(unittest.TestCase):
    def test_unit_conversion_for_force_and_delta_b(self) -> None:
        mechanics = {"reaction_force_x_N": 5.96e-5}
        self.assertAlmostEqual(reaction_force_uN(mechanics), 59.6)
        row = {"dB_sensor_x_T": "1.25e-6", "dB_sensor_y_uT": "-0.5", "dB_sensor_z_T": "2.0e-6"}
        self.assertAlmostEqual(component_value_uT(row, "x"), 1.25)
        self.assertEqual(magnetic_components_uT(row), {"x": 1.25, "y": -0.5, "z": 2.0})

    def test_selects_final_or_reliable_row_without_field_optimization(self) -> None:
        rows = [
            {
                "study": "sensor_sweep_best_signal",
                "dB_sensor_x_uT": "100",
                "dB_sensor_y_uT": "0",
                "dB_sensor_z_uT": "0",
                "rank_by_deltaB_norm": "1",
            },
            {
                "study": "final_candidate_run",
                "dB_sensor_x_uT": "1",
                "dB_sensor_y_uT": "2",
                "dB_sensor_z_uT": "3",
                "is_valid": "True",
            },
        ]
        selected = select_baseline_magnetic_row(rows, {})
        self.assertIsNotNone(selected)
        self.assertEqual(selected["study"], "final_candidate_run")

    def test_discovers_real_baseline_files(self) -> None:
        result_dir = Path(__file__).resolve().parents[4] / "results" / "experiment_baseline"
        if not result_dir.exists():
            self.skipTest("real baseline results are not present")
        files = discover_result_files(result_dir)
        self.assertIsNotNone(files.mechanics_json)
        self.assertIsNotNone(files.summary_csv)
        self.assertIsNotNone(files.pvd_file)

    @unittest.skipUnless(HAS_PLOTTING, "matplotlib/numpy are required for figure generation")
    def test_report_is_created_when_3d_fields_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "result"
            result_dir.mkdir()
            (result_dir / "mechanics_result.json").write_text(
                json.dumps(
                    {
                        "delta_x_m": 0.00055,
                        "reaction_force_x_N": 5.96e-5,
                        "reaction_force_x_uN": 59.6,
                        "reaction_error_to_60uN_percent": -0.67,
                        "J_min": 0.999,
                        "J_max": 1.001,
                        "von_mises_max_Pa": 215000.0,
                        "element_degree": 2,
                        "h_cilium_m": 20e-6,
                        "h_substrate_m": 100e-6,
                        "nu_pdms": 0.49,
                        "nu_magnetic": 0.49,
                    }
                ),
                encoding="utf-8",
            )
            with (result_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "study",
                        "dB_sensor_x_T",
                        "dB_sensor_y_T",
                        "dB_sensor_z_T",
                        "dB_sensor_norm_T",
                        "sensor_x_m",
                        "sensor_y_m",
                        "sensor_z_m",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "study": "final_candidate_run",
                        "dB_sensor_x_T": "1e-6",
                        "dB_sensor_y_T": "-2e-6",
                        "dB_sensor_z_T": "5e-7",
                        "dB_sensor_norm_T": "2.291287847e-6",
                        "sensor_x_m": "0.0",
                        "sensor_y_m": "0.0",
                        "sensor_z_m": "-5e-5",
                    }
                )

            outdir = root / "visual_report"
            result = generate_visual_report(result_dir, outdir, enable_3d=True)
            self.assertTrue((outdir / "report.md").exists())
            self.assertTrue((outdir / "report.html").exists())
            self.assertTrue((outdir / "figures" / "mechanics_summary_card.png").exists())
            self.assertTrue((outdir / "figures" / "deltaB_components.png").exists())
            self.assertTrue(any("3D" in message for message in result.diagnostics))


if __name__ == "__main__":
    unittest.main()
