from __future__ import annotations

import os
import tempfile
import unittest

from magnetic_cilium_pipeline.config.loader import load_simulation_config, validate_config_file
from magnetic_cilium_pipeline.io.run_context import RunContext


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ConfigLoaderTests(unittest.TestCase):
    def test_flat_mechanics_config_validates(self) -> None:
        path = os.path.join(ROOT, "configs", "mechanics_final.yaml")
        report = validate_config_file(path)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.mode, "mechanics")

    def test_nested_sweep_config_validates(self) -> None:
        path = os.path.join(ROOT, "configs", "magnetic_under_cilium_br_z_sweep.yaml")
        report = validate_config_file(path)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.mode, "magnetics-fem-validation")
        self.assertTrue(report.is_sweep)

    def test_invalid_config_reports_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("mode: mechanics\nD: -1\nn_steps: 0\n")
            report = validate_config_file(path)
            self.assertFalse(report.ok)
            self.assertGreaterEqual(len(report.errors), 2)

    def test_run_context_is_side_effect_free_when_requested(self) -> None:
        path = os.path.join(ROOT, "configs", "mechanics_final.yaml")
        config = load_simulation_config(path)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(config, root_dir=tmp, create_dirs=False, run_id="test_run")
            self.assertEqual(ctx.run_id, "test_run")
            self.assertFalse(os.path.exists(ctx.run_dir))


if __name__ == "__main__":
    unittest.main()
