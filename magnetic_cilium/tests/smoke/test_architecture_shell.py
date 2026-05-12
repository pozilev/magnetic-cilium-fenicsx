from __future__ import annotations

import os
import tempfile
import unittest

from magnetic_cilium.config.loader import load_simulation_config, validate_config_file
from magnetic_cilium.geometry.model import CiliumGeometry, HallSensorGeometry
from magnetic_cilium.io.run_context import RunContext


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class ArchitectureShellSmokeTests(unittest.TestCase):
    def test_new_config_facade_validates_existing_config(self) -> None:
        path = os.path.join(ROOT, "configs", "mechanics_final.yaml")
        report = validate_config_file(path)
        self.assertTrue(report.ok, report.errors)

    def test_run_context_dry_run_does_not_create_directories(self) -> None:
        path = os.path.join(ROOT, "configs", "mechanics_final.yaml")
        config = load_simulation_config(path)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(config, root_dir=tmp, create_dirs=False, run_id="smoke_run")
            self.assertEqual(ctx.run_id, "smoke_run")
            self.assertFalse(os.path.exists(ctx.run_dir))

    def test_geometry_value_objects_are_importable(self) -> None:
        cilium = CiliumGeometry(radius=1.0, lower_length=2.0, magnetic_length=3.0)
        sensor = HallSensorGeometry(center=(0.0, 0.0, -1.0), size=(0.5, 0.5, 0.1))
        self.assertEqual(cilium.total_length, 5.0)
        self.assertEqual(sensor.center[2], -1.0)


if __name__ == "__main__":
    unittest.main()
