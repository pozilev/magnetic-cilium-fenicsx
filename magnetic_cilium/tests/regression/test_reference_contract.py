from __future__ import annotations

import json
import os
import unittest

from magnetic_cilium.magnetics.state import MagneticResult
from magnetic_cilium.postprocess.quality import evaluate_quality


FIXTURE = os.path.join(os.path.dirname(__file__), "reference_magnetic_response.json")


class ReferenceContractTests(unittest.TestCase):
    def test_reference_fixture_matches_result_contract(self) -> None:
        with open(FIXTURE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        result = MagneticResult.from_result_dict(payload["result"], model_type=payload["model_type"])
        quality = evaluate_quality(result.values, model_type=payload["model_type"])
        self.assertTrue(quality.ok, quality.reasons)
        self.assertAlmostEqual(result.delta_B_norm_uT, 2.0615528128)


if __name__ == "__main__":
    unittest.main()
