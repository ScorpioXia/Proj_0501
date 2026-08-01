from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumbar_stability.clinical_mri import GLOBAL_MRI_FEATURES, _build_global_mri


class V11InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = PROJECT_ROOT / "configs" / "v11_clinical_mri.json"
        cls.config = json.loads(config_path.read_text(encoding="utf-8"))

    def test_all_configured_inputs_exist(self) -> None:
        for key, value in self.config["paths"].items():
            if key == "output_dir":
                continue
            self.assertTrue((PROJECT_ROOT / value).is_file(), f"Missing {key}: {value}")

    def test_global_mri_panel_shape_and_values(self) -> None:
        source = PROJECT_ROOT / self.config["paths"]["global_3d_feature_file"]
        panel = _build_global_mri(source)
        self.assertEqual(panel["patient_id"].nunique(), 312)
        self.assertEqual(panel.shape, (312, 1 + len(GLOBAL_MRI_FEATURES)))
        self.assertFalse(panel["patient_id"].duplicated().any())
        self.assertTrue(np.isfinite(panel[GLOBAL_MRI_FEATURES].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
