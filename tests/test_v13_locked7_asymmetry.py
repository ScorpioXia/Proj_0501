from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumbar_stability.locked7_asymmetry import (  # noqa: E402
    Locked7AsymmetryConfig,
    V13Data,
    _run_panel_stage,
    _run_stage4_nested_selection,
    _save_stage_outputs,
    _stage3_candidate_leaderboard,
    _stage3_cross_model_consensus,
    load_v13_data,
    stage1_panels,
    stage2_panels,
    stage3_panels,
)


class V13Locked7AsymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.real_config = Locked7AsymmetryConfig(
            label_file=str(ROOT / "data/labels/PATIENT_LIST_FILE.csv"),
            feature_universe_file=str(
                ROOT / "data/derived/v11_clinical_mri/patient_feature_universe_raw.csv"
            ),
            locked_selection_file=str(
                ROOT
                / "data/derived/v11_clinical_mri/optimistic_global_selection_NOT_VALID.csv"
            ),
            output_dir=str(ROOT / ".smoke_tmp/v13_real_contract"),
            model_grids={},
            fixed_model_params={},
            validate_only=True,
        )

    def test_real_data_and_panel_contract(self) -> None:
        data = load_v13_data(self.real_config)
        self.assertEqual(len(data.table), 219)
        self.assertEqual(data.table["patient_id"].nunique(), 219)
        self.assertEqual(data.table["label"].value_counts().sort_index().to_dict(), {0: 132, 1: 87})
        self.assertEqual(len(data.locked_features), 7)
        self.assertEqual(len(data.asymmetry_features), 78)
        self.assertEqual(len(data.overlapping_asymmetry_features), 3)
        self.assertEqual(len(data.asymmetry_candidates), 75)

        first = stage1_panels(data)
        second = stage2_panels(data)
        third, manifest = stage3_panels(data)
        self.assertEqual(len(first), 7)
        self.assertTrue(all(len(features) == 6 for features in first.values()))
        self.assertEqual(second, {"locked7_full": data.locked_features})
        self.assertEqual(len(third), 76)
        self.assertEqual((manifest["panel_role"] == "add_one_candidate").sum(), 75)
        self.assertTrue(all(len(features) == 8 for panel, features in third.items() if panel != "locked7_reference"))

    @staticmethod
    def _synthetic_data(rows: int = 40) -> V13Data:
        rng = np.random.default_rng(20260803)
        labels = np.asarray([0, 1] * (rows // 2))
        table = pd.DataFrame(
            {
                "patient_id": [f"S{index:03d}" for index in range(rows)],
                "label": labels,
            }
        )
        locked = []
        for index in range(1, 8):
            feature = f"locked_{index}"
            locked.append(feature)
            table[feature] = rng.normal(size=rows) + labels * (0.08 if index == 1 else 0.0)
        candidates = ["candidate_signal_asymmetry", "candidate_noise_asymmetry"]
        table[candidates[0]] = labels + rng.normal(scale=0.5, size=rows)
        table[candidates[1]] = rng.normal(size=rows)
        table.loc[3, "locked_2"] = np.nan
        return V13Data(
            table=table,
            locked_features=locked,
            asymmetry_features=candidates,
            asymmetry_candidates=candidates,
            overlapping_asymmetry_features=[],
            feature_manifest=pd.DataFrame(),
            audit=pd.DataFrame(),
            warnings=pd.DataFrame(),
        )

    @staticmethod
    def _synthetic_config(output_dir: Path) -> Locked7AsymmetryConfig:
        return Locked7AsymmetryConfig(
            label_file="unused",
            feature_universe_file="unused",
            locked_selection_file="unused",
            output_dir=str(output_dir),
            stages=(1, 2),
            repeats=1,
            outer_folds=2,
            inner_folds=2,
            stage3_repeats=1,
            stage4_repeats=1,
            stage4_inner_folds=2,
            bootstrap_iterations=20,
            screening_bootstrap_iterations=20,
            permutation_iterations=20,
            model_grids={
                "l2_logistic": {"C": [0.1]},
                "random_forest": {
                    "n_estimators": [10],
                    "max_depth": [3],
                    "min_samples_leaf": [2],
                    "max_features": ["sqrt"],
                },
                "xgboost": {
                    "n_estimators": [5],
                    "max_depth": [2],
                    "learning_rate": [0.1],
                },
            },
            fixed_model_params={
                "l2_logistic": {"C": 0.1},
                "random_forest": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                },
                "xgboost": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                },
            },
            resume=True,
        )

    def test_stage2_checkpoints_resume_on_synthetic_data(self) -> None:
        data = self._synthetic_data()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            config = self._synthetic_config(output_dir)
            stage_dir = output_dir / "stage2"
            first_log: list[str] = []
            first_predictions, first_tuning = _run_panel_stage(
                config,
                data,
                2,
                stage_dir,
                stage2_panels(data),
                1,
                True,
                first_log.append,
            )
            self.assertEqual(len(first_predictions), 40 * 3)
            self.assertEqual(len(first_tuning), 2 * 3)
            self.assertEqual(len(list(stage_dir.rglob("complete.json"))), 6)

            second_log: list[str] = []
            second_predictions, _ = _run_panel_stage(
                config,
                data,
                2,
                stage_dir,
                stage2_panels(data),
                1,
                True,
                second_log.append,
            )
            self.assertEqual(len(second_predictions), len(first_predictions))
            self.assertEqual(sum("RESUME task=" in line for line in second_log), 6)

    def test_stage4_training_only_selection_contract_on_synthetic_data(self) -> None:
        data = self._synthetic_data()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            config = self._synthetic_config(output_dir)
            log: list[str] = []
            predictions, scores, selections = _run_stage4_nested_selection(
                config, data, output_dir / "stage4", log.append
            )
            self.assertEqual(len(predictions), 40 * 2 * 3)
            self.assertEqual(len(selections), 2)
            self.assertEqual(len(scores), 2 * 2 * 3)
            self.assertTrue(
                (selections["selection_scope"] == "outer_training_inner_cv_only").all()
            )
            self.assertTrue(
                set(selections["selected_feature"]).issubset(set(data.asymmetry_candidates))
            )

    def test_stage3_fixed_screening_and_multiple_comparison_outputs(self) -> None:
        data = self._synthetic_data()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            config = self._synthetic_config(output_dir)
            stage_dir = output_dir / "stage3"
            panels, manifest = stage3_panels(data)
            log: list[str] = []
            predictions, tuning = _run_panel_stage(
                config,
                data,
                3,
                stage_dir,
                panels,
                1,
                False,
                log.append,
            )
            each_repeat, patient_mean, _ = _save_stage_outputs(
                stage_dir,
                predictions,
                tuning,
                20,
                17,
            )
            leaderboard = _stage3_candidate_leaderboard(
                each_repeat, patient_mean, manifest, config, log.append
            )
            consensus = _stage3_cross_model_consensus(leaderboard)
            self.assertEqual(len(leaderboard), 2 * 3)
            self.assertEqual(len(consensus), 2)
            self.assertTrue(leaderboard["fdr_q_within_model"].between(0, 1).all())
            self.assertIn("robust_candidate", consensus.columns)


if __name__ == "__main__":
    unittest.main()
