from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumbar_stability.prior_feature_ml import (  # noqa: E402
    FeatureBuildResult,
    PriorMLConfig,
    _model_and_grid,
    _preprocessor,
    _run_models,
    build_prior_feature_table,
)


class V12PriorFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PriorMLConfig(
            label_file=str(ROOT / "data/labels/PATIENT_LIST_FILE.csv"),
            annotation_file=str(
                ROOT / "data/annotations/slice_level_annotations_219_v2.csv"
            ),
            feature_2d_file=str(
                ROOT
                / "data/features/features_312_20260722/muscle_features_2d_v7.csv"
            ),
            locked_feature_universe_file=str(
                ROOT
                / "data/derived/v11_clinical_mri/patient_feature_universe_raw.csv"
            ),
            locked_selection_file=str(
                ROOT
                / "data/derived/v11_clinical_mri/optimistic_global_selection_NOT_VALID.csv"
            ),
            output_dir=str(ROOT / ".smoke_tmp/v12_test"),
            model_grids={},
            validate_only=True,
        )

    def test_real_feature_table_contract(self) -> None:
        result = build_prior_feature_table(self.config)
        self.assertEqual(len(result.table), 219)
        self.assertEqual(result.table["patient_id"].nunique(), 219)
        self.assertEqual(result.table["label"].value_counts().sort_index().to_dict(), {0: 132, 1: 87})
        self.assertEqual(len(result.core_features), 18)
        self.assertEqual(len(result.locked_features), 7)
        self.assertFalse(any("name" in column.lower() for column in result.table.columns))

    def test_three_model_interfaces_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(20260802)
        frame = pd.DataFrame(
            {
                "numeric_a": rng.normal(size=40),
                "numeric_b": rng.normal(size=40),
                "target_slip_segment": ["L4"] * 39 + ["L5"],
            }
        )
        frame.loc[3, "numeric_b"] = np.nan
        labels = np.asarray([0, 1] * 20)
        grids = {
            "xgboost": {"n_estimators": [5], "max_depth": [2], "learning_rate": [0.1]},
            "lightgbm": {"n_estimators": [5], "num_leaves": [7], "learning_rate": [0.1]},
            "random_forest": {
                "n_estimators": [10],
                "max_depth": [3],
                "min_samples_leaf": [2],
                "max_features": ["sqrt"],
            },
        }
        for model_name, grid in grids.items():
            model, _ = _model_and_grid(model_name, grid, 7, labels)
            pipeline = Pipeline(
                [
                    (
                        "preprocess",
                        _preprocessor(
                            ["numeric_a", "numeric_b"], ["target_slip_segment"]
                        ),
                    ),
                    ("model", model),
                ]
            )
            pipeline.fit(frame, labels)
            probability = pipeline.predict_proba(frame)[:, 1]
            self.assertEqual(probability.shape, (40,))
            self.assertTrue(np.isfinite(probability).all())

    def test_model_tasks_checkpoint_and_resume_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(17)
        rows = 40
        labels = np.asarray([0, 1] * (rows // 2))
        table = pd.DataFrame(
            {
                "patient_id": [f"S{index:03d}" for index in range(rows)],
                "target_slip_segment": ["L4"] * 39 + ["L5"],
                "label": labels,
                "core_a": rng.normal(size=rows),
                "core_b": rng.normal(size=rows),
                "locked_a": rng.normal(size=rows),
            }
        )
        table.loc[3, "core_b"] = np.nan
        build = FeatureBuildResult(
            table=table,
            core_features=["core_a", "core_b"],
            locked_features=["locked_a"],
            categorical_features=["target_slip_segment"],
            dictionary=pd.DataFrame(),
            audit=pd.DataFrame(),
            warnings=[],
        )
        grids = {
            "xgboost": {
                "n_estimators": [5],
                "max_depth": [2],
                "learning_rate": [0.1],
            },
            "lightgbm": {
                "n_estimators": [5],
                "num_leaves": [7],
                "learning_rate": [0.1],
            },
            "random_forest": {
                "n_estimators": [10],
                "max_depth": [3],
                "min_samples_leaf": [2],
                "max_features": ["sqrt"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            config = PriorMLConfig(
                label_file="unused",
                annotation_file="unused",
                feature_2d_file="unused",
                locked_feature_universe_file="unused",
                locked_selection_file="unused",
                output_dir=str(output_dir),
                repeats=1,
                outer_folds=2,
                inner_folds=2,
                bootstrap_iterations=10,
                permutation_repeats=0,
                model_grids=grids,
                resume=True,
            )
            first_log: list[str] = []
            first = _run_models(config, build, output_dir, first_log.append)
            self.assertEqual(len(first[0]), rows * 2 * 3)
            self.assertEqual(len(first[2]), 2 * 2 * 3)
            self.assertEqual(
                len(list((output_dir / "checkpoints").rglob("complete.json"))), 12
            )

            second_log: list[str] = []
            second = _run_models(config, build, output_dir, second_log.append)
            self.assertEqual(len(second[0]), len(first[0]))
            self.assertEqual(sum("RESUME task=" in line for line in second_log), 12)


if __name__ == "__main__":
    unittest.main()
