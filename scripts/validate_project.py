"""Read-only structural and result-integrity checks for the organized project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumbar_stability.clinical_mri import GLOBAL_MRI_FEATURES, _build_global_mri


VERSIONS = [
    "v01_elasticnet_311",
    "v02_multimodel_311",
    "v03_single_nested_cv_312",
    "v04_repeated_nested_cv_312",
    "v05_factor_analysis_312",
    "v06_segment_pilot_30",
    "v07_segment_validation_219",
    "v08_feature_discovery_219",
    "v09_pearson_factor_219",
    "v10_stability_lasso_219",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_archive() -> None:
    for version in VERSIONS:
        root = PROJECT_ROOT / "archive" / version
        require((root / "README.md").is_file(), f"{version}: README.md missing")
        require((root / "code").is_dir(), f"{version}: code directory missing")
        require((root / "results").is_dir(), f"{version}: results directory missing")
    print(f"archive: {len(VERSIONS)} canonical versions have README/code/results")


def validate_current_inputs() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "v11_clinical_mri.json").read_text(encoding="utf-8")
    )
    for key, value in config["paths"].items():
        if key != "output_dir":
            require((PROJECT_ROOT / value).is_file(), f"configured input missing: {key}")

    current = _build_global_mri(
        PROJECT_ROOT / config["paths"]["global_3d_feature_file"]
    ).sort_values("patient_id").reset_index(drop=True)
    require(current.shape == (312, 8), f"unexpected current MRI panel shape: {current.shape}")

    legacy_wide = (
        PROJECT_ROOT
        / "archive/v05_factor_analysis_312/results/v5_factor_analysis"
        / "patient_features_E3_combined_raw.csv"
    )
    if legacy_wide.is_file():
        previous = _build_global_mri(legacy_wide).sort_values("patient_id").reset_index(drop=True)
        require(current["patient_id"].equals(previous["patient_id"]), "MRI patient IDs changed")
        require(
            np.allclose(
                current[GLOBAL_MRI_FEATURES],
                previous[GLOBAL_MRI_FEATURES],
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            ),
            "MRI values differ from the completed-run wide table",
        )
    print("inputs: v11 files exist; current raw-3D MRI7 matches the completed-run panel")


def validate_current_results() -> None:
    output = PROJECT_ROOT / "results" / "v11_clinical_mri"
    repeated = pd.read_csv(output / "all_repeated_oof_predictions.csv")
    mean_saved = pd.read_csv(output / "mean_oof_predictions_by_patient.csv")
    aggregate = pd.read_csv(output / "aggregate_performance.csv")

    prediction_key = ["cohort", "repeat_index", "feature_set", "model", "patient_id"]
    require(not repeated.duplicated(prediction_key).any(), "duplicate repeated OOF prediction key")
    patient_key = ["cohort", "feature_set", "model", "patient_id"]
    counts = repeated.groupby(patient_key).size()
    require(counts.eq(10).all(), "each patient/model must have exactly 10 OOF probabilities")

    calculated = (
        repeated.groupby(patient_key, as_index=False)
        .agg(label=("label", "first"), mean_oof_probability=("probability", "mean"))
        .sort_values(patient_key)
        .reset_index(drop=True)
    )
    saved = mean_saved.sort_values(patient_key).reset_index(drop=True)
    require(calculated[patient_key + ["label"]].equals(saved[patient_key + ["label"]]), "saved mean OOF keys differ")
    require(
        np.allclose(calculated["mean_oof_probability"], saved["mean_oof_probability"], atol=1e-15),
        "saved mean OOF probabilities differ",
    )

    auc_rows = []
    for keys, frame in calculated.groupby(["cohort", "feature_set", "model"]):
        auc_rows.append((*keys, roc_auc_score(frame["label"], frame["mean_oof_probability"])))
    auc = pd.DataFrame(auc_rows, columns=["cohort", "feature_set", "model", "recomputed_auc"])
    check = aggregate.merge(auc, on=["cohort", "feature_set", "model"], validate="one_to_one")
    require(np.allclose(check["roc_auc"], check["recomputed_auc"], atol=1e-15), "aggregate AUC mismatch")
    require(len(repeated) == 24720, f"unexpected OOF row count: {len(repeated)}")
    print("results: 24,720 OOF rows, 10 predictions/patient/model, aggregate AUCs reproduced")


def main() -> None:
    validate_archive()
    validate_current_inputs()
    validate_current_results()
    print("validation: PASS")


if __name__ == "__main__":
    main()
