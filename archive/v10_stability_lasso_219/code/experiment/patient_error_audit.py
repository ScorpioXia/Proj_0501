"""Create a patient-level out-of-fold review audit across recent experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment.segment_pilot import normalise_patient_ids, read_csv_compatible


def _wrong(label: pd.Series, probability: pd.Series) -> pd.Series:
    return np.where(label.eq(1), probability.lt(0.5), probability.ge(0.5)).astype(int)


def build_patient_error_audit(
    *,
    label_file: Path,
    current_result_dir: Path,
    factor_result_dir: Path,
    subset_result_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_raw = pd.read_csv(
        current_result_dir / "all_repeated_oof_predictions.csv",
        dtype={"patient_id": "string"},
    )
    current_mean = pd.read_csv(
        current_result_dir / "mean_oof_predictions_by_patient.csv",
        dtype={"patient_id": "string"},
    )
    factor_mean = pd.read_csv(
        factor_result_dir / "mean_oof_predictions_by_patient.csv",
        dtype={"patient_id": "string"},
    )
    subset_mean = pd.read_csv(
        subset_result_dir / "mean_oof_predictions_by_patient.csv",
        dtype={"patient_id": "string"},
    )
    assignments = pd.read_csv(
        current_result_dir / "outer_fold_assignments.csv",
        dtype={"patient_id": "string"},
    )
    missing = pd.read_csv(
        current_result_dir / "source_missing_slice_muscle_rows.csv",
        dtype={"patient_id": "string"},
    )
    labels = read_csv_compatible(
        label_file,
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    for frame in (
        current_raw,
        current_mean,
        factor_mean,
        subset_mean,
        assignments,
        missing,
        labels,
    ):
        frame["patient_id"] = normalise_patient_ids(frame["patient_id"])

    current_raw = current_raw[current_raw["method"].eq("nested_train_only")].copy()
    current_mean = current_mean[current_mean["method"].eq("nested_train_only")].copy()
    factor_mean = factor_mean[factor_mean["method"].eq("nested_train_only")].copy()
    subset_mean = subset_mean[
        subset_mean["model"].eq("E0_locked22_baseline")
    ].copy()

    cohort_ids = set(current_raw["patient_id"])
    label_numeric = pd.to_numeric(labels["instability_label"], errors="coerce")
    cohort_labels = labels.loc[
        labels["patient_id"].isin(cohort_ids) & label_numeric.isin([0, 1]),
        [
            "patient_id",
            "instability_label",
            "target_slip_segment",
            "target_slip_vicesegment",
            "slice_count",
            "age_years",
            "height_cm",
            "weight_kg",
            "bmi_kg_m2",
            "clinical_source",
            "review_status",
            "reviewer",
            "notes",
        ],
    ].copy()
    cohort_labels["instability_label"] = pd.to_numeric(
        cohort_labels["instability_label"], errors="raise"
    ).astype(int)
    if cohort_labels["patient_id"].duplicated().any() or len(cohort_labels) != 219:
        raise ValueError("Expected 219 unique labeled patients for the audit")

    current_model_wide = current_mean.pivot(
        index=["patient_id", "true_label"],
        columns="model",
        values="mean_oof_probability",
    ).reset_index()
    current_model_wide = current_model_wide.rename(columns={
        "logistic_l2": "stability_lasso_logistic_probability",
        "svm_rbf": "stability_lasso_svm_probability",
        "xgboost": "stability_lasso_xgboost_probability",
    })
    factor_model_wide = factor_mean.pivot(
        index=["patient_id", "true_label"],
        columns="model",
        values="mean_oof_probability",
    ).reset_index()
    factor_model_wide = factor_model_wide.rename(columns={
        "logistic_l2": "factor_logistic_probability",
        "svm_rbf": "factor_svm_probability",
        "xgboost": "factor_xgboost_probability",
    }).drop(columns=["true_label"])
    subset_reference = subset_mean[
        ["patient_id", "mean_oof_probability"]
    ].rename(columns={
        "mean_oof_probability": "v8_locked22_probability",
    })

    current_patient = current_raw.groupby(
        ["patient_id", "true_label"], as_index=False
    ).agg(
        current_all_model_mean_probability=("predicted_probability", "mean"),
        current_all_model_probability_sd=("predicted_probability", "std"),
        current_prediction_count=("predicted_probability", "size"),
        current_positive_prediction_fraction=(
            "predicted_probability",
            lambda values: float((values >= 0.5).mean()),
        ),
    )
    current_patient["current_wrong_prediction_fraction"] = np.where(
        current_patient["true_label"].eq(1),
        1.0 - current_patient["current_positive_prediction_fraction"],
        current_patient["current_positive_prediction_fraction"],
    )
    current_patient["current_probability_assigned_to_true_class"] = np.where(
        current_patient["true_label"].eq(1),
        current_patient["current_all_model_mean_probability"],
        1.0 - current_patient["current_all_model_mean_probability"],
    )

    folds = assignments.sort_values(["patient_id", "repeat_index"]).groupby(
        "patient_id", as_index=False
    ).agg(
        heldout_repeats=("repeat_index", "nunique"),
        heldout_fold_trace=(
            "outer_fold",
            lambda values: ";".join(str(int(value)) for value in values),
        ),
    )
    missing_count = missing.groupby("patient_id", as_index=False).size().rename(
        columns={"size": "missing_slice_muscle_rows"}
    )

    audit = cohort_labels.merge(
        current_patient,
        left_on=["patient_id", "instability_label"],
        right_on=["patient_id", "true_label"],
        validate="one_to_one",
    ).merge(
        current_model_wide,
        on=["patient_id", "true_label"],
        validate="one_to_one",
    ).merge(
        factor_model_wide,
        on="patient_id",
        validate="one_to_one",
    ).merge(
        subset_reference,
        on="patient_id",
        validate="one_to_one",
    ).merge(
        folds,
        on="patient_id",
        validate="one_to_one",
    ).merge(
        missing_count,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    audit["missing_slice_muscle_rows"] = (
        audit["missing_slice_muscle_rows"].fillna(0).astype(int)
    )

    probability_columns = [
        "stability_lasso_logistic_probability",
        "stability_lasso_svm_probability",
        "stability_lasso_xgboost_probability",
        "factor_logistic_probability",
        "factor_svm_probability",
        "factor_xgboost_probability",
        "v8_locked22_probability",
    ]
    wrong_columns = []
    for column in probability_columns:
        wrong_column = column.replace("_probability", "_wrong_at_0_5")
        audit[wrong_column] = _wrong(audit["true_label"], audit[column])
        wrong_columns.append(wrong_column)
    audit["wrong_reference_model_count_of_7"] = audit[wrong_columns].sum(axis=1)
    audit["all_7_reference_models_wrong"] = (
        audit["wrong_reference_model_count_of_7"].eq(7).astype(int)
    )

    audit["extreme_rank_within_label"] = audit.groupby("true_label")[
        "current_all_model_mean_probability"
    ].rank(
        method="first",
        ascending=True,
    )
    unstable_count = int(audit["true_label"].eq(1).sum())
    stable_count = int(audit["true_label"].eq(0).sum())
    audit["extreme_current_mismatch"] = np.where(
        audit["true_label"].eq(1),
        audit["extreme_rank_within_label"].le(15),
        audit["extreme_rank_within_label"].gt(stable_count - 15),
    ).astype(int)
    audit["review_priority"] = "routine_model_error_audit"
    audit.loc[
        audit["current_wrong_prediction_fraction"].ge(0.80),
        "review_priority",
    ] = "current_pipeline_hard_case"
    audit.loc[
        audit["current_wrong_prediction_fraction"].ge(0.80)
        & audit["wrong_reference_model_count_of_7"].ge(6),
        "review_priority",
    ] = "high_consensus_review"
    audit.loc[
        audit["extreme_current_mismatch"].eq(1)
        & audit["review_priority"].eq("routine_model_error_audit"),
        "review_priority",
    ] = "extreme_probability_review"

    audit["review_reason"] = (
        "Model disagreement is not proof of a wrong label; review X-ray threshold, "
        "patient_id linkage, slip segment, MRI level mapping, and image quality."
    )
    audit = audit.sort_values(
        [
            "review_priority",
            "wrong_reference_model_count_of_7",
            "current_wrong_prediction_fraction",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    priority = audit[
        ~audit["review_priority"].eq("routine_model_error_audit")
    ].copy()
    priority["review_order"] = np.arange(1, len(priority) + 1)
    return audit, priority


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result_root = root / "results"
    audit, priority = build_patient_error_audit(
        label_file=root / "PATIENT_LIST_FILE.csv",
        current_result_dir=result_root / "v7_stability_lasso_replication",
        factor_result_dir=result_root / "v9_pearson_factor_replication",
        subset_result_dir=result_root / "v8_nested_feature_discovery",
    )
    output = result_root / "v7_stability_lasso_replication"
    audit.to_csv(
        output / "patient_oof_error_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    priority.to_csv(
        output / "priority_review_patients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"patients={len(audit)}; priority={len(priority)}; "
        f"high_consensus={(priority['review_priority'] == 'high_consensus_review').sum()}"
    )
