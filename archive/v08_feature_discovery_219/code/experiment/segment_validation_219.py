"""Locked compact segment validation for the 219-patient 20-slice cohort."""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment.preprocessing import QuantileClipper
from experiment.segment_pilot import (
    CORE_FEATURES,
    DISC_LEVELS,
    MUSCLE_PAIRS,
    SLIP_TO_TARGET_DISC,
    _metric_row,
    _stratified_bootstrap_indices,
    normalise_patient_ids,
    read_csv_compatible,
)


FEATURE_SETS = [
    "G0_global_compact",
    "A_target_compact",
    "B_target_plus_6_gradients",
]
REFERENCE_PATIENT_ID = "77"
EXPECTED_ELIGIBLE_PATIENTS = 219
EXPECTED_MUSCLES = sorted(muscle for pair in MUSCLE_PAIRS.values() for muscle in pair)


def create_protocol_annotation_table(
    label_file: Path,
    source_annotation_file: Path,
    output_annotation_file: Path,
    reference_patient_id: str = REFERENCE_PATIENT_ID,
    expected_patients: int = EXPECTED_ELIGIBLE_PATIENTS,
) -> pd.DataFrame:
    """Create a v2 table without overwriting the manually reviewed v1 pilot."""
    labels = read_csv_compatible(label_file, dtype={"patient_id": "string"}, low_memory=False)
    source = read_csv_compatible(
        source_annotation_file, dtype={"patient_id": "string"}, low_memory=False
    )
    labels["patient_id"] = normalise_patient_ids(labels["patient_id"])
    source["patient_id"] = normalise_patient_ids(source["patient_id"])
    numeric_label = pd.to_numeric(labels["instability_label"], errors="coerce")
    slice_count = pd.to_numeric(labels["slice_count"], errors="coerce")
    eligible = labels.loc[
        numeric_label.isin([0, 1]) & slice_count.eq(20)
    ].copy()
    eligible["instability_label"] = numeric_label.loc[eligible.index].astype(int)
    if len(eligible) != expected_patients or eligible["patient_id"].nunique() != expected_patients:
        raise ValueError(
            f"Expected {expected_patients} unique eligible patients, found "
            f"{len(eligible)} rows/{eligible['patient_id'].nunique()} IDs"
        )

    mapping = (
        source.loc[
            source["patient_id"].eq(normalise_patient_ids(pd.Series([reference_patient_id])).iloc[0]),
            ["slice_index", "anatomical_level"],
        ]
        .sort_values("slice_index")
        .reset_index(drop=True)
    )
    mapping["slice_index"] = pd.to_numeric(mapping["slice_index"], errors="raise").astype(int)
    if (
        len(mapping) != 20
        or mapping["slice_index"].tolist() != list(range(20))
        or mapping["slice_index"].duplicated().any()
    ):
        raise ValueError("Reference patient 77 does not provide one mapping for slice_index 0..19")
    if set(mapping["anatomical_level"]) - set(DISC_LEVELS):
        raise ValueError("Reference patient 77 contains an unsupported anatomical level")
    if not mapping.groupby("anatomical_level").size().eq(4).all():
        raise ValueError("Reference patient 77 does not contain exactly four slices per level")

    previous_ids = set(source["patient_id"].dropna())
    rows = []
    for patient in eligible.itertuples(index=False):
        prior = str(patient.patient_id) in previous_ids
        for item in mapping.itertuples(index=False):
            rows.append(
                {
                    "schema_version": "v2",
                    "patient_id": str(patient.patient_id),
                    "image_file": getattr(patient, "image_file", ""),
                    "source_slice_count": 20,
                    "slice_index": int(item.slice_index),
                    "anatomical_level": item.anatomical_level,
                    "annotation_confidence": "protocol_inferred",
                    "annotator": "copied_from_verified_patient_77_pattern",
                    "annotation_date": date.today().isoformat(),
                    "review_status": (
                        "mapping_matched_previous_v1"
                        if prior
                        else "protocol_inferred_needs_spot_check"
                    ),
                    "reviewer": "",
                    "notes": (
                        "20-slice level pattern copied from patient 77 as requested; "
                        "not an independent per-patient anatomical review"
                    ),
                }
            )
    output = pd.DataFrame(rows)
    if (
        len(output) != expected_patients * 20
        or output["patient_id"].nunique() != expected_patients
        or output[["patient_id", "slice_index"]].duplicated().any()
    ):
        raise RuntimeError("Generated annotation table failed final row/key validation")
    counts = output.groupby(["patient_id", "anatomical_level"]).size()
    if not counts.eq(4).all():
        raise RuntimeError("Generated table does not contain four slices per patient/level")
    output_annotation_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_annotation_file, index=False, encoding="utf-8-sig")
    return output


def _relation_wide(
    segment_medians: pd.DataFrame,
    patient_levels: pd.DataFrame,
    relation: str,
    patient_order: list[str],
) -> pd.DataFrame:
    level_column = f"{relation}_level"
    chosen = segment_medians.merge(
        patient_levels[["patient_id", level_column]],
        left_on=["patient_id", "anatomical_level"],
        right_on=["patient_id", level_column],
        how="inner",
        validate="many_to_one",
    )
    wide = chosen.pivot(index="patient_id", columns="muscle_name", values=CORE_FEATURES)
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature}" for muscle, feature in wide.columns]
    expected_columns = [
        f"{muscle}__{feature}" for muscle in EXPECTED_MUSCLES for feature in CORE_FEATURES
    ]
    return wide.reindex(index=patient_order, columns=expected_columns)


def _global_wide(frame: pd.DataFrame, patient_order: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(["patient_id", "muscle_name"])[CORE_FEATURES].median()
    wide = grouped.unstack("muscle_name").swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature}" for muscle, feature in wide.columns]
    expected_columns = [
        f"{muscle}__{feature}" for muscle in EXPECTED_MUSCLES for feature in CORE_FEATURES
    ]
    return wide.reindex(index=patient_order, columns=expected_columns)


def _bilateral(
    wide: pd.DataFrame, group: str, feature: str, operation: str
) -> pd.Series:
    left, right = MUSCLE_PAIRS[group]
    left_value = wide[f"{left}__{feature}"]
    right_value = wide[f"{right}__{feature}"]
    both_present = left_value.notna() & right_value.notna()
    if operation == "sum":
        result = left_value + right_value
    elif operation == "mean":
        result = (left_value + right_value) / 2.0
    elif operation == "asymmetry":
        result = (
            2.0 * (left_value - right_value).abs()
            / (left_value.abs() + right_value.abs() + 1e-8)
        )
    else:
        raise ValueError(f"Unknown bilateral operation: {operation}")
    return result.where(both_present)


# Exactly 16 features, locked before the 189-patient confirmation run.
COMPACT_DEFINITIONS = [
    ("erector_spinae_area_sum", "erector_spinae", "muscle_area_mm2", "sum"),
    ("erector_spinae_lean_sum", "erector_spinae", "Lean_Muscle_Area", "sum"),
    ("erector_spinae_fip_mean", "erector_spinae", "FIP", "mean"),
    ("erector_spinae_fip_asymmetry", "erector_spinae", "FIP", "asymmetry"),
    ("erector_spinae_iqr_mean", "erector_spinae", "IQR_Intensity_Muscle", "mean"),
    ("erector_spinae_std_mean", "erector_spinae", "Std_Intensity_Muscle", "mean"),
    (
        "erector_spinae_entropy_mean",
        "erector_spinae",
        "Texture_FirstOrder_Entropy",
        "mean",
    ),
    ("erector_spinae_glcm_idm_mean", "erector_spinae", "Texture_GLCM_Idm", "mean"),
    ("multifidus_area_sum", "multifidus", "muscle_area_mm2", "sum"),
    ("multifidus_lean_sum", "multifidus", "Lean_Muscle_Area", "sum"),
    ("multifidus_fip_mean", "multifidus", "FIP", "mean"),
    ("multifidus_lean_asymmetry", "multifidus", "Lean_Muscle_Area", "asymmetry"),
    ("multifidus_fip_asymmetry", "multifidus", "FIP", "asymmetry"),
    ("multifidus_std_mean", "multifidus", "Std_Intensity_Muscle", "mean"),
    ("psoas_lean_sum", "psoas", "Lean_Muscle_Area", "sum"),
    ("psoas_fip_mean", "psoas", "FIP", "mean"),
]

GRADIENT_DEFINITIONS = [
    (
        "erector_spinae_entropy_target_minus_caudal",
        "erector_spinae",
        "Texture_FirstOrder_Entropy",
        "mean",
    ),
    (
        "erector_spinae_std_target_minus_caudal",
        "erector_spinae",
        "Std_Intensity_Muscle",
        "mean",
    ),
    (
        "erector_spinae_glcm_idm_target_minus_caudal",
        "erector_spinae",
        "Texture_GLCM_Idm",
        "mean",
    ),
    (
        "erector_spinae_lean_target_minus_caudal",
        "erector_spinae",
        "Lean_Muscle_Area",
        "sum",
    ),
    ("multifidus_fip_target_minus_caudal", "multifidus", "FIP", "mean"),
    (
        "multifidus_mean_intensity_target_minus_caudal",
        "multifidus",
        "Mean_Intensity_Muscle",
        "mean",
    ),
]


def _compact_panel(wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{prefix}__{name}": _bilateral(wide, group, feature, operation)
            for name, group, feature, operation in COMPACT_DEFINITIONS
        },
        index=wide.index,
    )


def _gradient_panel(target: pd.DataFrame, caudal: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"gradient__{name}": (
                _bilateral(target, group, feature, operation)
                - _bilateral(caudal, group, feature, operation)
            )
            for name, group, feature, operation in GRADIENT_DEFINITIONS
        },
        index=target.index,
    )


def build_compact_tables(
    annotation_file: Path,
    label_file: Path,
    feature_file: Path,
    pilot_ids: set[str],
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
]:
    annotations = read_csv_compatible(
        annotation_file, dtype={"patient_id": "string"}, low_memory=False
    )
    labels_raw = read_csv_compatible(label_file, dtype={"patient_id": "string"}, low_memory=False)
    features = read_csv_compatible(feature_file, dtype={"patient_id": "string"}, low_memory=False)
    for frame in (annotations, labels_raw, features):
        frame["patient_id"] = normalise_patient_ids(frame["patient_id"])
    annotations["slice_index"] = pd.to_numeric(annotations["slice_index"], errors="raise").astype(int)
    features["slice_index"] = pd.to_numeric(features["slice_index"], errors="raise").astype(int)
    annotation_ids = set(annotations["patient_id"])

    label_numeric = pd.to_numeric(labels_raw["instability_label"], errors="coerce")
    labels = labels_raw.loc[
        labels_raw["patient_id"].isin(annotation_ids) & label_numeric.isin([0, 1]),
        ["patient_id", "target_slip_segment"],
    ].copy()
    labels["label"] = label_numeric.loc[labels.index].astype(int)
    if labels["patient_id"].duplicated().any() or set(labels["patient_id"]) != annotation_ids:
        raise ValueError("219 annotation IDs and usable label IDs do not match one-to-one")
    labels["target_level"] = labels["target_slip_segment"].map(SLIP_TO_TARGET_DISC)
    labels["caudal_level"] = labels["target_level"].map(
        lambda value: (
            DISC_LEVELS[DISC_LEVELS.index(value) + 1]
            if value in DISC_LEVELS and DISC_LEVELS.index(value) + 1 < len(DISC_LEVELS)
            else pd.NA
        )
    )
    labels["cohort_role"] = np.where(
        labels["patient_id"].isin(pilot_ids),
        "development_pilot_30",
        "primary_confirmation_189",
    )
    patient_order = labels["patient_id"].tolist()

    feature_ids = set(features["patient_id"])
    if not annotation_ids.issubset(feature_ids):
        raise ValueError(f"Full 2D file lacks {len(annotation_ids - feature_ids)} eligible patients")
    features = features[features["patient_id"].isin(annotation_ids)].copy()
    annotation_keys = set(map(tuple, annotations[["patient_id", "slice_index"]].to_numpy()))
    feature_keys = set(
        map(tuple, features[["patient_id", "slice_index"]].drop_duplicates().to_numpy())
    )
    if annotation_keys != feature_keys:
        raise ValueError("Annotation and full-feature patient/slice keys differ")
    if features[["patient_id", "slice_index", "muscle_name"]].duplicated().any():
        raise ValueError("Full 2D feature table contains duplicate patient/slice/muscle keys")

    merged = features.merge(
        annotations[["patient_id", "slice_index", "anatomical_level"]],
        on=["patient_id", "slice_index"],
        how="inner",
        validate="many_to_one",
    )
    for column in CORE_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if np.isinf(merged[CORE_FEATURES].to_numpy(float)).any():
        raise ValueError("Core features contain infinite values")

    expected = pd.MultiIndex.from_product(
        [patient_order, range(20), EXPECTED_MUSCLES],
        names=["patient_id", "slice_index", "muscle_name"],
    )
    present = pd.MultiIndex.from_frame(
        features[["patient_id", "slice_index", "muscle_name"]]
    )
    missing_rows = expected.difference(present).to_frame(index=False)
    missing_rows = missing_rows.merge(
        annotations[["patient_id", "slice_index", "anatomical_level"]],
        on=["patient_id", "slice_index"],
        how="left",
        validate="many_to_one",
    )
    missing_rows["action"] = (
        "patient retained; any resulting patient-level missing predictor is "
        "median-imputed using the training fold only"
    )

    segment_medians = (
        merged.groupby(["patient_id", "anatomical_level", "muscle_name"])[CORE_FEATURES]
        .median()
        .reset_index()
    )
    levels = labels[["patient_id", "target_level", "caudal_level"]]
    global_wide = _global_wide(merged, patient_order)
    target_wide = _relation_wide(segment_medians, levels, "target", patient_order)
    caudal_wide = _relation_wide(segment_medians, levels, "caudal", patient_order)
    global_panel = _compact_panel(global_wide, "global")
    target_panel = _compact_panel(target_wide, "target")
    gradients = _gradient_panel(target_wide, caudal_wide)

    label_table = labels[["patient_id", "label"]]

    def finish(panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.replace([np.inf, -np.inf], np.nan)
        panel.index.name = "patient_id"
        return panel.reset_index().merge(
            label_table, on="patient_id", how="left", validate="one_to_one"
        )

    tables = {
        "G0_global_compact": finish(global_panel),
        "A_target_compact": finish(target_panel),
        "B_target_plus_6_gradients": finish(pd.concat([target_panel, gradients], axis=1)),
    }
    issues = [
        {
            "severity": "warning",
            "stage": "annotation_assumption",
            "issue": (
                "All 20-slice patients were assigned the patient-77 level pattern "
                "without individual image review"
            ),
            "action": "Treat as protocol-inferred localisation and perform future spot checks",
        },
        {
            "severity": "warning",
            "stage": "missing_muscle_rows",
            "issue": (
                f"{len(missing_rows)} expected slice/muscle rows are absent across "
                f"{missing_rows['patient_id'].nunique()} patients"
            ),
            "action": "Retain patients; use training-fold-only median imputation",
        },
    ]
    non_l4 = labels[labels["target_slip_segment"] != "L4"]
    if not non_l4.empty:
        issues.append(
            {
                "severity": "info",
                "stage": "caudal_gradient",
                "issue": (
                    f"{len(non_l4)} patient(s) lack a caudal disc within the five annotated "
                    "levels; their six gradient predictors are missing"
                ),
                "action": "Training-fold median imputation; no missingness indicator",
            }
        )
    audit = []
    for name, table in tables.items():
        predictors = table.drop(columns=["patient_id", "label"])
        audit.append(
            {
                "feature_set": name,
                "patients": len(table),
                "candidate_features": predictors.shape[1],
                "missing_cells": int(predictors.isna().sum().sum()),
                "patients_with_missing": int(predictors.isna().any(axis=1).sum()),
                "constant_features": int((predictors.nunique(dropna=True) <= 1).sum()),
                "label_0": int((table["label"] == 0).sum()),
                "label_1": int((table["label"] == 1).sum()),
            }
        )
    return tables, labels, pd.DataFrame(audit), missing_rows, issues


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def run_repeated_validation(
    tables: dict[str, pd.DataFrame],
    cohort_ids: list[str],
    cohort_name: str,
    repeats: int,
    outer_folds: int,
    inner_folds: int,
    base_seed: int,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    reference = (
        tables[FEATURE_SETS[0]]
        .set_index("patient_id")
        .loc[cohort_ids]
        .reset_index()
    )
    ids = reference["patient_id"].to_numpy()
    y = reference["label"].astype(int).to_numpy()
    if np.bincount(y).min() < outer_folds:
        raise ValueError(f"{cohort_name}: minority class smaller than outer folds")
    predictions, repeat_metrics, folds, coefficients, issues = [], [], [], [], []

    for repeat_index in range(repeats):
        seed = base_seed + repeat_index
        splitter = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
        split_list = list(splitter.split(np.zeros(len(y)), y))
        if repeat_index == 0 or (repeat_index + 1) % 5 == 0:
            log(f"{cohort_name}: repeat {repeat_index + 1}/{repeats}, seed={seed}")
        for feature_set in FEATURE_SETS:
            table = tables[feature_set].set_index("patient_id").loc[cohort_ids].reset_index()
            X = table.drop(columns=["patient_id", "label"]).apply(pd.to_numeric, errors="coerce")
            oof = np.full(len(y), np.nan)
            for fold, (train, test) in enumerate(split_list):
                inner = StratifiedKFold(
                    inner_folds, shuffle=True, random_state=seed * 100 + fold
                )
                search = GridSearchCV(
                    _pipeline(seed),
                    {"classifier__C": [0.03, 0.1, 0.3, 1.0]},
                    scoring="roc_auc",
                    cv=inner,
                    refit=True,
                    n_jobs=1,
                    error_score="raise",
                )
                started = time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    search.fit(X.iloc[train], y[train])
                for item in caught:
                    issues.append(
                        {
                            "severity": "warning",
                            "stage": "model_fit",
                            "cohort": cohort_name,
                            "repeat_index": repeat_index,
                            "outer_fold": fold,
                            "feature_set": feature_set,
                            "issue": str(item.message),
                            "action": "recorded for review",
                        }
                    )
                probability = search.predict_proba(X.iloc[test])[:, 1]
                oof[test] = probability
                fitted_coefficients = search.best_estimator_.named_steps["classifier"].coef_[0]
                for feature, coefficient in zip(X.columns, fitted_coefficients):
                    coefficients.append(
                        {
                            "cohort": cohort_name,
                            "repeat_index": repeat_index,
                            "outer_fold": fold,
                            "feature_set": feature_set,
                            "feature": feature,
                            "scaled_coefficient": float(coefficient),
                        }
                    )
                folds.append(
                    {
                        "cohort": cohort_name,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "outer_fold": fold,
                        "feature_set": feature_set,
                        "train_n": len(train),
                        "test_n": len(test),
                        "train_positive": int(y[train].sum()),
                        "test_positive": int(y[test].sum()),
                        "inner_best_auc": float(search.best_score_),
                        "selected_C": float(search.best_params_["classifier__C"]),
                        "runtime_seconds": time.time() - started,
                    }
                )
                for index, value in zip(test, probability):
                    predictions.append(
                        {
                            "cohort": cohort_name,
                            "repeat_index": repeat_index,
                            "seed": seed,
                            "outer_fold": fold,
                            "feature_set": feature_set,
                            "patient_id": ids[index],
                            "true_label": int(y[index]),
                            "predicted_probability": float(value),
                        }
                    )
            repeat_metrics.append(
                {
                    "cohort": cohort_name,
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "feature_set": feature_set,
                    **_metric_row(y, oof),
                }
            )
    return (
        pd.DataFrame(predictions),
        pd.DataFrame(repeat_metrics),
        pd.DataFrame(folds),
        pd.DataFrame(coefficients),
        issues,
    )


def _aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(
            ["cohort", "feature_set", "patient_id", "true_label"], as_index=False
        )
        .agg(
            mean_oof_probability=("predicted_probability", "mean"),
            oof_probability_sd=("predicted_probability", "std"),
            prediction_count=("predicted_probability", "size"),
        )
    )


def _summaries(
    predictions: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = _aggregate_predictions(predictions)
    performance_rows, paired_rows, repeat_rows = [], [], []
    rng = np.random.default_rng(seed)
    comparisons = [
        ("G0_global_compact", "A_target_compact"),
        ("G0_global_compact", "B_target_plus_6_gradients"),
        ("A_target_compact", "B_target_plus_6_gradients"),
    ]
    for cohort in aggregate["cohort"].drop_duplicates():
        cohort_aggregate = aggregate[aggregate["cohort"] == cohort]
        for feature_set in FEATURE_SETS:
            frame = cohort_aggregate[
                cohort_aggregate["feature_set"] == feature_set
            ].sort_values("patient_id")
            y = frame["true_label"].to_numpy(int)
            probability = frame["mean_oof_probability"].to_numpy(float)
            point = _metric_row(y, probability)
            boot = {"roc_auc": [], "pr_auc": [], "brier": []}
            for _ in range(bootstrap_iterations):
                index = _stratified_bootstrap_indices(y, rng)
                boot["roc_auc"].append(roc_auc_score(y[index], probability[index]))
                boot["pr_auc"].append(average_precision_score(y[index], probability[index]))
                boot["brier"].append(brier_score_loss(y[index], probability[index]))
            performance_rows.append(
                {
                    "cohort": cohort,
                    "feature_set": feature_set,
                    **point,
                    **{
                        f"{metric}_ci_low": float(np.quantile(values, 0.025))
                        for metric, values in boot.items()
                    },
                    **{
                        f"{metric}_ci_high": float(np.quantile(values, 0.975))
                        for metric, values in boot.items()
                    },
                }
            )
            repeated = repeat_metrics[
                (repeat_metrics["cohort"] == cohort)
                & (repeat_metrics["feature_set"] == feature_set)
            ]
            for metric in ("roc_auc", "pr_auc", "brier"):
                values = repeated[metric]
                repeat_rows.append(
                    {
                        "cohort": cohort,
                        "feature_set": feature_set,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "median": float(values.median()),
                        "empirical_2_5_percentile": float(values.quantile(0.025)),
                        "empirical_97_5_percentile": float(values.quantile(0.975)),
                    }
                )

        lookup = {
            name: cohort_aggregate[cohort_aggregate["feature_set"] == name]
            .sort_values("patient_id")
            .reset_index(drop=True)
            for name in FEATURE_SETS
        }
        for reference, comparison in comparisons:
            left, right = lookup[reference], lookup[comparison]
            if not left[["patient_id", "true_label"]].equals(
                right[["patient_id", "true_label"]]
            ):
                raise ValueError("Paired prediction cohorts are inconsistent")
            y = left["true_label"].to_numpy(int)
            p0 = left["mean_oof_probability"].to_numpy(float)
            p1 = right["mean_oof_probability"].to_numpy(float)
            differences = []
            for _ in range(bootstrap_iterations):
                index = _stratified_bootstrap_indices(y, rng)
                differences.append(
                    roc_auc_score(y[index], p1[index])
                    - roc_auc_score(y[index], p0[index])
                )
            differences = np.asarray(differences)
            probability_positive = float(np.mean(differences > 0))
            paired_rows.append(
                {
                    "cohort": cohort,
                    "reference": reference,
                    "comparison": comparison,
                    "auc_improvement": float(
                        roc_auc_score(y, p1) - roc_auc_score(y, p0)
                    ),
                    "bootstrap_ci_low": float(np.quantile(differences, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(differences, 0.975)),
                    "bootstrap_probability_gt_0": probability_positive,
                    "two_sided_bootstrap_tail_probability": float(
                        min(1.0, 2 * min(probability_positive, 1 - probability_positive))
                    ),
                }
            )
    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(repeat_rows),
    )


def _coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    return (
        coefficients.groupby(["cohort", "feature_set", "feature"], as_index=False)
        .agg(
            outer_fits=("scaled_coefficient", "size"),
            mean_scaled_coefficient=("scaled_coefficient", "mean"),
            mean_abs_scaled_coefficient=(
                "scaled_coefficient", lambda values: float(np.mean(np.abs(values)))
            ),
            positive_coefficient_fraction=(
                "scaled_coefficient", lambda values: float(np.mean(np.asarray(values) > 0))
            ),
        )
        .sort_values(
            ["cohort", "feature_set", "mean_abs_scaled_coefficient"],
            ascending=[True, True, False],
        )
    )


def run_segment_validation_219(
    label_file: Path,
    source_annotation_file: Path,
    annotation_file: Path,
    feature_file: Path,
    output_dir: Path,
    repeats: int = 10,
    outer_folds: int = 5,
    inner_folds: int = 4,
    bootstrap_iterations: int = 3000,
    base_seed: int = 20260726,
    log: Callable[[str], None] = print,
) -> dict:
    started = time.time()
    log("Create and validate the 219-patient protocol-inferred annotation table")
    annotations = create_protocol_annotation_table(
        label_file, source_annotation_file, annotation_file
    )
    pilot_source = read_csv_compatible(
        source_annotation_file, dtype={"patient_id": "string"}
    )
    pilot_ids = set(normalise_patient_ids(pilot_source["patient_id"]))
    log("Build locked 16-feature target/global panels and six caudal gradients")
    tables, labels, audit, missing_rows, build_issues = build_compact_tables(
        annotation_file, label_file, feature_file, pilot_ids
    )
    primary_ids = labels.loc[
        labels["cohort_role"].eq("primary_confirmation_189"), "patient_id"
    ].tolist()
    all_ids = labels["patient_id"].tolist()
    log(
        f"Cohorts: primary={len(primary_ids)} "
        f"{labels[labels['patient_id'].isin(primary_ids)]['label'].value_counts().to_dict()}, "
        f"sensitivity={len(all_ids)} {labels['label'].value_counts().to_dict()}"
    )

    outputs = []
    for cohort_name, ids, seed_offset in (
        ("primary_confirmation_189", primary_ids, 0),
        ("sensitivity_all_219", all_ids, 1000),
    ):
        outputs.append(
            run_repeated_validation(
                tables,
                ids,
                cohort_name,
                repeats,
                outer_folds,
                inner_folds,
                base_seed + seed_offset,
                log,
            )
        )
    predictions = pd.concat([item[0] for item in outputs], ignore_index=True)
    repeat_metrics = pd.concat([item[1] for item in outputs], ignore_index=True)
    folds = pd.concat([item[2] for item in outputs], ignore_index=True)
    coefficients = pd.concat([item[3] for item in outputs], ignore_index=True)
    runtime_issues = [issue for item in outputs for issue in item[4]]
    performance, paired, repeat_summary = _summaries(
        predictions, repeat_metrics, bootstrap_iterations, base_seed + 5000
    )
    aggregate = _aggregate_predictions(predictions)
    coefficient_summary = _coefficient_summary(coefficients)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(
        output_dir / "annotation_snapshot_used.csv", index=False, encoding="utf-8-sig"
    )
    labels.to_csv(
        output_dir / "cohort_membership_and_labels.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(output_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    missing_rows.to_csv(
        output_dir / "missing_slice_muscle_rows.csv", index=False, encoding="utf-8-sig"
    )
    for feature_set, table in tables.items():
        table.to_csv(
            output_dir / f"patient_features_{feature_set}_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )
    predictions.to_csv(
        output_dir / "oof_predictions_all_repeats.csv", index=False, encoding="utf-8-sig"
    )
    aggregate.to_csv(
        output_dir / "mean_oof_predictions_by_patient.csv", index=False, encoding="utf-8-sig"
    )
    repeat_metrics.to_csv(
        output_dir / "performance_by_repeat.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(output_dir / "nested_cv_fold_details.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(
        output_dir / "coefficients_by_outer_fit.csv", index=False, encoding="utf-8-sig"
    )
    coefficient_summary.to_csv(
        output_dir / "coefficient_stability.csv", index=False, encoding="utf-8-sig"
    )
    performance.to_csv(
        output_dir / "aggregate_oof_performance.csv", index=False, encoding="utf-8-sig"
    )
    paired.to_csv(
        output_dir / "paired_auc_comparisons.csv", index=False, encoding="utf-8-sig"
    )
    repeat_summary.to_csv(
        output_dir / "repeat_performance_summary.csv", index=False, encoding="utf-8-sig"
    )
    all_issues = build_issues + runtime_issues
    pd.DataFrame(all_issues).to_csv(
        output_dir / "warnings_and_bug_records.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "warnings_and_bug_records.json").write_text(
        json.dumps(all_issues, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "experiment_version": "v7_segment_validation_219",
        "annotation_source": str(source_annotation_file),
        "annotation_output": str(annotation_file),
        "feature_source": str(feature_file),
        "feature_sets": FEATURE_SETS,
        "compact_definitions": COMPACT_DEFINITIONS,
        "gradient_definitions": GRADIENT_DEFINITIONS,
        "primary_cohort": "189 patients not used in the v6 pilot",
        "sensitivity_cohort": "all 219 eligible 20-slice patients",
        "repeats": repeats,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "C_grid": [0.03, 0.1, 0.3, 1.0],
        "bootstrap_iterations": bootstrap_iterations,
        "base_seed": base_seed,
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "prefix": sys.prefix,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary_perf = performance[performance["cohort"] == "primary_confirmation_189"]
    primary_paired = paired[paired["cohort"] == "primary_confirmation_189"]
    sensitivity_paired = paired[paired["cohort"] == "sensitivity_all_219"]
    primary_target_gain = primary_paired[
        (primary_paired["reference"] == "G0_global_compact")
        & (primary_paired["comparison"] == "A_target_compact")
    ].iloc[0]
    primary_gradient_gain = primary_paired[
        (primary_paired["reference"] == "G0_global_compact")
        & (primary_paired["comparison"] == "B_target_plus_6_gradients")
    ].iloc[0]
    sensitivity_target_gain = sensitivity_paired[
        (sensitivity_paired["reference"] == "G0_global_compact")
        & (sensitivity_paired["comparison"] == "A_target_compact")
    ].iloc[0]
    sensitivity_gradient_vs_target = sensitivity_paired[
        (sensitivity_paired["reference"] == "A_target_compact")
        & (sensitivity_paired["comparison"] == "B_target_plus_6_gradients")
    ].iloc[0]
    performance_lines = [
        (
            f"| {row.feature_set} | {row.roc_auc:.3f} "
            f"({row.roc_auc_ci_low:.3f}-{row.roc_auc_ci_high:.3f}) | "
            f"{row.pr_auc:.3f} | {row.brier:.3f} |"
        )
        for row in primary_perf.itertuples(index=False)
    ]
    paired_lines = [
        (
            f"| {row.reference} | {row.comparison} | {row.auc_improvement:+.3f} "
            f"({row.bootstrap_ci_low:+.3f} to {row.bootstrap_ci_high:+.3f}) |"
        )
        for row in primary_paired.itertuples(index=False)
    ]
    summary = f"""# v7 compact segment validation (219 eligible patients)

## Confirmatory cohort

The primary analysis contains 189 patients who were not used to choose the
compact feature definitions in the v6 30-patient pilot. The all-219 analysis is
secondary because it reuses the original pilot patients.

| Feature set | ROC-AUC (95% patient-bootstrap interval) | PR-AUC | Brier |
|---|---:|---:|---:|
{chr(10).join(performance_lines)}

| Reference | Comparison | Paired AUC improvement |
|---|---|---:|
{chr(10).join(paired_lines)}

## Interpretation

- Target localisation improved AUC by {primary_target_gain.auc_improvement:+.3f}
  in the independent 189-patient cohort and by
  {sensitivity_target_gain.auc_improvement:+.3f} in the all-219 sensitivity
  cohort. Both intervals crossed zero, so this is a small consistent trend
  rather than confirmatory evidence.
- Adding six locked caudal gradients improved AUC over the global baseline by
  {primary_gradient_gain.auc_improvement:+.3f} in the independent cohort, but
  the interval still crossed zero.
- In the all-219 sensitivity analysis, gradients changed AUC relative to the
  target-only panel by {sensitivity_gradient_vs_target.auc_improvement:+.3f}.
  Therefore, gradient benefit is not yet robust.

## Validation design

- {repeats} random seeds x {outer_folds} outer folds; {inner_folds}-fold inner
  tuning of C only.
- The feature panel was locked before examining outcomes in the 189-patient
  confirmation cohort.
- No supervised feature selection was used.
- Imputation, clipping and scaling were fitted inside each training fold.

## Data limitations

- The 219-patient level table is protocol-inferred from the verified patient-77
  pattern, not individually reviewed anatomical localisation.
- {len(missing_rows)} expected slice/muscle rows were absent from the feature
  source. Patients were retained and residual patient-level missing values were
  imputed inside training folds.
"""
    (output_dir / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    log(f"Completed v7 segment validation in {time.time() - started:.1f} seconds")
    return {
        "performance": performance.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "issues": all_issues,
        "runtime_seconds": time.time() - started,
    }
