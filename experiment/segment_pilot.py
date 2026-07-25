"""Segment-localised feasibility experiment for the 30-patient pilot cohort.

The module intentionally uses only the already available v7 slice-level 2D
features.  It compares four pre-specified feature sets on identical repeated
cross-validation splits:

G0_global
    Whole-lumbar summary of the same compact muscle feature panel.
A_target
    Target-disc summary only.
B_target_adjacent
    Target, cranial and caudal adjacent levels plus local differences/slopes.
C_mechanistic_combined
    B plus clinically interpretable area, fat, asymmetry and relative-deviation
    combinations that can be calculated without height or vertebral area.

All supervised operations are fitted inside the current training fold.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment.preprocessing import QuantileClipper


DISC_LEVELS = [
    "L1_L2_DISC",
    "L2_L3_DISC",
    "L3_L4_DISC",
    "L4_L5_DISC",
    "L5_S1_DISC",
]
SLIP_TO_TARGET_DISC = {
    "L1": "L1_L2_DISC",
    "L2": "L2_L3_DISC",
    "L3": "L3_L4_DISC",
    "L4": "L4_L5_DISC",
    "L5": "L5_S1_DISC",
}
MUSCLE_PAIRS = {
    "multifidus": ("multifidus_left", "multifidus_right"),
    "erector_spinae": ("erector_spinae_left", "erector_spinae_right"),
    "psoas": ("psoas_left", "psoas_right"),
}
EXPECTED_MUSCLES = [muscle for pair in MUSCLE_PAIRS.values() for muscle in pair]

# Compact, pre-specified panel. Area/functional area, fat fraction, intensity
# location/dispersion and three corrected/transparent texture measurements.
CORE_FEATURES = [
    "muscle_area_mm2",
    "Lean_Muscle_Area",
    "FIP",
    "Mean_Intensity_Muscle",
    "Std_Intensity_Muscle",
    "IQR_Intensity_Muscle",
    "Texture_FirstOrder_Entropy",
    "Texture_GLCM_Contrast",
    "Texture_GLCM_Idm",
]
ASYMMETRY_FEATURES = [
    "muscle_area_mm2",
    "Lean_Muscle_Area",
    "FIP",
    "Mean_Intensity_Muscle",
]
AREA_FEATURES = {"muscle_area_mm2", "Lean_Muscle_Area"}
FEATURE_SET_ORDER = [
    "G0_global",
    "A_target",
    "B_target_adjacent",
    "C_mechanistic_combined",
]


@dataclass
class PilotData:
    tables: dict[str, pd.DataFrame]
    labels: pd.DataFrame
    alignment_audit: pd.DataFrame
    segment_availability: pd.DataFrame
    feature_dictionary: pd.DataFrame
    outliers: pd.DataFrame
    issues: list[dict]


def read_csv_compatible(path: Path, **kwargs) -> pd.DataFrame:
    """Read UTF-8/GBK CSVs without altering the source file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or RuntimeError(f"Unable to read {path}")


def normalise_patient_ids(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    result = result.mask(result.eq(""))
    numeric = pd.to_numeric(result, errors="coerce")
    integer = numeric.notna() & np.isfinite(numeric) & np.equal(numeric % 1, 0)
    result = result.copy()
    result.loc[integer] = numeric.loc[integer].astype("Int64").astype("string")
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _metric_row(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _target_neighbours(target_disc: str) -> tuple[str | None, str | None]:
    index = DISC_LEVELS.index(target_disc)
    upper = DISC_LEVELS[index - 1] if index > 0 else None
    lower = DISC_LEVELS[index + 1] if index < len(DISC_LEVELS) - 1 else None
    return upper, lower


def _wide_level(
    segment_medians: pd.DataFrame,
    patient_levels: pd.DataFrame,
    relation: str,
) -> pd.DataFrame:
    level_column = f"{relation}_level"
    selected = segment_medians.merge(
        patient_levels[["patient_id", level_column]],
        left_on=["patient_id", "anatomical_level"],
        right_on=["patient_id", level_column],
        how="inner",
        validate="many_to_one",
    )
    wide = selected.pivot(
        index="patient_id",
        columns="muscle_name",
        values=CORE_FEATURES,
    )
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [
        f"{relation}__{muscle}__{feature}" for muscle, feature in wide.columns
    ]
    return wide


def _wide_global(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["patient_id", "muscle_name"], sort=False)[CORE_FEATURES].median()
    wide = grouped.unstack("muscle_name").swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"global__{muscle}__{feature}" for muscle, feature in wide.columns]
    return wide


def _add_asymmetry(wide: pd.DataFrame, relation: str) -> pd.DataFrame:
    output = wide.copy()
    epsilon = 1e-8
    for group, (left, right) in MUSCLE_PAIRS.items():
        for feature in ASYMMETRY_FEATURES:
            left_column = f"{relation}__{left}__{feature}"
            right_column = f"{relation}__{right}__{feature}"
            name = f"{relation}__{group}__{feature}__absolute_asymmetry"
            output[name] = (
                2.0 * (output[left_column] - output[right_column]).abs()
                / (output[left_column].abs() + output[right_column].abs() + epsilon)
            )
    return output


def _adjacent_change_features(level_wides: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for muscle in EXPECTED_MUSCLES:
        for feature in CORE_FEATURES:
            target = level_wides["target"][f"target__{muscle}__{feature}"]
            upper = level_wides["upper"][f"upper__{muscle}__{feature}"]
            lower = level_wides["lower"][f"lower__{muscle}__{feature}"]
            stem = f"{muscle}__{feature}"
            columns[f"change__target_minus_upper__{stem}"] = target - upper
            columns[f"change__target_minus_lower__{stem}"] = target - lower
            columns[f"change__upper_to_lower_slope__{stem}"] = (lower - upper) / 2.0
    return pd.DataFrame(columns, index=level_wides["target"].index)


def _bilateral_value(wide: pd.DataFrame, relation: str, group: str, feature: str) -> pd.Series:
    left, right = MUSCLE_PAIRS[group]
    left_value = wide[f"{relation}__{left}__{feature}"]
    right_value = wide[f"{relation}__{right}__{feature}"]
    if feature in AREA_FEATURES:
        return left_value + right_value
    return (left_value + right_value) / 2.0


def _mechanistic_features(
    level_wides: dict[str, pd.DataFrame],
    global_wide: pd.DataFrame,
) -> pd.DataFrame:
    target = level_wides["target"]
    output = pd.DataFrame(index=target.index)
    epsilon = 1e-8

    # Bilateral group summaries at the target disc.
    for group in MUSCLE_PAIRS:
        for feature in CORE_FEATURES:
            output[f"mechanism__target_bilateral__{group}__{feature}"] = _bilateral_value(
                target, "target", group, feature
            )

    # Posterior support and relative muscle-balance measures.
    for feature in ("muscle_area_mm2", "Lean_Muscle_Area"):
        multifidus = _bilateral_value(target, "target", "multifidus", feature)
        erector = _bilateral_value(target, "target", "erector_spinae", feature)
        psoas = _bilateral_value(target, "target", "psoas", feature)
        posterior = multifidus + erector
        output[f"mechanism__target_posterior_sum__{feature}"] = posterior
        output[f"mechanism__target_posterior_to_psoas__{feature}"] = posterior / (
            psoas.abs() + epsilon
        )
        output[f"mechanism__target_multifidus_to_erector__{feature}"] = multifidus / (
            erector.abs() + epsilon
        )

    # Functional-to-total area is available even though vertebral area is not.
    for group in MUSCLE_PAIRS:
        lean = _bilateral_value(target, "target", group, "Lean_Muscle_Area")
        area = _bilateral_value(target, "target", group, "muscle_area_mm2")
        output[f"mechanism__target_functional_fraction__{group}"] = lean / (
            area.abs() + epsilon
        )

    # Deviation of target muscle quality from the same patient's whole-lumbar
    # average. This controls some between-patient intensity/size variability.
    for group in MUSCLE_PAIRS:
        for feature in CORE_FEATURES:
            target_value = _bilateral_value(target, "target", group, feature)
            global_value = _bilateral_value(global_wide, "global", group, feature)
            output[f"mechanism__target_vs_global_relative__{group}__{feature}"] = (
                target_value - global_value
            ) / (global_value.abs() + epsilon)
    return output


def _feature_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for feature_set, table in tables.items():
        for feature in [column for column in table.columns if column not in {"patient_id", "label"}]:
            if "absolute_asymmetry" in feature:
                family = "bilateral_asymmetry"
            elif feature.startswith("change__"):
                family = "adjacent_level_change"
            elif feature.startswith("mechanism__"):
                family = "biomechanical_combination"
            elif feature.startswith("global__"):
                family = "whole_lumbar_baseline"
            elif feature.startswith("target__"):
                family = "target_level"
            elif feature.startswith("upper__") or feature.startswith("lower__"):
                family = "adjacent_level"
            else:
                family = "other"
            records.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "family": family,
                    "aggregation": "median of four annotated slices before derivation",
                    "uses_label_to_construct": False,
                }
            )
    return pd.DataFrame(records)


def _outlier_records(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for feature_set, table in tables.items():
        for feature in [column for column in table.columns if column not in {"patient_id", "label"}]:
            values = pd.to_numeric(table[feature], errors="coerce")
            median = values.median()
            mad = (values - median).abs().median()
            if not np.isfinite(mad) or mad <= 0:
                continue
            robust_z = 0.67448975 * (values - median) / mad
            for index in robust_z.index[robust_z.abs() > 5]:
                records.append(
                    {
                        "feature_set": feature_set,
                        "patient_id": table.at[index, "patient_id"],
                        "feature": feature,
                        "value": values.at[index],
                        "robust_z": robust_z.at[index],
                        "action": "logged_only; train-fold quantile clipping used in modeling",
                    }
                )
    columns = ["feature_set", "patient_id", "feature", "value", "robust_z", "action"]
    return pd.DataFrame(records, columns=columns)


def build_pilot_tables(
    annotation_file: Path,
    label_file: Path,
    feature_file: Path,
) -> PilotData:
    annotations = read_csv_compatible(annotation_file, dtype={"patient_id": "string"})
    labels_raw = read_csv_compatible(label_file, dtype={"patient_id": "string"}, low_memory=False)
    features = read_csv_compatible(feature_file, dtype={"patient_id": "string"}, low_memory=False)

    _require_columns(
        annotations,
        {"patient_id", "slice_index", "anatomical_level"},
        annotation_file.name,
    )
    _require_columns(
        labels_raw,
        {"patient_id", "instability_label", "target_slip_segment"},
        label_file.name,
    )
    _require_columns(
        features,
        {"patient_id", "slice_index", "muscle_name", *CORE_FEATURES},
        feature_file.name,
    )

    for frame in (annotations, labels_raw, features):
        frame["patient_id"] = normalise_patient_ids(frame["patient_id"])
    annotations["slice_index"] = pd.to_numeric(
        annotations["slice_index"], errors="raise"
    ).astype(int)
    features["slice_index"] = pd.to_numeric(features["slice_index"], errors="raise").astype(int)

    if annotations[["patient_id", "slice_index"]].duplicated().any():
        raise ValueError("Annotation table has duplicate patient_id/slice_index rows")
    if features[["patient_id", "slice_index", "muscle_name"]].duplicated().any():
        raise ValueError("2D feature table has duplicate patient_id/slice_index/muscle_name rows")
    invalid_levels = sorted(set(annotations["anatomical_level"].dropna()) - set(DISC_LEVELS))
    if invalid_levels:
        raise ValueError(f"Unknown anatomical_level values: {invalid_levels}")

    annotation_ids = set(annotations["patient_id"].dropna())
    label_numeric = pd.to_numeric(labels_raw["instability_label"], errors="coerce")
    labels = labels_raw.loc[
        labels_raw["patient_id"].isin(annotation_ids) & label_numeric.notna(),
        ["patient_id", "target_slip_segment"],
    ].copy()
    labels["label"] = label_numeric.loc[labels.index].astype(int)
    if labels["patient_id"].duplicated().any():
        duplicates = labels.loc[labels["patient_id"].duplicated(False), "patient_id"].tolist()
        raise ValueError(f"Duplicate pilot patient IDs in label table: {duplicates[:10]}")
    if set(labels["label"]) != {0, 1} or not labels["label"].isin([0, 1]).all():
        raise ValueError("Pilot labels must contain both binary classes 0 and 1")
    if set(labels["patient_id"]) != annotation_ids:
        missing = sorted(annotation_ids - set(labels["patient_id"]))
        raise ValueError(f"Annotated patients without a usable instability label: {missing}")
    invalid_slip = sorted(set(labels["target_slip_segment"]) - set(SLIP_TO_TARGET_DISC))
    if invalid_slip:
        raise ValueError(f"Unknown target_slip_segment values: {invalid_slip}")

    feature_ids = set(features["patient_id"].dropna())
    if feature_ids != annotation_ids:
        raise ValueError(
            "Annotation and 2D feature patient sets differ: "
            f"annotation_only={sorted(annotation_ids - feature_ids)[:10]}, "
            f"feature_only={sorted(feature_ids - annotation_ids)[:10]}"
        )

    annotation_keys = set(map(tuple, annotations[["patient_id", "slice_index"]].to_numpy()))
    feature_keys = set(
        map(tuple, features[["patient_id", "slice_index"]].drop_duplicates().to_numpy())
    )
    if annotation_keys != feature_keys:
        raise ValueError(
            "Annotation and feature slice keys differ: "
            f"annotation_only={len(annotation_keys - feature_keys)}, "
            f"feature_only={len(feature_keys - annotation_keys)}"
        )
    muscles_per_slice = (
        features.groupby(["patient_id", "slice_index"])["muscle_name"]
        .agg(lambda values: tuple(sorted(values)))
    )
    expected_tuple = tuple(sorted(EXPECTED_MUSCLES))
    if not muscles_per_slice.map(lambda value: value == expected_tuple).all():
        raise ValueError("At least one slice does not contain the expected six bilateral muscles")

    labels["target_level"] = labels["target_slip_segment"].map(SLIP_TO_TARGET_DISC)
    neighbours = labels["target_level"].map(_target_neighbours)
    labels["upper_level"] = [pair[0] for pair in neighbours]
    labels["lower_level"] = [pair[1] for pair in neighbours]
    if labels[["upper_level", "lower_level"]].isna().any().any():
        raise ValueError(
            "The current pilot requires both cranial and caudal adjacent disc levels"
        )

    merged = features.merge(
        annotations[["patient_id", "slice_index", "anatomical_level"]],
        on=["patient_id", "slice_index"],
        how="inner",
        validate="many_to_one",
    )
    for feature in CORE_FEATURES:
        merged[feature] = pd.to_numeric(merged[feature], errors="coerce")
    numeric = merged[CORE_FEATURES].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Core 2D features contain infinite values")

    level_counts = (
        annotations.groupby(["patient_id", "anatomical_level"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=DISC_LEVELS, fill_value=0)
    )
    availability = labels.set_index("patient_id")[
        ["label", "target_slip_segment", "target_level", "upper_level", "lower_level"]
    ].join(level_counts)
    for relation in ("target", "upper", "lower"):
        availability[f"{relation}_slice_count"] = [
            int(level_counts.at[patient_id, level])
            for patient_id, level in zip(availability.index, availability[f"{relation}_level"])
        ]
    availability = availability.reset_index()
    if (availability[["target_slice_count", "upper_slice_count", "lower_slice_count"]] <= 0).any().any():
        raise ValueError("At least one patient lacks a target or adjacent segment")

    segment_medians = (
        merged.groupby(["patient_id", "anatomical_level", "muscle_name"], sort=False)[
            CORE_FEATURES
        ]
        .median()
        .reset_index()
    )
    patient_levels = labels[
        ["patient_id", "target_level", "upper_level", "lower_level"]
    ]
    global_raw = _wide_global(merged)
    level_wides = {
        relation: _wide_level(segment_medians, patient_levels, relation)
        for relation in ("target", "upper", "lower")
    }
    global_with_asymmetry = _add_asymmetry(global_raw, "global")
    level_with_asymmetry = {
        relation: _add_asymmetry(wide, relation)
        for relation, wide in level_wides.items()
    }
    changes = _adjacent_change_features(level_wides)
    mechanisms = _mechanistic_features(level_wides, global_raw)

    patient_order = labels["patient_id"].tolist()
    label_table = labels[["patient_id", "label"]].copy()

    def finish(wide: pd.DataFrame) -> pd.DataFrame:
        clean = wide.replace([np.inf, -np.inf], np.nan).reindex(patient_order)
        clean.index.name = "patient_id"
        return (
            clean.reset_index()
            .merge(label_table, on="patient_id", how="left", validate="one_to_one")
        )

    tables = {
        "G0_global": finish(global_with_asymmetry),
        "A_target": finish(level_with_asymmetry["target"]),
        "B_target_adjacent": finish(
            pd.concat(
                [
                    level_with_asymmetry["target"],
                    level_with_asymmetry["upper"],
                    level_with_asymmetry["lower"],
                    changes,
                ],
                axis=1,
            )
        ),
        "C_mechanistic_combined": finish(
            pd.concat(
                [
                    level_with_asymmetry["target"],
                    level_with_asymmetry["upper"],
                    level_with_asymmetry["lower"],
                    changes,
                    mechanisms,
                ],
                axis=1,
            )
        ),
    }

    issues: list[dict] = []
    issues.append(
        {
            "severity": "resolved_bug",
            "stage": "development_self_test",
            "issue": (
                "Pandas 2.3 treated Series.eq(tuple) as a length-aligned comparison "
                "in the six-muscle completeness audit"
            ),
            "action": "Changed to explicit row-wise tuple comparison before model execution",
        }
    )
    label_counts = labels["label"].value_counts().to_dict()
    issues.append(
        {
            "severity": "warning",
            "stage": "cohort",
            "issue": f"Small imbalanced pilot cohort: label_counts={label_counts}",
            "action": "Use repeated stratified 3-fold CV and paired uncertainty; no final clinical claim",
        }
    )
    if labels["target_slip_segment"].nunique() == 1:
        only = labels["target_slip_segment"].iloc[0]
        issues.append(
            {
                "severity": "warning",
                "stage": "external_validity",
                "issue": f"All pilot patients have target_slip_segment={only}",
                "action": "Interpret only as a fixed-level localisation feasibility test",
            }
        )
    if "target_slip_vicesegment" in labels_raw.columns:
        secondary = labels_raw.loc[
            labels_raw["patient_id"].isin(annotation_ids)
            & labels_raw["target_slip_vicesegment"].notna(),
            ["patient_id", "target_slip_vicesegment"],
        ]
        if not secondary.empty:
            issues.append(
                {
                    "severity": "info",
                    "stage": "target_definition",
                    "issue": (
                        f"{len(secondary)} patient(s) have a secondary slip segment; "
                        "primary target only was used"
                    ),
                    "action": "Retain secondary segment for a future pre-specified sensitivity analysis",
                }
            )
    clinical_columns = [
        column
        for column in ("age_years", "height_cm", "weight_kg", "bmi_kg_m2")
        if column in labels_raw.columns
    ]
    pilot_clinical = labels_raw[labels_raw["patient_id"].isin(annotation_ids)]
    missing_clinical = {
        column: int(pd.to_numeric(pilot_clinical[column], errors="coerce").isna().sum())
        for column in clinical_columns
    }
    if missing_clinical and any(value for value in missing_clinical.values()):
        issues.append(
            {
                "severity": "info",
                "stage": "feature_availability",
                "issue": f"Clinical fields not usable in this pilot: {missing_clinical}",
                "action": "No clinical covariates or height-normalised area were constructed",
            }
        )
    issues.append(
        {
            "severity": "info",
            "stage": "feature_availability",
            "issue": "Vertebral-body area is unavailable",
            "action": "Functional muscle area / vertebral area was not constructed",
        }
    )

    audit_rows = [
        {
            "check": "annotated_patients",
            "value": len(annotation_ids),
            "status": "pass",
        },
        {
            "check": "annotated_slices",
            "value": len(annotations),
            "status": "pass",
        },
        {
            "check": "feature_rows",
            "value": len(features),
            "status": "pass",
        },
        {
            "check": "exact_patient_set_match",
            "value": feature_ids == annotation_ids == set(labels["patient_id"]),
            "status": "pass",
        },
        {
            "check": "exact_slice_key_match",
            "value": annotation_keys == feature_keys,
            "status": "pass",
        },
        {
            "check": "missing_core_feature_cells",
            "value": int(merged[CORE_FEATURES].isna().sum().sum()),
            "status": "pass" if not merged[CORE_FEATURES].isna().any().any() else "warning",
        },
        {
            "check": "positive_labels",
            "value": int((labels["label"] == 1).sum()),
            "status": "warning",
        },
        {
            "check": "negative_labels",
            "value": int((labels["label"] == 0).sum()),
            "status": "pass",
        },
    ]
    for feature_set, table in tables.items():
        predictors = table.drop(columns=["patient_id", "label"])
        audit_rows.extend(
            [
                {
                    "check": f"{feature_set}_candidate_features",
                    "value": predictors.shape[1],
                    "status": "pass",
                },
                {
                    "check": f"{feature_set}_missing_cells",
                    "value": int(predictors.isna().sum().sum()),
                    "status": "pass" if not predictors.isna().any().any() else "warning",
                },
                {
                    "check": f"{feature_set}_constant_features",
                    "value": int((predictors.nunique(dropna=True) <= 1).sum()),
                    "status": "pass",
                },
            ]
        )

    return PilotData(
        tables=tables,
        labels=labels,
        alignment_audit=pd.DataFrame(audit_rows),
        segment_availability=availability,
        feature_dictionary=_feature_dictionary(tables),
        outliers=_outlier_records(tables),
        issues=issues,
    )


def _make_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(lower=0.02, upper=0.98)),
            ("variance", VarianceThreshold(threshold=1e-12)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif)),
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


def _selected_features(best_estimator: Pipeline, source_features: list[str]) -> list[str]:
    after_variance = np.asarray(source_features, dtype=object)[
        best_estimator.named_steps["variance"].get_support()
    ]
    return after_variance[best_estimator.named_steps["selector"].get_support()].tolist()


def repeated_nested_cv(
    tables: dict[str, pd.DataFrame],
    repeats: int,
    outer_folds: int,
    inner_folds: int,
    base_seed: int,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    reference = tables[FEATURE_SET_ORDER[0]][["patient_id", "label"]].copy()
    patient_ids = reference["patient_id"].to_numpy()
    y = reference["label"].astype(int).to_numpy()
    if int(np.bincount(y).min()) < outer_folds:
        raise ValueError("Minority class is smaller than outer_folds")

    predictions = []
    repeat_metrics = []
    fold_records = []
    selected_records = []
    runtime_issues: list[dict] = []

    for repeat_index in range(repeats):
        seed = base_seed + repeat_index
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        splits = list(outer.split(np.zeros(len(y)), y))
        if repeat_index == 0 or (repeat_index + 1) % 10 == 0:
            log(f"Repeated nested CV: repeat {repeat_index + 1}/{repeats}, seed={seed}")

        for feature_set in FEATURE_SET_ORDER:
            table = tables[feature_set]
            if not np.array_equal(table["patient_id"].to_numpy(), patient_ids):
                raise ValueError(f"{feature_set} patient order differs from reference")
            X = (
                table.drop(columns=["patient_id", "label"])
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            oof = np.full(len(y), np.nan)
            for fold, (train_index, test_index) in enumerate(splits):
                train_counts = np.bincount(y[train_index], minlength=2)
                if int(train_counts.min()) < inner_folds:
                    raise ValueError(
                        f"Repeat {repeat_index}, fold {fold}: insufficient minority cases "
                        f"for {inner_folds}-fold inner CV"
                    )
                inner = StratifiedKFold(
                    n_splits=inner_folds,
                    shuffle=True,
                    random_state=seed * 100 + fold,
                )
                search = GridSearchCV(
                    _make_pipeline(seed),
                    {
                        "selector__k": [3, 5, 8],
                        "classifier__C": [0.03, 0.1, 0.3, 1.0],
                    },
                    scoring="roc_auc",
                    cv=inner,
                    refit=True,
                    n_jobs=1,
                    error_score="raise",
                )
                started = time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    search.fit(X.iloc[train_index], y[train_index])
                for item in caught:
                    runtime_issues.append(
                        {
                            "severity": "warning",
                            "stage": "model_fit",
                            "repeat_index": repeat_index,
                            "seed": seed,
                            "outer_fold": fold,
                            "feature_set": feature_set,
                            "category": item.category.__name__,
                            "issue": str(item.message),
                            "action": "recorded; inspect before interpretation",
                        }
                    )
                probability = search.predict_proba(X.iloc[test_index])[:, 1]
                oof[test_index] = probability
                names = _selected_features(search.best_estimator_, list(X.columns))
                coefficients = search.best_estimator_.named_steps["classifier"].coef_[0]
                for name, coefficient in zip(names, coefficients):
                    selected_records.append(
                        {
                            "repeat_index": repeat_index,
                            "seed": seed,
                            "outer_fold": fold,
                            "feature_set": feature_set,
                            "feature": name,
                            "coefficient_after_scaling": float(coefficient),
                        }
                    )
                fold_records.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "outer_fold": fold,
                        "feature_set": feature_set,
                        "train_n": len(train_index),
                        "test_n": len(test_index),
                        "train_positive": int(y[train_index].sum()),
                        "test_positive": int(y[test_index].sum()),
                        "inner_best_auc": float(search.best_score_),
                        "selected_k": int(search.best_params_["selector__k"]),
                        "selected_C": float(search.best_params_["classifier__C"]),
                        "runtime_seconds": time.time() - started,
                    }
                )
                for row_index, probability_value in zip(test_index, probability):
                    predictions.append(
                        {
                            "repeat_index": repeat_index,
                            "seed": seed,
                            "outer_fold": fold,
                            "feature_set": feature_set,
                            "patient_id": patient_ids[row_index],
                            "true_label": int(y[row_index]),
                            "predicted_probability": float(probability_value),
                        }
                    )
            if np.isnan(oof).any():
                raise RuntimeError(f"{feature_set}, repeat {repeat_index}: incomplete OOF predictions")
            repeat_metrics.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "feature_set": feature_set,
                    **_metric_row(y, oof),
                }
            )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(repeat_metrics),
        pd.DataFrame(fold_records),
        pd.DataFrame(selected_records),
        runtime_issues,
    )


def aggregate_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["feature_set", "patient_id", "true_label"], as_index=False)
        .agg(
            mean_oof_probability=("predicted_probability", "mean"),
            oof_probability_sd=("predicted_probability", "std"),
            oof_prediction_count=("predicted_probability", "size"),
        )
    )


def _stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    return np.concatenate(
        [
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ]
    )


def aggregate_performance(
    aggregate_predictions: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for feature_set in FEATURE_SET_ORDER:
        frame = (
            aggregate_predictions[aggregate_predictions["feature_set"] == feature_set]
            .sort_values("patient_id")
        )
        y = frame["true_label"].to_numpy(int)
        probability = frame["mean_oof_probability"].to_numpy(float)
        point = _metric_row(y, probability)
        boot = {metric: [] for metric in ("roc_auc", "pr_auc", "brier")}
        for _ in range(bootstrap_iterations):
            index = _stratified_bootstrap_indices(y, rng)
            metrics = _metric_row(y[index], probability[index])
            for metric in boot:
                boot[metric].append(metrics[metric])
        rows.append(
            {
                "feature_set": feature_set,
                **point,
                **{
                    f"{metric}_bootstrap_ci_low": float(np.quantile(values, 0.025))
                    for metric, values in boot.items()
                },
                **{
                    f"{metric}_bootstrap_ci_high": float(np.quantile(values, 0.975))
                    for metric, values in boot.items()
                },
            }
        )
    return pd.DataFrame(rows)


def repeat_performance_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for feature_set in FEATURE_SET_ORDER:
        frame = repeat_metrics[repeat_metrics["feature_set"] == feature_set]
        for metric in ("roc_auc", "pr_auc", "brier", "balanced_accuracy"):
            values = frame[metric].to_numpy(float)
            records.append(
                {
                    "feature_set": feature_set,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "empirical_2_5_percentile": float(np.quantile(values, 0.025)),
                    "empirical_97_5_percentile": float(np.quantile(values, 0.975)),
                    "repeats": len(values),
                }
            )
    return pd.DataFrame(records)


def paired_comparisons(
    repeat_metrics: pd.DataFrame,
    aggregate_predictions: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = [
        ("G0_global", "A_target"),
        ("G0_global", "B_target_adjacent"),
        ("G0_global", "C_mechanistic_combined"),
        ("A_target", "C_mechanistic_combined"),
    ]
    repeat_rows = []
    for reference, comparison in comparisons:
        left = repeat_metrics[repeat_metrics["feature_set"] == reference].set_index(
            "repeat_index"
        )
        right = repeat_metrics[repeat_metrics["feature_set"] == comparison].set_index(
            "repeat_index"
        )
        for metric in ("roc_auc", "pr_auc", "brier"):
            # Positive always means the comparison is better.
            difference = (
                left[metric] - right[metric]
                if metric == "brier"
                else right[metric] - left[metric]
            )
            repeat_rows.append(
                {
                    "reference": reference,
                    "comparison": comparison,
                    "metric": metric,
                    "mean_improvement": float(difference.mean()),
                    "sd": float(difference.std(ddof=1)),
                    "median_improvement": float(difference.median()),
                    "empirical_2_5_percentile": float(difference.quantile(0.025)),
                    "empirical_97_5_percentile": float(difference.quantile(0.975)),
                    "fraction_repeats_improved": float((difference > 0).mean()),
                    "repeats": len(difference),
                    "interval_note": "empirical split variability; not an independent-sample CI",
                }
            )

    prediction_lookup = {
        feature_set: (
            aggregate_predictions[
                aggregate_predictions["feature_set"] == feature_set
            ]
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        for feature_set in FEATURE_SET_ORDER
    }
    bootstrap_rows = []
    rng = np.random.default_rng(seed + 77)
    for reference, comparison in comparisons:
        left = prediction_lookup[reference]
        right = prediction_lookup[comparison]
        if not left[["patient_id", "true_label"]].equals(
            right[["patient_id", "true_label"]]
        ):
            raise ValueError(f"Prediction cohorts differ: {reference} vs {comparison}")
        y = left["true_label"].to_numpy(int)
        p_left = left["mean_oof_probability"].to_numpy(float)
        p_right = right["mean_oof_probability"].to_numpy(float)
        point = roc_auc_score(y, p_right) - roc_auc_score(y, p_left)
        differences = []
        for _ in range(bootstrap_iterations):
            index = _stratified_bootstrap_indices(y, rng)
            differences.append(
                roc_auc_score(y[index], p_right[index])
                - roc_auc_score(y[index], p_left[index])
            )
        differences = np.asarray(differences)
        probability_gt_zero = float(np.mean(differences > 0))
        bootstrap_rows.append(
            {
                "reference": reference,
                "comparison": comparison,
                "metric": "roc_auc",
                "point_auc_improvement": float(point),
                "bootstrap_ci_low": float(np.quantile(differences, 0.025)),
                "bootstrap_ci_high": float(np.quantile(differences, 0.975)),
                "bootstrap_probability_improvement_gt_0": probability_gt_zero,
                "two_sided_bootstrap_tail_probability": float(
                    min(1.0, 2.0 * min(probability_gt_zero, 1.0 - probability_gt_zero))
                ),
                "bootstrap_iterations": bootstrap_iterations,
            }
        )
    return pd.DataFrame(repeat_rows), pd.DataFrame(bootstrap_rows)


def feature_stability(
    selected_records: pd.DataFrame,
    repeats: int,
    outer_folds: int,
) -> pd.DataFrame:
    denominator = repeats * outer_folds
    return (
        selected_records.groupby(["feature_set", "feature"], as_index=False)
        .agg(
            selected_outer_fits=("feature", "size"),
            mean_scaled_coefficient=("coefficient_after_scaling", "mean"),
            mean_abs_scaled_coefficient=(
                "coefficient_after_scaling",
                lambda values: float(np.mean(np.abs(values))),
            ),
            positive_coefficient_fraction=(
                "coefficient_after_scaling",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
        )
        .assign(selection_frequency=lambda frame: frame["selected_outer_fits"] / denominator)
        .sort_values(
            ["feature_set", "selection_frequency", "mean_abs_scaled_coefficient"],
            ascending=[True, False, False],
        )
    )


def save_results(
    data: PilotData,
    output_dir: Path,
    predictions: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    fold_records: pd.DataFrame,
    selected_records: pd.DataFrame,
    runtime_issues: list[dict],
    repeats: int,
    outer_folds: int,
    inner_folds: int,
    bootstrap_iterations: int,
    base_seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_set, table in data.tables.items():
        table.to_csv(
            output_dir / f"patient_features_{feature_set}_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )
    data.alignment_audit.to_csv(
        output_dir / "data_alignment_audit.csv", index=False, encoding="utf-8-sig"
    )
    data.segment_availability.to_csv(
        output_dir / "patient_segment_availability.csv", index=False, encoding="utf-8-sig"
    )
    data.feature_dictionary.to_csv(
        output_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig"
    )
    data.outliers.to_csv(
        output_dir / "outlier_records.csv", index=False, encoding="utf-8-sig"
    )

    predictions.to_csv(
        output_dir / "oof_predictions_all_repeats.csv", index=False, encoding="utf-8-sig"
    )
    repeat_metrics.to_csv(
        output_dir / "performance_by_repeat.csv", index=False, encoding="utf-8-sig"
    )
    fold_records.to_csv(
        output_dir / "nested_cv_fold_details.csv", index=False, encoding="utf-8-sig"
    )
    selected_records.to_csv(
        output_dir / "selected_features_by_outer_fit.csv", index=False, encoding="utf-8-sig"
    )

    aggregate_predictions = aggregate_oof_predictions(predictions)
    aggregate_predictions.to_csv(
        output_dir / "mean_oof_predictions_by_patient.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate_perf = aggregate_performance(
        aggregate_predictions, bootstrap_iterations, base_seed + 10000
    )
    aggregate_perf.to_csv(
        output_dir / "aggregate_oof_performance.csv", index=False, encoding="utf-8-sig"
    )
    repeat_summary = repeat_performance_summary(repeat_metrics)
    repeat_summary.to_csv(
        output_dir / "repeat_performance_summary.csv", index=False, encoding="utf-8-sig"
    )
    paired_repeat, paired_bootstrap = paired_comparisons(
        repeat_metrics,
        aggregate_predictions,
        bootstrap_iterations,
        base_seed + 20000,
    )
    paired_repeat.to_csv(
        output_dir / "paired_improvement_by_repeat.csv", index=False, encoding="utf-8-sig"
    )
    paired_bootstrap.to_csv(
        output_dir / "paired_auc_patient_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability = feature_stability(selected_records, repeats, outer_folds)
    stability.to_csv(
        output_dir / "feature_selection_stability.csv", index=False, encoding="utf-8-sig"
    )

    all_issues = data.issues + runtime_issues
    pd.DataFrame(all_issues).to_csv(
        output_dir / "warnings_and_bug_records.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "warnings_and_bug_records.json").write_text(
        json.dumps(all_issues, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary = paired_bootstrap[
        (paired_bootstrap["reference"] == "G0_global")
        & (paired_bootstrap["comparison"] == "C_mechanistic_combined")
    ].iloc[0]
    direct_localisation = paired_bootstrap[
        (paired_bootstrap["reference"] == "G0_global")
        & (paired_bootstrap["comparison"] == "A_target")
    ].iloc[0]
    if primary["bootstrap_ci_low"] > 0:
        conclusion = (
            "The pre-specified combined segment model showed a positive paired AUC "
            "difference whose patient-bootstrap interval stayed above zero. This is "
            "preliminary evidence only and requires a larger, multi-level cohort."
        )
    elif primary["point_auc_improvement"] > 0:
        conclusion = (
            "The pre-specified combined segment model had a higher point AUC than the "
            "global baseline, but the paired interval crossed zero. The pilot is "
            "directionally encouraging but statistically inconclusive."
        )
    else:
        conclusion = (
            "The pre-specified combined segment model did not improve point AUC over "
            "the matched global baseline. This pilot provides no evidence of segment "
            "localisation benefit under the current feature definitions."
        )
    if direct_localisation["bootstrap_ci_low"] > 0:
        direct_conclusion = (
            "The target-only model improved over the global baseline with a paired "
            "interval above zero."
        )
    elif direct_localisation["point_auc_improvement"] > 0:
        direct_conclusion = (
            "The target-only model was the strongest segment formulation and improved "
            "the point AUC over the global baseline, but its paired interval still "
            "crossed zero."
        )
    else:
        direct_conclusion = (
            "The target-only model did not improve point AUC over the global baseline."
        )

    environment = {
        "python_executable": sys.executable,
        "python": sys.version,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    config = {
        "experiment_version": "v6_segment_pilot_test30",
        "feature_sets": FEATURE_SET_ORDER,
        "core_features": CORE_FEATURES,
        "slip_to_target_disc": SLIP_TO_TARGET_DISC,
        "repeats": repeats,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "base_seed": base_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "classifier": "L2 logistic regression with balanced class weights",
        "selector_k_grid": [3, 5, 8],
        "classifier_C_grid": [0.03, 0.1, 0.3, 1.0],
        "environment": environment,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    performance_lines = []
    for row in aggregate_perf.itertuples(index=False):
        performance_lines.append(
            f"| {row.feature_set} | {row.roc_auc:.3f} "
            f"({row.roc_auc_bootstrap_ci_low:.3f}-{row.roc_auc_bootstrap_ci_high:.3f}) "
            f"| {row.pr_auc:.3f} | {row.brier:.3f} |"
        )
    comparison_lines = []
    for row in paired_bootstrap.itertuples(index=False):
        comparison_lines.append(
            f"| {row.reference} | {row.comparison} | "
            f"{row.point_auc_improvement:+.3f} "
            f"({row.bootstrap_ci_low:+.3f} to {row.bootstrap_ci_high:+.3f}) |"
        )
    summary_markdown = f"""# v6 segment-localisation pilot (30 patients)

## Design

- Cohort: {len(data.labels)} patients; label counts =
  {data.labels['label'].value_counts().sort_index().to_dict()}.
- Target definition: slipped vertebra mapped to the caudal disc
  (for example L4 -> L4/L5).
- Validation: {repeats} repeated stratified {outer_folds}-fold outer CV with
  {inner_folds}-fold inner tuning; identical splits for all feature sets.
- Model: balanced L2 logistic regression. Imputation, clipping, scaling,
  top-k selection and C tuning were fitted inside training folds only.
- The 95% intervals below are stratified patient-bootstrap intervals on each
  patient's mean repeated OOF probability.

## Aggregate repeated-OOF performance

| Feature set | ROC-AUC (95% bootstrap interval) | PR-AUC | Brier |
|---|---:|---:|---:|
{chr(10).join(performance_lines)}

## Paired AUC improvement

Positive values favour the comparison model.

| Reference | Comparison | AUC improvement (95% bootstrap interval) |
|---|---|---:|
{chr(10).join(comparison_lines)}

## Pre-specified primary interpretation

{conclusion}

## Direct target-localisation comparison

{direct_conclusion} This comparison is more directly related to the localisation
hypothesis, while the larger B/C candidate pools test whether adjacent gradients
and additional mechanism combinations add further predictive value.

## Important limitations

- Only 7 patients have instability_label=1.
- All 30 patients have primary target_slip_segment=L4; performance at other
  slipped levels is untested.
- Age, height, weight and BMI were unavailable for these 30 patients.
- Vertebral-body area was unavailable, so height- and vertebral-area-normalised
  muscle measures were not calculated.
- Repeated splits reduce dependence on one lucky split but do not create more
  independent patients. This is a feasibility signal, not a clinical model.
"""
    (output_dir / "RESULTS_SUMMARY.md").write_text(summary_markdown, encoding="utf-8")

    return {
        "conclusion": conclusion,
        "direct_localisation_conclusion": direct_conclusion,
        "aggregate_performance": aggregate_perf.to_dict(orient="records"),
        "paired_auc": paired_bootstrap.to_dict(orient="records"),
        "issues": all_issues,
    }


def run_segment_pilot(
    annotation_file: Path,
    label_file: Path,
    feature_file: Path,
    output_dir: Path,
    repeats: int = 100,
    outer_folds: int = 3,
    inner_folds: int = 2,
    bootstrap_iterations: int = 5000,
    base_seed: int = 20260725,
    log: Callable[[str], None] = print,
) -> dict:
    started = time.time()
    log(f"Build segment-localised tables from {feature_file}")
    data = build_pilot_tables(annotation_file, label_file, feature_file)
    log(
        "Aligned cohort: "
        f"patients={len(data.labels)}, "
        f"labels={data.labels['label'].value_counts().sort_index().to_dict()}, "
        f"feature_counts={{"
        + ", ".join(
            f"{name}: {table.shape[1] - 2}" for name, table in data.tables.items()
        )
        + "}"
    )
    predictions, repeat_metrics, fold_records, selected_records, runtime_issues = (
        repeated_nested_cv(
            data.tables,
            repeats=repeats,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            base_seed=base_seed,
            log=log,
        )
    )
    result = save_results(
        data=data,
        output_dir=output_dir,
        predictions=predictions,
        repeat_metrics=repeat_metrics,
        fold_records=fold_records,
        selected_records=selected_records,
        runtime_issues=runtime_issues,
        repeats=repeats,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        bootstrap_iterations=bootstrap_iterations,
        base_seed=base_seed,
    )
    result["runtime_seconds"] = time.time() - started
    log(f"Completed segment pilot in {result['runtime_seconds']:.1f} seconds")
    return result
