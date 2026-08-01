"""v8 nested feature-subset discovery for lumbar instability prediction.

This is an exploratory feature-discovery experiment, not a confirmatory model.
All outcome-guided screening and subset search occur inside outer training
folds. Outer-fold predictions are used only for performance estimation.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment.preprocessing import QuantileClipper
from experiment.segment_pilot import (
    DISC_LEVELS,
    MUSCLE_PAIRS,
    SLIP_TO_TARGET_DISC,
    _metric_row,
    _stratified_bootstrap_indices,
    normalise_patient_ids,
    read_csv_compatible,
)
from experiment.segment_validation_219 import build_compact_tables


EXPECTED_MUSCLES = sorted(muscle for pair in MUSCLE_PAIRS.values() for muscle in pair)
MODEL_ORDER = [
    "E0_locked22_baseline",
    "E10_best_subset",
    "E15_best_subset",
    "E20_best_subset",
    "E15_PCA5",
    "N15_random_subset",
]

FEATURE_2D_FAMILIES = {
    # Shape / quantity
    "muscle_area_mm2": "quantity_shape",
    "Perimeter": "quantity_shape",
    "Equivalent_Diameter": "quantity_shape",
    "Aspect_Ratio": "quantity_shape",
    "Eccentricity": "quantity_shape",
    "Max_Transverse_Diameter": "quantity_shape",
    "Max_AP_Diameter": "quantity_shape",
    "Circularity": "quantity_shape",
    "Solidity": "quantity_shape",
    "Max_Inscribed_Circle_Diameter": "quantity_shape",
    "Shape_Complexity": "quantity_shape",
    # Intensity
    "Mean_Intensity_Muscle": "intensity_distribution",
    "Std_Intensity_Muscle": "intensity_distribution",
    "Median_Intensity_Muscle": "intensity_distribution",
    "IQR_Intensity_Muscle": "intensity_distribution",
    "Skewness_Intensity_Muscle": "intensity_distribution",
    "Kurtosis_Intensity_Muscle": "intensity_distribution",
    "Mean_Intensity_Lean_Muscle": "intensity_distribution",
    # Fat / functional muscle quality
    "Fat_Area": "fat_infiltration",
    "FIP": "fat_infiltration",
    "Lean_Muscle_Area": "fat_infiltration",
    "Fat_to_Lean_Ratio": "fat_infiltration",
    "Radial_FIP_Ring1": "fat_infiltration",
    "Radial_FIP_Ring2": "fat_infiltration",
    "Radial_FIP_Ring3": "fat_infiltration",
    "Fascial_Fat_Ratio": "fat_infiltration",
    "Fat_Entropy": "fat_infiltration",
    "Fat_Centroid_Offset": "fat_infiltration",
    "Fat_Clustering_Index": "fat_infiltration",
    # Corrected/retained texture families
    "Texture_FirstOrder_Entropy": "texture",
    "Texture_FirstOrder_Skewness": "texture",
    "Texture_FirstOrder_Kurtosis": "texture",
    "Texture_GLCM_Contrast": "texture",
    "Texture_GLCM_Correlation": "texture",
    "Texture_GLCM_Id": "texture",
    "Texture_GLCM_Idm": "texture",
    "Texture_GLDM_DependenceEntropy": "texture",
    "Texture_GLDM_DependenceNonUniformity": "texture",
    "Texture_GLDM_GrayLevelNonUniformity": "texture",
}
AREA_LIKE_2D = {"muscle_area_mm2", "Fat_Area", "Lean_Muscle_Area"}
ASYMMETRY_2D = {
    "muscle_area_mm2",
    "Lean_Muscle_Area",
    "Fat_Area",
    "FIP",
    "Mean_Intensity_Muscle",
    "Std_Intensity_Muscle",
    "IQR_Intensity_Muscle",
    "Texture_FirstOrder_Entropy",
    "Texture_GLCM_Contrast",
    "Texture_GLCM_Idm",
}

FEATURE_3D_FAMILIES = {
    "3D_Volume": "three_d_quantity",
    "3D_Func_Volume": "three_d_quantity",
    "3D_FIP": "three_d_fat",
    "Mean_Area": "three_d_quantity",
    "Max_Area": "three_d_quantity",
    "Min_Area": "three_d_quantity",
    "Std_Area": "three_d_quantity",
    "Mean_Func_CSA": "three_d_quantity",
    "Max_Func_CSA": "three_d_quantity",
    "Min_Func_CSA": "three_d_quantity",
    "Mean_FIP": "three_d_fat",
    "Max_FIP": "three_d_fat",
    "Min_FIP": "three_d_fat",
    "Std_FIP": "three_d_fat",
    "CV_Area_Z": "three_d_longitudinal",
    "CV_FIP_Z": "three_d_longitudinal",
    "SA_V": "three_d_shape",
    "3D_Shape_Index": "three_d_shape",
}
AREA_LIKE_3D = {
    "3D_Volume",
    "3D_Func_Volume",
    "Mean_Area",
    "Max_Area",
    "Min_Area",
    "Std_Area",
    "Mean_Func_CSA",
    "Max_Func_CSA",
    "Min_Func_CSA",
}
ASYMMETRY_3D = {
    "3D_Volume",
    "3D_Func_Volume",
    "3D_FIP",
    "Mean_Area",
    "Mean_Func_CSA",
    "Mean_FIP",
}
FEATURE_CROSS = {
    "FIP_Slope": "three_d_longitudinal",
    "Area_Z_Gradient": "three_d_longitudinal",
    "Func_Area_Z_Gradient": "three_d_longitudinal",
    "Centroid_Z_Drift": "three_d_longitudinal",
    "Shape_Z_Deformation": "three_d_longitudinal",
}

SOURCE_EXCLUSIONS = {
    "Area": "exact alias of muscle_area_mm2",
    "Func_CSA": "exact alias of Lean_Muscle_Area",
    "Convex_Hull_Area": "redundant geometric intermediate",
    "Min_BBox_Orientation": "orientation depends on image coordinate convention",
    "Mean_Intensity_Fat": "fat-threshold-dependent technical intensity",
    "Deep_Fat_Ratio": "formula retained as suspect in earlier audit",
    "Texture_GLRLM_ShortRunEmphasis": "exact duplicate of GLSZM SmallAreaEmphasis",
    "Texture_GLRLM_LongRunEmphasis": "exact duplicate of GLSZM LargeAreaEmphasis",
    "Texture_GLRLM_RunLengthNonUniformity": "exact duplicate of GLSZM SizeZoneNonUniformity",
    "Texture_GLSZM_SmallAreaEmphasis": "paired suspect duplicate",
    "Texture_GLSZM_LargeAreaEmphasis": "paired suspect duplicate",
    "Texture_GLSZM_SizeZoneNonUniformity": "paired suspect duplicate",
    "pixel_spacing_x": "technical acquisition field",
    "pixel_spacing_y": "technical acquisition field",
    "slice_thickness": "technical acquisition field",
    "Total_CSA": "ambiguous extraction/QC field",
    "fat_threshold_used": "algorithm/QC field",
    "fat_threshold_decision": "algorithm/QC field",
    "slice_index": "technical index",
    "Peak_Area_Slice_Index": "non-comparable technical index",
    "Peak_FIP_Slice_Index": "non-comparable technical index",
    "csf_value": "normalization/QC field",
    "muscle_name": "identifier",
    "patient_id": "identifier",
}


def _target_levels(labels: pd.DataFrame) -> pd.DataFrame:
    output = labels.copy()
    output["target_level"] = output["target_slip_segment"].map(SLIP_TO_TARGET_DISC)

    def neighbour(level: str, offset: int):
        if level not in DISC_LEVELS:
            return pd.NA
        index = DISC_LEVELS.index(level) + offset
        return DISC_LEVELS[index] if 0 <= index < len(DISC_LEVELS) else pd.NA

    output["cranial_level"] = output["target_level"].map(lambda value: neighbour(value, -1))
    output["caudal_level"] = output["target_level"].map(lambda value: neighbour(value, 1))
    return output


def _relation_muscle_wide(
    medians: pd.DataFrame,
    levels: pd.DataFrame,
    relation: str,
    patient_order: list[str],
    base_features: list[str],
) -> pd.DataFrame:
    level_column = f"{relation}_level"
    selected = medians.merge(
        levels[["patient_id", level_column]],
        left_on=["patient_id", "anatomical_level"],
        right_on=["patient_id", level_column],
        how="inner",
        validate="many_to_one",
    )
    wide = selected.pivot(index="patient_id", columns="muscle_name", values=base_features)
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature}" for muscle, feature in wide.columns]
    expected = [
        f"{muscle}__{feature}" for muscle in EXPECTED_MUSCLES for feature in base_features
    ]
    return wide.reindex(index=patient_order, columns=expected)


def _global_muscle_wide(
    frame: pd.DataFrame, patient_order: list[str], base_features: list[str]
) -> pd.DataFrame:
    grouped = frame.groupby(["patient_id", "muscle_name"])[base_features].median()
    wide = grouped.unstack("muscle_name").swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature}" for muscle, feature in wide.columns]
    expected = [
        f"{muscle}__{feature}" for muscle in EXPECTED_MUSCLES for feature in base_features
    ]
    return wide.reindex(index=patient_order, columns=expected)


def _bilateral(
    wide: pd.DataFrame, group: str, feature: str, operation: str
) -> pd.Series:
    left, right = MUSCLE_PAIRS[group]
    left_value = wide[f"{left}__{feature}"]
    right_value = wide[f"{right}__{feature}"]
    present = left_value.notna() & right_value.notna()
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
        raise ValueError(operation)
    return result.where(present)


def _add_candidate(
    columns: dict[str, pd.Series],
    dictionary: list[dict],
    name: str,
    values: pd.Series,
    family: str,
    source: str,
    muscle_group: str,
    base_feature: str,
    relation: str,
    aggregation: str,
) -> None:
    columns[name] = values
    dictionary.append(
        {
            "feature": name,
            "family": family,
            "source": source,
            "muscle_group": muscle_group,
            "base_feature": base_feature,
            "relation": relation,
            "aggregation": aggregation,
        }
    )


def _wide_precomputed(
    frame: pd.DataFrame, base_features: list[str], patient_order: list[str]
) -> pd.DataFrame:
    pivot = frame.pivot(index="patient_id", columns="muscle_name", values=base_features)
    pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    pivot.columns = [f"{muscle}__{feature}" for muscle, feature in pivot.columns]
    expected = [
        f"{muscle}__{feature}" for muscle in EXPECTED_MUSCLES for feature in base_features
    ]
    return pivot.reindex(index=patient_order, columns=expected)


def build_feature_universe(
    annotation_file: Path,
    label_file: Path,
    feature_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    annotations = read_csv_compatible(annotation_file, dtype={"patient_id": "string"})
    labels_raw = read_csv_compatible(label_file, dtype={"patient_id": "string"}, low_memory=False)
    two_d = read_csv_compatible(
        feature_dir / "muscle_features_2d_v7.csv",
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    for frame in (annotations, labels_raw, two_d):
        frame["patient_id"] = normalise_patient_ids(frame["patient_id"])
    annotations["slice_index"] = pd.to_numeric(annotations["slice_index"], errors="raise").astype(int)
    two_d["slice_index"] = pd.to_numeric(two_d["slice_index"], errors="raise").astype(int)
    annotation_ids = set(annotations["patient_id"])
    numeric_label = pd.to_numeric(labels_raw["instability_label"], errors="coerce")
    labels = labels_raw.loc[
        labels_raw["patient_id"].isin(annotation_ids) & numeric_label.isin([0, 1]),
        ["patient_id", "target_slip_segment"],
    ].copy()
    labels["label"] = numeric_label.loc[labels.index].astype(int)
    if len(labels) != 219 or labels["patient_id"].duplicated().any():
        raise ValueError("Expected 219 unique annotated/labeled patients")
    labels = _target_levels(labels)
    patient_order = labels["patient_id"].tolist()
    levels = labels[["patient_id", "target_level", "cranial_level", "caudal_level"]]

    two_d = two_d[two_d["patient_id"].isin(annotation_ids)].copy()
    two_d = two_d.merge(
        annotations[["patient_id", "slice_index", "anatomical_level"]],
        on=["patient_id", "slice_index"],
        how="inner",
        validate="many_to_one",
    )
    base_2d = list(FEATURE_2D_FAMILIES)
    for feature in base_2d:
        two_d[feature] = pd.to_numeric(two_d[feature], errors="coerce")
    segment_medians = (
        two_d.groupby(["patient_id", "anatomical_level", "muscle_name"])[base_2d]
        .median()
        .reset_index()
    )
    relation_wides = {
        "global": _global_muscle_wide(two_d, patient_order, base_2d),
        "target": _relation_muscle_wide(
            segment_medians, levels, "target", patient_order, base_2d
        ),
        "cranial": _relation_muscle_wide(
            segment_medians, levels, "cranial", patient_order, base_2d
        ),
        "caudal": _relation_muscle_wide(
            segment_medians, levels, "caudal", patient_order, base_2d
        ),
    }
    columns: dict[str, pd.Series] = {}
    dictionary: list[dict] = []
    for relation in ("global", "target"):
        for group in MUSCLE_PAIRS:
            for feature, family in FEATURE_2D_FAMILIES.items():
                operation = "sum" if feature in AREA_LIKE_2D else "mean"
                name = f"2d__{relation}__{group}__{feature}__{operation}"
                _add_candidate(
                    columns,
                    dictionary,
                    name,
                    _bilateral(relation_wides[relation], group, feature, operation),
                    family,
                    "muscle_features_2d_v7.csv",
                    group,
                    feature,
                    relation,
                    operation,
                )
    for neighbour in ("cranial", "caudal"):
        for group in MUSCLE_PAIRS:
            for feature, family in FEATURE_2D_FAMILIES.items():
                operation = "sum" if feature in AREA_LIKE_2D else "mean"
                values = _bilateral(
                    relation_wides["target"], group, feature, operation
                ) - _bilateral(relation_wides[neighbour], group, feature, operation)
                name = f"2d__target_minus_{neighbour}__{group}__{feature}__{operation}"
                _add_candidate(
                    columns,
                    dictionary,
                    name,
                    values,
                    "segment_gradient",
                    "muscle_features_2d_v7.csv",
                    group,
                    feature,
                    f"target_minus_{neighbour}",
                    operation,
                )
    for relation in ("global", "target"):
        for group in MUSCLE_PAIRS:
            for feature in ASYMMETRY_2D:
                name = f"2d__{relation}__{group}__{feature}__asymmetry"
                _add_candidate(
                    columns,
                    dictionary,
                    name,
                    _bilateral(relation_wides[relation], group, feature, "asymmetry"),
                    "bilateral_asymmetry",
                    "muscle_features_2d_v7.csv",
                    group,
                    feature,
                    relation,
                    "absolute_asymmetry",
                )

    # 3D and cross-layer features.
    three_d = read_csv_compatible(
        feature_dir / "muscle_features_3d_v7.csv",
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    cross = read_csv_compatible(
        feature_dir / "muscle_features_level3_cross_v7.csv",
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    for frame in (three_d, cross):
        frame["patient_id"] = normalise_patient_ids(frame["patient_id"])
        frame.drop(frame.index[~frame["patient_id"].isin(annotation_ids)], inplace=True)
    for feature in FEATURE_3D_FAMILIES:
        three_d[feature] = pd.to_numeric(three_d[feature], errors="coerce")
    three_wide = _wide_precomputed(three_d, list(FEATURE_3D_FAMILIES), patient_order)
    for group in MUSCLE_PAIRS:
        for feature, family in FEATURE_3D_FAMILIES.items():
            operation = "sum" if feature in AREA_LIKE_3D else "mean"
            name = f"3d__global__{group}__{feature}__{operation}"
            _add_candidate(
                columns,
                dictionary,
                name,
                _bilateral(three_wide, group, feature, operation),
                family,
                "muscle_features_3d_v7.csv",
                group,
                feature,
                "global_3d",
                operation,
            )
            if feature in ASYMMETRY_3D:
                asymmetry_name = f"3d__global__{group}__{feature}__asymmetry"
                _add_candidate(
                    columns,
                    dictionary,
                    asymmetry_name,
                    _bilateral(three_wide, group, feature, "asymmetry"),
                    "bilateral_asymmetry",
                    "muscle_features_3d_v7.csv",
                    group,
                    feature,
                    "global_3d",
                    "absolute_asymmetry",
                )
    for feature in FEATURE_CROSS:
        cross[feature] = pd.to_numeric(cross[feature], errors="coerce")
    cross_wide = _wide_precomputed(cross, list(FEATURE_CROSS), patient_order)
    for group in MUSCLE_PAIRS:
        for feature, family in FEATURE_CROSS.items():
            name = f"3d_cross__{group}__{feature}__mean"
            _add_candidate(
                columns,
                dictionary,
                name,
                _bilateral(cross_wide, group, feature, "mean"),
                family,
                "muscle_features_level3_cross_v7.csv",
                group,
                feature,
                "global_3d",
                "bilateral_mean",
            )

    # Precomputed multi-muscle features with stable transforms for raw ratios.
    multi = read_csv_compatible(
        feature_dir / "muscle_features_level3_multi_v7.csv",
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    multi["patient_id"] = normalise_patient_ids(multi["patient_id"])
    multi = multi[multi["patient_id"].isin(annotation_ids)].set_index("patient_id").reindex(
        patient_order
    )
    for feature in [column for column in multi.columns if column != "csf_value"]:
        values = pd.to_numeric(multi[feature], errors="coerce")
        if "Ratio" in feature or feature.startswith(("Rat_", "Psoas_", "ES_", "MF_")):
            transformed = np.sign(values) * np.log1p(np.abs(values))
            aggregation = "signed_log1p"
        elif feature.startswith("Symmetry_Index"):
            transformed = 2.0 * np.abs(values - 1.0) / (np.abs(values) + 1.0 + 1e-8)
            aggregation = "bounded_deviation_from_1"
        else:
            transformed = values
            aggregation = "precomputed"
        name = f"multi__{feature}__{aggregation}"
        _add_candidate(
            columns,
            dictionary,
            name,
            transformed,
            "intermuscle_biomechanics",
            "muscle_features_level3_multi_v7.csv",
            "multi_muscle",
            feature,
            "global_3d",
            aggregation,
        )

    universe = pd.DataFrame(columns, index=patient_order)
    universe.index.name = "patient_id"
    universe = universe.replace([np.inf, -np.inf], np.nan)
    feature_dictionary = pd.DataFrame(dictionary)
    exclusions = [
        {
            "stage": "source_formula_exclusion",
            "feature": feature,
            "reason": reason,
            "action": "exclude_before_modeling",
        }
        for feature, reason in SOURCE_EXCLUSIONS.items()
    ]

    # Label-independent full-cohort hard exclusions: constants, severe missingness,
    # and exact duplicates. Correlation filtering remains inside outer folds.
    drop_features: set[str] = set()
    for feature in universe.columns:
        values = universe[feature]
        if values.notna().mean() < 0.80:
            drop_features.add(feature)
            exclusions.append(
                {
                    "stage": "derived_quality_exclusion",
                    "feature": feature,
                    "reason": f"missing_fraction={values.isna().mean():.4f} > 0.20",
                    "action": "exclude_before_modeling",
                }
            )
        elif values.nunique(dropna=True) <= 1:
            drop_features.add(feature)
            exclusions.append(
                {
                    "stage": "derived_quality_exclusion",
                    "feature": feature,
                    "reason": "constant or all-missing feature",
                    "action": "exclude_before_modeling",
                }
            )
    kept = [feature for feature in universe.columns if feature not in drop_features]
    hashes: dict[int, list[str]] = defaultdict(list)
    for feature in kept:
        hashes[int(pd.util.hash_pandas_object(universe[feature], index=False).sum())].append(
            feature
        )
    for group in hashes.values():
        if len(group) < 2:
            continue
        representative = group[0]
        for candidate in group[1:]:
            if universe[representative].equals(universe[candidate]):
                drop_features.add(candidate)
                exclusions.append(
                    {
                        "stage": "derived_quality_exclusion",
                        "feature": candidate,
                        "reason": f"exact duplicate of {representative}",
                        "action": "exclude_before_modeling",
                    }
                )
    universe = universe.drop(columns=sorted(drop_features))
    feature_dictionary = feature_dictionary[
        feature_dictionary["feature"].isin(universe.columns)
    ].reset_index(drop=True)
    table = universe.reset_index().merge(
        labels[["patient_id", "label"]], on="patient_id", validate="one_to_one"
    )
    audit = pd.DataFrame(
        [
            {
                "patients": len(table),
                "label_0": int((table["label"] == 0).sum()),
                "label_1": int((table["label"] == 1).sum()),
                "constructed_features": len(columns),
                "hard_excluded_derived": len(drop_features),
                "candidate_features": universe.shape[1],
                "missing_cells": int(universe.isna().sum().sum()),
                "patients_with_any_missing": int(universe.isna().any(axis=1).sum()),
                "infinite_cells": 0,
            }
        ]
    )
    issues = [
        {
            "severity": "warning",
            "stage": "study_design",
            "issue": "All 219 labels have been used in earlier exploratory analyses",
            "action": "Treat v8 as exploratory feature discovery; require future independent confirmation",
        },
        {
            "severity": "warning",
            "stage": "annotation",
            "issue": "219-patient anatomical levels are protocol-inferred from patient 77",
            "action": "Do not interpret selected segment features as individually verified localisation",
        },
    ]
    return table, feature_dictionary, pd.DataFrame(exclusions), audit, issues


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _point_biserial_scores(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = values - values.mean(axis=0)
    centered_y = y - y.mean()
    numerator = centered_x.T @ centered_y
    denominator = np.sqrt(
        np.sum(centered_x**2, axis=0) * np.sum(centered_y**2)
    )
    return np.divide(
        np.abs(numerator),
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _correlation_representatives(
    X: pd.DataFrame, y: np.ndarray, threshold: float
) -> tuple[list[str], pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(X)
    names = np.asarray(X.columns, dtype=object)
    variances = np.nanvar(values, axis=0)
    keep = variances > 1e-12
    values, names = values[:, keep], names[keep]
    correlations = (
        pd.DataFrame(values).corr(method="spearman").abs().fillna(0.0).to_numpy()
    )
    union = _UnionFind(len(names))
    rows, columns = np.where(np.triu(correlations, 1) > threshold)
    for left, right in zip(rows, columns):
        union.union(int(left), int(right))
    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(names)):
        clusters[union.find(index)].append(index)
    scores = _point_biserial_scores(values, y.astype(float))
    representatives, records = [], []
    for cluster_id, indices in enumerate(clusters.values()):
        representative_index = max(indices, key=lambda index: (scores[index], -index))
        representative = str(names[representative_index])
        representatives.append(representative)
        for index in indices:
            records.append(
                {
                    "cluster_id": cluster_id,
                    "feature": str(names[index]),
                    "representative": representative,
                    "cluster_size": len(indices),
                    "train_abs_point_biserial": float(scores[index]),
                }
            )
    return representatives, pd.DataFrame(records)


def _stability_screen(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    corr_threshold: float,
    subsamples: int,
    top_per_subsample: int,
    pool_size: int,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    representatives, clusters = _correlation_representatives(X, y, corr_threshold)
    reduced = X[representatives]
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(reduced)
    scaler = StandardScaler()
    values = scaler.fit_transform(values)
    splitter = StratifiedShuffleSplit(
        n_splits=subsamples, train_size=0.75, random_state=seed
    )
    counts = np.zeros(len(representatives), dtype=int)
    coefficient_sum = np.zeros(len(representatives), dtype=float)
    sign_sum = np.zeros(len(representatives), dtype=float)
    for sample_index, _ in splitter.split(values, y):
        classifier = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
        classifier.fit(values[sample_index], y[sample_index])
        coefficients = classifier.coef_[0]
        ranking = np.argsort(np.abs(coefficients), kind="stable")[
            -min(top_per_subsample, len(coefficients)) :
        ]
        counts[ranking] += 1
        coefficient_sum[ranking] += coefficients[ranking]
        sign_sum[ranking] += np.sign(coefficients[ranking])
    stability = pd.DataFrame(
        {
            "feature": representatives,
            "stability_frequency": counts / subsamples,
            "mean_selected_coefficient": np.divide(
                coefficient_sum,
                counts,
                out=np.zeros_like(coefficient_sum),
                where=counts > 0,
            ),
            "sign_consistency_proxy": np.divide(
                np.abs(sign_sum),
                counts,
                out=np.zeros_like(sign_sum),
                where=counts > 0,
            ),
        }
    ).sort_values(
        ["stability_frequency", "sign_consistency_proxy", "feature"],
        ascending=[False, False, True],
    )
    pool = stability.head(min(pool_size, len(stability)))["feature"].tolist()
    return pool, stability, clusters


def _generate_subsets(
    pool: list[str],
    family_map: dict[str, str],
    size: int,
    count: int,
    seed: int,
    weights: dict[str, float],
) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    probabilities = np.asarray([max(weights.get(feature, 0.01), 0.01) for feature in pool])
    probabilities = probabilities / probabilities.sum()
    subsets: set[tuple[str, ...]] = set()
    attempts = 0
    max_per_family = max(3, int(np.ceil(size * 0.40)))
    while len(subsets) < count and attempts < count * 500:
        attempts += 1
        chosen = rng.choice(pool, size=size, replace=False, p=probabilities)
        families = [family_map.get(str(feature), "unknown") for feature in chosen]
        family_counts = pd.Series(families).value_counts()
        if len(family_counts) < min(4, size) or family_counts.max() > max_per_family:
            continue
        subsets.add(tuple(sorted(map(str, chosen))))
    if len(subsets) < count:
        while len(subsets) < count and attempts < count * 1000:
            attempts += 1
            chosen = tuple(sorted(map(str, rng.choice(pool, size=size, replace=False))))
            subsets.add(chosen)
    if len(subsets) < count:
        raise RuntimeError(f"Could generate only {len(subsets)}/{count} subsets of size {size}")
    return sorted(subsets)


def _subset_pipeline() -> Pipeline:
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
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                ),
            ),
        ]
    )


def _score_subsets(
    X: pd.DataFrame,
    y: np.ndarray,
    subsets: list[tuple[str, ...]],
    cv: StratifiedKFold,
) -> pd.DataFrame:
    records = []
    for subset in subsets:
        scores = cross_val_score(
            _subset_pipeline(),
            X[list(subset)],
            y,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        records.append(
            {
                "subset_size": len(subset),
                "features_json": json.dumps(list(subset), ensure_ascii=False),
                "inner_mean_auc": float(scores.mean()),
                "inner_auc_sd": float(scores.std(ddof=1)),
                "inner_auc_se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["inner_mean_auc", "inner_auc_sd"], ascending=[False, True]
    )


def _final_pipeline(seed: int, pca_components: int | None = None) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(lower=0.01, upper=0.99)),
        ("scaler", StandardScaler()),
    ]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=seed)))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=10000,
                tol=1e-3,
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def _fit_final(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    seed: int,
    inner_folds: int,
    pca_components: int | None = None,
) -> tuple[np.ndarray, GridSearchCV, list[warnings.WarningMessage]]:
    inner = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed + 777
    )
    search = GridSearchCV(
        _final_pipeline(seed, pca_components),
        {
            "classifier__C": [0.03, 0.1, 0.3, 1.0],
            "classifier__l1_ratio": [0.25, 0.5, 0.75],
        },
        scoring="roc_auc",
        cv=inner,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(X_train, y_train)
    return search.predict_proba(X_test)[:, 1], search, caught


def nested_subset_tournament(
    candidate_table: pd.DataFrame,
    baseline_table: pd.DataFrame,
    dictionary: pd.DataFrame,
    repeats: int,
    outer_folds: int,
    inner_folds: int,
    candidates_per_size: int,
    stability_subsamples: int,
    base_seed: int,
    log: Callable[[str], None],
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    candidate_table = candidate_table.copy()
    candidate_table["patient_id"] = normalise_patient_ids(candidate_table["patient_id"])
    baseline_table = baseline_table.copy()
    baseline_table["patient_id"] = normalise_patient_ids(baseline_table["patient_id"])
    baseline_table = baseline_table.set_index("patient_id").loc[
        candidate_table["patient_id"]
    ].reset_index()
    patient_ids = candidate_table["patient_id"].to_numpy()
    y = candidate_table["label"].astype(int).to_numpy()
    X = candidate_table.drop(columns=["patient_id", "label"]).apply(
        pd.to_numeric, errors="coerce"
    )
    X_baseline = baseline_table.drop(columns=["patient_id", "label"]).apply(
        pd.to_numeric, errors="coerce"
    )
    family_map = dictionary.set_index("feature")["family"].to_dict()

    predictions, repeat_metrics, folds = [], [], []
    selected, coefficients, candidate_scores = [], [], []
    stability_records, cluster_records = [], []
    issues: list[dict] = []
    for repeat_index in range(repeats):
        seed = base_seed + repeat_index
        splitter = StratifiedKFold(
            n_splits=outer_folds, shuffle=True, random_state=seed
        )
        split_list = list(splitter.split(X, y))
        log(f"v8 tournament repeat {repeat_index + 1}/{repeats}, seed={seed}")
        repeat_oof = {
            model: np.full(len(y), np.nan, dtype=float) for model in MODEL_ORDER
        }
        for outer_fold, (train, test) in enumerate(split_list):
            fold_seed = seed * 100 + outer_fold
            pool, stability, clusters = _stability_screen(
                X.iloc[train],
                y[train],
                fold_seed,
                corr_threshold=0.90,
                subsamples=stability_subsamples,
                top_per_subsample=20,
                pool_size=40,
            )
            stability.insert(0, "outer_fold", outer_fold)
            stability.insert(0, "repeat_index", repeat_index)
            stability_records.append(stability)
            clusters.insert(0, "outer_fold", outer_fold)
            clusters.insert(0, "repeat_index", repeat_index)
            cluster_records.append(clusters)
            weights = stability.set_index("feature")["stability_frequency"].to_dict()
            inner = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=fold_seed + 31,
            )
            best_by_size: dict[int, list[str]] = {}
            all_fold_candidates = []
            for size in (10, 15, 20):
                subsets = _generate_subsets(
                    pool,
                    family_map,
                    size,
                    candidates_per_size,
                    fold_seed + size,
                    weights,
                )
                scores = _score_subsets(X.iloc[train], y[train], subsets, inner)
                scores.insert(0, "outer_fold", outer_fold)
                scores.insert(0, "repeat_index", repeat_index)
                scores["rank_within_size"] = np.arange(1, len(scores) + 1)
                candidate_scores.append(scores)
                all_fold_candidates.append(scores)
                best_by_size[size] = json.loads(scores.iloc[0]["features_json"])

            random15 = list(
                _generate_subsets(
                    pool,
                    family_map,
                    15,
                    1,
                    fold_seed + 9999,
                    {feature: 1.0 for feature in pool},
                )[0]
            )
            model_features = {
                "E0_locked22_baseline": list(X_baseline.columns),
                "E10_best_subset": best_by_size[10],
                "E15_best_subset": best_by_size[15],
                "E20_best_subset": best_by_size[20],
                "E15_PCA5": best_by_size[15],
                "N15_random_subset": random15,
            }
            for model_name, features in model_features.items():
                source = X_baseline if model_name == "E0_locked22_baseline" else X
                pca_components = 5 if model_name == "E15_PCA5" else None
                probability, search, caught = _fit_final(
                    source.iloc[train][features],
                    y[train],
                    source.iloc[test][features],
                    fold_seed,
                    inner_folds,
                    pca_components,
                )
                repeat_oof[model_name][test] = probability
                for item in caught:
                    issues.append(
                        {
                            "severity": "warning",
                            "stage": "final_model_fit",
                            "repeat_index": repeat_index,
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "issue": str(item.message),
                            "action": "recorded for review",
                        }
                    )
                folds.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "train_n": len(train),
                        "test_n": len(test),
                        "train_positive": int(y[train].sum()),
                        "test_positive": int(y[test].sum()),
                        "feature_count": len(features),
                        "inner_best_auc": float(search.best_score_),
                        "best_C": float(search.best_params_["classifier__C"]),
                        "best_l1_ratio": float(
                            search.best_params_["classifier__l1_ratio"]
                        ),
                    }
                )
                for feature in features:
                    selected.append(
                        {
                            "repeat_index": repeat_index,
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "feature": feature,
                        }
                    )
                if pca_components is None:
                    model_coefficients = search.best_estimator_.named_steps[
                        "classifier"
                    ].coef_[0]
                    for feature, coefficient in zip(features, model_coefficients):
                        coefficients.append(
                            {
                                "repeat_index": repeat_index,
                                "outer_fold": outer_fold,
                                "model": model_name,
                                "feature": feature,
                                "scaled_coefficient": float(coefficient),
                            }
                        )
                for local, row_index in enumerate(test):
                    predictions.append(
                        {
                            "repeat_index": repeat_index,
                            "seed": seed,
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "patient_id": patient_ids[row_index],
                            "true_label": int(y[row_index]),
                            "predicted_probability": float(probability[local]),
                        }
                    )
        for model_name, probability in repeat_oof.items():
            if np.isnan(probability).any():
                raise RuntimeError(f"Incomplete OOF predictions: {repeat_index}/{model_name}")
            repeat_metrics.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "model": model_name,
                    **_metric_row(y, probability),
                }
            )
    return (
        {
            "predictions": pd.DataFrame(predictions),
            "repeat_metrics": pd.DataFrame(repeat_metrics),
            "folds": pd.DataFrame(folds),
            "selected": pd.DataFrame(selected),
            "coefficients": pd.DataFrame(coefficients),
            "candidate_scores": pd.concat(candidate_scores, ignore_index=True),
            "stability": pd.concat(stability_records, ignore_index=True),
            "clusters": pd.concat(cluster_records, ignore_index=True),
        },
        issues,
    )


def _aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["model", "patient_id", "true_label"], as_index=False)
        .agg(
            mean_oof_probability=("predicted_probability", "mean"),
            oof_probability_sd=("predicted_probability", "std"),
            prediction_count=("predicted_probability", "size"),
        )
    )


def performance_summaries(
    predictions: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = _aggregate_predictions(predictions)
    rng = np.random.default_rng(seed)
    performance_rows = []
    for model in MODEL_ORDER:
        frame = aggregate[aggregate["model"] == model].sort_values("patient_id")
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
                "model": model,
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
    comparisons = [
        ("E0_locked22_baseline", "E10_best_subset"),
        ("E0_locked22_baseline", "E15_best_subset"),
        ("E0_locked22_baseline", "E20_best_subset"),
        ("E15_best_subset", "E15_PCA5"),
        ("N15_random_subset", "E15_best_subset"),
    ]
    lookup = {
        model: aggregate[aggregate["model"] == model]
        .sort_values("patient_id")
        .reset_index(drop=True)
        for model in MODEL_ORDER
    }
    paired_rows = []
    for reference, comparison in comparisons:
        left, right = lookup[reference], lookup[comparison]
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
                "reference": reference,
                "comparison": comparison,
                "auc_improvement": float(
                    roc_auc_score(y, p1) - roc_auc_score(y, p0)
                ),
                "ci_low": float(np.quantile(differences, 0.025)),
                "ci_high": float(np.quantile(differences, 0.975)),
                "bootstrap_probability_gt_0": probability_positive,
                "two_sided_bootstrap_tail_probability": float(
                    min(1.0, 2 * min(probability_positive, 1 - probability_positive))
                ),
            }
        )
    repeat_rows = []
    for model in MODEL_ORDER:
        frame = repeat_metrics[repeat_metrics["model"] == model]
        for metric in ("roc_auc", "pr_auc", "brier"):
            values = frame[metric]
            repeat_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "empirical_2_5_percentile": float(values.quantile(0.025)),
                    "empirical_97_5_percentile": float(values.quantile(0.975)),
                }
            )
    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(repeat_rows),
    )


def feature_evidence(
    outputs: dict[str, pd.DataFrame],
    dictionary: pd.DataFrame,
    repeats: int,
    outer_folds: int,
) -> pd.DataFrame:
    denominator = repeats * outer_folds
    selected = outputs["selected"]
    best15 = selected[selected["model"] == "E15_best_subset"]
    selection = (
        best15.groupby("feature").size().rename("selected_outer_fits").to_frame()
    )
    coefficient_frame = outputs["coefficients"]
    coefficient_frame = coefficient_frame[
        coefficient_frame["model"] == "E15_best_subset"
    ]
    coefficient = coefficient_frame.groupby("feature").agg(
        mean_scaled_coefficient=("scaled_coefficient", "mean"),
        mean_abs_scaled_coefficient=(
            "scaled_coefficient", lambda values: float(np.mean(np.abs(values)))
        ),
        positive_coefficient_fraction=(
            "scaled_coefficient", lambda values: float(np.mean(np.asarray(values) > 0))
        ),
    )
    coefficient["sign_consistency"] = np.maximum(
        coefficient["positive_coefficient_fraction"],
        1 - coefficient["positive_coefficient_fraction"],
    )
    candidates = outputs["candidate_scores"].copy()
    candidates["feature"] = candidates["features_json"].map(json.loads)
    long = candidates.explode("feature")
    all_mean = candidates.groupby(["repeat_index", "outer_fold"])[
        "inner_mean_auc"
    ].mean()
    included = long.groupby(["repeat_index", "outer_fold", "feature"])[
        "inner_mean_auc"
    ].mean()
    lift_records = []
    for (repeat_index, outer_fold, feature), value in included.items():
        lift_records.append(
            {
                "feature": feature,
                "repeat_index": repeat_index,
                "outer_fold": outer_fold,
                "inner_auc_lift_vs_fold_candidate_mean": float(
                    value - all_mean.loc[(repeat_index, outer_fold)]
                ),
            }
        )
    lift = pd.DataFrame(lift_records).groupby("feature").agg(
        mean_inner_auc_lift=("inner_auc_lift_vs_fold_candidate_mean", "mean"),
        positive_lift_fraction=(
            "inner_auc_lift_vs_fold_candidate_mean",
            lambda values: float(np.mean(np.asarray(values) > 0)),
        ),
    )
    top = candidates[
        candidates.groupby(["repeat_index", "outer_fold"])["inner_mean_auc"].transform(
            lambda values: values >= values.quantile(0.90)
        )
    ].copy()
    top["feature"] = top["features_json"].map(json.loads)
    top_long = top.explode("feature")
    top_frequency = (
        top_long.groupby("feature")
        .size()
        .div(top.groupby(["repeat_index", "outer_fold"]).size().sum())
        .rename("top_decile_subset_frequency")
    )
    evidence = (
        dictionary.set_index("feature")
        .join(selection)
        .join(coefficient)
        .join(lift)
        .join(top_frequency)
        .reset_index()
    )
    evidence["selected_outer_fits"] = evidence["selected_outer_fits"].fillna(0).astype(int)
    evidence["best15_selection_frequency"] = evidence["selected_outer_fits"] / denominator
    for column in (
        "mean_scaled_coefficient",
        "mean_abs_scaled_coefficient",
        "positive_coefficient_fraction",
        "sign_consistency",
        "mean_inner_auc_lift",
        "positive_lift_fraction",
        "top_decile_subset_frequency",
    ):
        evidence[column] = evidence[column].fillna(0.0)

    def grade(row) -> tuple[str, str]:
        if (
            row.best15_selection_frequency >= 0.60
            and row.sign_consistency >= 0.80
            and row.positive_lift_fraction >= 0.60
            and row.mean_inner_auc_lift > 0
        ):
            return "A_retain", "stable selection, direction and positive subset lift"
        if (
            row.best15_selection_frequency >= 0.30
            and row.sign_consistency >= 0.70
            and row.mean_inner_auc_lift > 0
        ):
            return "B_candidate", "moderate selection stability with consistent direction"
        if (
            row.best15_selection_frequency < 0.10
            and row.mean_inner_auc_lift <= 0
            and row.top_decile_subset_frequency < 0.01
        ):
            return (
                "D_exclude_low_evidence",
                "rarely selected and no positive conditional subset evidence",
            )
        return "C_defer", "insufficient or unstable evidence; retain only as correlated proxy"

    grades = evidence.apply(grade, axis=1, result_type="expand")
    evidence["recommendation"] = grades[0]
    evidence["recommendation_reason"] = grades[1]
    return evidence.sort_values(
        [
            "recommendation",
            "best15_selection_frequency",
            "mean_inner_auc_lift",
        ],
        ascending=[True, False, False],
    )


def reduced_permutation_control(
    candidate_table: pd.DataFrame,
    dictionary: pd.DataFrame,
    permutations: int,
    base_seed: int,
    log: Callable[[str], None],
) -> pd.DataFrame:
    X = candidate_table.drop(columns=["patient_id", "label"]).apply(
        pd.to_numeric, errors="coerce"
    )
    original_y = candidate_table["label"].astype(int).to_numpy()
    family_map = dictionary.set_index("feature")["family"].to_dict()
    rng = np.random.default_rng(base_seed)
    records = []
    for permutation_index in range(permutations):
        y = rng.permutation(original_y)
        splitter = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=base_seed + permutation_index
        )
        oof = np.full(len(y), np.nan)
        for outer_fold, (train, test) in enumerate(splitter.split(X, y)):
            fold_seed = base_seed + permutation_index * 100 + outer_fold
            pool, stability, _ = _stability_screen(
                X.iloc[train],
                y[train],
                fold_seed,
                corr_threshold=0.90,
                subsamples=10,
                top_per_subsample=20,
                pool_size=40,
            )
            weights = stability.set_index("feature")["stability_frequency"].to_dict()
            subsets = _generate_subsets(
                pool,
                family_map,
                15,
                30,
                fold_seed + 15,
                weights,
            )
            inner = StratifiedKFold(
                n_splits=4, shuffle=True, random_state=fold_seed + 31
            )
            scores = _score_subsets(X.iloc[train], y[train], subsets, inner)
            features = json.loads(scores.iloc[0]["features_json"])
            model = _subset_pipeline()
            model.fit(X.iloc[train][features], y[train])
            oof[test] = model.predict_proba(X.iloc[test][features])[:, 1]
        records.append(
            {
                "permutation_index": permutation_index,
                "permuted_oof_auc": float(roc_auc_score(y, oof)),
                "permuted_pr_auc": float(average_precision_score(y, oof)),
            }
        )
        if permutation_index == 0 or (permutation_index + 1) % 5 == 0:
            log(f"Reduced full-search permutation {permutation_index + 1}/{permutations}")
    return pd.DataFrame(records)


def run_feature_discovery(
    annotation_file: Path,
    label_file: Path,
    feature_dir: Path,
    pilot_annotation_file: Path,
    output_dir: Path,
    repeats: int = 10,
    outer_folds: int = 5,
    inner_folds: int = 4,
    candidates_per_size: int = 40,
    stability_subsamples: int = 20,
    bootstrap_iterations: int = 3000,
    permutation_iterations: int = 20,
    base_seed: int = 20260729,
    log: Callable[[str], None] = print,
) -> dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Build label-independent 2D/3D/gradient candidate universe")
    candidate_table, dictionary, exclusions, audit, build_issues = build_feature_universe(
        annotation_file, label_file, feature_dir
    )
    pilot_ids = set(
        normalise_patient_ids(
            read_csv_compatible(pilot_annotation_file, dtype={"patient_id": "string"})[
                "patient_id"
            ]
        )
    )
    baseline_tables, _, _, missing_rows, baseline_issues = build_compact_tables(
        annotation_file,
        label_file,
        feature_dir / "muscle_features_2d_v7.csv",
        pilot_ids,
    )
    baseline = baseline_tables["B_target_plus_6_gradients"]
    log(
        f"Candidate universe: patients={len(candidate_table)}, "
        f"features={candidate_table.shape[1] - 2}"
    )
    outputs, runtime_issues = nested_subset_tournament(
        candidate_table,
        baseline,
        dictionary,
        repeats,
        outer_folds,
        inner_folds,
        candidates_per_size,
        stability_subsamples,
        base_seed,
        log,
    )
    log("Summarize repeated OOF performance and paired uncertainty")
    performance, paired, repeat_summary = performance_summaries(
        outputs["predictions"],
        outputs["repeat_metrics"],
        bootstrap_iterations,
        base_seed + 5000,
    )
    evidence = feature_evidence(outputs, dictionary, repeats, outer_folds)
    log("Run reduced full-search label-permutation negative control")
    permutations = reduced_permutation_control(
        candidate_table,
        dictionary,
        permutation_iterations,
        base_seed + 10000,
        log,
    )
    best15_auc = float(
        performance.loc[
            performance["model"] == "E15_best_subset", "roc_auc"
        ].iloc[0]
    )
    empirical_p = float(
        (1 + (permutations["permuted_oof_auc"] >= best15_auc).sum())
        / (1 + len(permutations))
    )
    baseline_auc = float(
        performance.loc[
            performance["model"] == "E0_locked22_baseline", "roc_auc"
        ].iloc[0]
    )
    search_passed_validation_gate = bool(
        best15_auc > baseline_auc and empirical_p <= 0.05
    )
    evidence["inner_screen_grade"] = evidence["recommendation"]
    evidence["predictive_validation_gate"] = (
        "passed" if search_passed_validation_gate else "failed"
    )
    if search_passed_validation_gate:
        evidence["final_recommendation"] = evidence["inner_screen_grade"].map(
            {
                "A_retain": "retain_for_next_predictive_stage",
                "B_candidate": "exploratory_candidate",
                "C_defer": "defer_correlated_or_unstable",
                "D_exclude_low_evidence": "exclude_low_evidence",
            }
        )
    else:
        evidence["final_recommendation"] = evidence["inner_screen_grade"].map(
            {
                "A_retain": "exploratory_inner_signal_not_predictively_validated",
                "B_candidate": "exploratory_inner_signal_not_predictively_validated",
                "C_defer": "defer_correlated_or_unstable",
                "D_exclude_low_evidence": "exclude_low_evidence",
            }
        )

    candidate_table.to_csv(
        output_dir / "patient_feature_universe_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dictionary.to_csv(
        output_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig"
    )
    exclusions.to_csv(
        output_dir / "hard_exclusion_report.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(
        output_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig"
    )
    missing_rows.to_csv(
        output_dir / "source_missing_slice_muscle_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    _aggregate_predictions(outputs["predictions"]).to_csv(
        output_dir / "mean_oof_predictions_by_patient.csv",
        index=False,
        encoding="utf-8-sig",
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
    evidence.to_csv(
        output_dir / "feature_evidence_and_recommendations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evidence[
        evidence["final_recommendation"] == "retain_for_next_predictive_stage"
    ].to_csv(
        output_dir / "features_retain_for_next_stage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evidence[
        evidence["final_recommendation"]
        == "exploratory_inner_signal_not_predictively_validated"
    ].to_csv(
        output_dir / "features_exploratory_inner_signals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evidence[
        evidence["final_recommendation"] == "exclude_low_evidence"
    ].to_csv(
        output_dir / "features_exclude_low_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    permutations.to_csv(
        output_dir / "label_permutation_control.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_issues = build_issues + baseline_issues + runtime_issues
    pd.DataFrame(all_issues).to_csv(
        output_dir / "warnings_and_bug_records.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "warnings_and_bug_records.json").write_text(
        json.dumps(all_issues, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "experiment_version": "v8_nested_feature_discovery",
        "purpose": "exploratory feature discovery; not confirmatory clinical model",
        "repeats": repeats,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "candidates_per_size": candidates_per_size,
        "subset_sizes": [10, 15, 20],
        "stability_subsamples": stability_subsamples,
        "correlation_cluster_threshold": 0.90,
        "stability_pool_size": 40,
        "final_model": "ElasticNet logistic regression",
        "bootstrap_iterations": bootstrap_iterations,
        "permutation_iterations": permutation_iterations,
        "permutation_note": "reduced one-repeat full-search negative control",
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

    perf_lines = [
        (
            f"| {row.model} | {row.roc_auc:.3f} "
            f"({row.roc_auc_ci_low:.3f}-{row.roc_auc_ci_high:.3f}) | "
            f"{row.pr_auc:.3f} | {row.brier:.3f} |"
        )
        for row in performance.itertuples(index=False)
    ]
    grade_counts = evidence["final_recommendation"].value_counts().to_dict()
    top_features = evidence[
        evidence["inner_screen_grade"].isin(["A_retain", "B_candidate"])
    ].head(20)
    top_lines = [
        (
            f"| {row.feature} | {row.family} | {row.final_recommendation} | "
            f"{row.best15_selection_frequency:.2f} | {row.sign_consistency:.2f} | "
            f"{row.mean_inner_auc_lift:+.4f} |"
        )
        for row in top_features.itertuples(index=False)
    ]
    summary = f"""# v8 nested feature discovery

## Scope

This is an exploratory feature-discovery analysis. All 219 labels were used in
earlier project stages, so these results require future independent validation.

## Repeated nested-CV performance

| Model | ROC-AUC (95% patient-bootstrap interval) | PR-AUC | Brier |
|---|---:|---:|---:|
{chr(10).join(perf_lines)}

## Feature evidence

Recommendation counts: {grade_counts}

| Feature | Family | Recommendation | Best15 selection frequency | Sign consistency | Mean inner AUC lift |
|---|---|---|---:|---:|---:|
{chr(10).join(top_lines)}

## Negative control

- Best15 repeated-OOF AUC: {best15_auc:.4f}
- Reduced full-search permutation median AUC:
  {permutations['permuted_oof_auc'].median():.4f}
- Maximum permuted AUC: {permutations['permuted_oof_auc'].max():.4f}
- Empirical permutation tail probability: {empirical_p:.4f}
- Predictive validation gate:
  {'passed' if search_passed_validation_gate else 'failed'}

Because the selected Best15 model
{'outperformed' if best15_auc > baseline_auc else 'did not outperform'} the
locked baseline and the permutation tail probability was {empirical_p:.4f},
inner-screen A/B features are
{'eligible for predictive retention' if search_passed_validation_gate else 'reported only as exploratory inner signals; no feature is retained as predictively validated'}.

The permutation procedure uses one 5-fold repeat and 30 candidate subsets per
fold, so it is a reduced search-capacity control rather than an exact p-value for
the larger repeated experiment.
"""
    (output_dir / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    log(f"Completed v8 feature discovery in {time.time() - started:.1f} seconds")
    return {
        "performance": performance.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "grade_counts": grade_counts,
        "best15_auc": best15_auc,
        "permutation_empirical_p": empirical_p,
        "runtime_seconds": time.time() - started,
    }
