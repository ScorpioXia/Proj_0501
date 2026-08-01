"""Build leakage-free, unstandardized patient-level feature tables.

This module performs only deterministic, label-independent data cleaning and
aggregation.  No imputation, scaling, correlation filtering, or supervised
feature selection is applied here; those operations belong inside CV folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MUSCLE_PAIRS = {
    "multifidus": ("multifidus_left", "multifidus_right"),
    "erector_spinae": ("erector_spinae_left", "erector_spinae_right"),
    "psoas": ("psoas_left", "psoas_right"),
}

CORE_2D_FEATURES = [
    "Area",
    "Max_AP_Diameter",
    "Max_Transverse_Diameter",
    "Solidity",
    "Max_Inscribed_Circle_Diameter",
    "Lean_Muscle_Area",
    "FIP",
    "Std_Intensity_Muscle",
    "Skewness_Intensity_Muscle",
    "Kurtosis_Intensity_Muscle",
    "Deep_Fat_Ratio",
    "Radial_FIP_Ring1",
    "Fat_Entropy",
    "Fat_Centroid_Offset",
]

ASYMMETRY_2D_FEATURES = [
    "Area",
    "Lean_Muscle_Area",
    "FIP",
    "Deep_Fat_Ratio",
    "Fat_Centroid_Offset",
]

KNOWN_EXCLUSIONS = {
    "2d": {
        "Texture_GLCM_Contrast": "constant in all 40,146 retained rows",
        "muscle_area_mm2": "exact duplicate of Area",
        "Func_CSA": "exact duplicate of Lean_Muscle_Area",
        "Texture_GLCM_Correlation": "exactly duplicates other GLCM fields; mapping is suspect",
        "Texture_GLCM_Id": "exactly duplicates other GLCM fields; mapping is suspect",
        "Texture_GLCM_Idm": "exactly duplicates other GLCM fields; mapping is suspect",
        "Texture_GLRLM_ShortRunEmphasis": "exact duplicate of GLSZM SmallAreaEmphasis",
        "Texture_GLRLM_LongRunEmphasis": "exact duplicate of GLSZM LargeAreaEmphasis",
        "Texture_GLRLM_RunLengthNonUniformity": "exact duplicate of GLSZM SizeZoneNonUniformity",
        "Texture_GLSZM_SmallAreaEmphasis": "duplicate/suspect matrix texture",
        "Texture_GLSZM_LargeAreaEmphasis": "duplicate/suspect matrix texture",
        "Texture_GLSZM_SizeZoneNonUniformity": "duplicate/suspect matrix texture",
        "pixel_spacing_x": "technical acquisition field",
        "pixel_spacing_y": "technical acquisition field",
        "fat_threshold_used": "algorithm/QC field, not a biological predictor",
        "fat_threshold_decision": "algorithm/QC field, not a biological predictor",
        "slice_index": "non-comparable absolute slice index",
        "Total_CSA": "ambiguous extraction/QC field",
    },
    "3d": {
        "SA_V": "100% missing",
        "3D_Shape_Index": "100% missing",
        "pixel_spacing_x": "technical acquisition field",
        "pixel_spacing_y": "technical acquisition field",
    },
}


@dataclass
class BuildResult:
    tables: dict[str, pd.DataFrame]
    audit: pd.DataFrame
    outliers: pd.DataFrame
    feature_dictionary: pd.DataFrame
    bug_records: list[dict]


def _normalise_ids(values: pd.Series) -> pd.Series:
    s = values.astype("string").str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    finite = numeric.dropna()
    is_int = finite == finite.astype("Int64")
    mask = pd.Series(False, index=s.index, dtype=bool)
    mask.loc[finite.index] = is_int.values
    s = s.copy()
    s[mask] = numeric[mask].astype("Int64").astype("string")
    return s


def _validate_cohort(df: pd.DataFrame, target_ids: list[str], name: str) -> None:
    ids = set(_normalise_ids(df["patient_id"]).dropna())
    target = set(target_ids)
    if ids != target:
        raise ValueError(
            f"{name}: cohort mismatch; missing={len(target - ids)}, extra={len(ids - target)}"
        )


def _flatten_pivot(df: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    wide = df.pivot(index="patient_id", columns="muscle_name", values=list(value_columns))
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature}" for muscle, feature in wide.columns]
    return wide


def _aggregate_2d(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    missing = sorted(set(CORE_2D_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"2D source is missing required fields: {missing}")

    grouped = df.groupby(["patient_id", "muscle_name"], sort=False)[CORE_2D_FEATURES]
    median = grouped.median().add_suffix("__median")
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    iqr = (q75 - q25).add_suffix("__IQR")
    p90 = grouped.quantile(0.90).add_suffix("__P90")
    long_agg = pd.concat([median, iqr, p90], axis=1)

    wide = long_agg.unstack("muscle_name").swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature_stat}" for muscle, feature_stat in wide.columns]

    dictionary = []
    for muscle in sorted(df["muscle_name"].unique()):
        for feature in CORE_2D_FEATURES:
            for stat in ("median", "IQR", "P90"):
                dictionary.append(
                    {
                        "feature": f"{muscle}__{feature}__{stat}",
                        "feature_set": "2d",
                        "source": "muscle_features_2d_v6.csv",
                        "muscle": muscle,
                        "base_feature": feature,
                        "aggregation": stat,
                        "role": "candidate_predictor",
                    }
                )

    eps = 1e-8
    for muscle_group, (left, right) in MUSCLE_PAIRS.items():
        for feature in ASYMMETRY_2D_FEATURES:
            left_col = f"{left}__{feature}__median"
            right_col = f"{right}__{feature}__median"
            out_col = f"{muscle_group}__{feature}__median_asymmetry"
            numerator = 2.0 * (wide[left_col] - wide[right_col]).abs()
            denominator = wide[left_col].abs() + wide[right_col].abs() + eps
            wide[out_col] = numerator / denominator
            dictionary.append(
                {
                    "feature": out_col,
                    "feature_set": "2d",
                    "source": "derived from bilateral 2D medians",
                    "muscle": muscle_group,
                    "base_feature": feature,
                    "aggregation": "normalised_absolute_asymmetry",
                    "role": "candidate_predictor",
                }
            )

    return wide, dictionary


def _robust_outlier_records(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    records = []
    for feature in [c for c in table.columns if c not in {"patient_id", "label"}]:
        values = pd.to_numeric(table[feature], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan)
        median = finite.median()
        mad = (finite - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        robust_z = 0.67448975 * (finite - median) / mad
        mask = robust_z.abs() > 5
        for idx in table.index[mask]:
            records.append(
                {
                    "feature_set": feature_set,
                    "patient_id": table.at[idx, "patient_id"],
                    "feature": feature,
                    "value": values.at[idx],
                    "robust_z": robust_z.at[idx],
                    "action": "logged_only; train-fold winsorisation handles extremes",
                }
            )
    return pd.DataFrame(records)


def build_feature_tables(
    project_dir: Path,
    output_dir: Path,
    feature_dir: Path | None = None,
    label_file: Path | None = None,
    feature_version: str = "v6",
    expected_patients: int | None = 311,
) -> BuildResult:
    source_dir = feature_dir or (project_dir / "features_311")

    if label_file and label_file.suffix.lower() == ".csv":
        try:
            labels = pd.read_csv(label_file, usecols=["patient_id", "label"])
        except UnicodeDecodeError:
            labels = pd.read_csv(label_file, usecols=["patient_id", "label"], encoding="gbk")
    elif label_file:
        labels = pd.read_excel(label_file, usecols=["patient_id", "label"])
    else:
        labels = pd.read_excel(project_dir / "patient_stable_311.xlsx", usecols=["patient_id", "label"])

    labels["patient_id"] = _normalise_ids(labels["patient_id"])
    labels = labels.dropna(subset=["label"]).copy()
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce").astype(int)
    labels = labels[labels["label"].isin([0, 1])].copy()

    if expected_patients is not None:
        if len(labels) != expected_patients or labels["patient_id"].nunique() != expected_patients:
            raise ValueError(
                f"Label file must contain exactly {expected_patients} unique patients; "
                f"got {len(labels)} rows, {labels['patient_id'].nunique()} unique"
            )
    if set(labels["label"].dropna().unique()) != {0, 1}:
        raise ValueError("Labels must be binary values 0 and 1")
    target_ids = labels["patient_id"].tolist()

    def _read_csv_safe(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(path, low_memory=False, encoding="gbk")

    df2 = _read_csv_safe(source_dir / f"muscle_features_2d_{feature_version}.csv")
    df3 = _read_csv_safe(source_dir / f"muscle_features_3d_{feature_version}.csv")
    cross = _read_csv_safe(source_dir / f"muscle_features_level3_cross_{feature_version}.csv")
    multi = _read_csv_safe(source_dir / f"muscle_features_level3_multi_{feature_version}.csv")
    for name, frame in (("2d", df2), ("3d", df3), ("cross", cross), ("multi", multi)):
        frame["patient_id"] = _normalise_ids(frame["patient_id"])
        _validate_cohort(frame, target_ids, name)

    bug_records = []
    for feature, reason in KNOWN_EXCLUSIONS["2d"].items():
        bug_records.append({"severity": "warning", "stage": "source_audit", "file": f"muscle_features_2d_{feature_version}.csv", "feature": feature, "issue": reason, "action": "excluded_from_primary_experiment"})
    for feature, reason in KNOWN_EXCLUSIONS["3d"].items():
        bug_records.append({"severity": "warning", "stage": "source_audit", "file": f"muscle_features_3d_{feature_version}.csv", "feature": feature, "issue": reason, "action": "excluded_from_primary_experiment"})

    two_d_wide, feature_dictionary = _aggregate_2d(df2)

    excluded_3d = {"patient_id", "muscle_name", *KNOWN_EXCLUSIONS["3d"].keys()}
    features_3d = [c for c in df3.select_dtypes(include=np.number).columns if c not in excluded_3d]
    three_d_wide = _flatten_pivot(df3, features_3d)
    for muscle in sorted(df3["muscle_name"].unique()):
        for feature in features_3d:
            feature_dictionary.append({"feature": f"{muscle}__{feature}", "feature_set": "3d_level3", "source": f"muscle_features_3d_{feature_version}.csv", "muscle": muscle, "base_feature": feature, "aggregation": "precomputed_3d", "role": "candidate_predictor"})

    cross_features = [c for c in cross.select_dtypes(include=np.number).columns]
    cross_wide = _flatten_pivot(cross, cross_features)
    dup_in_cross = [c for c in cross_wide.columns if c in three_d_wide.columns]
    if dup_in_cross:
        cross_wide = cross_wide.drop(columns=dup_in_cross)
        bug_records.append({
            "severity": "warning", "stage": "feature_merge",
            "file": f"muscle_features_level3_cross_{feature_version}.csv",
            "feature": ", ".join(dup_in_cross),
            "issue": "duplicate columns also present in 3d features; cross-layer duplicates dropped",
            "action": "dropped from cross-layer table",
        })
    for muscle in sorted(cross["muscle_name"].unique()):
        for feature in cross_features:
            col_name = f"{muscle}__{feature}"
            if col_name in dup_in_cross:
                continue
            feature_dictionary.append({"feature": col_name, "feature_set": "3d_level3", "source": f"muscle_features_level3_cross_{feature_version}.csv", "muscle": muscle, "base_feature": feature, "aggregation": "precomputed_cross_layer", "role": "candidate_predictor"})

    multi_wide = multi.set_index("patient_id").drop(columns=[], errors="ignore")
    multi_features = [c for c in multi_wide.select_dtypes(include=np.number).columns]
    multi_wide = multi_wide[multi_features]
    for feature in multi_features:
        feature_dictionary.append({"feature": feature, "feature_set": "3d_level3", "source": f"muscle_features_level3_multi_{feature_version}.csv", "muscle": "multi_muscle", "base_feature": feature, "aggregation": "precomputed_multi_muscle", "role": "candidate_predictor"})

    def finish(wide: pd.DataFrame) -> pd.DataFrame:
        table = wide.reindex(target_ids).reset_index().rename(columns={"index": "patient_id"})
        return table.merge(labels, on="patient_id", how="left", validate="one_to_one")

    table_2d = finish(two_d_wide)
    table_3d = finish(pd.concat([three_d_wide, cross_wide, multi_wide], axis=1))
    table_combined = table_3d.drop(columns="label").merge(
        table_2d.drop(columns="label"), on="patient_id", validate="one_to_one"
    ).merge(labels, on="patient_id", validate="one_to_one")

    tables = {"E1_3d_level3": table_3d, "E2_2d": table_2d, "E3_combined": table_combined}
    audit_rows = []
    for name, table in tables.items():
        features = table.drop(columns=["patient_id", "label"])
        numeric = features.apply(pd.to_numeric, errors="coerce")
        audit_rows.append(
            {
                "feature_set": name,
                "patients": len(table),
                "candidate_features": features.shape[1],
                "missing_cells": int(numeric.isna().sum().sum()),
                "infinite_cells": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
                "constant_features_full_cohort": int((numeric.nunique(dropna=True) <= 1).sum()),
                "label_0": int((table["label"] == 0).sum()),
                "label_1": int((table["label"] == 1).sum()),
            }
        )

    outliers = pd.concat(
        [_robust_outlier_records(table, name) for name, table in tables.items()],
        ignore_index=True,
    )
    audit = pd.DataFrame(audit_rows)
    feature_dictionary_df = pd.DataFrame(feature_dictionary).drop_duplicates("feature", keep="first")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"patient_features_{name}_raw.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    outliers.to_csv(output_dir / "outlier_records.csv", index=False, encoding="utf-8-sig")
    feature_dictionary_df.to_csv(output_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "bug_records.json").write_text(json.dumps(bug_records, ensure_ascii=False, indent=2), encoding="utf-8")

    return BuildResult(tables, audit, outliers, feature_dictionary_df, bug_records)
