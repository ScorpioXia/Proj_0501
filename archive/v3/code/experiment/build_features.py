"""Build unstandardized patient-level feature tables from v7 CSV files.

Only deterministic, label-independent aggregation is performed here. Missing
value imputation, clipping, redundancy removal, supervised selection and
scaling remain inside the cross-validation pipeline.
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

# Curated v7 fields. Corrected GLCM fields are intentionally included; the
# still duplicated GLRLM/GLSZM pairs remain excluded pending formula review.
CORE_2D_FEATURES = [
    "Area", "Max_AP_Diameter", "Max_Transverse_Diameter", "Solidity",
    "Max_Inscribed_Circle_Diameter", "Lean_Muscle_Area", "FIP",
    "Std_Intensity_Muscle", "Skewness_Intensity_Muscle",
    "Kurtosis_Intensity_Muscle", "Deep_Fat_Ratio", "Radial_FIP_Ring1",
    "Radial_FIP_Ring2", "Radial_FIP_Ring3", "Fat_Entropy",
    "Fat_Centroid_Offset", "Texture_FirstOrder_Entropy",
    "Texture_GLCM_Contrast", "Texture_GLCM_Correlation", "Texture_GLCM_Id",
    "Texture_GLCM_Idm", "Texture_GLDM_DependenceEntropy",
    "Texture_GLDM_DependenceNonUniformity", "Texture_GLDM_GrayLevelNonUniformity",
]

ASYMMETRY_2D_FEATURES = [
    "Area", "Lean_Muscle_Area", "FIP", "Deep_Fat_Ratio", "Fat_Centroid_Offset",
]

EXCLUSIONS = {
    "muscle_area_mm2": "exact alias of Area",
    "Func_CSA": "exact alias of Lean_Muscle_Area",
    "Texture_GLRLM_ShortRunEmphasis": "exact duplicate of GLSZM SmallAreaEmphasis",
    "Texture_GLRLM_LongRunEmphasis": "exact duplicate of GLSZM LargeAreaEmphasis",
    "Texture_GLRLM_RunLengthNonUniformity": "exact duplicate of GLSZM SizeZoneNonUniformity",
    "Texture_GLSZM_SmallAreaEmphasis": "paired with a still-suspect duplicate mapping",
    "Texture_GLSZM_LargeAreaEmphasis": "paired with a still-suspect duplicate mapping",
    "Texture_GLSZM_SizeZoneNonUniformity": "paired with a still-suspect duplicate mapping",
    "pixel_spacing_x": "technical acquisition field",
    "pixel_spacing_y": "technical acquisition field",
    "slice_thickness": "technical acquisition field",
    "csf_value": "normalization/QC field rather than a biological predictor",
    "fat_threshold_used": "algorithm/QC field rather than a biological predictor",
    "fat_threshold_decision": "algorithm/QC field rather than a biological predictor",
    "slice_index": "non-comparable absolute slice index",
    "Peak_Area_Slice_Index": "non-comparable absolute slice index",
    "Peak_FIP_Slice_Index": "non-comparable absolute slice index",
    "Total_CSA": "ambiguous extraction/QC field",
}

MULTI_RATIO_FIELDS = {
    "Psoas_Posterior_Ratio", "ES_MF_Area_Ratio", "Psoas_ES_Area_Ratio",
    "MF_Psoas_Area_Ratio", "Rat_FIP_MF_Psoas", "Mean_Intensity_ES_MF_Ratio",
}
MULTI_SYMMETRY_FIELDS = {
    "Symmetry_Index_Area_MF", "Symmetry_Index_Area_ES", "Symmetry_Index_Area_Psoas",
    "Symmetry_Index_FIP_MF", "Symmetry_Index_FIP_ES", "Symmetry_Index_FIP_Psoas",
}


@dataclass
class BuildResult:
    tables: dict[str, pd.DataFrame]
    audit: pd.DataFrame
    outliers: pd.DataFrame
    feature_dictionary: pd.DataFrame
    bug_records: list[dict]
    labels: pd.DataFrame


def normalise_patient_ids(values: pd.Series) -> pd.Series:
    """Canonicalize numeric IDs without ever consulting patient names."""
    result = values.astype("string").str.strip()
    result = result.mask(result.eq(""))
    numeric = pd.to_numeric(result, errors="coerce")
    integer_mask = numeric.notna() & np.isfinite(numeric) & np.equal(numeric % 1, 0)
    result = result.copy()
    result.loc[integer_mask] = numeric.loc[integer_mask].astype("Int64").astype("string")
    return result


def read_csv_compatible(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    last_error = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error  # pragma: no cover


def load_labels(label_file: Path) -> pd.DataFrame:
    """Read the dynamic labeled cohort from PATIENT_LIST_FILE.csv."""
    if label_file.suffix.lower() != ".csv":
        raise ValueError("The current pipeline accepts labels only from PATIENT_LIST_FILE.csv")
    raw = read_csv_compatible(label_file)
    required = {"patient_id", "label"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Label CSV is missing columns: {sorted(required - set(raw.columns))}")

    label_text = raw["label"].astype("string").str.strip()
    numeric_label = pd.to_numeric(raw["label"], errors="coerce")
    invalid_nonblank = label_text.notna() & label_text.ne("") & numeric_label.isna()
    invalid_binary = numeric_label.notna() & ~numeric_label.isin([0, 1])
    if invalid_nonblank.any() or invalid_binary.any():
        rows = raw.index[invalid_nonblank | invalid_binary].tolist()[:10]
        raise ValueError(f"Label column contains non-binary values at source rows: {rows}")

    labels = raw.loc[numeric_label.notna(), ["patient_id"]].copy()
    labels["label"] = numeric_label.loc[numeric_label.notna()].astype(int).to_numpy()
    labels["patient_id"] = normalise_patient_ids(labels["patient_id"])
    if labels["patient_id"].isna().any():
        raise ValueError("A labeled row has an empty patient_id")
    duplicates = labels[labels["patient_id"].duplicated(keep=False)]["patient_id"].unique()
    if len(duplicates):
        raise ValueError(f"Duplicate labeled patient_id values: {duplicates[:10].tolist()}")
    if set(labels["label"].unique()) != {0, 1}:
        raise ValueError("The labeled cohort must contain both label 0 and label 1")
    return labels.reset_index(drop=True)


def _validate_and_subset(frame: pd.DataFrame, labels: pd.DataFrame, name: str) -> tuple[pd.DataFrame, int]:
    if "patient_id" not in frame.columns:
        raise ValueError(f"{name} has no patient_id column")
    frame = frame.copy()
    frame["patient_id"] = normalise_patient_ids(frame["patient_id"])
    if frame["patient_id"].isna().any():
        raise ValueError(f"{name} contains empty patient_id values")
    source_ids = set(frame["patient_id"])
    target_ids = set(labels["patient_id"])
    missing = sorted(target_ids - source_ids)
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} labeled patients: {missing[:10]}")
    extras = len(source_ids - target_ids)
    return frame[frame["patient_id"].isin(target_ids)].copy(), extras


def _flatten_pivot(frame: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    duplicated = frame.duplicated(["patient_id", "muscle_name"], keep=False)
    if duplicated.any():
        raise ValueError("Patient/muscle rows must be unique before wide pivot")
    wide = frame.pivot(index="patient_id", columns="muscle_name", values=list(value_columns))
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature}" for muscle, feature in wide.columns]
    return wide


def _aggregate_2d(frame: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, list[dict]]:
    missing = sorted(set(CORE_2D_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"2D source is missing required v7 fields: {missing}")
    grouped = frame.groupby(["patient_id", "muscle_name"], sort=False)[CORE_2D_FEATURES]
    median = grouped.median().add_suffix("__median")
    iqr = (grouped.quantile(0.75) - grouped.quantile(0.25)).add_suffix("__IQR")
    p90 = grouped.quantile(0.90).add_suffix("__P90")
    long_aggregate = pd.concat([median, iqr, p90], axis=1)
    wide = long_aggregate.unstack("muscle_name").swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [f"{muscle}__{feature_stat}" for muscle, feature_stat in wide.columns]

    dictionary = []
    for muscle in sorted(frame["muscle_name"].unique()):
        for feature in CORE_2D_FEATURES:
            for statistic in ("median", "IQR", "P90"):
                dictionary.append({
                    "feature": f"{muscle}__{feature}__{statistic}", "feature_set": "2d",
                    "source": source_name, "muscle": muscle, "base_feature": feature,
                    "aggregation": statistic, "role": "candidate_predictor",
                })

    epsilon = 1e-8
    for muscle_group, (left, right) in MUSCLE_PAIRS.items():
        for feature in ASYMMETRY_2D_FEATURES:
            left_column = f"{left}__{feature}__median"
            right_column = f"{right}__{feature}__median"
            output_column = f"{muscle_group}__{feature}__median_asymmetry"
            numerator = 2.0 * (wide[left_column] - wide[right_column]).abs()
            denominator = wide[left_column].abs() + wide[right_column].abs() + epsilon
            wide[output_column] = numerator / denominator
            dictionary.append({
                "feature": output_column, "feature_set": "2d",
                "source": "derived from bilateral 2D medians", "muscle": muscle_group,
                "base_feature": feature, "aggregation": "bounded_absolute_asymmetry",
                "role": "candidate_predictor",
            })
    return wide, dictionary


def _transform_multi_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Replace denominator-sensitive raw ratios with stable transforms."""
    output = pd.DataFrame(index=frame.index)
    records: list[dict] = []
    for column in frame.select_dtypes(include=np.number).columns:
        if column in EXCLUSIONS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if column in MULTI_RATIO_FIELDS:
            new_name = f"{column}__signed_log1p"
            output[new_name] = np.sign(values) * np.log1p(np.abs(values))
            aggregation = "signed_log1p_of_raw_ratio"
        elif column in MULTI_SYMMETRY_FIELDS:
            new_name = f"{column}__bounded_deviation_from_1"
            output[new_name] = 2.0 * np.abs(values - 1.0) / (np.abs(values) + 1.0 + 1e-8)
            aggregation = "bounded_ratio_asymmetry"
        else:
            new_name = column
            output[new_name] = values
            aggregation = "precomputed_multi_muscle"
        records.append({
            "feature": new_name, "feature_set": "3d_level3",
            "source": "multi-muscle v7 table", "muscle": "multi_muscle",
            "base_feature": column, "aggregation": aggregation,
            "role": "candidate_predictor",
        })
    return output, records


def _robust_outlier_records(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    records = []
    for feature in [c for c in table.columns if c not in {"patient_id", "label"}]:
        values = pd.to_numeric(table[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = values.median()
        mad = (values - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        robust_z = 0.67448975 * (values - median) / mad
        for index in table.index[robust_z.abs() > 5]:
            records.append({
                "feature_set": feature_set, "patient_id": table.at[index, "patient_id"],
                "feature": feature, "value": values.at[index], "robust_z": robust_z.at[index],
                "action": "logged_only; train-fold winsorisation handles extremes",
            })
    return pd.DataFrame(records)


def build_feature_tables(
    feature_dir: Path,
    label_file: Path,
    output_dir: Path,
    feature_version: str = "v7",
) -> BuildResult:
    labels = load_labels(label_file)
    names = {
        "2d": f"muscle_features_2d_{feature_version}.csv",
        "3d": f"muscle_features_3d_{feature_version}.csv",
        "cross": f"muscle_features_level3_cross_{feature_version}.csv",
        "multi": f"muscle_features_level3_multi_{feature_version}.csv",
    }
    frames = {key: read_csv_compatible(feature_dir / name, low_memory=False) for key, name in names.items()}
    bug_records: list[dict] = []
    for key, frame in list(frames.items()):
        frames[key], extras = _validate_and_subset(frame, labels, key)
        if extras:
            bug_records.append({
                "severity": "info", "stage": "cohort_alignment", "file": names[key],
                "feature": "patient_id", "issue": f"ignored {extras} feature-only patients without labels",
                "action": "subset to labeled patient_id values",
            })

    for feature, reason in EXCLUSIONS.items():
        present_in = [names[key] for key, frame in frames.items() if feature in frame.columns]
        if present_in:
            bug_records.append({
                "severity": "warning", "stage": "predictor_exclusion",
                "file": ", ".join(present_in), "feature": feature,
                "issue": reason, "action": "excluded from modeling",
            })

    two_d_wide, dictionary = _aggregate_2d(frames["2d"], names["2d"])

    df3 = frames["3d"]
    excluded = {"patient_id", "muscle_name", *EXCLUSIONS}
    features_3d = [column for column in df3.select_dtypes(include=np.number).columns if column not in excluded]
    three_d_wide = _flatten_pivot(df3, features_3d)
    for muscle in sorted(df3["muscle_name"].unique()):
        for feature in features_3d:
            dictionary.append({
                "feature": f"{muscle}__{feature}", "feature_set": "3d_level3",
                "source": names["3d"], "muscle": muscle, "base_feature": feature,
                "aggregation": "precomputed_3d", "role": "candidate_predictor",
            })

    cross = frames["cross"]
    cross_features = [
        column for column in cross.select_dtypes(include=np.number).columns
        if column not in {"patient_id", "muscle_name", *EXCLUSIONS}
    ]
    cross_wide = _flatten_pivot(cross, cross_features)
    duplicate_columns = sorted(set(cross_wide.columns) & set(three_d_wide.columns))
    if duplicate_columns:
        cross_wide = cross_wide.drop(columns=duplicate_columns)
        bug_records.append({
            "severity": "warning", "stage": "feature_merge", "file": names["cross"],
            "feature": ", ".join(duplicate_columns), "issue": "duplicate 3D/cross-layer columns",
            "action": "dropped cross-layer copies",
        })
    for muscle in sorted(cross["muscle_name"].unique()):
        for feature in cross_features:
            name = f"{muscle}__{feature}"
            if name not in duplicate_columns:
                dictionary.append({
                    "feature": name, "feature_set": "3d_level3", "source": names["cross"],
                    "muscle": muscle, "base_feature": feature,
                    "aggregation": "precomputed_cross_layer", "role": "candidate_predictor",
                })

    multi = frames["multi"].set_index("patient_id")
    multi_wide, multi_dictionary = _transform_multi_features(multi)
    dictionary.extend(multi_dictionary)

    target_ids = labels["patient_id"].tolist()

    def finish(wide: pd.DataFrame) -> pd.DataFrame:
        table = wide.reindex(target_ids).reset_index().rename(columns={"index": "patient_id"})
        return table.merge(labels, on="patient_id", how="left", validate="one_to_one")

    table_2d = finish(two_d_wide)
    table_3d = finish(pd.concat([three_d_wide, cross_wide, multi_wide], axis=1))
    table_combined = (
        table_3d.drop(columns="label")
        .merge(table_2d.drop(columns="label"), on="patient_id", validate="one_to_one")
        .merge(labels, on="patient_id", validate="one_to_one")
    )
    tables = {"E1_3d_level3": table_3d, "E2_2d": table_2d, "E3_combined": table_combined}

    audits = []
    for name, table in tables.items():
        numeric = table.drop(columns=["patient_id", "label"]).apply(pd.to_numeric, errors="coerce")
        audits.append({
            "feature_set": name, "patients": len(table), "candidate_features": numeric.shape[1],
            "missing_cells": int(numeric.isna().sum().sum()),
            "infinite_cells": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
            "constant_features_full_cohort": int((numeric.nunique(dropna=True) <= 1).sum()),
            "label_0": int((table["label"] == 0).sum()), "label_1": int((table["label"] == 1).sum()),
        })
    audit = pd.DataFrame(audits)
    outlier_frames = [_robust_outlier_records(table, name) for name, table in tables.items()]
    outliers = pd.concat(outlier_frames, ignore_index=True) if any(not x.empty for x in outlier_frames) else pd.DataFrame(
        columns=["feature_set", "patient_id", "feature", "value", "robust_z", "action"]
    )
    feature_dictionary = pd.DataFrame(dictionary).drop_duplicates("feature", keep="first")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"patient_features_{name}_raw.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    outliers.to_csv(output_dir / "outlier_records.csv", index=False, encoding="utf-8-sig")
    feature_dictionary.to_csv(output_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(output_dir / "labeled_cohort_used.csv", index=False, encoding="utf-8-sig")
    (output_dir / "bug_records.json").write_text(
        json.dumps(bug_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return BuildResult(tables, audit, outliers, feature_dictionary, bug_records, labels)
