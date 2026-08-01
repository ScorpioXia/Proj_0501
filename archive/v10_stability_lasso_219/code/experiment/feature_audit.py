"""Dynamic numerical audit for the current feature CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment.build_features import load_labels, normalise_patient_ids, read_csv_compatible


def _row(check_id, file_name, severity, category, description, tested, violations,
         max_error=np.nan, examples="", recommendation=""):
    return {
        "check_id": check_id, "file": file_name, "severity": severity,
        "category": category, "description": description, "rows_tested": int(tested),
        "violations": int(violations),
        "violation_rate": violations / tested if tested else np.nan,
        "max_absolute_error": max_error, "examples": examples,
        "recommendation": recommendation,
    }


def _examples(frame: pd.DataFrame, mask, columns, n=5):
    columns = [column for column in columns if column in frame.columns]
    return frame.loc[np.asarray(mask), columns].head(n).to_json(orient="records", force_ascii=False)


def run_feature_accuracy_audit(
    feature_dir: Path,
    label_file: Path,
    feature_version: str,
    output_dir: Path,
) -> pd.DataFrame:
    labels = load_labels(label_file)
    expected_ids = set(labels["patient_id"])
    file_names = {
        "2d": f"muscle_features_2d_{feature_version}.csv",
        "3d": f"muscle_features_3d_{feature_version}.csv",
        "cross": f"muscle_features_level3_cross_{feature_version}.csv",
        "multi": f"muscle_features_level3_multi_{feature_version}.csv",
    }
    frames = {
        key: read_csv_compatible(feature_dir / name, low_memory=False)
        for key, name in file_names.items()
    }
    for frame in frames.values():
        frame["patient_id"] = normalise_patient_ids(frame["patient_id"])
    rows = []

    for key, frame in frames.items():
        source_ids = set(frame["patient_id"].dropna())
        missing = expected_ids - source_ids
        rows.append(_row(
            f"cohort_{key}", file_names[key], "error" if missing else "info", "cohort",
            "Every labeled patient_id must occur in the feature file",
            len(expected_ids), len(missing),
            examples=f"labeled={len(expected_ids)}; feature_patients={len(source_ids)}; missing={sorted(missing)[:10]}",
            recommendation="Recover missing labeled patients before modeling" if missing else "Cohort coverage verified",
        ))

    for key in ("3d", "cross"):
        frame = frames[key][frames[key]["patient_id"].isin(expected_ids)]
        counts = frame.groupby("patient_id")["muscle_name"].nunique()
        bad = counts.ne(6)
        rows.append(_row(
            f"six_muscles_{key}", file_names[key], "error" if bad.any() else "info", "completeness",
            "Each labeled patient should have six bilateral muscle records",
            len(counts), int(bad.sum()), examples=counts[bad].head().to_json(),
            recommendation="Recover missing/duplicate muscle records" if bad.any() else "Completeness verified",
        ))

    df2 = frames["2d"][frames["2d"]["patient_id"].isin(expected_ids)].copy()
    df3 = frames["3d"][frames["3d"]["patient_id"].isin(expected_ids)].copy()
    multi = frames["multi"][frames["multi"]["patient_id"].isin(expected_ids)].copy()
    slice_counts = df2.groupby(["patient_id", "muscle_name"]).size()
    low_slice = slice_counts.lt(3)
    rows.append(_row(
        "2d_slice_count", file_names["2d"], "warning" if low_slice.any() else "info", "completeness",
        "Patient/muscle records with fewer than three slices have unstable IQR/P90 summaries",
        len(slice_counts), int(low_slice.sum()),
        examples=f"min={slice_counts.min()}; median={slice_counts.median()}; max={slice_counts.max()}; low={slice_counts[low_slice].head().to_dict()}",
        recommendation="Visually review and flag patients with fewer than three slices" if low_slice.any() else "Slice coverage verified",
    ))

    area_error = (df2["Area"] - df2["Fat_Area"] - df2["Lean_Muscle_Area"]).abs()
    bad = area_error.gt(1e-6)
    rows.append(_row(
        "2d_area_partition", file_names["2d"], "error" if bad.any() else "info", "formula",
        "Area = Fat_Area + Lean_Muscle_Area", len(df2), int(bad.sum()),
        float(area_error.max()), _examples(df2, bad, ["patient_id", "slice_index", "muscle_name", "Area", "Fat_Area", "Lean_Muscle_Area"]),
        "Audit area and fat-mask calculations" if bad.any() else "Identity verified",
    ))

    valid_area = df2["Area"].gt(1e-12)
    fip_error = (df2.loc[valid_area, "FIP"] - df2.loc[valid_area, "Fat_Area"] / df2.loc[valid_area, "Area"]).abs()
    bad_valid = fip_error.gt(1e-6)
    rows.append(_row(
        "2d_fip_formula", file_names["2d"], "error" if bad_valid.any() else "info", "formula",
        "FIP = Fat_Area / Area", int(valid_area.sum()), int(bad_valid.sum()),
        float(fip_error.max()), recommendation="Audit FIP calculation" if bad_valid.any() else "Identity verified",
    ))

    for feature in ("FIP", "Radial_FIP_Ring1", "Radial_FIP_Ring2", "Radial_FIP_Ring3", "Deep_Fat_Ratio"):
        values = pd.to_numeric(df2[feature], errors="coerce")
        bad = values.lt(0) | values.gt(1)
        rows.append(_row(
            f"range_{feature}", file_names["2d"], "error" if bad.any() else "info", "range",
            f"{feature} must be within [0, 1]", int(values.notna().sum()), int(bad.sum()),
            examples=_examples(df2, bad, ["patient_id", "slice_index", "muscle_name", feature]),
            recommendation="Verify denominator and zero handling" if bad.any() else "Range verified",
        ))

    valid_hull = df2["Convex_Hull_Area"].gt(1e-12)
    solidity_error = (
        df2.loc[valid_hull, "Solidity"]
        - df2.loc[valid_hull, "Area"] / df2.loc[valid_hull, "Convex_Hull_Area"]
    ).abs()
    bad_valid = solidity_error.gt(1e-6)
    bad_range = df2["Solidity"].lt(0) | df2["Solidity"].gt(1 + 1e-9)
    rows.append(_row(
        "2d_solidity", file_names["2d"], "error" if bad_valid.any() or bad_range.any() else "info", "geometry",
        "Solidity = Area / Convex_Hull_Area and lies within [0, 1]",
        int(valid_hull.sum()), int(bad_valid.sum() + bad_range.sum()),
        float(solidity_error.max()), recommendation="Audit convex-hull implementation" if bad_valid.any() or bad_range.any() else "Formula and range verified",
    ))

    duplicate_pairs = [
        ("Area", "muscle_area_mm2", False), ("Lean_Muscle_Area", "Func_CSA", False),
        ("Texture_GLCM_Correlation", "Texture_GLCM_Id", True),
        ("Texture_GLCM_Correlation", "Texture_GLCM_Idm", True),
        ("Texture_GLRLM_ShortRunEmphasis", "Texture_GLSZM_SmallAreaEmphasis", True),
        ("Texture_GLRLM_LongRunEmphasis", "Texture_GLSZM_LargeAreaEmphasis", True),
        ("Texture_GLRLM_RunLengthNonUniformity", "Texture_GLSZM_SizeZoneNonUniformity", True),
    ]
    for left, right, suspicious in duplicate_pairs:
        if left not in df2 or right not in df2:
            continue
        equal = np.allclose(df2[left], df2[right], equal_nan=True, rtol=0, atol=0)
        rows.append(_row(
            f"duplicate_{left}_{right}", file_names["2d"],
            "warning" if equal and suspicious else "info", "duplicate",
            f"Exact-column comparison: {left} versus {right}", len(df2), len(df2) if equal else 0,
            examples=f"exact_equal={equal}",
            recommendation="Exclude and review mapping" if equal and suspicious else ("Keep one alias only" if equal else "Distinct fields verified"),
        ))

    valid_volume = df3["3D_Volume"].gt(1e-12)
    fip3_error = (
        df3.loc[valid_volume, "3D_FIP"]
        - (1 - df3.loc[valid_volume, "3D_Func_Volume"] / df3.loc[valid_volume, "3D_Volume"])
    ).abs()
    bad_valid = fip3_error.gt(1e-6)
    rows.append(_row(
        "3d_fip_formula", file_names["3d"], "error" if bad_valid.any() else "info", "formula",
        "3D_FIP = 1 - 3D_Func_Volume / 3D_Volume", int(valid_volume.sum()), int(bad_valid.sum()),
        float(fip3_error.max()), recommendation="Audit 3D integration" if bad_valid.any() else "Identity verified",
    ))

    for prefix in ("Area", "Func_CSA", "FIP"):
        minimum, mean, maximum = f"Min_{prefix}", f"Mean_{prefix}", f"Max_{prefix}"
        bad = (df3[minimum] > df3[mean]) | (df3[mean] > df3[maximum])
        rows.append(_row(
            f"3d_order_{prefix}", file_names["3d"], "error" if bad.any() else "info", "summary_order",
            f"{minimum} <= {mean} <= {maximum}", len(df3), int(bad.sum()),
            examples=_examples(df3, bad, ["patient_id", "muscle_name", minimum, mean, maximum]),
            recommendation="Recompute slice summaries" if bad.any() else "Order verified",
        ))

    for feature in ("SA_V", "3D_Shape_Index"):
        missing = int(df3[feature].isna().sum())
        rows.append(_row(
            f"missing_{feature}", file_names["3d"], "warning" if missing else "info", "missingness",
            f"Missing values in corrected {feature}", len(df3), missing,
            recommendation="Review extraction failures" if missing else "Availability verified",
        ))

    ratio_columns = [column for column in multi.columns if "Ratio" in column or column.startswith("Rat_") or column.startswith("Symmetry_Index")]
    for column in ratio_columns:
        values = pd.to_numeric(multi[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        extreme = values.abs().gt(10)
        rows.append(_row(
            f"extreme_multi_{column}", file_names["multi"], "warning" if extreme.any() else "info", "outlier",
            f"Raw denominator-sensitive field {column}: absolute value > 10", int(values.notna().sum()), int(extreme.sum()),
            examples=_examples(multi, extreme, ["patient_id", column]),
            recommendation="Use the pipeline's log/bounded transform and inspect denominators" if extreme.any() else "No extreme raw ratio",
        ))

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "feature_accuracy_audit.csv", index=False, encoding="utf-8-sig")
    serious = result[result["severity"].isin(["error", "warning"])]
    lines = [
        "# 特征准确性数值审计", "",
        "本审计仅基于当前 CSV 的数学一致性，不能替代原始 MRI、分割掩膜和叠加图的人工复核。", "",
        f"动态纳入有标签患者 {len(labels)} 人；共执行 {len(result)} 项检查，错误 {int((result.severity == 'error').sum())} 项，警告 {int((result.severity == 'warning').sum())} 项。",
        "", "|级别|检查|违规数/检查数|建议|", "|---|---|---:|---|",
    ]
    for row in serious.itertuples():
        lines.append(f"|{row.severity}|{row.description}|{row.violations}/{row.rows_tested}|{row.recommendation}|")
    (output_dir / "FEATURE_ACCURACY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
