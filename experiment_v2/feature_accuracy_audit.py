"""Numerical consistency audit for retained feature files.

This does not replace visual review of source MRI masks.  It checks cohort
completeness, mathematical identities, physical range constraints and exact
duplicate columns that can be assessed from the retained CSV files alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _row(check_id, file, severity, category, description, tested, violations,
         max_error=np.nan, examples="", recommendation=""):
    rate = violations / tested if tested else np.nan
    return {
        "check_id": check_id,
        "file": file,
        "severity": severity,
        "category": category,
        "description": description,
        "rows_tested": int(tested),
        "violations": int(violations),
        "violation_rate": rate,
        "max_absolute_error": max_error,
        "examples": examples,
        "recommendation": recommendation,
    }


def _examples(frame: pd.DataFrame, mask, columns, n=5):
    cols = [c for c in columns if c in frame.columns]
    return frame.loc[np.asarray(mask), cols].head(n).to_json(orient="records", force_ascii=False)


def run_feature_accuracy_audit(project_dir: Path, output_dir: Path) -> pd.DataFrame:
    source = project_dir / "features_311"
    df2 = pd.read_csv(source / "muscle_features_2d_v6.csv", low_memory=False)
    df3 = pd.read_csv(source / "muscle_features_3d_v6.csv", low_memory=False)
    cross = pd.read_csv(source / "muscle_features_level3_cross_v6.csv", low_memory=False)
    multi = pd.read_csv(source / "muscle_features_level3_multi_v6.csv", low_memory=False)
    rows = []

    for name, frame, expected_rows in (
        ("muscle_features_2d_v6.csv", df2, None),
        ("muscle_features_3d_v6.csv", df3, 311 * 6),
        ("muscle_features_level3_cross_v6.csv", cross, 311 * 6),
        ("muscle_features_level3_multi_v6.csv", multi, 311),
    ):
        patient_count = frame["patient_id"].astype(str).str.strip().nunique()
        violations = int(patient_count != 311 or (expected_rows is not None and len(frame) != expected_rows))
        rows.append(_row(
            f"cohort_{name}", name, "error" if violations else "info", "cohort",
            f"Expected 311 patients" + (f" and {expected_rows} rows" if expected_rows else ""),
            1, violations, examples=f"patients={patient_count}; rows={len(frame)}",
            recommendation="Repair patient/muscle alignment before modeling" if violations else "No action",
        ))

    for name, frame in (("muscle_features_3d_v6.csv", df3), ("muscle_features_level3_cross_v6.csv", cross)):
        counts = frame.groupby("patient_id")["muscle_name"].nunique()
        bad = counts.ne(6)
        rows.append(_row(
            f"six_muscles_{name}", name, "error" if bad.any() else "info", "completeness",
            "Each patient should have all six bilateral muscle records", len(counts), int(bad.sum()),
            examples=counts[bad].head().to_json(),
            recommendation="Recover missing muscle rows" if bad.any() else "No action",
        ))

    slice_counts = df2.groupby(["patient_id", "muscle_name"]).size()
    rows.append(_row(
        "2d_slice_count_variability", "muscle_features_2d_v6.csv", "warning", "completeness",
        "2D slice counts vary across patient-muscle records; aggregation reduces but does not remove this acquisition effect",
        len(slice_counts), int((slice_counts < 3).sum()),
        examples=f"min={slice_counts.min()}; median={slice_counts.median()}; max={slice_counts.max()}",
        recommendation="Review cases with <3 slices and consider slice-count sensitivity analysis",
    ))

    valid_area = df2["Area"] > 1e-12
    area_identity_error = (df2["Area"] - df2["Fat_Area"] - df2["Lean_Muscle_Area"]).abs()
    bad = area_identity_error > 1e-6
    rows.append(_row(
        "2d_area_partition", "muscle_features_2d_v6.csv", "error" if bad.any() else "info", "formula",
        "Area should equal Fat_Area + Lean_Muscle_Area", len(df2), int(bad.sum()),
        float(area_identity_error.max()), _examples(df2, bad, ["patient_id", "slice_index", "muscle_name", "Area", "Fat_Area", "Lean_Muscle_Area"]),
        "Inspect area-unit conversion and fat-mask calculation" if bad.any() else "Identity verified",
    ))

    expected_fip = df2.loc[valid_area, "Fat_Area"] / df2.loc[valid_area, "Area"]
    fip_error = (df2.loc[valid_area, "FIP"] - expected_fip).abs()
    bad_valid = fip_error > 1e-6
    bad = np.zeros(len(df2), dtype=bool); bad[np.flatnonzero(valid_area)] = bad_valid.to_numpy()
    rows.append(_row(
        "2d_fip_formula", "muscle_features_2d_v6.csv", "error" if bad_valid.any() else "info", "formula",
        "FIP should equal Fat_Area / Area for nonzero masks", int(valid_area.sum()), int(bad_valid.sum()),
        float(fip_error.max()), _examples(df2, bad, ["patient_id", "slice_index", "muscle_name", "Area", "Fat_Area", "FIP"]),
        "Inspect FIP calculation" if bad_valid.any() else "Identity verified",
    ))

    duplicate_pairs = [
        ("Area", "muscle_area_mm2"), ("Lean_Muscle_Area", "Func_CSA"),
        ("Texture_GLCM_Correlation", "Texture_GLCM_Id"),
        ("Texture_GLCM_Correlation", "Texture_GLCM_Idm"),
        ("Texture_GLRLM_ShortRunEmphasis", "Texture_GLSZM_SmallAreaEmphasis"),
        ("Texture_GLRLM_LongRunEmphasis", "Texture_GLSZM_LargeAreaEmphasis"),
        ("Texture_GLRLM_RunLengthNonUniformity", "Texture_GLSZM_SizeZoneNonUniformity"),
    ]
    for left, right in duplicate_pairs:
        equal = np.allclose(df2[left], df2[right], equal_nan=True, rtol=0, atol=0)
        suspicious = left.startswith("Texture")
        rows.append(_row(
            f"duplicate_{left}_{right}", "muscle_features_2d_v6.csv",
            "error" if equal and suspicious else ("warning" if equal else "info"), "duplicate",
            f"Exact-column comparison: {left} versus {right}", len(df2), len(df2) if equal else 0,
            examples=f"exact_equal={equal}",
            recommendation=("Audit texture feature name-to-formula mapping" if equal and suspicious else
                            "Keep only one copy in modeling" if equal else "No action"),
        ))

    for feature in ("FIP", "Radial_FIP_Ring1", "Radial_FIP_Ring2", "Radial_FIP_Ring3", "Deep_Fat_Ratio"):
        values = pd.to_numeric(df2[feature], errors="coerce")
        bad = values.lt(0) | values.gt(1)
        severity = "error" if feature == "Deep_Fat_Ratio" and bad.any() else ("warning" if bad.any() else "info")
        rows.append(_row(
            f"range_{feature}", "muscle_features_2d_v6.csv", severity, "range",
            f"{feature} expected in [0, 1] if defined as a proportion", values.notna().sum(), int(bad.sum()),
            examples=_examples(df2, bad, ["patient_id", "slice_index", "muscle_name", feature]),
            recommendation="Confirm the denominator and zero-denominator handling in extraction" if bad.any() else "Range verified",
        ))

    valid_hull = df2["Convex_Hull_Area"] > 1e-12
    solidity_expected = df2.loc[valid_hull, "Area"] / df2.loc[valid_hull, "Convex_Hull_Area"]
    solidity_error = (df2.loc[valid_hull, "Solidity"] - solidity_expected).abs()
    solidity_bad_range = df2["Solidity"].gt(1 + 1e-9)
    rows.append(_row(
        "2d_solidity_range", "muscle_features_2d_v6.csv", "error" if solidity_bad_range.any() else "info", "geometry",
        "Solidity=area/convex-hull-area should not exceed 1", int(valid_hull.sum()), int(solidity_bad_range.sum()),
        float(max(solidity_error.max(), (df2["Solidity"] - 1).max())),
        _examples(df2, solidity_bad_range, ["patient_id", "slice_index", "muscle_name", "Area", "Convex_Hull_Area", "Solidity"]),
        "Inspect convex-hull pixel count and physical-area conversion; exclude/correct Solidity until resolved" if solidity_bad_range.any() else "Range verified",
    ))

    valid_vol = df3["3D_Volume"] > 1e-12
    fip3_error = (df3.loc[valid_vol, "3D_FIP"] - (1 - df3.loc[valid_vol, "3D_Func_Volume"] / df3.loc[valid_vol, "3D_Volume"])).abs()
    bad_valid = fip3_error > 1e-6
    rows.append(_row(
        "3d_fip_formula", "muscle_features_3d_v6.csv", "error" if bad_valid.any() else "info", "formula",
        "3D_FIP should equal 1 - 3D_Func_Volume/3D_Volume", int(valid_vol.sum()), int(bad_valid.sum()),
        float(fip3_error.max()),
        recommendation="Inspect volume integration/FIP calculation" if bad_valid.any() else "Identity verified",
    ))

    for prefix in ("Area", "Func_CSA", "FIP"):
        minimum, mean, maximum = f"Min_{prefix}", f"Mean_{prefix}", f"Max_{prefix}"
        bad = (df3[minimum] > df3[mean]) | (df3[mean] > df3[maximum])
        rows.append(_row(
            f"3d_order_{prefix}", "muscle_features_3d_v6.csv", "error" if bad.any() else "info", "summary_order",
            f"Expected {minimum} <= {mean} <= {maximum}", len(df3), int(bad.sum()),
            examples=_examples(df3, bad, ["patient_id", "muscle_name", minimum, mean, maximum]),
            recommendation="Recompute slice summaries" if bad.any() else "Order verified",
        ))

    missing = df3.isna().mean()
    for feature in missing[missing == 1].index:
        rows.append(_row(
            f"all_missing_{feature}", "muscle_features_3d_v6.csv", "warning", "missingness",
            f"{feature} is 100% missing", len(df3), len(df3),
            recommendation="Remove from modeling or repair extraction implementation",
        ))

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "feature_accuracy_audit.csv", index=False, encoding="utf-8-sig")

    serious = result[result["severity"].isin(["error", "warning"])]
    lines = [
        "# 特征准确性数值审计",
        "",
        "本审计仅使用当前保留的 CSV，不能代替对原始 MRI、分割掩膜与逐例叠加图的人工复核。",
        "",
        f"共执行 {len(result)} 项检查；发现 {int((result.severity == 'error').sum())} 项错误级问题、{int((result.severity == 'warning').sum())} 项警告。",
        "",
        "|级别|检查|违规数/检查数|建议|",
        "|---|---|---:|---|",
    ]
    for row in serious.itertuples():
        lines.append(f"|{row.severity}|{row.description}|{row.violations}/{row.rows_tested}|{row.recommendation}|")
    (output_dir / "FEATURE_ACCURACY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
