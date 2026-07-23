"""Post-hoc sensitivity analysis excluding numerically suspect 2D fields.

This analysis was motivated by the retained-feature audit and must not be
reported as a preregistered primary comparison.  It uses the same outer and
inner folds and the same prespecified XGBoost/LightGBM candidates as stage 2.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment_v1.build_features import build_feature_tables
from experiment_v2.modeling_v2 import (
    bootstrap_performance,
    importance_summary,
    nested_cv_model,
)


def paired_auc(frame: pd.DataFrame, references: list[str], comparisons: list[str], iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for comparison in comparisons:
        right = frame[frame.experiment_id == comparison].set_index("patient_id")
        for reference in references:
            left = frame[frame.experiment_id == reference].set_index("patient_id")
            common = left.index.intersection(right.index)
            y = left.loc[common, "true_label"].to_numpy(int)
            p0 = left.loc[common, "predicted_probability"].to_numpy(float)
            p1 = right.loc[common, "predicted_probability"].to_numpy(float)
            point = roc_auc_score(y, p1) - roc_auc_score(y, p0)
            values = []
            for _ in range(iterations):
                idx = rng.integers(0, len(y), len(y))
                if np.unique(y[idx]).size < 2:
                    continue
                values.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
            rows.append({
                "reference": reference, "comparison": comparison,
                "auc_difference": point, "ci_low": np.quantile(values, 0.025),
                "ci_high": np.quantile(values, 0.975),
                "bootstrap_probability_gt_reference": float(np.mean(np.asarray(values) > 0)),
            })
    return pd.DataFrame(rows)


def main():
    if "nnunet-master" not in sys.executable.lower():
        raise RuntimeError("Sensitivity analysis must run in nnUNet-master")
    output_dir = PROJECT_DIR / "experiment_results_v2_sensitivity"
    output_dir.mkdir(exist_ok=True)
    config = json.loads((PROJECT_DIR / "experiment_v2" / "config.json").read_text(encoding="utf-8"))
    log_path = output_dir / "experiment_log.txt"

    def log(message):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("START post-hoc suspect-feature sensitivity analysis")
    build = build_feature_tables(PROJECT_DIR, output_dir)
    table = build.tables["E2_2d"].copy()
    suspect = [c for c in table.columns if "Solidity" in c or "Deep_Fat_Ratio" in c]
    clean = table.drop(columns=suspect)
    clean_name = "E2_2d_without_Solidity_DeepFat"
    log(f"Dropped {len(suspect)} features; retained {clean.shape[1] - 2} predictors")
    pd.DataFrame({"dropped_feature": suspect, "reason": "numerical audit range violation"}).to_csv(
        output_dir / "dropped_suspect_features.csv", index=False, encoding="utf-8-sig"
    )

    predictions, folds, importances, issues = [], [], [], []
    for model in ("xgboost", "lightgbm"):
        pred, fold, importance, model_issues = nested_cv_model(
            clean, clean_name, model, config, output_dir, log
        )
        predictions.append(pred); folds.append(fold); importances.append(importance); issues.extend(model_issues)
    sensitivity_predictions = pd.concat(predictions, ignore_index=True)

    primary = pd.read_csv(PROJECT_DIR / "experiment_results_v2" / "nested_cv_predictions.csv")
    references = ["PrevalenceBaseline", "XGBoost__E2_2d", "LightGBM__E2_2d"]
    reference_predictions = primary[primary.experiment_id.isin(references)].copy()
    combined = pd.concat([reference_predictions, sensitivity_predictions], ignore_index=True)
    comparison_ids = sensitivity_predictions.experiment_id.drop_duplicates().tolist()
    performance = bootstrap_performance(combined, config["bootstrap_iterations"], config["random_seed"] + 401)
    comparisons = paired_auc(combined, references, comparison_ids, config["bootstrap_iterations"], config["random_seed"] + 402)
    importance_rows = pd.concat(importances, ignore_index=True)

    sensitivity_predictions.to_csv(output_dir / "nested_cv_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(folds, ignore_index=True).to_csv(output_dir / "outer_fold_performance.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output_dir / "model_performance_with_references.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output_dir / "paired_auc_comparisons.csv", index=False, encoding="utf-8-sig")
    importance_rows.to_csv(output_dir / "permutation_importance_each_fold.csv", index=False, encoding="utf-8-sig")
    importance_summary(importance_rows, config["outer_folds"]).to_csv(
        output_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(issues, columns=["experiment_id", "model", "feature_set", "outer_fold", "category", "message"]).to_csv(
        output_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "status": "completed", "analysis_type": "post_hoc_sensitivity",
        "dropped_feature_count": len(suspect), "retained_predictor_count": clean.shape[1] - 2,
        "performance": performance.to_dict(orient="records"),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("FINAL\n" + performance.to_string(index=False))
    log("END sensitivity analysis completed")


if __name__ == "__main__":
    main()
