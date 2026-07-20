"""Run the complete, reproducible v1 experiment from raw retained features."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_features import build_feature_tables
from modeling import (
    bootstrap_metrics,
    dummy_prevalence_oof,
    environment_info,
    nested_cv_feature_set,
    paired_bootstrap_comparisons,
    save_plots,
    selection_summary,
    univariate_associations,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("config.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = (args.output_dir or project_dir / "experiment_results_v1").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    shutil.copy2(args.config, output_dir / "config_used.json")
    (output_dir / "environment.json").write_text(json.dumps(environment_info(), ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = output_dir / "experiment_log.txt"

    def log(message):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    failures = []
    try:
        log("START experiment_v1")
        log("Build deterministic patient-level feature tables")
        build = build_feature_tables(project_dir, output_dir)
        log(f"Data audit complete: {build.audit.to_dict(orient='records')}")
        log(f"Robust outliers logged: {len(build.outliers)} records; no patients deleted")

        associations = univariate_associations(build.tables)
        associations.to_csv(output_dir / "univariate_feature_association.csv", index=False, encoding="utf-8-sig")
        log(f"Univariate association table saved: {len(associations)} feature rows")

        baseline_predictions, baseline_folds = dummy_prevalence_oof(
            build.tables["E1_3d_level3"], config
        )
        all_predictions, all_folds, all_coefficients, runtime_issues = [baseline_predictions], [baseline_folds], [], []
        for feature_set in ("E1_3d_level3", "E2_2d", "E3_combined"):
            predictions, folds, coefficients, issues = nested_cv_feature_set(build.tables[feature_set], feature_set, config, output_dir, log)
            all_predictions.append(predictions); all_folds.append(folds); all_coefficients.append(coefficients); runtime_issues.extend(issues)

        predictions = pd.concat(all_predictions, ignore_index=True)
        fold_performance = pd.concat(all_folds, ignore_index=True)
        coefficients = pd.concat(all_coefficients, ignore_index=True)
        performance = bootstrap_metrics(predictions, config["bootstrap_iterations"], config["random_seed"])
        comparisons = paired_bootstrap_comparisons(predictions, config["bootstrap_iterations"], config["random_seed"])
        selection = selection_summary(coefficients, config["outer_folds"])

        predictions.to_csv(output_dir / "nested_cv_predictions.csv", index=False, encoding="utf-8-sig")
        fold_performance.to_csv(output_dir / "outer_fold_performance.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output_dir / "model_performance.csv", index=False, encoding="utf-8-sig")
        comparisons.to_csv(output_dir / "feature_set_comparisons.csv", index=False, encoding="utf-8-sig")
        coefficients.to_csv(output_dir / "selected_features_each_fold.csv", index=False, encoding="utf-8-sig")
        selection.to_csv(output_dir / "feature_selection_frequency.csv", index=False, encoding="utf-8-sig")
        warning_columns = ["feature_set", "outer_fold", "category", "message"]
        pd.DataFrame(runtime_issues, columns=warning_columns).to_csv(
            output_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig"
        )
        save_plots(predictions, associations, selection, output_dir)

        summary = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "patients": 311,
            "label_distribution": {"0_stable": 189, "1_unstable": 122},
            "performance": performance.to_dict(orient="records"),
            "paired_auc_comparisons": comparisons.to_dict(orient="records"),
            "runtime_warning_count": len(runtime_issues),
            "outlier_record_count": len(build.outliers),
        }
        (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log("FINAL PERFORMANCE\n" + performance.to_string(index=False))
        log("END experiment_v1 completed successfully")
    except Exception as exc:
        failures.append({"time": datetime.now().isoformat(), "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        (output_dir / "fatal_errors.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
