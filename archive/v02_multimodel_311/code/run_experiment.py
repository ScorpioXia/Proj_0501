"""Run all stage-2 models from the retained 311-patient feature files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DEFAULT = Path(__file__).resolve().parents[1]
if str(PROJECT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(PROJECT_DEFAULT))

from experiment_v1.build_features import build_feature_tables
from experiment_v2.feature_accuracy_audit import run_feature_accuracy_audit
from experiment_v2.modeling_v2 import (
    bootstrap_performance,
    environment_info,
    importance_summary,
    nested_cv_model,
    paired_auc_vs_references,
    prevalence_predictions,
    save_performance_plots,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("config.json"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset, e.g. xgboost svm_rbf")
    parser.add_argument("--feature-sets", nargs="*", default=None, help="Optional subset, e.g. E1_3d_level3 E3_combined")
    parser.add_argument("--feature-dir", type=Path, default=None, help="Directory containing feature CSV files")
    parser.add_argument("--label-file", type=Path, default=None, help="Label file (CSV or Excel) with patient_id and label columns")
    parser.add_argument("--feature-version", type=str, default="v6", help="Feature file version suffix, e.g. v6 or v7")
    parser.add_argument("--expected-patients", type=str, default="311", help="Expected number of patients, or 'none' to skip check")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip nnUNet-master environment check")
    return parser.parse_args()


def load_v1_references(project_dir: Path):
    path = project_dir / "experiment_results_v1" / "nested_cv_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    old = pd.read_csv(path)
    old = old[old.feature_set.isin(["E1_3d_level3", "E3_combined"])].copy()
    if old.empty:
        return old
    old["experiment_id"] = "ElasticNet__" + old["feature_set"]
    old["model"] = "elastic_net_v1_reference"
    return old[["experiment_id", "model", "feature_set", "outer_fold", "patient_id", "true_label", "predicted_probability", "predicted_label_0_5"]]


def main():
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = (args.output_dir or project_dir / "experiment_results_v2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = args.models or config["models"]
    feature_sets = args.feature_sets or config["feature_sets"]
    shutil.copy2(args.config, output_dir / "config_used.json")
    (output_dir / "environment.json").write_text(json.dumps(environment_info(), ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = output_dir / "experiment_log.txt"

    def log(message):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        log(f"START experiment_v2; executable={sys.executable}")
        if not args.skip_env_check and "nnUNet-master".lower() not in sys.executable.lower():
            raise RuntimeError("Stage 2 must run in the nnUNet-master Conda environment (use --skip-env-check to bypass)")

        if args.feature_dir or args.label_file or args.feature_version != "v6" or args.expected_patients is not None:
            log(f"Custom dataset: feature_dir={args.feature_dir}, label_file={args.label_file}, "
                f"feature_version={args.feature_version}, expected_patients={args.expected_patients}")

        log("Run retained-feature numerical accuracy audit")
        if args.feature_dir or args.label_file:
            log("Skipping feature accuracy audit for custom dataset (audit is calibrated for 311-patient v6 features)")
            accuracy_audit = pd.DataFrame(columns=["severity", "stage", "file", "feature", "issue", "action"])
        else:
            accuracy_audit = run_feature_accuracy_audit(project_dir, output_dir)
            log(f"Feature audit: errors={(accuracy_audit.severity == 'error').sum()}, warnings={(accuracy_audit.severity == 'warning').sum()}")

        log("Rebuild deterministic patient-level feature tables")
        expected_patients = None if args.expected_patients.lower() == "none" else int(args.expected_patients)
        build = build_feature_tables(
            project_dir, output_dir,
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            feature_version=args.feature_version,
            expected_patients=expected_patients,
        )
        baseline_pred, baseline_folds = prevalence_predictions(build.tables["E1_3d_level3"], config)
        all_pred, all_folds, all_importance, issues = [baseline_pred], [baseline_folds], [], []

        v1_ref = load_v1_references(project_dir)
        if not v1_ref.empty:
            all_pred.append(v1_ref)
            log("Loaded v1 ElasticNet OOF predictions as historical references; not refitted in stage 2")

        for feature_set in feature_sets:
            if feature_set not in build.tables:
                raise ValueError(f"Unknown feature set {feature_set}")
            for model in models:
                log(f"BEGIN {model} on {feature_set}")
                pred, folds, importance, model_issues = nested_cv_model(
                    build.tables[feature_set], feature_set, model, config, output_dir, log
                )
                all_pred.append(pred); all_folds.append(folds); all_importance.append(importance); issues.extend(model_issues)

        predictions = pd.concat(all_pred, ignore_index=True)
        fold_performance = pd.concat(all_folds, ignore_index=True)
        importance_rows = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
        performance = bootstrap_performance(predictions, config["bootstrap_iterations"], config["random_seed"])
        if args.feature_dir or args.label_file:
            comparisons = pd.DataFrame(columns=["reference", "comparison", "auc_difference", "ci_low", "ci_high", "bootstrap_probability_gt_reference"])
            log("Skipping paired AUC vs v1 references for custom dataset")
        else:
            comparisons = paired_auc_vs_references(predictions, config["bootstrap_iterations"], config["random_seed"])
        summary_importance = importance_summary(importance_rows, config["outer_folds"])

        predictions.to_csv(output_dir / "nested_cv_predictions.csv", index=False, encoding="utf-8-sig")
        fold_performance.to_csv(output_dir / "outer_fold_performance.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output_dir / "model_performance.csv", index=False, encoding="utf-8-sig")
        comparisons.to_csv(output_dir / "paired_auc_comparisons.csv", index=False, encoding="utf-8-sig")
        importance_rows.to_csv(output_dir / "permutation_importance_each_fold.csv", index=False, encoding="utf-8-sig")
        summary_importance.to_csv(output_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(issues, columns=["experiment_id", "model", "feature_set", "outer_fold", "category", "message"]).to_csv(
            output_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig"
        )
        save_performance_plots(predictions, performance, output_dir)

        n_patients = len(build.tables[list(build.tables.keys())[0]])
        payload = {
            "status": "completed", "completed_at": datetime.now().isoformat(timespec="seconds"),
            "python_executable": sys.executable, "patients": n_patients,
            "models_run": models, "feature_sets_run": feature_sets,
            "performance": performance.to_dict(orient="records"),
            "feature_audit_errors": int((accuracy_audit.severity == "error").sum()),
            "feature_audit_warnings": int((accuracy_audit.severity == "warning").sum()),
            "runtime_warning_count": len(issues),
        }
        (output_dir / "experiment_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log("FINAL PERFORMANCE\n" + performance.to_string(index=False))
        log("END experiment_v2 completed successfully")
    except Exception as exc:
        failure = {
            "time": datetime.now().isoformat(timespec="seconds"), "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
        }
        (output_dir / "fatal_errors.json").write_text(json.dumps([failure], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
