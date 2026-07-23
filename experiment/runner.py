"""Orchestration for data validation and nested-CV experiments."""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiment.build_features import build_feature_tables
from experiment.feature_audit import run_feature_accuracy_audit
from experiment.modeling import (
    MODEL_LABELS,
    bootstrap_performance,
    environment_info,
    importance_summary,
    make_outer_fold_assignments,
    nested_cv_model,
    paired_auc_vs_prevalence,
    prevalence_predictions,
    save_performance_plots,
)


def _merge_config(base: dict, overrides: dict | None) -> dict:
    result = dict(base)
    for key, value in (overrides or {}).items():
        if key == "candidates":
            merged = {name: list(rows) for name, rows in result["candidates"].items()}
            merged.update(value)
            result["candidates"] = merged
        else:
            result[key] = value
    return result


def _validate_config(config: dict, models: list[str], feature_sets: list[str]) -> None:
    unknown_models = sorted(set(models) - set(MODEL_LABELS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    unknown_sets = sorted(set(feature_sets) - {"E1_3d_level3", "E2_2d", "E3_combined"})
    if unknown_sets:
        raise ValueError(f"Unknown feature sets: {unknown_sets}")
    for model in models:
        if model not in config["candidates"] or not config["candidates"][model]:
            raise ValueError(f"No hyperparameter candidates configured for {model}")
    if not 0 <= config["winsor_lower_quantile"] < config["winsor_upper_quantile"] <= 1:
        raise ValueError("Winsor quantiles are invalid")
    if config["outer_folds"] < 2 or config["inner_folds"] < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2")


def run_experiment(
    *,
    project_dir: Path,
    feature_dir: Path,
    label_file: Path,
    output_dir: Path,
    config_file: Path,
    feature_version: str = "v7",
    models: list[str] | None = None,
    feature_sets: list[str] | None = None,
    config_overrides: dict | None = None,
    reuse_outer_folds: Path | None = None,
    validate_only: bool = False,
    require_nnunet_environment: bool = True,
    allow_existing_output: bool = False,
) -> None:
    project_dir = project_dir.resolve()
    feature_dir = feature_dir.resolve()
    label_file = label_file.resolve()
    output_dir = output_dir.resolve()
    config_file = config_file.resolve()
    reuse_outer_folds = reuse_outer_folds.resolve() if reuse_outer_folds else None

    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing_output:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new OUTPUT_DIR or set ALLOW_EXISTING_OUTPUT=True."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _merge_config(json.loads(config_file.read_text(encoding="utf-8")), config_overrides)
    selected_models = models or list(config["models"])
    selected_feature_sets = feature_sets or list(config["feature_sets"])
    _validate_config(config, selected_models, selected_feature_sets)
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment_info(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = output_dir / "experiment_log.txt"

    def log(message: str):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        log(f"START current pipeline; executable={sys.executable}")
        if require_nnunet_environment and Path(sys.prefix).name.lower() != "nnunet-master":
            raise RuntimeError(
                "This pipeline must run in the nnUNet-master Conda environment. "
                "Select C:\\ProgramData\\anaconda3\\envs\\nnUNet-master\\python.exe in PyCharm."
            )
        log(f"Feature source={feature_dir}; labels={label_file}; version={feature_version}")

        log("Run dynamic feature accuracy audit")
        accuracy_audit = run_feature_accuracy_audit(feature_dir, label_file, feature_version, output_dir)
        error_count = int((accuracy_audit["severity"] == "error").sum())
        warning_count = int((accuracy_audit["severity"] == "warning").sum())
        log(f"Feature audit: errors={error_count}, warnings={warning_count}")
        if error_count:
            raise ValueError("Feature audit found error-level violations; see feature_accuracy_audit.csv")

        log("Build deterministic patient-level tables from labeled patient_id values")
        build = build_feature_tables(feature_dir, label_file, output_dir, feature_version)
        cohort_size = len(build.labels)
        label_counts = build.labels["label"].value_counts().sort_index().to_dict()
        log(f"Dynamic cohort: patients={cohort_size}; label_counts={label_counts}")

        assignments = make_outer_fold_assignments(
            build.tables["E1_3d_level3"], config, reuse_file=reuse_outer_folds
        )
        assignments.to_csv(output_dir / "outer_fold_assignments.csv", index=False, encoding="utf-8-sig")
        log("Saved patient_id-based outer-fold assignments for exact future reuse")

        manifest = {
            "project_dir": str(project_dir), "feature_dir": str(feature_dir),
            "label_file": str(label_file), "feature_version": feature_version,
            "patients": cohort_size, "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "models": selected_models, "feature_sets": selected_feature_sets,
            "reuse_outer_folds": str(reuse_outer_folds) if reuse_outer_folds else None,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if validate_only:
            summary = {
                "status": "validation_completed", "completed_at": datetime.now().isoformat(timespec="seconds"),
                "patients": cohort_size, "label_counts": manifest["label_counts"],
                "feature_audit_errors": error_count, "feature_audit_warnings": warning_count,
                "message": "No model was fitted because VALIDATE_ONLY=True.",
            }
            (output_dir / "experiment_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log("END validation-only run completed; no model fitted")
            return

        baseline_predictions, baseline_folds = prevalence_predictions(
            build.tables["E1_3d_level3"], assignments
        )
        all_predictions = [baseline_predictions]
        all_folds = [baseline_folds]
        all_importance = []
        issues: list[dict] = []

        for feature_set in selected_feature_sets:
            for model in selected_models:
                log(f"BEGIN {model} on {feature_set}")
                predictions, folds, importance, model_issues = nested_cv_model(
                    build.tables[feature_set], feature_set, model, config, assignments, log
                )
                all_predictions.append(predictions)
                all_folds.append(folds)
                all_importance.append(importance)
                issues.extend(model_issues)

        predictions = pd.concat(all_predictions, ignore_index=True)
        fold_performance = pd.concat(all_folds, ignore_index=True)
        importance_rows = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
        performance = bootstrap_performance(predictions, config["bootstrap_iterations"], config["random_seed"])
        comparisons = paired_auc_vs_prevalence(
            predictions, config["bootstrap_iterations"], config["random_seed"]
        )
        summary_importance = importance_summary(importance_rows, config["outer_folds"])

        predictions.to_csv(output_dir / "nested_cv_predictions.csv", index=False, encoding="utf-8-sig")
        fold_performance.to_csv(output_dir / "outer_fold_performance.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output_dir / "model_performance.csv", index=False, encoding="utf-8-sig")
        comparisons.to_csv(output_dir / "paired_auc_vs_prevalence.csv", index=False, encoding="utf-8-sig")
        importance_rows.to_csv(output_dir / "permutation_importance_each_fold.csv", index=False, encoding="utf-8-sig")
        summary_importance.to_csv(output_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            issues,
            columns=["experiment_id", "model", "feature_set", "outer_fold", "category", "message"],
        ).to_csv(output_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig")
        save_performance_plots(predictions, performance, output_dir)

        payload = {
            "status": "completed", "completed_at": datetime.now().isoformat(timespec="seconds"),
            "python_executable": sys.executable, "patients": cohort_size,
            "label_counts": manifest["label_counts"], "models_run": selected_models,
            "feature_sets_run": selected_feature_sets,
            "performance": performance.to_dict(orient="records"),
            "feature_audit_errors": error_count, "feature_audit_warnings": warning_count,
            "runtime_warning_count": len(issues),
        }
        (output_dir / "experiment_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log("FINAL PERFORMANCE\n" + performance.to_string(index=False))
        log("END experiment completed successfully")
    except Exception as exc:
        failure = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "exception": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "fatal_errors.json").write_text(
            json.dumps([failure], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise
