"""v5 repeated nested-CV orchestration for Pearson-screened factor models."""

from __future__ import annotations

import copy
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.build_features import build_feature_tables
from experiment.factor_modeling import fit_descriptive_factor_model, nested_cv_factor_model
from experiment.feature_audit import run_feature_accuracy_audit
from experiment.modeling import (
    bootstrap_performance,
    environment_info,
    paired_auc_vs_prevalence,
    prevalence_predictions,
    save_performance_plots,
)
from experiment.repeated_runner import (
    _add_repeat_columns,
    _mean_oof_predictions,
    _repeat_performance,
    hierarchical_paired_auc,
    hierarchical_repeated_performance,
)


MODELS = ("elastic_net", "xgboost", "lightgbm")
FEATURE_SET = "E3_combined"
WARNING_COLUMNS = ["experiment_id", "model", "feature_set", "outer_fold", "category", "message"]


def _validate_factor_config(config: dict) -> None:
    threshold = float(config["factor_pearson_threshold"])
    if not 0 < threshold < 1:
        raise ValueError("factor_pearson_threshold must be between 0 and 1")
    counts = [int(value) for value in config["factor_counts"]]
    if not counts or min(counts) < 1 or max(counts) > 10:
        raise ValueError("factor_counts must contain values from 1 through 10")
    for model in MODELS:
        if not config["factor_model_candidates"].get(model):
            raise ValueError(f"Missing factor_model_candidates for {model}")


def _selection_stability(selected: pd.DataFrame, total_splits: int) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    result = selected.groupby(
        ["experiment_id", "model", "feature_set", "source_feature"], as_index=False
    ).agg(
        selected_outer_splits=("repeat_fold_id", "nunique"),
        mean_train_abs_pearson=("train_fold_abs_pearson", "mean"),
        sd_train_abs_pearson=("train_fold_abs_pearson", "std"),
        mean_train_signed_pearson=("train_fold_signed_pearson", "mean"),
        mean_factor_uniqueness=("factor_uniqueness", "mean"),
    )
    result["selection_frequency"] = result["selected_outer_splits"] / int(total_splits)
    return result.sort_values(
        ["experiment_id", "selection_frequency", "mean_train_abs_pearson"],
        ascending=[True, False, False],
    )


def run_factor_experiment(
    *,
    project_dir: Path,
    feature_dir: Path,
    label_file: Path,
    output_dir: Path,
    config_file: Path,
    feature_version: str,
    repeat_seeds: list[int],
    config_overrides: dict | None = None,
    require_nnunet_environment: bool = True,
    resume: bool = True,
) -> None:
    project_dir = project_dir.resolve()
    feature_dir = feature_dir.resolve()
    label_file = label_file.resolve()
    output_dir = output_dir.resolve()
    config_file = config_file.resolve()
    if not repeat_seeds or len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("repeat_seeds must be a nonempty unique list")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_file.read_text(encoding="utf-8"))
    for key, value in (config_overrides or {}).items():
        if key in {"factor_model_candidates"}:
            merged = copy.deepcopy(config[key]); merged.update(value); config[key] = merged
        else:
            config[key] = value
    _validate_factor_config(config)
    requested_manifest = {
        "pipeline_version": "v5",
        "feature_dir": str(feature_dir),
        "label_file": str(label_file),
        "feature_version": feature_version,
        "feature_set": FEATURE_SET,
        "models": list(MODELS),
        "pearson_threshold": float(config["factor_pearson_threshold"]),
        "factor_counts": [int(value) for value in config["factor_counts"]],
        "repeat_seeds": [int(value) for value in repeat_seeds],
        "outer_folds": int(config["outer_folds"]),
        "inner_folds": int(config["inner_folds"]),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in requested_manifest}
        if comparable != requested_manifest:
            raise ValueError("Existing v5 output manifest does not match the requested protocol")
    else:
        manifest_path.write_text(json.dumps(requested_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment_info(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = output_dir / "experiment_log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        log(f"START v5 Pearson-factor repeated nested CV; executable={sys.executable}")
        if require_nnunet_environment and Path(sys.prefix).name.lower() != "nnunet-master":
            raise RuntimeError("v5 must run in the nnUNet-master Conda environment")
        audit = run_feature_accuracy_audit(feature_dir, label_file, feature_version, output_dir)
        audit_errors = int((audit["severity"] == "error").sum())
        audit_warnings = int((audit["severity"] == "warning").sum())
        if audit_errors:
            raise ValueError("Feature audit found error-level violations")
        build = build_feature_tables(feature_dir, label_file, output_dir, feature_version)
        table = build.tables[FEATURE_SET]
        requested_manifest["patients"] = int(len(table))
        requested_manifest["input_features"] = int(len(table.columns) - 2)
        requested_manifest["label_counts"] = {
            str(key): int(value) for key, value in table["label"].value_counts().sort_index().items()
        }
        manifest_path.write_text(json.dumps(requested_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log(
            f"Cohort={len(table)}; input_features={len(table.columns)-2}; "
            f"threshold={config['factor_pearson_threshold']}; repeats={len(repeat_seeds)}"
        )

        all_predictions, all_folds, all_importances = [], [], []
        all_loadings, all_selected, all_issues, all_assignments = [], [], [], []
        repeats_root = output_dir / "repeats"
        repeats_root.mkdir(exist_ok=True)
        for repeat_index, seed in enumerate(repeat_seeds, start=1):
            repeat_dir = repeats_root / f"repeat_{repeat_index:02d}_seed_{seed}"
            repeat_dir.mkdir(exist_ok=True)
            completion_file = repeat_dir / "repeat_summary.json"
            if resume and completion_file.exists():
                completion = json.loads(completion_file.read_text(encoding="utf-8"))
                if completion.get("status") == "completed":
                    log(f"REPEAT {repeat_index}/{len(repeat_seeds)} seed={seed}: load checkpoint")
                    all_predictions.append(pd.read_csv(repeat_dir / "oof_predictions.csv"))
                    all_folds.append(pd.read_csv(repeat_dir / "outer_fold_performance.csv"))
                    all_importances.append(pd.read_csv(repeat_dir / "factor_permutation_importance.csv"))
                    all_loadings.append(pd.read_csv(repeat_dir / "factor_loadings.csv"))
                    all_selected.append(pd.read_csv(repeat_dir / "selected_original_features.csv"))
                    all_issues.append(pd.read_csv(repeat_dir / "runtime_warnings.csv"))
                    all_assignments.append(pd.read_csv(repeat_dir / "outer_fold_assignments.csv"))
                    continue

            log(f"REPEAT {repeat_index}/{len(repeat_seeds)} seed={seed}: begin")
            repeat_config = copy.deepcopy(config)
            repeat_config["random_seed"] = int(seed)
            from experiment.modeling import make_outer_fold_assignments

            assignments = make_outer_fold_assignments(table, repeat_config)
            assignments = _add_repeat_columns(assignments, repeat_index, seed)
            predictions, fold_rows, importance_rows = [], [], []
            loading_rows, selected_rows, issues = [], [], []
            baseline_predictions, baseline_folds = prevalence_predictions(
                table, assignments[["patient_id", "label", "outer_fold"]]
            )
            predictions.append(_add_repeat_columns(baseline_predictions, repeat_index, seed))
            fold_rows.append(_add_repeat_columns(baseline_folds, repeat_index, seed))
            for model in MODELS:
                log(f"REPEAT {repeat_index}/{len(repeat_seeds)}: {model} + factors")
                pred, fold, importance, loadings, selected, model_issues = nested_cv_factor_model(
                    table,
                    FEATURE_SET,
                    model,
                    repeat_config,
                    assignments[["patient_id", "label", "outer_fold"]],
                    log,
                )
                predictions.append(_add_repeat_columns(pred, repeat_index, seed))
                fold_rows.append(_add_repeat_columns(fold, repeat_index, seed))
                importance_rows.append(_add_repeat_columns(importance, repeat_index, seed))
                loading_rows.append(_add_repeat_columns(loadings, repeat_index, seed))
                selected_rows.append(_add_repeat_columns(selected, repeat_index, seed))
                issues.extend(model_issues)

            repeat_predictions = pd.concat(predictions, ignore_index=True)
            repeat_folds = pd.concat(fold_rows, ignore_index=True)
            repeat_importance = pd.concat(importance_rows, ignore_index=True)
            repeat_loadings = pd.concat(loading_rows, ignore_index=True)
            repeat_selected = pd.concat(selected_rows, ignore_index=True)
            repeat_issues = pd.DataFrame(issues, columns=WARNING_COLUMNS)
            repeat_predictions.to_csv(repeat_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
            repeat_folds.to_csv(repeat_dir / "outer_fold_performance.csv", index=False, encoding="utf-8-sig")
            repeat_importance.to_csv(
                repeat_dir / "factor_permutation_importance.csv", index=False, encoding="utf-8-sig"
            )
            repeat_loadings.to_csv(repeat_dir / "factor_loadings.csv", index=False, encoding="utf-8-sig")
            repeat_selected.to_csv(
                repeat_dir / "selected_original_features.csv", index=False, encoding="utf-8-sig"
            )
            repeat_issues.to_csv(repeat_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig")
            assignments.to_csv(repeat_dir / "outer_fold_assignments.csv", index=False, encoding="utf-8-sig")
            completion_file.write_text(json.dumps({
                "status": "completed",
                "repeat_index": repeat_index,
                "seed": int(seed),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "warning_count": int(len(repeat_issues)),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            all_predictions.append(repeat_predictions)
            all_folds.append(repeat_folds)
            all_importances.append(repeat_importance)
            all_loadings.append(repeat_loadings)
            all_selected.append(repeat_selected)
            all_issues.append(repeat_issues)
            all_assignments.append(assignments)
            log(f"REPEAT {repeat_index}/{len(repeat_seeds)} seed={seed}: completed")

        predictions = pd.concat(all_predictions, ignore_index=True)
        folds = pd.concat(all_folds, ignore_index=True)
        importances = pd.concat(all_importances, ignore_index=True)
        loadings = pd.concat(all_loadings, ignore_index=True)
        selected = pd.concat(all_selected, ignore_index=True)
        issues = pd.concat(all_issues, ignore_index=True)
        assignments = pd.concat(all_assignments, ignore_index=True)
        repeat_performance = _repeat_performance(predictions)
        repeated_performance = hierarchical_repeated_performance(
            predictions, config["bootstrap_iterations"], config["random_seed"]
        )
        repeated_comparisons = hierarchical_paired_auc(
            predictions, config["bootstrap_iterations"], config["random_seed"]
        )
        mean_predictions = _mean_oof_predictions(predictions)
        mean_performance = bootstrap_performance(
            mean_predictions, config["bootstrap_iterations"], config["random_seed"] + 1
        )
        mean_vs_baseline = paired_auc_vs_prevalence(
            mean_predictions, config["bootstrap_iterations"], config["random_seed"] + 1
        )
        total_splits = len(repeat_seeds) * int(config["outer_folds"])
        selection_stability = _selection_stability(selected, total_splits)

        model_folds = folds[folds["model"] != "prevalence"].copy()
        descriptive_factor_count = int(model_folds["actual_factor_count"].mode().iloc[0])
        descriptive_scores, descriptive_loadings, _ = fit_descriptive_factor_model(
            table, config, descriptive_factor_count
        )

        predictions.to_csv(output_dir / "all_repeated_oof_predictions.csv", index=False, encoding="utf-8-sig")
        folds.to_csv(output_dir / "all_outer_fold_performance.csv", index=False, encoding="utf-8-sig")
        assignments.to_csv(output_dir / "all_outer_fold_assignments.csv", index=False, encoding="utf-8-sig")
        importances.to_csv(output_dir / "all_factor_permutation_importance.csv", index=False, encoding="utf-8-sig")
        loadings.to_csv(output_dir / "all_outer_fold_factor_loadings.csv", index=False, encoding="utf-8-sig")
        selected.to_csv(output_dir / "all_outer_fold_selected_features.csv", index=False, encoding="utf-8-sig")
        selection_stability.to_csv(
            output_dir / "source_feature_selection_stability.csv", index=False, encoding="utf-8-sig"
        )
        issues.to_csv(output_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig")
        repeat_performance.to_csv(output_dir / "performance_each_repeat.csv", index=False, encoding="utf-8-sig")
        repeated_performance.to_csv(
            output_dir / "repeated_nested_cv_performance.csv", index=False, encoding="utf-8-sig"
        )
        repeated_comparisons.to_csv(
            output_dir / "repeated_paired_auc_comparisons.csv", index=False, encoding="utf-8-sig"
        )
        mean_predictions.to_csv(output_dir / "mean_oof_predictions.csv", index=False, encoding="utf-8-sig")
        mean_performance.to_csv(output_dir / "mean_oof_performance.csv", index=False, encoding="utf-8-sig")
        mean_vs_baseline.to_csv(
            output_dir / "mean_oof_paired_vs_prevalence.csv", index=False, encoding="utf-8-sig"
        )
        descriptive_scores.to_csv(
            output_dir / "descriptive_full_cohort_factor_scores_NOT_FOR_PERFORMANCE.csv",
            index=False,
            encoding="utf-8-sig",
        )
        descriptive_loadings.to_csv(
            output_dir / "descriptive_full_cohort_factor_loadings_NOT_FOR_PERFORMANCE.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_performance_plots(mean_predictions, mean_performance, output_dir)

        summary = {
            "status": "completed",
            "pipeline_version": "v5",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "patients": int(len(table)),
            "input_features": int(len(table.columns) - 2),
            "pearson_threshold": float(config["factor_pearson_threshold"]),
            "repeat_seeds": [int(value) for value in repeat_seeds],
            "outer_folds_per_repeat": int(config["outer_folds"]),
            "total_outer_validation_splits_per_model": int(total_splits),
            "models": list(MODELS),
            "feature_set": FEATURE_SET,
            "descriptive_full_cohort_factor_count": descriptive_factor_count,
            "descriptive_outputs_are_not_performance_estimates": True,
            "feature_audit_errors": audit_errors,
            "feature_audit_warnings": audit_warnings,
            "runtime_warning_count": int(len(issues)),
            "primary_repeated_performance": repeated_performance.to_dict(orient="records"),
            "secondary_mean_oof_performance": mean_performance.to_dict(orient="records"),
        }
        (output_dir / "experiment_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log("PRIMARY REPEATED PERFORMANCE\n" + repeated_performance.to_string(index=False))
        log("END v5 Pearson-factor repeated nested CV completed successfully")
    except Exception as exc:
        failure = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "fatal_errors.json").write_text(
            json.dumps([failure], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise
