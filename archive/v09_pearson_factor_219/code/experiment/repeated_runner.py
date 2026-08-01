"""Repeated nested-CV orchestration for the locked v4 candidate models."""

from __future__ import annotations

import copy
import json
import sys
import traceback
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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
from experiment.preprocessing import metric_row


def _core_probability_metrics(y_true, probability) -> dict[str, float]:
    """Only the three metrics needed by the repeated bootstrap."""
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def _validate_experiments(experiments: list[tuple[str, str]], config: dict) -> None:
    if not experiments:
        raise ValueError("At least one locked model/feature-set combination is required")
    if len(set(experiments)) != len(experiments):
        raise ValueError("Locked experiments contain duplicates")
    valid_sets = {"E1_3d_level3", "E2_2d", "E3_combined"}
    for model, feature_set in experiments:
        if model not in MODEL_LABELS:
            raise ValueError(f"Unknown model: {model}")
        if model not in config["candidates"] or not config["candidates"][model]:
            raise ValueError(f"No tuning candidates configured for {model}")
        if feature_set not in valid_sets:
            raise ValueError(f"Unknown feature set: {feature_set}")


def _add_repeat_columns(frame: pd.DataFrame, repeat_index: int, seed: int) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "repeat_index", repeat_index)
    result.insert(1, "repeat_seed", seed)
    if "outer_fold" in result.columns:
        result.insert(2, "repeat_fold_id", result["repeat_seed"].astype(str) + "__" + result["outer_fold"].astype(str))
    return result


def _repeat_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (repeat_index, seed, experiment_id), frame in predictions.groupby(
        ["repeat_index", "repeat_seed", "experiment_id"], sort=False
    ):
        values = metric_row(frame["true_label"], frame["predicted_probability"])
        rows.append({
            "repeat_index": int(repeat_index), "repeat_seed": int(seed),
            "experiment_id": experiment_id, "model": frame["model"].iloc[0],
            "feature_set": frame["feature_set"].iloc[0], **values,
        })
    return pd.DataFrame(rows)


def _prediction_matrices(predictions: pd.DataFrame, experiment_id: str):
    frame = predictions[predictions["experiment_id"] == experiment_id].copy()
    labels = frame.pivot(index="patient_id", columns="repeat_seed", values="true_label").sort_index()
    probabilities = frame.pivot(index="patient_id", columns="repeat_seed", values="predicted_probability").reindex(labels.index)
    if labels.isna().any().any() or probabilities.isna().any().any():
        raise ValueError(f"Incomplete repeated OOF matrix for {experiment_id}")
    if not labels.nunique(axis=1).eq(1).all():
        raise ValueError(f"Labels changed between repeats for {experiment_id}")
    return labels.iloc[:, 0].to_numpy(int), probabilities.to_numpy(float), labels.index


def hierarchical_repeated_performance(
    predictions: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    """Mean repeat metrics with patient-and-repeat hierarchical bootstrap CIs."""
    rng = np.random.default_rng(seed + 1701)
    repeat_metrics = _repeat_performance(predictions)
    rows = []
    for experiment_id in predictions["experiment_id"].drop_duplicates():
        y, matrix, _ = _prediction_matrices(predictions, experiment_id)
        observed = repeat_metrics[repeat_metrics["experiment_id"] == experiment_id]
        bootstrap = {key: [] for key in ("roc_auc", "pr_auc", "brier")}
        for _ in range(iterations):
            patient_index = rng.integers(0, len(y), len(y))
            if np.unique(y[patient_index]).size < 2:
                continue
            repeat_index = rng.integers(0, matrix.shape[1], matrix.shape[1])
            sampled_metrics = [
                _core_probability_metrics(y[patient_index], matrix[patient_index, column])
                for column in repeat_index
            ]
            for key in bootstrap:
                bootstrap[key].append(float(np.mean([item[key] for item in sampled_metrics])))
        first = predictions[predictions["experiment_id"] == experiment_id].iloc[0]
        row = {
            "experiment_id": experiment_id, "model": first["model"],
            "feature_set": first["feature_set"], "repeats": matrix.shape[1],
        }
        for key in ("roc_auc", "pr_auc", "brier"):
            values = observed[key].to_numpy(float)
            row[f"mean_{key}"] = float(values.mean())
            row[f"sd_{key}"] = float(values.std(ddof=1))
            row[f"min_{key}"] = float(values.min())
            row[f"max_{key}"] = float(values.max())
            row[f"{key}_hierarchical_ci_low"] = float(np.quantile(bootstrap[key], 0.025))
            row[f"{key}_hierarchical_ci_high"] = float(np.quantile(bootstrap[key], 0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False)


def hierarchical_paired_auc(
    predictions: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2701)
    experiment_ids = predictions["experiment_id"].drop_duplicates().tolist()
    baseline = "PrevalenceBaseline"
    pairs = [(baseline, item) for item in experiment_ids if item != baseline]
    candidates = [item for item in experiment_ids if item != baseline]
    pairs.extend(combinations(candidates, 2))
    rows = []
    for reference, comparison in pairs:
        y0, matrix0, ids0 = _prediction_matrices(predictions, reference)
        y1, matrix1, ids1 = _prediction_matrices(predictions, comparison)
        if not ids0.equals(ids1) or not np.array_equal(y0, y1):
            raise ValueError(f"Repeated OOF cohorts differ: {reference} versus {comparison}")
        repeat_differences = []
        for column in range(matrix0.shape[1]):
            repeat_differences.append(
                roc_auc_score(y0, matrix1[:, column])
                - roc_auc_score(y0, matrix0[:, column])
            )
        bootstrap = []
        for _ in range(iterations):
            patient_index = rng.integers(0, len(y0), len(y0))
            if np.unique(y0[patient_index]).size < 2:
                continue
            repeat_index = rng.integers(0, matrix0.shape[1], matrix0.shape[1])
            differences = []
            for column in repeat_index:
                differences.append(
                    roc_auc_score(y0[patient_index], matrix1[patient_index, column])
                    - roc_auc_score(y0[patient_index], matrix0[patient_index, column])
                )
            bootstrap.append(float(np.mean(differences)))
        rows.append({
            "reference": reference, "comparison": comparison,
            "mean_repeat_auc_difference": float(np.mean(repeat_differences)),
            "sd_repeat_auc_difference": float(np.std(repeat_differences, ddof=1)),
            "hierarchical_ci_low": float(np.quantile(bootstrap, 0.025)),
            "hierarchical_ci_high": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_probability_gt_reference": float(np.mean(np.asarray(bootstrap) > 0)),
        })
    return pd.DataFrame(rows)


def _mean_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = predictions.groupby(
        ["experiment_id", "model", "feature_set", "patient_id", "true_label"],
        as_index=False,
    )["predicted_probability"].agg(["mean", "std", "min", "max"]).reset_index()
    grouped = grouped.rename(columns={
        "mean": "predicted_probability", "std": "prediction_sd_across_repeats",
        "min": "prediction_min_across_repeats", "max": "prediction_max_across_repeats",
    })
    grouped["predicted_label_0_5"] = (grouped["predicted_probability"] >= 0.5).astype(int)
    return grouped


def run_repeated_experiment(
    *,
    project_dir: Path,
    feature_dir: Path,
    label_file: Path,
    output_dir: Path,
    config_file: Path,
    feature_version: str,
    experiments: list[tuple[str, str]],
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
    if len(repeat_seeds) != len(set(repeat_seeds)) or not repeat_seeds:
        raise ValueError("repeat_seeds must be a nonempty unique list")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_file.read_text(encoding="utf-8"))
    for key, value in (config_overrides or {}).items():
        if key == "candidates":
            merged = copy.deepcopy(config["candidates"]); merged.update(value); config["candidates"] = merged
        else:
            config[key] = value
    _validate_experiments(experiments, config)
    if config["outer_folds"] != 5:
        raise ValueError("v4 protocol is locked to five outer folds")

    requested_manifest = {
        "pipeline_version": "v4", "feature_dir": str(feature_dir),
        "label_file": str(label_file), "feature_version": feature_version,
        "experiments": [{"model": model, "feature_set": feature_set} for model, feature_set in experiments],
        "repeat_seeds": repeat_seeds, "outer_folds": config["outer_folds"],
        "inner_folds": config["inner_folds"],
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in requested_manifest}
        if comparable != requested_manifest:
            raise ValueError("Existing v4 output manifest does not match the requested protocol")
    else:
        manifest_path.write_text(json.dumps(requested_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(environment_info(), ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = output_dir / "experiment_log.txt"

    def log(message: str):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        log(f"START v4 repeated nested CV; executable={sys.executable}")
        if require_nnunet_environment and Path(sys.prefix).name.lower() != "nnunet-master":
            raise RuntimeError("v4 must run in the nnUNet-master Conda environment")
        accuracy = run_feature_accuracy_audit(feature_dir, label_file, feature_version, output_dir)
        errors = int((accuracy["severity"] == "error").sum())
        warnings_count = int((accuracy["severity"] == "warning").sum())
        if errors:
            raise ValueError("Feature audit found error-level violations")
        build = build_feature_tables(feature_dir, label_file, output_dir, feature_version)
        requested_manifest["patients"] = len(build.labels)
        requested_manifest["label_counts"] = {
            str(key): int(value) for key, value in build.labels["label"].value_counts().sort_index().items()
        }
        manifest_path.write_text(json.dumps(requested_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Cohort={len(build.labels)}; repeats={len(repeat_seeds)}; locked experiments={experiments}")

        all_predictions, all_folds, all_importances, all_issues, all_assignments = [], [], [], [], []
        repeats_root = output_dir / "repeats"
        repeats_root.mkdir(exist_ok=True)
        for repeat_index, seed in enumerate(repeat_seeds, start=1):
            repeat_dir = repeats_root / f"repeat_{repeat_index:02d}_seed_{seed}"
            repeat_dir.mkdir(exist_ok=True)
            completion_file = repeat_dir / "repeat_summary.json"
            if resume and completion_file.exists():
                completion = json.loads(completion_file.read_text(encoding="utf-8"))
                if completion.get("status") == "completed":
                    log(f"REPEAT {repeat_index}/{len(repeat_seeds)} seed={seed}: load completed checkpoint")
                    all_predictions.append(pd.read_csv(repeat_dir / "oof_predictions.csv"))
                    all_folds.append(pd.read_csv(repeat_dir / "outer_fold_performance.csv"))
                    all_importances.append(pd.read_csv(repeat_dir / "permutation_importance.csv"))
                    issue_file = repeat_dir / "runtime_warnings.csv"
                    if issue_file.exists():
                        all_issues.append(pd.read_csv(issue_file))
                    all_assignments.append(pd.read_csv(repeat_dir / "outer_fold_assignments.csv"))
                    continue

            log(f"REPEAT {repeat_index}/{len(repeat_seeds)} seed={seed}: begin")
            repeat_config = copy.deepcopy(config)
            repeat_config["random_seed"] = int(seed)
            assignments = make_outer_fold_assignments(build.tables["E1_3d_level3"], repeat_config)
            assignments = _add_repeat_columns(assignments, repeat_index, seed)
            predictions, fold_rows, importance_rows, issues = [], [], [], []
            baseline_predictions, baseline_folds = prevalence_predictions(
                build.tables["E1_3d_level3"], assignments[["patient_id", "label", "outer_fold"]]
            )
            predictions.append(_add_repeat_columns(baseline_predictions, repeat_index, seed))
            fold_rows.append(_add_repeat_columns(baseline_folds, repeat_index, seed))
            for model, feature_set in experiments:
                log(f"REPEAT {repeat_index}/{len(repeat_seeds)}: {model} + {feature_set}")
                pred, fold, imp, model_issues = nested_cv_model(
                    build.tables[feature_set], feature_set, model, repeat_config,
                    assignments[["patient_id", "label", "outer_fold"]], log,
                )
                predictions.append(_add_repeat_columns(pred, repeat_index, seed))
                fold_rows.append(_add_repeat_columns(fold, repeat_index, seed))
                importance_rows.append(_add_repeat_columns(imp, repeat_index, seed))
                issues.extend(model_issues)

            repeat_predictions = pd.concat(predictions, ignore_index=True)
            repeat_folds = pd.concat(fold_rows, ignore_index=True)
            repeat_importance = pd.concat(importance_rows, ignore_index=True)
            repeat_issues = pd.DataFrame(
                issues, columns=["experiment_id", "model", "feature_set", "outer_fold", "category", "message"]
            )
            repeat_predictions.to_csv(repeat_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
            repeat_folds.to_csv(repeat_dir / "outer_fold_performance.csv", index=False, encoding="utf-8-sig")
            repeat_importance.to_csv(repeat_dir / "permutation_importance.csv", index=False, encoding="utf-8-sig")
            repeat_issues.to_csv(repeat_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig")
            assignments.to_csv(repeat_dir / "outer_fold_assignments.csv", index=False, encoding="utf-8-sig")
            completion_file.write_text(json.dumps({
                "status": "completed", "repeat_index": repeat_index, "seed": seed,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "warning_count": len(repeat_issues),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            all_predictions.append(repeat_predictions); all_folds.append(repeat_folds)
            all_importances.append(repeat_importance); all_issues.append(repeat_issues)
            all_assignments.append(assignments)
            log(f"REPEAT {repeat_index}/{len(repeat_seeds)} seed={seed}: completed")

        predictions = pd.concat(all_predictions, ignore_index=True)
        folds = pd.concat(all_folds, ignore_index=True)
        importances = pd.concat(all_importances, ignore_index=True)
        issues = pd.concat(all_issues, ignore_index=True) if all_issues else pd.DataFrame()
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
        stability = importance_summary(importances, len(repeat_seeds) * config["outer_folds"])

        predictions.to_csv(output_dir / "all_repeated_oof_predictions.csv", index=False, encoding="utf-8-sig")
        folds.to_csv(output_dir / "all_outer_fold_performance.csv", index=False, encoding="utf-8-sig")
        assignments.to_csv(output_dir / "all_outer_fold_assignments.csv", index=False, encoding="utf-8-sig")
        importances.to_csv(output_dir / "all_permutation_importance.csv", index=False, encoding="utf-8-sig")
        issues.to_csv(output_dir / "runtime_warnings.csv", index=False, encoding="utf-8-sig")
        repeat_performance.to_csv(output_dir / "performance_each_repeat.csv", index=False, encoding="utf-8-sig")
        repeated_performance.to_csv(output_dir / "repeated_nested_cv_performance.csv", index=False, encoding="utf-8-sig")
        repeated_comparisons.to_csv(output_dir / "repeated_paired_auc_comparisons.csv", index=False, encoding="utf-8-sig")
        mean_predictions.to_csv(output_dir / "mean_oof_predictions.csv", index=False, encoding="utf-8-sig")
        mean_performance.to_csv(output_dir / "mean_oof_performance.csv", index=False, encoding="utf-8-sig")
        mean_vs_baseline.to_csv(output_dir / "mean_oof_paired_vs_prevalence.csv", index=False, encoding="utf-8-sig")
        stability.to_csv(output_dir / "feature_stability_across_50_folds.csv", index=False, encoding="utf-8-sig")
        save_performance_plots(mean_predictions, mean_performance, output_dir)

        summary = {
            "status": "completed", "pipeline_version": "v4",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "patients": len(build.labels), "repeat_seeds": repeat_seeds,
            "outer_folds_per_repeat": config["outer_folds"],
            "total_outer_validation_splits_per_model": len(repeat_seeds) * config["outer_folds"],
            "locked_experiments": requested_manifest["experiments"],
            "feature_audit_errors": errors, "feature_audit_warnings": warnings_count,
            "runtime_warning_count": len(issues),
            "primary_repeated_performance": repeated_performance.to_dict(orient="records"),
            "secondary_mean_oof_performance": mean_performance.to_dict(orient="records"),
        }
        (output_dir / "experiment_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log("PRIMARY REPEATED PERFORMANCE\n" + repeated_performance.to_string(index=False))
        log("END v4 repeated nested CV completed successfully")
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
