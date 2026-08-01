"""Reproduce and audit a Pearson-screened six-factor modeling workflow.

The experiment deliberately reports two versions:

1. ``optimistic_global`` fits outcome-guided Pearson screening and factor
   analysis once on the full cohort before cross-validation.  This mirrors a
   common radiomics workflow but leaks outcome information and is not a valid
   generalization estimate.
2. ``nested_train_only`` refits imputation, clipping, variance filtering,
   Pearson screening, scaling, factor analysis, and model tuning using only the
   current training partition.

The contrast quantifies how much the apparently successful workflow can be
inflated by analysis order.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from experiment.factor_modeling import (
    AdaptiveRotatedFactorAnalysis,
    FixedPearsonThresholdSelector,
)
from experiment.feature_discovery import build_feature_universe
from experiment.preprocessing import QuantileClipper, VarianceFilter
from experiment.segment_validation_219 import build_compact_tables


MODELS = ("logistic_l2", "svm_rbf", "xgboost")
METHODS = ("optimistic_global", "nested_train_only")


def _classifier_and_grid(model_name: str, seed: int):
    if model_name == "logistic_l2":
        estimator = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
        grid = {"classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    elif model_name == "svm_rbf":
        estimator = SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
        )
        grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", 0.03, 0.1, 0.3],
        }
    elif model_name == "xgboost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
        )
        grid = [
            {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [1, 2],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
                "classifier__min_child_weight": [3],
                "classifier__reg_lambda": [5.0],
            },
            {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [1, 2],
                "classifier__learning_rate": [0.03],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
                "classifier__min_child_weight": [5],
                "classifier__reg_lambda": [10.0],
            },
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return estimator, grid


def _nested_pipeline(model_name: str, threshold: float, factor_count: int, seed: int):
    classifier, grid = _classifier_and_grid(model_name, seed)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(0.01, 0.99)),
        ("variance", VarianceFilter(1e-12)),
        ("pearson", FixedPearsonThresholdSelector(threshold)),
        ("scaler", StandardScaler()),
        (
            "factors",
            AdaptiveRotatedFactorAnalysis(
                n_components=factor_count,
                random_state=seed,
            ),
        ),
        ("classifier", classifier),
    ])
    return pipeline, grid


def _factor_score_classifier(model_name: str, seed: int):
    classifier, raw_grid = _classifier_and_grid(model_name, seed)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return pipeline, raw_grid


def _metric_row(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def _prepare_audit_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    clipper = QuantileClipper(0.01, 0.99)
    variance = VarianceFilter(1e-12)
    values = imputer.fit_transform(frame)
    values = clipper.fit_transform(values)
    values = variance.fit_transform(values)
    return values, variance.keep_mask_


def _signed_pearson(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = values - values.mean(axis=0)
    y_centered = y.astype(float) - y.mean()
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )


def _threshold_audit(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_names: list[str],
    thresholds: tuple[float, ...],
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = table[feature_names].apply(pd.to_numeric, errors="coerce")
    y = table["label"].astype(int).to_numpy()
    rows: list[dict] = []
    feature_rows: list[dict] = []

    def add_partition(
        scope: str,
        values_index: np.ndarray,
        repeat_index: int | None,
        seed: int | None,
        outer_fold: int | None,
        inner_fold: int | None,
    ) -> None:
        values, variance_mask = _prepare_audit_values(X.iloc[values_index])
        names = np.asarray(feature_names, dtype=object)[variance_mask]
        signed = _signed_pearson(values, y[values_index])
        absolute = np.abs(signed)
        for threshold in thresholds:
            selected = absolute >= threshold
            rows.append({
                "scope": scope,
                "repeat_index": repeat_index,
                "seed": seed,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "patients": int(len(values_index)),
                "threshold": float(threshold),
                "available_after_variance": int(len(names)),
                "selected_features": int(selected.sum()),
                "maximum_abs_pearson": float(absolute.max()),
            })
            if scope in {"full_cohort", "outer_train"}:
                for name, score, signed_score in zip(
                    names[selected], absolute[selected], signed[selected]
                ):
                    feature_rows.append({
                        "scope": scope,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "threshold": float(threshold),
                        "feature": str(name),
                        "abs_pearson": float(score),
                        "signed_pearson": float(signed_score),
                    })

    add_partition(
        "full_cohort",
        np.arange(len(table)),
        None,
        None,
        None,
        None,
    )
    for (repeat_index, seed), group in assignments.groupby(["repeat_index", "seed"]):
        fold_by_id = group.set_index("patient_id")["outer_fold"]
        mapped_fold = table["patient_id"].map(fold_by_id).to_numpy()
        for outer_fold in sorted(group["outer_fold"].unique()):
            outer_train_index = np.flatnonzero(mapped_fold != outer_fold)
            add_partition(
                "outer_train",
                outer_train_index,
                int(repeat_index),
                int(seed),
                int(outer_fold),
                None,
            )
            inner = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=int(seed) + int(outer_fold),
            )
            outer_y = y[outer_train_index]
            for inner_fold, (relative_train, _) in enumerate(
                inner.split(np.zeros((len(outer_y), 1)), outer_y),
                start=1,
            ):
                add_partition(
                    "inner_train",
                    outer_train_index[relative_train],
                    int(repeat_index),
                    int(seed),
                    int(outer_fold),
                    int(inner_fold),
                )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)


def _make_assignments(table: pd.DataFrame, repeats: int, outer_folds: int, seed: int):
    y = table["label"].astype(int).to_numpy()
    rows: list[dict] = []
    for repeat_index in range(repeats):
        repeat_seed = seed + repeat_index
        splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for outer_fold, (_, test_index) in enumerate(
            splitter.split(np.zeros((len(y), 1)), y)
        ):
            for index in test_index:
                rows.append({
                    "repeat_index": int(repeat_index),
                    "seed": int(repeat_seed),
                    "outer_fold": int(outer_fold),
                    "patient_id": table.iloc[index]["patient_id"],
                    "label": int(y[index]),
                })
    return pd.DataFrame(rows)


def _pre_selector_names(best: Pipeline, feature_names: list[str]) -> np.ndarray:
    names = np.asarray(feature_names, dtype=object)
    variance = best.named_steps["variance"]
    return names[variance.keep_mask_]


def _full_cohort_transformer(
    table: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
    factor_count: int,
    seed: int,
):
    transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(0.01, 0.99)),
        ("variance", VarianceFilter(1e-12)),
        ("pearson", FixedPearsonThresholdSelector(threshold)),
        ("scaler", StandardScaler()),
        (
            "factors",
            AdaptiveRotatedFactorAnalysis(
                n_components=factor_count,
                random_state=seed,
            ),
        ),
    ])
    X = table[feature_names].apply(pd.to_numeric, errors="coerce")
    y = table["label"].astype(int).to_numpy()
    scores = transformer.fit_transform(X, y)
    names = _pre_selector_names(transformer, feature_names)
    selector = transformer.named_steps["pearson"]
    selected_names = selector.get_feature_names_out(names)
    return scores, transformer, selected_names


def _bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for label in (0, 1):
        positions = np.flatnonzero(y == label)
        parts.append(rng.choice(positions, size=len(positions), replace=True))
    index = np.concatenate(parts)
    rng.shuffle(index)
    return index


def _aggregate_performance(
    predictions: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    for keys, group in predictions.groupby(["method", "model", "repeat_index", "seed"]):
        repeat_rows.append({
            "method": keys[0],
            "model": keys[1],
            "repeat_index": int(keys[2]),
            "seed": int(keys[3]),
            **_metric_row(
                group["true_label"].to_numpy(int),
                group["predicted_probability"].to_numpy(float),
            ),
        })
    repeat_metrics = pd.DataFrame(repeat_rows)
    mean_predictions = predictions.groupby(
        ["method", "model", "patient_id", "true_label"],
        as_index=False,
    ).agg(
        mean_oof_probability=("predicted_probability", "mean"),
        oof_probability_sd=("predicted_probability", "std"),
        prediction_count=("predicted_probability", "size"),
    )
    rng = np.random.default_rng(seed)
    aggregate_rows = []
    for (method, model), group in mean_predictions.groupby(["method", "model"]):
        y = group["true_label"].to_numpy(int)
        p = group["mean_oof_probability"].to_numpy(float)
        point = _metric_row(y, p)
        boot = {"roc_auc": [], "pr_auc": [], "brier": []}
        for _ in range(bootstrap_iterations):
            index = _bootstrap_indices(y, rng)
            row = _metric_row(y[index], p[index])
            for metric in boot:
                boot[metric].append(row[metric])
        aggregate_rows.append({
            "method": method,
            "model": model,
            **point,
            "roc_auc_ci_low": float(np.quantile(boot["roc_auc"], 0.025)),
            "roc_auc_ci_high": float(np.quantile(boot["roc_auc"], 0.975)),
            "pr_auc_ci_low": float(np.quantile(boot["pr_auc"], 0.025)),
            "pr_auc_ci_high": float(np.quantile(boot["pr_auc"], 0.975)),
            "brier_ci_low": float(np.quantile(boot["brier"], 0.025)),
            "brier_ci_high": float(np.quantile(boot["brier"], 0.975)),
        })
    return repeat_metrics, mean_predictions, pd.DataFrame(aggregate_rows)


def _paired_method_comparisons(
    mean_predictions: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model in MODELS:
        subset = mean_predictions[mean_predictions["model"].eq(model)]
        wide = subset.pivot(
            index=["patient_id", "true_label"],
            columns="method",
            values="mean_oof_probability",
        ).reset_index()
        y = wide["true_label"].to_numpy(int)
        nested = wide["nested_train_only"].to_numpy(float)
        optimistic = wide["optimistic_global"].to_numpy(float)
        point = roc_auc_score(y, optimistic) - roc_auc_score(y, nested)
        values = []
        for _ in range(bootstrap_iterations):
            index = _bootstrap_indices(y, rng)
            values.append(
                roc_auc_score(y[index], optimistic[index])
                - roc_auc_score(y[index], nested[index])
            )
        rows.append({
            "model": model,
            "comparison": "optimistic_global_minus_nested_train_only",
            "auc_difference": float(point),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_probability_gt_0": float(np.mean(np.asarray(values) > 0)),
        })
    return pd.DataFrame(rows)


def _write_summary(
    output_dir: Path,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    audit: pd.DataFrame,
    selection_stability: pd.DataFrame,
) -> None:
    full = audit[audit["scope"].eq("full_cohort")]
    count_025 = int(full.loc[full["threshold"].eq(0.25), "selected_features"].iloc[0])
    count_015 = int(full.loc[full["threshold"].eq(0.15), "selected_features"].iloc[0])
    lines = [
        "# v9 Pearson-to-six-factor replication and leakage audit",
        "",
        "## Exact threshold feasibility",
        "",
        f"- Full-cohort eligible features: 631.",
        f"- Features with absolute Pearson correlation >= 0.25: **{count_025}**.",
        f"- Features with absolute Pearson correlation >= 0.15: **{count_015}**.",
        "- The requested 0.25 -> six-factor workflow is infeasible because no feature reaches 0.25.",
        "",
        "## Performance",
        "",
        "| Analysis order | Model | ROC-AUC (95% CI) | PR-AUC | Brier |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in aggregate.sort_values(["method", "model"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['model']} | "
            f"{row['roc_auc']:.3f} ({row['roc_auc_ci_low']:.3f}-{row['roc_auc_ci_high']:.3f}) | "
            f"{row['pr_auc']:.3f} | {row['brier']:.3f} |"
        )
    lines.extend([
        "",
        "The optimistic_global rows are intentionally invalid pseudo-OOF estimates: "
        "all labels were used before cross-validation for Pearson screening and factor construction.",
        "",
        "## Leakage inflation",
        "",
        "| Model | Optimistic minus nested AUC | 95% CI |",
        "|---|---:|---:|",
    ])
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['model']} | {row['auc_difference']:+.3f} | "
            f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f} |"
        )
    stable = selection_stability.head(15)
    lines.extend([
        "",
        "## Most frequently Pearson-selected features in outer training folds",
        "",
        "| Feature | Selection frequency | Mean abs Pearson | Sign consistency |",
        "|---|---:|---:|---:|",
    ])
    for _, row in stable.iterrows():
        lines.append(
            f"| {row['feature']} | {row['selection_frequency']:.2f} | "
            f"{row['mean_abs_pearson']:.3f} | {row['sign_consistency']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- nested_train_only is the valid estimate of generalization.",
        "- optimistic_global is retained only to demonstrate analysis-order bias.",
        "- Selection frequency does not make a feature clinically validated.",
        "- The 219-patient anatomical-level mapping was protocol-inferred from patient 77, "
        "not independently reviewed for every patient.",
    ])
    (output_dir / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_pearson_factor_replication(
    *,
    annotation_file: Path,
    label_file: Path,
    feature_dir: Path,
    output_dir: Path,
    pearson_threshold: float = 0.15,
    exact_requested_threshold: float = 0.25,
    factor_count: int = 6,
    repeats: int = 10,
    outer_folds: int = 5,
    inner_folds: int = 4,
    bootstrap_iterations: int = 3000,
    base_seed: int = 20260730,
    log: Callable[[str], None] = print,
) -> dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Build the 219-patient 2D/3D/segment feature universe")
    table, dictionary, hard_exclusions, data_quality, source_warnings = build_feature_universe(
        annotation_file,
        label_file,
        feature_dir,
    )
    _, _, _, missing_rows, compact_warnings = build_compact_tables(
        annotation_file,
        label_file,
        feature_dir / "muscle_features_2d_v7.csv",
        set(),
    )
    source_warnings.extend(compact_warnings)
    table.to_csv(
        output_dir / "patient_feature_universe_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dictionary.to_csv(
        output_dir / "feature_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hard_exclusions.to_csv(
        output_dir / "hard_exclusion_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data_quality.to_csv(
        output_dir / "data_quality_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_rows.to_csv(
        output_dir / "source_missing_slice_muscle_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_names = [c for c in table.columns if c not in {"patient_id", "label"}]
    assignments = _make_assignments(table, repeats, outer_folds, base_seed)
    assignments.to_csv(
        output_dir / "outer_fold_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("Audit exact 0.25 and formal 0.15 Pearson thresholds")
    audit, audit_features = _threshold_audit(
        table,
        assignments,
        feature_names,
        (exact_requested_threshold, pearson_threshold),
        inner_folds,
    )
    audit.to_csv(output_dir / "pearson_threshold_audit.csv", index=False, encoding="utf-8-sig")
    audit_features.to_csv(
        output_dir / "pearson_selected_features_by_partition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exact_full_count = int(
        audit.loc[
            audit["scope"].eq("full_cohort")
            & audit["threshold"].eq(exact_requested_threshold),
            "selected_features",
        ].iloc[0]
    )
    if exact_full_count != 0:
        raise AssertionError("The exact 0.25 feasibility result changed unexpectedly")
    log("Exact 0.25 workflow is infeasible: full cohort retained zero features")

    X = table[feature_names].apply(pd.to_numeric, errors="coerce")
    y = table["label"].astype(int).to_numpy()
    global_scores, global_transformer, global_selected_names = _full_cohort_transformer(
        table,
        feature_names,
        pearson_threshold,
        factor_count,
        base_seed,
    )
    global_selector = global_transformer.named_steps["pearson"]
    global_factor_model = global_transformer.named_steps["factors"]
    pd.DataFrame(
        global_scores,
        columns=[
            f"Factor_{index + 1:02d}"
            for index in range(global_factor_model.actual_n_components_)
        ],
    ).assign(patient_id=table["patient_id"], label=y).to_csv(
        output_dir / "optimistic_full_cohort_factor_scores_NOT_VALID_FOR_PERFORMANCE.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_loading_rows = []
    for factor_index in range(global_factor_model.actual_n_components_):
        for feature_index, feature in enumerate(global_selected_names):
            global_loading_rows.append({
                "factor": f"Factor_{factor_index + 1:02d}",
                "feature": str(feature),
                "loading": float(global_factor_model.components_[factor_index, feature_index]),
                "absolute_loading": float(
                    abs(global_factor_model.components_[factor_index, feature_index])
                ),
                "full_cohort_abs_pearson": float(
                    global_selector.scores_[global_selector.keep_indices_[feature_index]]
                ),
                "warning": "Full-cohort descriptive output; not valid predictive evidence",
            })
    pd.DataFrame(global_loading_rows).to_csv(
        output_dir / "optimistic_full_cohort_factor_loadings_NOT_VALID_FOR_PERFORMANCE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions: list[dict] = []
    fold_rows: list[dict] = []
    selected_rows: list[dict] = []
    loading_rows: list[dict] = []
    runtime_warnings: list[dict] = list(source_warnings)

    for (repeat_index, seed), assignment in assignments.groupby(["repeat_index", "seed"]):
        fold_by_id = assignment.set_index("patient_id")["outer_fold"]
        mapped_fold = table["patient_id"].map(fold_by_id).to_numpy()
        for outer_fold in sorted(assignment["outer_fold"].unique()):
            train_index = np.flatnonzero(mapped_fold != outer_fold)
            test_index = np.flatnonzero(mapped_fold == outer_fold)
            inner = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=int(seed) + int(outer_fold),
            )
            for model_name in MODELS:
                for method in METHODS:
                    fold_started = time.time()
                    if method == "nested_train_only":
                        pipeline, grid = _nested_pipeline(
                            model_name,
                            pearson_threshold,
                            factor_count,
                            int(seed),
                        )
                        fit_X = X.iloc[train_index]
                        test_X = X.iloc[test_index]
                    else:
                        pipeline, grid = _factor_score_classifier(model_name, int(seed))
                        fit_X = global_scores[train_index]
                        test_X = global_scores[test_index]
                    search = GridSearchCV(
                        pipeline,
                        grid,
                        scoring="roc_auc",
                        cv=inner,
                        n_jobs=1,
                        refit=True,
                        error_score="raise",
                        return_train_score=True,
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        search.fit(fit_X, y[train_index])
                    for item in caught:
                        runtime_warnings.append({
                            "severity": "warning",
                            "stage": "model_fit",
                            "issue": item.category.__name__,
                            "action": str(item.message),
                            "method": method,
                            "model": model_name,
                            "repeat_index": int(repeat_index),
                            "outer_fold": int(outer_fold),
                        })
                    probability = search.predict_proba(test_X)[:, 1]
                    fold_metric = _metric_row(y[test_index], probability)
                    best_index = int(search.best_index_)
                    if method == "nested_train_only":
                        best = search.best_estimator_
                        selector = best.named_steps["pearson"]
                        pre_names = _pre_selector_names(best, feature_names)
                        fold_selected = selector.get_feature_names_out(pre_names)
                        factor_model = best.named_steps["factors"]
                        for local_index, feature in enumerate(fold_selected):
                            selected_rows.append({
                                "repeat_index": int(repeat_index),
                                "seed": int(seed),
                                "outer_fold": int(outer_fold),
                                "model": model_name,
                                "feature": str(feature),
                                "abs_pearson": float(
                                    selector.scores_[selector.keep_indices_[local_index]]
                                ),
                                "signed_pearson": float(
                                    selector.signed_scores_[selector.keep_indices_[local_index]]
                                ),
                            })
                        for factor_index in range(factor_model.actual_n_components_):
                            for feature_index, feature in enumerate(fold_selected):
                                loading_rows.append({
                                    "repeat_index": int(repeat_index),
                                    "seed": int(seed),
                                    "outer_fold": int(outer_fold),
                                    "model": model_name,
                                    "factor": f"Factor_{factor_index + 1:02d}",
                                    "feature": str(feature),
                                    "loading": float(
                                        factor_model.components_[factor_index, feature_index]
                                    ),
                                    "absolute_loading": float(
                                        abs(factor_model.components_[factor_index, feature_index])
                                    ),
                                })
                        selected_count = int(len(fold_selected))
                        actual_factor_count = int(factor_model.actual_n_components_)
                    else:
                        selected_count = int(len(global_selected_names))
                        actual_factor_count = int(global_factor_model.actual_n_components_)
                    fold_rows.append({
                        "method": method,
                        "model": model_name,
                        "repeat_index": int(repeat_index),
                        "seed": int(seed),
                        "outer_fold": int(outer_fold),
                        "train_n": int(len(train_index)),
                        "test_n": int(len(test_index)),
                        "selected_features": selected_count,
                        "actual_factor_count": actual_factor_count,
                        "inner_best_auc": float(search.best_score_),
                        "inner_train_auc": float(
                            search.cv_results_["mean_train_score"][best_index]
                        ),
                        "inner_train_validation_gap": float(
                            search.cv_results_["mean_train_score"][best_index]
                            - search.best_score_
                        ),
                        **fold_metric,
                        "runtime_seconds": float(time.time() - fold_started),
                        "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                    })
                    for local, row_index in enumerate(test_index):
                        predictions.append({
                            "method": method,
                            "model": model_name,
                            "repeat_index": int(repeat_index),
                            "seed": int(seed),
                            "outer_fold": int(outer_fold),
                            "patient_id": table.iloc[row_index]["patient_id"],
                            "true_label": int(y[row_index]),
                            "predicted_probability": float(probability[local]),
                        })
            log(
                f"repeat {int(repeat_index) + 1}/{repeats}, fold {int(outer_fold) + 1}/"
                f"{outer_folds} completed"
            )

    predictions_frame = pd.DataFrame(predictions)
    folds_frame = pd.DataFrame(fold_rows)
    selected_frame = pd.DataFrame(selected_rows)
    loadings_frame = pd.DataFrame(loading_rows)
    warnings_frame = pd.DataFrame(runtime_warnings)
    predictions_frame.to_csv(
        output_dir / "all_repeated_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds_frame.to_csv(
        output_dir / "outer_fold_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_frame.to_csv(
        output_dir / "nested_selected_original_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loadings_frame.to_csv(
        output_dir / "nested_factor_loadings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    warnings_frame.to_csv(
        output_dir / "warnings_and_bug_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeat_metrics, mean_predictions, aggregate = _aggregate_performance(
        predictions_frame,
        bootstrap_iterations,
        base_seed + 1000,
    )
    paired = _paired_method_comparisons(
        mean_predictions,
        bootstrap_iterations,
        base_seed + 2000,
    )
    repeat_metrics.to_csv(
        output_dir / "performance_each_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mean_predictions.to_csv(
        output_dir / "mean_oof_predictions_by_patient.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        output_dir / "aggregate_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir / "paired_leakage_inflation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    total_splits = repeats * outer_folds
    grouped = selected_frame.groupby(["model", "feature"], as_index=False).agg(
        selected_outer_splits=("outer_fold", "size"),
        mean_abs_pearson=("abs_pearson", "mean"),
        mean_signed_pearson=("signed_pearson", "mean"),
        positive_sign_fraction=("signed_pearson", lambda s: float((s > 0).mean())),
    )
    grouped["selection_frequency"] = grouped["selected_outer_splits"] / total_splits
    grouped["sign_consistency"] = np.maximum(
        grouped["positive_sign_fraction"],
        1 - grouped["positive_sign_fraction"],
    )
    stability = grouped.groupby("feature", as_index=False).agg(
        selection_frequency=("selection_frequency", "mean"),
        mean_abs_pearson=("mean_abs_pearson", "mean"),
        mean_signed_pearson=("mean_signed_pearson", "mean"),
        sign_consistency=("sign_consistency", "mean"),
    ).sort_values(
        ["selection_frequency", "mean_abs_pearson"],
        ascending=[False, False],
    )
    stability.to_csv(
        output_dir / "nested_source_feature_selection_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "experiment_version": "v9_pearson_factor_replication",
        "purpose": "replicate Pearson-to-six-factor workflow and quantify leakage inflation",
        "patients": int(len(table)),
        "label_counts": {
            str(k): int(v) for k, v in table["label"].value_counts().sort_index().items()
        },
        "candidate_features": int(len(feature_names)),
        "exact_requested_threshold": float(exact_requested_threshold),
        "formal_fallback_threshold": float(pearson_threshold),
        "factor_count": int(factor_count),
        "models": list(MODELS),
        "methods": list(METHODS),
        "repeats": int(repeats),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "bootstrap_iterations": int(bootstrap_iterations),
        "base_seed": int(base_seed),
        "runtime_seconds": float(time.time() - started),
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "prefix": sys.prefix,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(output_dir, aggregate, paired, audit, stability)
    log(f"Completed in {config['runtime_seconds']:.1f} seconds")
    return config
