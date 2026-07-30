"""Near-zero variance -> redundancy -> univariate logistic -> Stability LASSO.

The primary ``nested_train_only`` analysis refits every supervised selection
step without seeing the outer test patients. ``optimistic_global`` intentionally
selects seven variables once using all labels and is retained only as a leakage
diagnostic.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.special import expit
from scipy.stats import norm, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from experiment.feature_discovery import build_feature_universe
from experiment.pearson_factor_replication import (
    _aggregate_performance,
    _make_assignments,
    _paired_method_comparisons,
)
from experiment.preprocessing import QuantileClipper, VarianceFilter
from experiment.segment_validation_219 import build_compact_tables


MODELS = ("logistic_l2", "svm_rbf", "xgboost")
METHODS = ("optimistic_global", "nested_train_only")


@dataclass
class SelectionResult:
    imputer: SimpleImputer
    clipper: QuantileClipper
    variance: VarianceFilter
    correlation_indices: np.ndarray
    scaler: StandardScaler
    univariate_indices: np.ndarray
    selected_indices: np.ndarray
    feature_names_after_variance: np.ndarray
    feature_names_after_correlation: np.ndarray
    feature_names_after_univariate: np.ndarray
    selected_feature_names: np.ndarray
    univariate_p_values: np.ndarray
    univariate_coefficients: np.ndarray
    lasso_c: float | None
    lasso_cv_auc: float | None
    stability_frequency: np.ndarray
    stability_positive_fraction: np.ndarray
    stable_above_threshold: int

    def transform(self, X) -> np.ndarray:
        values = self.imputer.transform(X)
        values = self.clipper.transform(values)
        values = self.variance.transform(values)
        values = values[:, self.correlation_indices]
        values = self.scaler.transform(values)
        values = values[:, self.univariate_indices]
        return values[:, self.selected_indices]


def _univariate_logistic_wald(
    values: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast one-predictor logistic Wald tests with a tiny numerical ridge."""
    y_float = y.astype(float)
    p_values = np.ones(values.shape[1], dtype=float)
    coefficients = np.zeros(values.shape[1], dtype=float)
    ridge = np.eye(2) * 1e-8
    for column in range(values.shape[1]):
        design = np.column_stack([np.ones(len(y)), values[:, column]])
        beta = np.zeros(2, dtype=float)
        hessian = None
        for _ in range(40):
            probability = expit(np.clip(design @ beta, -30, 30))
            weight = np.clip(probability * (1.0 - probability), 1e-8, None)
            hessian = design.T @ (weight[:, None] * design) + ridge
            step = np.linalg.solve(hessian, design.T @ (y_float - probability))
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        covariance = np.linalg.inv(hessian)
        standard_error = np.sqrt(max(covariance[1, 1], 1e-16))
        z_value = beta[1] / standard_error
        p_values[column] = float(2.0 * norm.sf(abs(z_value)))
        coefficients[column] = float(beta[1])
    return p_values, coefficients


def _greedy_spearman_filter(values: np.ndarray, threshold: float) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 0, values)
    correlation = np.corrcoef(ranks, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    keep: list[int] = []
    for index in range(correlation.shape[0]):
        if all(abs(correlation[index, earlier]) <= threshold for earlier in keep):
            keep.append(index)
    return np.asarray(keep, dtype=int)


def _fit_selector(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | np.ndarray,
    *,
    seed: int,
    correlation_threshold: float,
    univariate_p_threshold: float,
    lasso_cs: tuple[float, ...],
    stability_subsamples: int,
    stability_train_fraction: float,
    stability_frequency_threshold: float,
    max_features: int,
) -> SelectionResult:
    imputer = SimpleImputer(strategy="median")
    clipper = QuantileClipper(0.01, 0.99)
    variance = VarianceFilter(1e-12)
    values = imputer.fit_transform(X)
    values = clipper.fit_transform(values)
    values = variance.fit_transform(values)
    names_after_variance = np.asarray(feature_names, dtype=object)[variance.keep_mask_]

    correlation_indices = _greedy_spearman_filter(values, correlation_threshold)
    values = values[:, correlation_indices]
    names_after_correlation = names_after_variance[correlation_indices]
    scaler = StandardScaler()
    values = scaler.fit_transform(values)

    p_values, coefficients = _univariate_logistic_wald(values, y)
    univariate_indices = np.flatnonzero(p_values < univariate_p_threshold)
    names_after_univariate = names_after_correlation[univariate_indices]
    if not len(univariate_indices):
        return SelectionResult(
            imputer, clipper, variance, correlation_indices, scaler,
            univariate_indices, np.asarray([], dtype=int),
            names_after_variance, names_after_correlation, names_after_univariate,
            np.asarray([], dtype=object), p_values, coefficients,
            None, None, np.asarray([], dtype=float), np.asarray([], dtype=float), 0,
        )
    candidate_values = values[:, univariate_indices]

    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    c_rows = []
    for c_value in lasso_cs:
        classifier = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=float(c_value),
            max_iter=5000,
            random_state=seed,
        )
        scores = cross_val_score(
            classifier,
            candidate_values,
            y,
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
        )
        c_rows.append((float(c_value), float(scores.mean())))
    best_c, best_auc = max(c_rows, key=lambda item: (item[1], -item[0]))

    frequency = np.zeros(candidate_values.shape[1], dtype=float)
    positive = np.zeros(candidate_values.shape[1], dtype=float)
    splitter = StratifiedShuffleSplit(
        n_splits=stability_subsamples,
        train_size=stability_train_fraction,
        random_state=seed,
    )
    for subsample_index, (train_index, _) in enumerate(
        splitter.split(candidate_values, y)
    ):
        classifier = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=best_c,
            max_iter=5000,
            random_state=seed + subsample_index,
        )
        classifier.fit(candidate_values[train_index], y[train_index])
        coefficient = classifier.coef_[0]
        nonzero = np.abs(coefficient) > 1e-8
        frequency += nonzero
        positive += nonzero & (coefficient > 0)
    frequency /= stability_subsamples
    positive_fraction = np.divide(
        positive,
        frequency * stability_subsamples,
        out=np.full_like(positive, 0.5),
        where=frequency > 0,
    )
    stable = np.flatnonzero(frequency >= stability_frequency_threshold)
    stable_count = int(len(stable))
    if stable_count:
        local_p = p_values[univariate_indices]
        ordering = sorted(
            stable.tolist(),
            key=lambda index: (
                -frequency[index],
                -max(positive_fraction[index], 1.0 - positive_fraction[index]),
                local_p[index],
                str(names_after_univariate[index]),
            ),
        )
        selected_indices = np.asarray(ordering[:max_features], dtype=int)
    else:
        selected_indices = np.asarray([], dtype=int)
    return SelectionResult(
        imputer=imputer,
        clipper=clipper,
        variance=variance,
        correlation_indices=correlation_indices,
        scaler=scaler,
        univariate_indices=univariate_indices,
        selected_indices=selected_indices,
        feature_names_after_variance=names_after_variance,
        feature_names_after_correlation=names_after_correlation,
        feature_names_after_univariate=names_after_univariate,
        selected_feature_names=names_after_univariate[selected_indices],
        univariate_p_values=p_values,
        univariate_coefficients=coefficients,
        lasso_c=best_c,
        lasso_cv_auc=best_auc,
        stability_frequency=frequency,
        stability_positive_fraction=positive_fraction,
        stable_above_threshold=stable_count,
    )


def _classifier_and_grid(model_name: str, seed: int):
    if model_name == "logistic_l2":
        classifier = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
        grid = {"classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    elif model_name == "svm_rbf":
        classifier = SVC(kernel="rbf", probability=True, random_state=seed)
        grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", 0.03, 0.1, 0.3],
        }
    elif model_name == "xgboost":
        classifier = XGBClassifier(
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
        raise ValueError(model_name)
    return Pipeline([("classifier", classifier)]), grid


def _selection_rows(
    result: SelectionResult,
    *,
    method: str,
    repeat_index: int | None,
    seed: int,
    outer_fold: int | None,
) -> list[dict]:
    if not len(result.univariate_indices):
        return []
    local_p = result.univariate_p_values[result.univariate_indices]
    local_coef = result.univariate_coefficients[result.univariate_indices]
    selected_set = set(result.selected_indices.tolist())
    rows = []
    for index, feature in enumerate(result.feature_names_after_univariate):
        positive_fraction = float(result.stability_positive_fraction[index])
        rows.append({
            "method": method,
            "repeat_index": repeat_index,
            "seed": int(seed),
            "outer_fold": outer_fold,
            "feature": str(feature),
            "univariate_logistic_p": float(local_p[index]),
            "univariate_logistic_coefficient": float(local_coef[index]),
            "stability_frequency": float(result.stability_frequency[index]),
            "stability_positive_fraction": positive_fraction,
            "stability_sign_consistency": float(
                max(positive_fraction, 1.0 - positive_fraction)
            ),
            "selected_final_max7": int(index in selected_set),
            "lasso_c": result.lasso_c,
            "lasso_cv_auc": result.lasso_cv_auc,
        })
    return rows


def _write_summary(
    output_dir: Path,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    folds: pd.DataFrame,
    stability: pd.DataFrame,
    global_result: SelectionResult,
) -> None:
    lines = [
        "# v7 Stability LASSO <=7 feature experiment",
        "",
        "## Protocol",
        "",
        "- 219 patients; 631 candidate 2D/3D/segment-derived features.",
        "- Near-zero variance removal, Spearman redundancy filter at 0.90.",
        "- Univariate logistic pre-screen at p < 0.10.",
        "- L1 penalty selected from 4-fold training CV.",
        "- 100 stratified 80% subsamples for Stability LASSO.",
        "- Final frequency threshold >=60%; at most seven variables.",
        "- Ten seeds x five outer folds; three downstream models.",
        "",
        "## Performance",
        "",
        "| Analysis order | Model | ROC-AUC (95% CI) | PR-AUC | Brier |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in aggregate.sort_values(["method", "model"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['model']} | "
            f"{row['roc_auc']:.3f} ({row['roc_auc_ci_low']:.3f}-"
            f"{row['roc_auc_ci_high']:.3f}) | {row['pr_auc']:.3f} | "
            f"{row['brier']:.3f} |"
        )
    lines.extend([
        "",
        "## Leakage diagnostic",
        "",
        "| Model | Optimistic minus nested AUC | 95% CI |",
        "|---|---:|---:|",
    ])
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['model']} | {row['auc_difference']:+.3f} | "
            f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f} |"
        )
    nested = folds[folds["method"].eq("nested_train_only")]
    lines.extend([
        "",
        "## Selection audit",
        "",
        f"- Global optimistic final variables: {len(global_result.selected_feature_names)}.",
        f"- Outer-fold final variable count min/median/max: "
        f"{nested['selected_features'].min():.0f}/"
        f"{nested['selected_features'].median():.1f}/"
        f"{nested['selected_features'].max():.0f}.",
        f"- Outer-fold univariate candidates min/median/max: "
        f"{nested['univariate_features'].min():.0f}/"
        f"{nested['univariate_features'].median():.1f}/"
        f"{nested['univariate_features'].max():.0f}.",
        "",
        "## Most reproducibly retained final variables",
        "",
        "| Feature | Outer-fold retention frequency | Mean stability frequency | Sign consistency |",
        "|---|---:|---:|---:|",
    ])
    for _, row in stability.head(15).iterrows():
        lines.append(
            f"| {row['feature']} | {row['outer_fold_retention_frequency']:.2f} | "
            f"{row['mean_stability_frequency']:.2f} | {row['mean_sign_consistency']:.2f} |"
        )
    lines.extend([
        "",
        "Only nested_train_only is a valid generalization estimate. "
        "optimistic_global uses all labels before cross-validation and is invalid.",
    ])
    (output_dir / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_stability_lasso_experiment(
    *,
    annotation_file: Path,
    label_file: Path,
    feature_dir: Path,
    output_dir: Path,
    repeats: int = 10,
    outer_folds: int = 5,
    inner_folds: int = 4,
    correlation_threshold: float = 0.90,
    univariate_p_threshold: float = 0.10,
    lasso_cs: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0),
    stability_subsamples: int = 100,
    stability_train_fraction: float = 0.80,
    stability_frequency_threshold: float = 0.60,
    max_features: int = 7,
    bootstrap_iterations: int = 3000,
    base_seed: int = 20260730,
    log: Callable[[str], None] = print,
) -> dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Build the 219-patient candidate universe")
    table, dictionary, hard_exclusions, data_quality, source_warnings = (
        build_feature_universe(annotation_file, label_file, feature_dir)
    )
    _, _, _, missing_rows, compact_warnings = build_compact_tables(
        annotation_file,
        label_file,
        feature_dir / "muscle_features_2d_v7.csv",
        set(),
    )
    source_warnings.extend(compact_warnings)
    for item in source_warnings:
        if item.get("stage") == "study_design":
            item["action"] = (
                "Treat the current Stability LASSO experiment as exploratory; "
                "require future independent confirmation"
            )
    table.to_csv(
        output_dir / "patient_feature_universe_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dictionary.to_csv(
        output_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig"
    )
    hard_exclusions.to_csv(
        output_dir / "hard_exclusion_report.csv", index=False, encoding="utf-8-sig"
    )
    data_quality.to_csv(
        output_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig"
    )
    missing_rows.to_csv(
        output_dir / "source_missing_slice_muscle_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_names = [c for c in table.columns if c not in {"patient_id", "label"}]
    X = table[feature_names].apply(pd.to_numeric, errors="coerce")
    y = table["label"].astype(int).to_numpy()
    assignments = _make_assignments(table, repeats, outer_folds, base_seed)
    assignments.to_csv(
        output_dir / "outer_fold_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("Fit the intentionally optimistic full-cohort Stability LASSO selector")
    global_result = _fit_selector(
        X,
        y,
        feature_names,
        seed=base_seed,
        correlation_threshold=correlation_threshold,
        univariate_p_threshold=univariate_p_threshold,
        lasso_cs=lasso_cs,
        stability_subsamples=stability_subsamples,
        stability_train_fraction=stability_train_fraction,
        stability_frequency_threshold=stability_frequency_threshold,
        max_features=max_features,
    )
    if not len(global_result.selected_feature_names):
        raise ValueError("Global Stability LASSO selected no variables")
    global_values = global_result.transform(X)
    global_selection_rows = _selection_rows(
        global_result,
        method="optimistic_global",
        repeat_index=None,
        seed=base_seed,
        outer_fold=None,
    )
    pd.DataFrame(global_selection_rows).to_csv(
        output_dir / "optimistic_global_selection_NOT_VALID.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions: list[dict] = []
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    runtime_warnings: list[dict] = list(source_warnings)
    for (repeat_index, seed), assignment in assignments.groupby(["repeat_index", "seed"]):
        fold_by_id = assignment.set_index("patient_id")["outer_fold"]
        mapped_fold = table["patient_id"].map(fold_by_id).to_numpy()
        for outer_fold in sorted(assignment["outer_fold"].unique()):
            train_index = np.flatnonzero(mapped_fold != outer_fold)
            test_index = np.flatnonzero(mapped_fold == outer_fold)
            nested_result = _fit_selector(
                X.iloc[train_index],
                y[train_index],
                feature_names,
                seed=int(seed) + int(outer_fold),
                correlation_threshold=correlation_threshold,
                univariate_p_threshold=univariate_p_threshold,
                lasso_cs=lasso_cs,
                stability_subsamples=stability_subsamples,
                stability_train_fraction=stability_train_fraction,
                stability_frequency_threshold=stability_frequency_threshold,
                max_features=max_features,
            )
            selection_rows.extend(_selection_rows(
                nested_result,
                method="nested_train_only",
                repeat_index=int(repeat_index),
                seed=int(seed),
                outer_fold=int(outer_fold),
            ))
            for method, result, train_values, test_values in [
                (
                    "nested_train_only",
                    nested_result,
                    nested_result.transform(X.iloc[train_index])
                    if len(nested_result.selected_feature_names)
                    else None,
                    nested_result.transform(X.iloc[test_index])
                    if len(nested_result.selected_feature_names)
                    else None,
                ),
                (
                    "optimistic_global",
                    global_result,
                    global_values[train_index],
                    global_values[test_index],
                ),
            ]:
                for model_name in MODELS:
                    fold_started = time.time()
                    if train_values is None:
                        probability = np.repeat(y[train_index].mean(), len(test_index))
                        inner_best_auc = 0.5
                        inner_train_auc = 0.5
                        gap = 0.0
                        best_params = {"fallback": "training_prevalence"}
                    else:
                        pipeline, grid = _classifier_and_grid(model_name, int(seed))
                        inner = StratifiedKFold(
                            n_splits=inner_folds,
                            shuffle=True,
                            random_state=int(seed) + int(outer_fold),
                        )
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
                            search.fit(train_values, y[train_index])
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
                        probability = search.predict_proba(test_values)[:, 1]
                        best_index = int(search.best_index_)
                        inner_best_auc = float(search.best_score_)
                        inner_train_auc = float(
                            search.cv_results_["mean_train_score"][best_index]
                        )
                        gap = inner_train_auc - inner_best_auc
                        best_params = search.best_params_
                    fold_rows.append({
                        "method": method,
                        "model": model_name,
                        "repeat_index": int(repeat_index),
                        "seed": int(seed),
                        "outer_fold": int(outer_fold),
                        "train_n": int(len(train_index)),
                        "test_n": int(len(test_index)),
                        "after_variance_features": int(
                            len(result.feature_names_after_variance)
                        ),
                        "after_correlation_features": int(
                            len(result.feature_names_after_correlation)
                        ),
                        "univariate_features": int(
                            len(result.feature_names_after_univariate)
                        ),
                        "stable_above_60pct": int(result.stable_above_threshold),
                        "selected_features": int(len(result.selected_feature_names)),
                        "lasso_c": result.lasso_c,
                        "lasso_cv_auc": result.lasso_cv_auc,
                        "inner_best_auc": inner_best_auc,
                        "inner_train_auc": inner_train_auc,
                        "inner_train_validation_gap": gap,
                        "outer_roc_auc": float(roc_auc_score(y[test_index], probability)),
                        "outer_pr_auc": float(
                            average_precision_score(y[test_index], probability)
                        ),
                        "outer_brier": float(
                            brier_score_loss(y[test_index], probability)
                        ),
                        "runtime_seconds": float(time.time() - fold_started),
                        "best_params": json.dumps(best_params, ensure_ascii=False),
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
                f"repeat {int(repeat_index) + 1}/{repeats}, fold "
                f"{int(outer_fold) + 1}/{outer_folds}: selected "
                f"{len(nested_result.selected_feature_names)} variables"
            )

    predictions_frame = pd.DataFrame(predictions)
    folds_frame = pd.DataFrame(fold_rows)
    selection_frame = pd.DataFrame(selection_rows)
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
    selection_frame.to_csv(
        output_dir / "nested_selection_details.csv",
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

    final_nested = selection_frame[
        selection_frame["selected_final_max7"].eq(1)
    ].copy()
    stability = final_nested.groupby("feature", as_index=False).agg(
        retained_outer_folds=("outer_fold", "size"),
        mean_stability_frequency=("stability_frequency", "mean"),
        mean_univariate_logistic_p=("univariate_logistic_p", "mean"),
        mean_sign_consistency=("stability_sign_consistency", "mean"),
    )
    stability["outer_fold_retention_frequency"] = (
        stability["retained_outer_folds"] / (repeats * outer_folds)
    )
    stability = stability.sort_values(
        ["outer_fold_retention_frequency", "mean_stability_frequency"],
        ascending=[False, False],
    )
    stability.to_csv(
        output_dir / "final_feature_selection_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "experiment_version": "v7_stability_lasso_replication",
        "patients": int(len(table)),
        "label_counts": {
            str(k): int(v) for k, v in table["label"].value_counts().sort_index().items()
        },
        "candidate_features": int(len(feature_names)),
        "correlation_threshold": float(correlation_threshold),
        "univariate_logistic_p_threshold": float(univariate_p_threshold),
        "lasso_cs": list(lasso_cs),
        "stability_subsamples": int(stability_subsamples),
        "stability_train_fraction": float(stability_train_fraction),
        "stability_frequency_threshold": float(stability_frequency_threshold),
        "max_features": int(max_features),
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
    _write_summary(
        output_dir,
        aggregate,
        paired,
        folds_frame,
        stability,
        global_result,
    )
    log(f"Completed in {config['runtime_seconds']:.1f} seconds")
    return config
