"""Leakage-safe multi-model nested cross-validation for stage 2."""

from __future__ import annotations

import json
import hashlib
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path

import lightgbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from interpret import __version__ as interpret_version
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from experiment_v1.modeling import (
    QuantileClipper,
    SpearmanCorrelationFilter,
    VarianceFilter,
    metric_row,
)


MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "RandomForest",
    "svm_rbf": "RBF-SVM",
    "lightgbm": "LightGBM",
    "ebm": "EBM",
}


class StablePointBiserialSelector(BaseEstimator, TransformerMixin):
    """Top-k absolute binary-label correlations without NumPy cov/corrcoef.

    NumPy 2.2.6 in the nnUNet-master Windows environment raised a native DLL
    exception inside np.corrcoef for this repeated pipeline call.  The direct
    centered dot-product formula is mathematically equivalent and avoids that
    native code path.
    """

    def __init__(self, k=20):
        self.k = k

    def fit(self, X, y):
        arr = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        x_centered = arr - arr.mean(axis=0)
        y_centered = target - target.mean()
        numerator = x_centered.T @ y_centered
        denominator = np.sqrt(np.sum(x_centered ** 2, axis=0) * np.sum(y_centered ** 2))
        scores = np.divide(np.abs(numerator), denominator, out=np.zeros_like(numerator), where=denominator > 0)
        self.scores_ = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        keep_count = min(int(self.k), arr.shape[1])
        self.keep_indices_ = np.argsort(self.scores_, kind="stable")[-keep_count:]
        self.keep_indices_.sort()
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_indices_]


def make_classifier(model_name: str, seed: int):
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=1, random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(n_jobs=1, random_state=seed)
    if model_name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary", verbosity=-1, n_jobs=1, random_state=seed,
            deterministic=True, force_col_wise=True,
        )
    if model_name == "ebm":
        return ExplainableBoostingClassifier(
            interactions=0, outer_bags=4, inner_bags=0, max_rounds=1500,
            early_stopping_rounds=50, validation_size=0.15, n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline(model_name: str, config: dict):
    scaler = StandardScaler() if model_name == "svm_rbf" else "passthrough"
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(config["winsor_lower_quantile"], config["winsor_upper_quantile"])),
        ("variance", VarianceFilter()),
        ("correlation", SpearmanCorrelationFilter(config["correlation_threshold"])),
        ("selector", StablePointBiserialSelector()),
        ("scaler", scaler),
        ("classifier", make_classifier(model_name, config["random_seed"])),
    ])


def candidate_grid(model_name: str, config: dict):
    result = []
    for candidate in config["candidates"][model_name]:
        candidate = dict(candidate)
        selector_k = candidate.pop("selector_k")
        row = {"selector__k": [selector_k]}
        row.update({f"classifier__{key}": [value] for key, value in candidate.items()})
        result.append(row)
    return result


def selected_feature_names(model: Pipeline, input_features: list[str]):
    names = np.asarray(input_features, dtype=object)
    for step_name in ("imputer", "clipper", "variance", "correlation", "selector"):
        step = model.named_steps[step_name]
        if hasattr(step, "get_feature_names_out"):
            names = step.get_feature_names_out(names)
    return names


def nested_cv_model(table: pd.DataFrame, feature_set: str, model_name: str,
                    config: dict, output_dir: Path, log):
    experiment_id = f"{MODEL_LABELS[model_name]}__{feature_set}"
    features = [c for c in table.columns if c not in {"patient_id", "label"}]
    X = table[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = table["label"].astype(int).to_numpy()
    outer = StratifiedKFold(n_splits=config["outer_folds"], shuffle=True, random_state=config["random_seed"])
    predictions, folds, importances, issues = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        started = time.time()
        fold_config = dict(config)
        cache_key = hashlib.sha1(experiment_id.encode("utf-8")).hexdigest()[:10]
        cache_dir = output_dir / ".pipeline_cache" / cache_key / f"f{fold}"
        fold_config["_cache_dir"] = str(cache_dir)
        inner = StratifiedKFold(
            n_splits=config["inner_folds"], shuffle=True,
            random_state=config["random_seed"] + fold,
        )
        search = GridSearchCV(
            estimator=make_pipeline(model_name, fold_config),
            param_grid=candidate_grid(model_name, config),
            scoring=config["primary_scoring"], cv=inner,
            n_jobs=config["search_n_jobs"], refit=True,
            error_score="raise", return_train_score=True,
        )
        log(f"{experiment_id} outer fold {fold}: {len(train_idx)} train / {len(test_idx)} test")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X.iloc[train_idx], y[train_idx])
        for item in caught:
            issues.append({
                "experiment_id": experiment_id, "model": model_name,
                "feature_set": feature_set, "outer_fold": fold,
                "category": item.category.__name__, "message": str(item.message),
            })

        probability = search.predict_proba(X.iloc[test_idx])[:, 1]
        metrics = metric_row(y[test_idx], probability)
        best_index = search.best_index_
        metrics.update({
            "experiment_id": experiment_id, "model": model_name,
            "feature_set": feature_set, "outer_fold": fold,
            "inner_best_auc": search.best_score_,
            "inner_best_train_auc": search.cv_results_["mean_train_score"][best_index],
            "inner_train_validation_auc_gap": (
                search.cv_results_["mean_train_score"][best_index] - search.best_score_
            ),
            "runtime_seconds": time.time() - started,
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        })
        folds.append(metrics)
        for local, row_idx in enumerate(test_idx):
            predictions.append({
                "experiment_id": experiment_id, "model": model_name,
                "feature_set": feature_set, "outer_fold": fold,
                "patient_id": table.iloc[row_idx]["patient_id"],
                "true_label": int(y[row_idx]),
                "predicted_probability": float(probability[local]),
                "predicted_label_0_5": int(probability[local] >= 0.5),
            })

        names = selected_feature_names(search.best_estimator_, features)
        transformed_test = search.best_estimator_[:-1].transform(X.iloc[test_idx])
        pi = permutation_importance(
            search.best_estimator_.named_steps["classifier"], transformed_test, y[test_idx],
            scoring="roc_auc", n_repeats=config["permutation_importance_repeats"],
            random_state=config["random_seed"] + fold, n_jobs=1,
        )
        selector_scores = search.best_estimator_.named_steps["selector"].scores_
        for name, mean_imp, std_imp, pearson_score in zip(names, pi.importances_mean, pi.importances_std, selector_scores[search.best_estimator_.named_steps["selector"].keep_indices_]):
            importances.append({
                "experiment_id": experiment_id, "model": model_name,
                "feature_set": feature_set, "outer_fold": fold, "feature": name,
                "train_fold_abs_pearson": float(pearson_score),
                "test_fold_permutation_auc_decrease_mean": float(mean_imp),
                "test_fold_permutation_auc_decrease_std": float(std_imp),
            })
        log(
            f"{experiment_id} fold {fold}: outer AUC={metrics['roc_auc']:.4f}; "
            f"inner={search.best_score_:.4f}; gap={metrics['inner_train_validation_auc_gap']:.4f}; "
            f"selected={len(names)}; seconds={metrics['runtime_seconds']:.1f}"
        )
        if cache_dir.exists():
            for attempt in range(5):
                try:
                    shutil.rmtree(cache_dir)
                    break
                except PermissionError:
                    time.sleep(0.5)
            else:
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception:
                    pass
    return pd.DataFrame(predictions), pd.DataFrame(folds), pd.DataFrame(importances), issues


def prevalence_predictions(table: pd.DataFrame, config: dict):
    y = table["label"].astype(int).to_numpy()
    outer = StratifiedKFold(n_splits=config["outer_folds"], shuffle=True, random_state=config["random_seed"])
    rows, folds = [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros((len(y), 1)), y), 1):
        probability = np.full(len(test_idx), y[train_idx].mean())
        metrics = metric_row(y[test_idx], probability)
        metrics.update({
            "experiment_id": "PrevalenceBaseline", "model": "prevalence",
            "feature_set": "none", "outer_fold": fold,
            "inner_best_auc": np.nan, "inner_best_train_auc": np.nan,
            "inner_train_validation_auc_gap": np.nan, "runtime_seconds": 0.0,
            "best_params": "training-fold prevalence",
        })
        folds.append(metrics)
        for local, row_idx in enumerate(test_idx):
            rows.append({
                "experiment_id": "PrevalenceBaseline", "model": "prevalence",
                "feature_set": "none", "outer_fold": fold,
                "patient_id": table.iloc[row_idx]["patient_id"],
                "true_label": int(y[row_idx]),
                "predicted_probability": float(probability[local]),
                "predicted_label_0_5": int(probability[local] >= 0.5),
            })
    return pd.DataFrame(rows), pd.DataFrame(folds)


def bootstrap_performance(predictions: pd.DataFrame, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for experiment_id, frame in predictions.groupby("experiment_id", sort=False):
        y = frame.true_label.to_numpy(); p = frame.predicted_probability.to_numpy()
        point = metric_row(y, p)
        boot = {key: [] for key in ("roc_auc", "pr_auc", "brier")}
        for _ in range(iterations):
            idx = rng.integers(0, len(frame), len(frame))
            if np.unique(y[idx]).size < 2:
                continue
            sample = metric_row(y[idx], p[idx])
            for key in boot:
                boot[key].append(sample[key])
        rows.append({
            "experiment_id": experiment_id,
            "model": frame.model.iloc[0], "feature_set": frame.feature_set.iloc[0],
            **point,
            **{f"{key}_ci_low": np.quantile(values, 0.025) for key, values in boot.items()},
            **{f"{key}_ci_high": np.quantile(values, 0.975) for key, values in boot.items()},
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def paired_auc_vs_references(predictions: pd.DataFrame, iterations: int, seed: int):
    refs = [x for x in ("PrevalenceBaseline", "ElasticNet__E1_3d_level3", "ElasticNet__E3_combined") if x in set(predictions.experiment_id)]
    rng = np.random.default_rng(seed + 99)
    rows = []
    for comparison in predictions.experiment_id.drop_duplicates():
        if comparison in refs:
            continue
        for reference in refs:
            left = predictions[predictions.experiment_id == reference].set_index("patient_id")
            right = predictions[predictions.experiment_id == comparison].set_index("patient_id")
            common = left.index.intersection(right.index)
            y = left.loc[common, "true_label"].to_numpy(int)
            p_ref = left.loc[common, "predicted_probability"].to_numpy(float)
            p_cmp = right.loc[common, "predicted_probability"].to_numpy(float)
            point = sklearn.metrics.roc_auc_score(y, p_cmp) - sklearn.metrics.roc_auc_score(y, p_ref)
            diffs = []
            for _ in range(iterations):
                idx = rng.integers(0, len(y), len(y))
                if np.unique(y[idx]).size < 2:
                    continue
                diffs.append(
                    sklearn.metrics.roc_auc_score(y[idx], p_cmp[idx]) -
                    sklearn.metrics.roc_auc_score(y[idx], p_ref[idx])
                )
            rows.append({
                "reference": reference, "comparison": comparison,
                "auc_difference": point, "ci_low": np.quantile(diffs, 0.025),
                "ci_high": np.quantile(diffs, 0.975),
                "bootstrap_probability_gt_reference": float(np.mean(np.asarray(diffs) > 0)),
            })
    return pd.DataFrame(rows)


def importance_summary(importances: pd.DataFrame, outer_folds: int):
    return (importances.groupby(["experiment_id", "model", "feature_set", "feature"], as_index=False)
            .agg(selected_folds=("outer_fold", "nunique"),
                 mean_train_abs_pearson=("train_fold_abs_pearson", "mean"),
                 mean_test_permutation_auc_decrease=("test_fold_permutation_auc_decrease_mean", "mean"),
                 median_test_permutation_auc_decrease=("test_fold_permutation_auc_decrease_mean", "median"))
            .assign(selection_frequency=lambda x: x.selected_folds / outer_folds)
            .sort_values(["experiment_id", "selection_frequency", "mean_test_permutation_auc_decrease"], ascending=[True, False, False]))


def save_performance_plots(predictions: pd.DataFrame, performance: pd.DataFrame, output_dir: Path):
    from sklearn.metrics import precision_recall_curve, roc_curve
    ordered = performance.experiment_id.tolist()
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 9))
    for i, exp in enumerate(ordered):
        frame = predictions[predictions.experiment_id == exp]
        fpr, tpr, _ = roc_curve(frame.true_label, frame.predicted_probability)
        auc = performance.loc[performance.experiment_id == exp, "roc_auc"].iloc[0]
        ax.plot(fpr, tpr, lw=1.5, color=cmap(i % 20), label=f"{exp} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Stage-2 nested-CV OOF ROC")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(output_dir / "stage2_roc_all_models.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    for i, exp in enumerate(ordered):
        frame = predictions[predictions.experiment_id == exp]
        precision, recall, _ = precision_recall_curve(frame.true_label, frame.predicted_probability)
        ap = performance.loc[performance.experiment_id == exp, "pr_auc"].iloc[0]
        ax.plot(recall, precision, lw=1.5, color=cmap(i % 20), label=f"{exp} ({ap:.3f})")
    ax.axhline(predictions.true_label.mean(), ls="--", color="grey")
    ax.set(xlabel="Recall", ylabel="Precision", title="Stage-2 nested-CV OOF precision-recall")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(output_dir / "stage2_pr_all_models.png", dpi=180); plt.close(fig)

    plot = performance[performance.model != "prevalence"].copy().sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(10, 9))
    xerr = np.vstack([plot.roc_auc - plot.roc_auc_ci_low, plot.roc_auc_ci_high - plot.roc_auc])
    ax.errorbar(plot.roc_auc, np.arange(len(plot)), xerr=xerr, fmt="o", color="#24557a", capsize=3)
    ax.set_yticks(np.arange(len(plot)), plot.experiment_id)
    ax.axvline(0.5, ls="--", color="grey")
    ax.set(xlabel="ROC-AUC with 95% bootstrap CI", title="Stage-2 model comparison")
    fig.tight_layout(); fig.savefig(output_dir / "stage2_auc_forest_plot.png", dpi=180); plt.close(fig)


def environment_info():
    return {
        "python_executable": sys.executable, "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__, "interpret": interpret_version,
    }
