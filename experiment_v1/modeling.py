"""Nested-CV modeling utilities for experiment v1."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.stats import pearsonr
import sklearn
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature using quantiles learned only from the training fold."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class VarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1e-12):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.keep_mask_ = np.nanvar(arr, axis=0) > self.threshold
        if not self.keep_mask_.any():
            raise ValueError("VarianceFilter removed every feature")
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_mask_]


class SpearmanCorrelationFilter(BaseEstimator, TransformerMixin):
    """Greedy, label-independent correlation filter fitted within a CV fold."""

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = pd.DataFrame(np.asarray(X, dtype=float))
        corr = frame.corr(method="spearman").abs().fillna(0.0).to_numpy()
        keep = []
        for idx in range(corr.shape[0]):
            if all(corr[idx, chosen] <= self.threshold for chosen in keep):
                keep.append(idx)
        self.keep_indices_ = np.asarray(keep, dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_indices_]


class PointBiserialSelector(BaseEstimator, TransformerMixin):
    """Select top-k absolute Pearson correlations with a binary label."""

    def __init__(self, k=20):
        self.k = k

    def fit(self, X, y):
        arr = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        scores = np.zeros(arr.shape[1], dtype=float)
        for index in range(arr.shape[1]):
            column = arr[:, index]
            if np.nanstd(column) == 0:
                scores[index] = 0.0
            else:
                scores[index] = abs(np.corrcoef(column, target)[0, 1])
        scores = np.nan_to_num(scores, nan=0.0)
        keep_count = min(int(self.k), arr.shape[1])
        self.scores_ = scores
        self.keep_indices_ = np.argsort(scores, kind="stable")[-keep_count:]
        self.keep_indices_.sort()
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_indices_]


def make_pipeline(config: dict) -> Pipeline:
    cache_dir = config.get("_cache_dir")
    memory = Memory(location=cache_dir, verbose=0) if cache_dir else None
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(config["winsor_lower_quantile"], config["winsor_upper_quantile"])),
            ("variance", VarianceFilter()),
            ("correlation", SpearmanCorrelationFilter(config["correlation_threshold"])),
            ("selector", PointBiserialSelector()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=10000,
                    random_state=config["random_seed"],
                    tol=1e-4,
                ),
            ),
        ],
        memory=memory,
    )


def parameter_grid(config: dict) -> dict:
    return {
        "selector__k": config["pearson_top_k"],
        "classifier__C": config["logistic_c"],
        "classifier__l1_ratio": config["l1_ratio"],
        "classifier__class_weight": config["class_weight"],
    }


def _feature_names_after_pipeline(model: Pipeline, input_features: list[str]) -> np.ndarray:
    names = np.asarray(input_features, dtype=object)
    for step_name in ("imputer", "clipper", "variance", "correlation", "selector", "scaler"):
        step = model.named_steps[step_name]
        if hasattr(step, "get_feature_names_out"):
            names = step.get_feature_names_out(names)
    return names


def metric_row(y_true, probability, threshold=0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y_true, prediction, zero_division=0),
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "f1": f1_score(y_true, prediction, zero_division=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def nested_cv_feature_set(table: pd.DataFrame, feature_set: str, config: dict, output_dir: Path, log) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    features = [c for c in table.columns if c not in {"patient_id", "label"}]
    X = table[features].replace([np.inf, -np.inf], np.nan)
    y = table["label"].astype(int).to_numpy()
    outer = StratifiedKFold(n_splits=config["outer_folds"], shuffle=True, random_state=config["random_seed"])
    prediction_rows, fold_rows, coefficient_rows, runtime_issues = [], [], [], []

    for fold, (train_index, test_index) in enumerate(outer.split(X, y), start=1):
        started = time.time()
        fold_config = dict(config)
        cache_dir = output_dir / ".pipeline_cache" / feature_set / f"fold_{fold}"
        fold_config["_cache_dir"] = str(cache_dir)
        inner = StratifiedKFold(n_splits=config["inner_folds"], shuffle=True, random_state=config["random_seed"] + fold)
        search = GridSearchCV(
            make_pipeline(fold_config),
            parameter_grid(config),
            scoring=config["primary_scoring"],
            cv=inner,
            n_jobs=config["n_jobs"],
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        log(f"{feature_set} outer fold {fold}: fit {len(train_index)} train / {len(test_index)} test")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X.iloc[train_index], y[train_index])
        for item in caught:
            runtime_issues.append({"feature_set": feature_set, "outer_fold": fold, "category": item.category.__name__, "message": str(item.message)})

        probability = search.predict_proba(X.iloc[test_index])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics = metric_row(y[test_index], probability)
        metrics.update({"feature_set": feature_set, "outer_fold": fold, "inner_best_auc": search.best_score_, "runtime_seconds": time.time() - started, "best_params": json.dumps(search.best_params_, ensure_ascii=False)})
        fold_rows.append(metrics)

        for local, row_index in enumerate(test_index):
            prediction_rows.append({"feature_set": feature_set, "outer_fold": fold, "patient_id": table.iloc[row_index]["patient_id"], "true_label": int(y[row_index]), "predicted_probability": float(probability[local]), "predicted_label_0_5": int(prediction[local])})

        selected_names = _feature_names_after_pipeline(search.best_estimator_, features)
        coefficients = search.best_estimator_.named_steps["classifier"].coef_[0]
        for name, coef in zip(selected_names, coefficients):
            coefficient_rows.append({"feature_set": feature_set, "outer_fold": fold, "feature": name, "coefficient": float(coef), "abs_coefficient": float(abs(coef)), "selected_nonzero": bool(abs(coef) > 1e-10)})
        log(f"{feature_set} outer fold {fold}: AUC={metrics['roc_auc']:.4f}, selected={len(selected_names)}, inner_best={search.best_score_:.4f}")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows), runtime_issues


def dummy_prevalence_oof(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF baseline using only the positive prevalence in each outer train fold."""
    y = table["label"].astype(int).to_numpy()
    outer = StratifiedKFold(
        n_splits=config["outer_folds"],
        shuffle=True,
        random_state=config["random_seed"],
    )
    prediction_rows, fold_rows = [], []
    dummy_x = np.zeros((len(table), 1))
    for fold, (train_index, test_index) in enumerate(outer.split(dummy_x, y), start=1):
        probability = np.full(len(test_index), y[train_index].mean(), dtype=float)
        metrics = metric_row(y[test_index], probability)
        metrics.update({
            "feature_set": "E0_prevalence_baseline",
            "outer_fold": fold,
            "inner_best_auc": np.nan,
            "runtime_seconds": 0.0,
            "best_params": "training-fold prevalence only",
        })
        fold_rows.append(metrics)
        for local, row_index in enumerate(test_index):
            prediction_rows.append({
                "feature_set": "E0_prevalence_baseline",
                "outer_fold": fold,
                "patient_id": table.iloc[row_index]["patient_id"],
                "true_label": int(y[row_index]),
                "predicted_probability": float(probability[local]),
                "predicted_label_0_5": int(probability[local] >= 0.5),
            })
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def univariate_associations(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for feature_set, table in tables.items():
        y = table["label"].astype(int)
        for feature in [c for c in table.columns if c not in {"patient_id", "label"}]:
            x = pd.to_numeric(table[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = x.notna() & y.notna()
            if valid.sum() < 50 or x[valid].nunique() < 2:
                continue
            xv, yv = x[valid], y[valid]
            r, p_value = pearsonr(xv, yv)
            auc = roc_auc_score(yv, xv)
            pooled_sd = xv.std(ddof=1)
            smd = (xv[yv == 1].mean() - xv[yv == 0].mean()) / pooled_sd if pooled_sd else np.nan
            rows.append({"feature_set": feature_set, "feature": feature, "n": int(valid.sum()), "stable_mean": xv[yv == 0].mean(), "unstable_mean": xv[yv == 1].mean(), "stable_median": xv[yv == 0].median(), "unstable_median": xv[yv == 1].median(), "pearson_r": r, "abs_pearson_r": abs(r), "p_value": p_value, "auc": auc, "directionless_auc": max(auc, 1 - auc), "smd": smd})
    result = pd.DataFrame(rows)
    result["q_value_bh"] = np.nan
    for feature_set, indices in result.groupby("feature_set").groups.items():
        result.loc[indices, "q_value_bh"] = _bh_fdr(result.loc[indices, "p_value"].to_numpy())
    return result.sort_values(["feature_set", "abs_pearson_r"], ascending=[True, False])


def bootstrap_metrics(predictions: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for feature_set, frame in predictions.groupby("feature_set", sort=False):
        y = frame["true_label"].to_numpy()
        p = frame["predicted_probability"].to_numpy()
        point = metric_row(y, p)
        boot = {key: [] for key in ("roc_auc", "pr_auc", "brier")}
        for _ in range(iterations):
            idx = rng.integers(0, len(frame), size=len(frame))
            if len(np.unique(y[idx])) < 2:
                continue
            sample = metric_row(y[idx], p[idx])
            for key in boot:
                boot[key].append(sample[key])
        row = {"feature_set": feature_set, **point}
        for key, values in boot.items():
            row[f"{key}_ci_low"] = np.quantile(values, 0.025)
            row[f"{key}_ci_high"] = np.quantile(values, 0.975)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_comparisons(predictions: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    pivot = predictions.pivot(index="patient_id", columns="feature_set", values=["true_label", "predicted_probability"])
    sets = list(predictions["feature_set"].drop_duplicates())
    rng = np.random.default_rng(seed + 77)
    rows = []
    for i, first in enumerate(sets):
        for second in sets[i + 1:]:
            y = pivot[("true_label", first)].to_numpy(dtype=int)
            p1 = pivot[("predicted_probability", first)].to_numpy(dtype=float)
            p2 = pivot[("predicted_probability", second)].to_numpy(dtype=float)
            point = roc_auc_score(y, p2) - roc_auc_score(y, p1)
            diffs = []
            for _ in range(iterations):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) < 2:
                    continue
                diffs.append(roc_auc_score(y[idx], p2[idx]) - roc_auc_score(y[idx], p1[idx]))
            rows.append({"reference_feature_set": first, "comparison_feature_set": second, "auc_difference_comparison_minus_reference": point, "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975), "bootstrap_probability_difference_gt_0": float(np.mean(np.asarray(diffs) > 0))})
    return pd.DataFrame(rows)


def selection_summary(coefficients: pd.DataFrame, outer_folds: int) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    result = coefficients.groupby(["feature_set", "feature"], as_index=False).agg(selected_folds=("outer_fold", "nunique"), nonzero_folds=("selected_nonzero", "sum"), mean_coefficient=("coefficient", "mean"), median_coefficient=("coefficient", "median"), mean_abs_coefficient=("abs_coefficient", "mean"))
    result["selection_frequency"] = result["selected_folds"] / outer_folds
    def direction_consistent(group):
        nonzero = group.loc[group["coefficient"].abs() > 1e-10, "coefficient"]
        return bool(nonzero.empty or np.sign(nonzero).nunique() <= 1)
    consistency = coefficients.groupby(["feature_set", "feature"]).apply(direction_consistent)
    result["coefficient_direction_consistent"] = consistency.reindex(
        pd.MultiIndex.from_frame(result[["feature_set", "feature"]])
    ).to_numpy()
    return result.sort_values(["feature_set", "selection_frequency", "mean_abs_coefficient"], ascending=[True, False, False])


def save_plots(predictions: pd.DataFrame, univariate: pd.DataFrame, selection: pd.DataFrame, output_dir: Path) -> None:
    palette = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    colors = {name: color for name, color in zip(predictions["feature_set"].unique(), palette)}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, frame in predictions.groupby("feature_set", sort=False):
        fpr, tpr, _ = roc_curve(frame.true_label, frame.predicted_probability)
        auc = roc_auc_score(frame.true_label, frame.predicted_probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Nested-CV out-of-fold ROC")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(output_dir / "feature_set_roc_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    prevalence = predictions.drop_duplicates(["feature_set", "patient_id"])["true_label"].mean()
    for name, frame in predictions.groupby("feature_set", sort=False):
        precision, recall, _ = precision_recall_curve(frame.true_label, frame.predicted_probability)
        ap = average_precision_score(frame.true_label, frame.predicted_probability)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=colors[name], linewidth=2)
    ax.axhline(prevalence, linestyle="--", color="grey", label=f"Prevalence={prevalence:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Nested-CV out-of-fold precision-recall")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(output_dir / "feature_set_pr_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, frame in predictions.groupby("feature_set", sort=False):
        observed, predicted = calibration_curve(frame.true_label, frame.predicted_probability, n_bins=8, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name, color=colors[name])
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed proportion", title="OOF calibration (descriptive)")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(output_dir / "calibration_comparison.png", dpi=180); plt.close(fig)

    top = univariate.sort_values("abs_pearson_r", ascending=False).head(25).sort_values("abs_pearson_r")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"].str.slice(0, 65), top["pearson_r"], color=np.where(top["pearson_r"] >= 0, "#d95f02", "#1b9e77"))
    ax.set(xlabel="Point-biserial Pearson r", title="Top univariate associations (exploratory)")
    fig.tight_layout(); fig.savefig(output_dir / "top_univariate_features.png", dpi=180); plt.close(fig)

    if not selection.empty:
        for name, frame in selection.groupby("feature_set", sort=False):
            top_sel = frame.head(25).sort_values(["selection_frequency", "mean_abs_coefficient"])
            fig, ax = plt.subplots(figsize=(9, 8))
            ax.barh(top_sel["feature"].str.slice(0, 65), top_sel["selection_frequency"], color="#4c78a8")
            ax.set(xlim=(0, 1), xlabel="Selection frequency across outer folds", title=f"Feature stability: {name}")
            fig.tight_layout(); fig.savefig(output_dir / f"feature_selection_frequency_{name}.png", dpi=180); plt.close(fig)


def environment_info() -> dict:
    import matplotlib as mpl
    import numpy
    return {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": mpl.__version__}
