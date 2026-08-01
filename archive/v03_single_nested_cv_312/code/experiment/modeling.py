"""Leakage-safe multi-model nested cross-validation."""

from __future__ import annotations

import json
import platform
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from experiment.build_features import normalise_patient_ids, read_csv_compatible
from experiment.preprocessing import (
    QuantileClipper,
    SpearmanCorrelationFilter,
    StablePointBiserialSelector,
    VarianceFilter,
    metric_row,
)


MODEL_LABELS = {
    "elastic_net": "ElasticNet",
    "xgboost": "XGBoost",
    "random_forest": "RandomForest",
    "svm_rbf": "RBF-SVM",
    "lightgbm": "LightGBM",
    "ebm": "EBM",
}


def make_classifier(model_name: str, seed: int):
    if model_name == "elastic_net":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=10000,
            tol=1e-4, random_state=seed,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=1, random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(n_jobs=1, random_state=seed)
    if model_name == "svm_rbf":
        base = SVC(kernel="rbf", probability=False, random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=False)
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


def make_pipeline(model_name: str, config: dict) -> Pipeline:
    scaler = StandardScaler() if model_name in {"elastic_net", "svm_rbf"} else "passthrough"
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(config["winsor_lower_quantile"], config["winsor_upper_quantile"])),
        ("variance", VarianceFilter()),
        ("correlation", SpearmanCorrelationFilter(config["correlation_threshold"])),
        ("selector", StablePointBiserialSelector()),
        ("scaler", scaler),
        ("classifier", make_classifier(model_name, config["random_seed"])),
    ])


def candidate_grid(model_name: str, config: dict) -> list[dict]:
    result = []
    classifier_prefix = "classifier__estimator__" if model_name == "svm_rbf" else "classifier__"
    for raw_candidate in config["candidates"][model_name]:
        candidate = dict(raw_candidate)
        selector_k = candidate.pop("selector_k")
        row = {"selector__k": [selector_k]}
        row.update({f"{classifier_prefix}{key}": [value] for key, value in candidate.items()})
        result.append(row)
    return result


def selected_feature_names(model: Pipeline, input_features: list[str]) -> np.ndarray:
    names = np.asarray(input_features, dtype=object)
    for step_name in ("imputer", "clipper", "variance", "correlation", "selector"):
        step = model.named_steps[step_name]
        if hasattr(step, "get_feature_names_out"):
            names = step.get_feature_names_out(names)
    return names


def make_outer_fold_assignments(
    table: pd.DataFrame,
    config: dict,
    reuse_file: Path | None = None,
) -> pd.DataFrame:
    """Create or validate a patient_id-based reusable outer-fold map."""
    cohort = table[["patient_id", "label"]].copy()
    cohort["patient_id"] = normalise_patient_ids(cohort["patient_id"])
    if reuse_file is not None:
        saved = read_csv_compatible(reuse_file)
        required = {"patient_id", "label", "outer_fold"}
        if not required.issubset(saved.columns):
            raise ValueError(f"Fold file is missing columns: {sorted(required - set(saved.columns))}")
        saved = saved[list(required)].copy()
        saved["patient_id"] = normalise_patient_ids(saved["patient_id"])
        saved["label"] = pd.to_numeric(saved["label"], errors="raise").astype(int)
        saved["outer_fold"] = pd.to_numeric(saved["outer_fold"], errors="raise").astype(int)
        if saved["patient_id"].duplicated().any():
            raise ValueError("Fold assignment file contains duplicate patient_id values")
        merged = cohort.merge(saved, on="patient_id", how="outer", suffixes=("_current", "_saved"), indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError("Fold assignment patient_id set does not exactly match the current labeled cohort")
        if not merged["label_current"].eq(merged["label_saved"]).all():
            raise ValueError("Fold assignment labels differ from PATIENT_LIST_FILE.csv")
        result = merged[["patient_id", "label_current", "outer_fold"]].rename(columns={"label_current": "label"})
        result = cohort[["patient_id"]].merge(result, on="patient_id", validate="one_to_one")
    else:
        y = cohort["label"].to_numpy(int)
        counts = pd.Series(y).value_counts()
        if counts.min() < config["outer_folds"]:
            raise ValueError("The minority class is smaller than outer_folds")
        splitter = StratifiedKFold(
            n_splits=config["outer_folds"], shuffle=True, random_state=config["random_seed"]
        )
        fold = np.zeros(len(cohort), dtype=int)
        for fold_number, (_, test_index) in enumerate(splitter.split(np.zeros((len(y), 1)), y), start=1):
            fold[test_index] = fold_number
        result = cohort.assign(outer_fold=fold)
    expected_folds = set(range(1, int(config["outer_folds"]) + 1))
    if set(result["outer_fold"].unique()) != expected_folds:
        raise ValueError(f"Outer-fold values must be exactly {sorted(expected_folds)}")
    return result


def _fold_indices(table: pd.DataFrame, assignments: pd.DataFrame):
    mapped = table[["patient_id"]].merge(assignments, on="patient_id", how="left", validate="one_to_one")
    if mapped["outer_fold"].isna().any():
        raise ValueError("Some model rows have no outer-fold assignment")
    for fold in sorted(mapped["outer_fold"].unique()):
        test_index = np.flatnonzero(mapped["outer_fold"].to_numpy() == fold)
        train_index = np.flatnonzero(mapped["outer_fold"].to_numpy() != fold)
        yield int(fold), train_index, test_index


def nested_cv_model(
    table: pd.DataFrame,
    feature_set: str,
    model_name: str,
    config: dict,
    assignments: pd.DataFrame,
    log,
):
    experiment_id = f"{MODEL_LABELS[model_name]}__{feature_set}"
    features = [column for column in table.columns if column not in {"patient_id", "label"}]
    X = table[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = table["label"].astype(int).to_numpy()
    predictions, folds, importances, issues = [], [], [], []

    for fold, train_index, test_index in _fold_indices(table, assignments):
        started = time.time()
        train_class_counts = pd.Series(y[train_index]).value_counts()
        if train_class_counts.min() < config["inner_folds"]:
            raise ValueError(f"Outer fold {fold}: minority training class is smaller than inner_folds")
        inner = StratifiedKFold(
            n_splits=config["inner_folds"], shuffle=True,
            random_state=config["random_seed"] + fold,
        )
        search = GridSearchCV(
            estimator=make_pipeline(model_name, config),
            param_grid=candidate_grid(model_name, config), scoring=config["primary_scoring"],
            cv=inner, n_jobs=config["search_n_jobs"], refit=True,
            error_score="raise", return_train_score=True,
        )
        log(f"{experiment_id} outer fold {fold}: {len(train_index)} train / {len(test_index)} test")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X.iloc[train_index], y[train_index])
        for item in caught:
            issues.append({
                "experiment_id": experiment_id, "model": model_name,
                "feature_set": feature_set, "outer_fold": fold,
                "category": item.category.__name__, "message": str(item.message),
            })

        probability = search.predict_proba(X.iloc[test_index])[:, 1]
        metrics = metric_row(y[test_index], probability)
        best_index = search.best_index_
        metrics.update({
            "experiment_id": experiment_id, "model": model_name,
            "feature_set": feature_set, "outer_fold": fold,
            "inner_best_auc": search.best_score_,
            "inner_best_train_auc": search.cv_results_["mean_train_score"][best_index],
            "inner_train_validation_auc_gap": search.cv_results_["mean_train_score"][best_index] - search.best_score_,
            "runtime_seconds": time.time() - started,
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        })
        folds.append(metrics)
        for local, row_index in enumerate(test_index):
            predictions.append({
                "experiment_id": experiment_id, "model": model_name,
                "feature_set": feature_set, "outer_fold": fold,
                "patient_id": table.iloc[row_index]["patient_id"],
                "true_label": int(y[row_index]),
                "predicted_probability": float(probability[local]),
                "predicted_label_0_5": int(probability[local] >= 0.5),
            })

        names = selected_feature_names(search.best_estimator_, features)
        transformed_test = search.best_estimator_[:-1].transform(X.iloc[test_index])
        importance = permutation_importance(
            search.best_estimator_.named_steps["classifier"], transformed_test, y[test_index],
            scoring="roc_auc", n_repeats=config["permutation_importance_repeats"],
            random_state=config["random_seed"] + fold, n_jobs=1,
        )
        selector = search.best_estimator_.named_steps["selector"]
        selected_scores = selector.scores_[selector.keep_indices_]
        for name, mean_importance, std_importance, score in zip(
            names, importance.importances_mean, importance.importances_std, selected_scores
        ):
            importances.append({
                "experiment_id": experiment_id, "model": model_name,
                "feature_set": feature_set, "outer_fold": fold, "feature": name,
                "train_fold_abs_pearson": float(score),
                "test_fold_permutation_auc_decrease_mean": float(mean_importance),
                "test_fold_permutation_auc_decrease_std": float(std_importance),
            })
        log(
            f"{experiment_id} fold {fold}: outer AUC={metrics['roc_auc']:.4f}; "
            f"inner={search.best_score_:.4f}; gap={metrics['inner_train_validation_auc_gap']:.4f}; "
            f"selected={len(names)}; seconds={metrics['runtime_seconds']:.1f}"
        )
    return pd.DataFrame(predictions), pd.DataFrame(folds), pd.DataFrame(importances), issues


def prevalence_predictions(table: pd.DataFrame, assignments: pd.DataFrame):
    y = table["label"].astype(int).to_numpy()
    rows, folds = [], []
    for fold, train_index, test_index in _fold_indices(table, assignments):
        probability = np.full(len(test_index), y[train_index].mean())
        metrics = metric_row(y[test_index], probability)
        metrics.update({
            "experiment_id": "PrevalenceBaseline", "model": "prevalence",
            "feature_set": "none", "outer_fold": fold,
            "inner_best_auc": np.nan, "inner_best_train_auc": np.nan,
            "inner_train_validation_auc_gap": np.nan, "runtime_seconds": 0.0,
            "best_params": "training-fold prevalence",
        })
        folds.append(metrics)
        for local, row_index in enumerate(test_index):
            rows.append({
                "experiment_id": "PrevalenceBaseline", "model": "prevalence",
                "feature_set": "none", "outer_fold": fold,
                "patient_id": table.iloc[row_index]["patient_id"],
                "true_label": int(y[row_index]),
                "predicted_probability": float(probability[local]),
                "predicted_label_0_5": int(probability[local] >= 0.5),
            })
    return pd.DataFrame(rows), pd.DataFrame(folds)


def bootstrap_performance(predictions: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for experiment_id, frame in predictions.groupby("experiment_id", sort=False):
        y = frame["true_label"].to_numpy(); probability = frame["predicted_probability"].to_numpy()
        point = metric_row(y, probability)
        boot = {key: [] for key in ("roc_auc", "pr_auc", "brier")}
        for _ in range(iterations):
            index = rng.integers(0, len(frame), len(frame))
            if np.unique(y[index]).size < 2:
                continue
            sample = metric_row(y[index], probability[index])
            for key in boot:
                boot[key].append(sample[key])
        rows.append({
            "experiment_id": experiment_id, "model": frame["model"].iloc[0],
            "feature_set": frame["feature_set"].iloc[0], **point,
            **{f"{key}_ci_low": np.quantile(values, 0.025) for key, values in boot.items()},
            **{f"{key}_ci_high": np.quantile(values, 0.975) for key, values in boot.items()},
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def paired_auc_vs_prevalence(predictions: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    baseline = predictions[predictions["experiment_id"] == "PrevalenceBaseline"].set_index("patient_id")
    rng = np.random.default_rng(seed + 99)
    rows = []
    for experiment_id in predictions["experiment_id"].drop_duplicates():
        if experiment_id == "PrevalenceBaseline":
            continue
        comparison = predictions[predictions["experiment_id"] == experiment_id].set_index("patient_id")
        if set(comparison.index) != set(baseline.index):
            raise ValueError(f"{experiment_id} OOF cohort differs from prevalence baseline")
        comparison = comparison.reindex(baseline.index)
        y = baseline["true_label"].to_numpy(int)
        p0 = baseline["predicted_probability"].to_numpy(float)
        p1 = comparison["predicted_probability"].to_numpy(float)
        point = sklearn.metrics.roc_auc_score(y, p1) - sklearn.metrics.roc_auc_score(y, p0)
        differences = []
        for _ in range(iterations):
            index = rng.integers(0, len(y), len(y))
            if np.unique(y[index]).size < 2:
                continue
            differences.append(
                sklearn.metrics.roc_auc_score(y[index], p1[index])
                - sklearn.metrics.roc_auc_score(y[index], p0[index])
            )
        rows.append({
            "reference": "PrevalenceBaseline", "comparison": experiment_id,
            "auc_difference": point, "ci_low": np.quantile(differences, 0.025),
            "ci_high": np.quantile(differences, 0.975),
            "bootstrap_probability_gt_reference": float(np.mean(np.asarray(differences) > 0)),
        })
    return pd.DataFrame(rows)


def importance_summary(importances: pd.DataFrame, outer_folds: int) -> pd.DataFrame:
    if importances.empty:
        return pd.DataFrame(columns=[
            "experiment_id", "model", "feature_set", "feature", "selected_folds",
            "mean_train_abs_pearson", "mean_test_permutation_auc_decrease",
            "median_test_permutation_auc_decrease", "selection_frequency",
        ])
    return (
        importances.groupby(["experiment_id", "model", "feature_set", "feature"], as_index=False)
        .agg(
            selected_folds=("outer_fold", "nunique"),
            mean_train_abs_pearson=("train_fold_abs_pearson", "mean"),
            mean_test_permutation_auc_decrease=("test_fold_permutation_auc_decrease_mean", "mean"),
            median_test_permutation_auc_decrease=("test_fold_permutation_auc_decrease_mean", "median"),
        )
        .assign(selection_frequency=lambda frame: frame["selected_folds"] / outer_folds)
        .sort_values(
            ["experiment_id", "selection_frequency", "mean_test_permutation_auc_decrease"],
            ascending=[True, False, False],
        )
    )


def save_performance_plots(predictions: pd.DataFrame, performance: pd.DataFrame, output_dir: Path):
    from sklearn.metrics import precision_recall_curve, roc_curve

    ordered = performance["experiment_id"].tolist()
    colors = plt.get_cmap("tab20")
    fig, axis = plt.subplots(figsize=(11, 9))
    for index, experiment_id in enumerate(ordered):
        frame = predictions[predictions["experiment_id"] == experiment_id]
        fpr, tpr, _ = roc_curve(frame["true_label"], frame["predicted_probability"])
        auc = performance.loc[performance["experiment_id"] == experiment_id, "roc_auc"].iloc[0]
        axis.plot(fpr, tpr, lw=1.5, color=colors(index % 20), label=f"{experiment_id} ({auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Nested-CV OOF ROC")
    axis.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(output_dir / "roc_all_models.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 9))
    for index, experiment_id in enumerate(ordered):
        frame = predictions[predictions["experiment_id"] == experiment_id]
        precision, recall, _ = precision_recall_curve(frame["true_label"], frame["predicted_probability"])
        ap = performance.loc[performance["experiment_id"] == experiment_id, "pr_auc"].iloc[0]
        axis.plot(recall, precision, lw=1.5, color=colors(index % 20), label=f"{experiment_id} ({ap:.3f})")
    axis.axhline(predictions["true_label"].mean(), ls="--", color="grey")
    axis.set(xlabel="Recall", ylabel="Precision", title="Nested-CV OOF precision-recall")
    axis.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(output_dir / "pr_all_models.png", dpi=180); plt.close(fig)

    plot = performance[performance["model"] != "prevalence"].copy().sort_values("roc_auc")
    if not plot.empty:
        fig, axis = plt.subplots(figsize=(10, max(5, 0.45 * len(plot) + 2)))
        x_error = np.vstack([plot["roc_auc"] - plot["roc_auc_ci_low"], plot["roc_auc_ci_high"] - plot["roc_auc"]])
        axis.errorbar(plot["roc_auc"], np.arange(len(plot)), xerr=x_error, fmt="o", color="#24557a", capsize=3)
        axis.set_yticks(np.arange(len(plot)), plot["experiment_id"])
        axis.axvline(0.5, ls="--", color="grey")
        axis.set(xlabel="ROC-AUC with 95% bootstrap CI", title="Model comparison")
        fig.tight_layout(); fig.savefig(output_dir / "auc_forest_plot.png", dpi=180); plt.close(fig)


def environment_info() -> dict:
    return {
        "python_executable": sys.executable, "python": sys.version,
        "conda_prefix": str(Path(sys.prefix)), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__, "interpret": interpret_version,
    }
