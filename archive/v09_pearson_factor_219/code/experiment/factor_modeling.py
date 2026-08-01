"""Leakage-safe Pearson screening, rotated factor analysis, and classification."""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment.modeling import MODEL_LABELS, make_classifier
from experiment.preprocessing import QuantileClipper, VarianceFilter, metric_row


class FixedPearsonThresholdSelector(BaseEstimator, TransformerMixin):
    """Keep features with absolute point-biserial Pearson correlation >= threshold."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        x_centered = values - values.mean(axis=0)
        y_centered = target - target.mean()
        numerator = x_centered.T @ y_centered
        denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
        signed = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        self.signed_scores_ = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
        self.scores_ = np.abs(self.signed_scores_)
        self.keep_indices_ = np.flatnonzero(self.scores_ >= float(self.threshold))
        if not len(self.keep_indices_):
            raise ValueError(
                f"No feature reached the locked absolute Pearson threshold {self.threshold:.3f}"
            )
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_indices_]


class AdaptiveRotatedFactorAnalysis(BaseEstimator, TransformerMixin):
    """Varimax factor analysis capped by the available training dimensions."""

    def __init__(self, n_components: int = 5, random_state: int = 2026):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.actual_n_components_ = min(
            int(self.n_components), values.shape[1], max(1, values.shape[0] - 1)
        )
        rotation = "varimax" if self.actual_n_components_ > 1 else None
        self.model_ = FactorAnalysis(
            n_components=self.actual_n_components_,
            rotation=rotation,
            random_state=int(self.random_state),
            svd_method="randomized",
            max_iter=1000,
        )
        self.model_.fit(values)
        self.components_ = self.model_.components_
        self.noise_variance_ = self.model_.noise_variance_
        self.n_iter_ = self.model_.n_iter_
        return self

    def transform(self, X):
        return self.model_.transform(np.asarray(X, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [f"Factor_{index:02d}" for index in range(1, self.actual_n_components_ + 1)],
            dtype=object,
        )


def make_factor_pipeline(model_name: str, config: dict) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(config["winsor_lower_quantile"], config["winsor_upper_quantile"])),
        ("variance", VarianceFilter()),
        ("pearson", FixedPearsonThresholdSelector(config["factor_pearson_threshold"])),
        ("scaler", StandardScaler()),
        ("factors", AdaptiveRotatedFactorAnalysis(random_state=config["random_seed"])),
        ("classifier", make_classifier(model_name, config["random_seed"])),
    ])


def factor_candidate_grid(model_name: str, config: dict) -> list[dict]:
    classifier_prefix = "classifier__"
    rows: list[dict] = []
    for factor_count in config["factor_counts"]:
        for candidate in config["factor_model_candidates"][model_name]:
            row = {"factors__n_components": [int(factor_count)]}
            row.update({f"{classifier_prefix}{key}": [value] for key, value in candidate.items()})
            rows.append(row)
    return rows


def _names_before_pearson(model: Pipeline, input_features: list[str]) -> np.ndarray:
    names = np.asarray(input_features, dtype=object)
    for step_name in ("imputer", "clipper", "variance"):
        step = model.named_steps[step_name]
        if hasattr(step, "get_feature_names_out"):
            names = step.get_feature_names_out(names)
    return names


def nested_cv_factor_model(
    table: pd.DataFrame,
    feature_set: str,
    model_name: str,
    config: dict,
    assignments: pd.DataFrame,
    log,
):
    experiment_id = f"{MODEL_LABELS[model_name]}__FactorAnalysis__{feature_set}"
    features = [column for column in table.columns if column not in {"patient_id", "label"}]
    X = table[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = table["label"].astype(int).to_numpy()
    mapped = table[["patient_id"]].merge(
        assignments[["patient_id", "outer_fold"]], on="patient_id", how="left", validate="one_to_one"
    )
    predictions: list[dict] = []
    folds: list[dict] = []
    factor_importances: list[dict] = []
    loadings: list[dict] = []
    selected_features: list[dict] = []
    issues: list[dict] = []

    for fold in sorted(mapped["outer_fold"].unique()):
        started = time.time()
        test_index = np.flatnonzero(mapped["outer_fold"].to_numpy() == fold)
        train_index = np.flatnonzero(mapped["outer_fold"].to_numpy() != fold)
        inner = StratifiedKFold(
            n_splits=config["inner_folds"],
            shuffle=True,
            random_state=config["random_seed"] + int(fold),
        )
        search = GridSearchCV(
            estimator=make_factor_pipeline(model_name, config),
            param_grid=factor_candidate_grid(model_name, config),
            scoring=config["primary_scoring"],
            cv=inner,
            n_jobs=config["search_n_jobs"],
            refit=True,
            error_score="raise",
            return_train_score=True,
        )
        log(f"{experiment_id} outer fold {fold}: {len(train_index)} train / {len(test_index)} test")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X.iloc[train_index], y[train_index])
        for item in caught:
            issues.append({
                "experiment_id": experiment_id,
                "model": model_name,
                "feature_set": feature_set,
                "outer_fold": int(fold),
                "category": item.category.__name__,
                "message": str(item.message),
            })

        best = search.best_estimator_
        probability = search.predict_proba(X.iloc[test_index])[:, 1]
        metrics = metric_row(y[test_index], probability)
        best_index = search.best_index_
        selector = best.named_steps["pearson"]
        factor_model = best.named_steps["factors"]
        pre_selector_names = _names_before_pearson(best, features)
        source_names = selector.get_feature_names_out(pre_selector_names)
        metrics.update({
            "experiment_id": experiment_id,
            "model": model_name,
            "feature_set": feature_set,
            "outer_fold": int(fold),
            "inner_best_auc": float(search.best_score_),
            "inner_best_train_auc": float(search.cv_results_["mean_train_score"][best_index]),
            "inner_train_validation_auc_gap": float(
                search.cv_results_["mean_train_score"][best_index] - search.best_score_
            ),
            "selected_original_features": int(len(source_names)),
            "requested_factor_count": int(search.best_params_["factors__n_components"]),
            "actual_factor_count": int(factor_model.actual_n_components_),
            "factor_analysis_iterations": int(factor_model.n_iter_),
            "runtime_seconds": time.time() - started,
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        })
        folds.append(metrics)
        for local, row_index in enumerate(test_index):
            predictions.append({
                "experiment_id": experiment_id,
                "model": model_name,
                "feature_set": feature_set,
                "outer_fold": int(fold),
                "patient_id": table.iloc[row_index]["patient_id"],
                "true_label": int(y[row_index]),
                "predicted_probability": float(probability[local]),
                "predicted_label_0_5": int(probability[local] >= 0.5),
            })

        transformed_test = best[:-1].transform(X.iloc[test_index])
        importance = permutation_importance(
            best.named_steps["classifier"],
            transformed_test,
            y[test_index],
            scoring="roc_auc",
            n_repeats=config["permutation_importance_repeats"],
            random_state=config["random_seed"] + int(fold),
            n_jobs=1,
        )
        factor_names = factor_model.get_feature_names_out()
        for name, mean_importance, std_importance in zip(
            factor_names, importance.importances_mean, importance.importances_std
        ):
            factor_importances.append({
                "experiment_id": experiment_id,
                "model": model_name,
                "feature_set": feature_set,
                "outer_fold": int(fold),
                "feature": str(name),
                "train_fold_abs_pearson": np.nan,
                "test_fold_permutation_auc_decrease_mean": float(mean_importance),
                "test_fold_permutation_auc_decrease_std": float(std_importance),
            })
        kept_scores = selector.scores_[selector.keep_indices_]
        kept_signed = selector.signed_scores_[selector.keep_indices_]
        for feature_name, score, signed_score, uniqueness in zip(
            source_names, kept_scores, kept_signed, factor_model.noise_variance_
        ):
            selected_features.append({
                "experiment_id": experiment_id,
                "model": model_name,
                "feature_set": feature_set,
                "outer_fold": int(fold),
                "source_feature": str(feature_name),
                "train_fold_abs_pearson": float(score),
                "train_fold_signed_pearson": float(signed_score),
                "factor_uniqueness": float(uniqueness),
            })
        for factor_index, factor_name in enumerate(factor_names):
            for source_index, feature_name in enumerate(source_names):
                loading = float(factor_model.components_[factor_index, source_index])
                loadings.append({
                    "experiment_id": experiment_id,
                    "model": model_name,
                    "feature_set": feature_set,
                    "outer_fold": int(fold),
                    "factor": str(factor_name),
                    "source_feature": str(feature_name),
                    "loading": loading,
                    "absolute_loading": abs(loading),
                })
        log(
            f"{experiment_id} fold {fold}: outer AUC={metrics['roc_auc']:.4f}; "
            f"selected={len(source_names)}; factors={factor_model.actual_n_components_}; "
            f"seconds={metrics['runtime_seconds']:.1f}"
        )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(folds),
        pd.DataFrame(factor_importances),
        pd.DataFrame(loadings),
        pd.DataFrame(selected_features),
        issues,
    )


def fit_descriptive_factor_model(
    table: pd.DataFrame,
    config: dict,
    factor_count: int,
):
    """Fit a full-cohort descriptive model; outputs must not be used as test performance."""
    features = [column for column in table.columns if column not in {"patient_id", "label"}]
    X = table[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = table["label"].astype(int).to_numpy()
    transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(config["winsor_lower_quantile"], config["winsor_upper_quantile"])),
        ("variance", VarianceFilter()),
        ("pearson", FixedPearsonThresholdSelector(config["factor_pearson_threshold"])),
        ("scaler", StandardScaler()),
        ("factors", AdaptiveRotatedFactorAnalysis(factor_count, config["random_seed"])),
    ])
    scores_array = transformer.fit_transform(X, y)
    selector = transformer.named_steps["pearson"]
    factor_model = transformer.named_steps["factors"]
    source_names = selector.get_feature_names_out(_names_before_pearson(transformer, features))
    factor_names = factor_model.get_feature_names_out()
    scores = pd.DataFrame(scores_array, columns=factor_names)
    scores.insert(0, "label", table["label"].to_numpy())
    scores.insert(0, "patient_id", table["patient_id"].to_numpy())
    loading_rows = []
    for factor_index, factor_name in enumerate(factor_names):
        for source_index, feature_name in enumerate(source_names):
            value = float(factor_model.components_[factor_index, source_index])
            loading_rows.append({
                "factor": str(factor_name),
                "source_feature": str(feature_name),
                "loading": value,
                "absolute_loading": abs(value),
                "full_cohort_abs_pearson": float(selector.scores_[selector.keep_indices_[source_index]]),
                "full_cohort_signed_pearson": float(
                    selector.signed_scores_[selector.keep_indices_[source_index]]
                ),
                "factor_uniqueness": float(factor_model.noise_variance_[source_index]),
            })
    return scores, pd.DataFrame(loading_rows), transformer
