"""Leakage-safe preprocessing components used inside cross-validation folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip using quantiles learned only from the current training fold."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class VarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 1e-12):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.keep_mask_ = np.nanvar(values, axis=0) > self.threshold
        if not self.keep_mask_.any():
            raise ValueError("VarianceFilter removed every feature")
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_mask_]


class SpearmanCorrelationFilter(BaseEstimator, TransformerMixin):
    """Greedy label-independent redundancy filter fitted within each fold."""

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = pd.DataFrame(np.asarray(X, dtype=float))
        correlations = frame.corr(method="spearman").abs().fillna(0.0).to_numpy()
        keep: list[int] = []
        for index in range(correlations.shape[0]):
            if all(correlations[index, chosen] <= self.threshold for chosen in keep):
                keep.append(index)
        self.keep_indices_ = np.asarray(keep, dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_indices_]


class StablePointBiserialSelector(BaseEstimator, TransformerMixin):
    """Select top-k absolute feature/label correlations within a training fold."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        x_centered = values - values.mean(axis=0)
        y_centered = target - target.mean()
        numerator = x_centered.T @ y_centered
        denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
        scores = np.divide(
            np.abs(numerator), denominator,
            out=np.zeros_like(numerator), where=denominator > 0,
        )
        self.scores_ = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        keep_count = min(int(self.k), values.shape[1])
        self.keep_indices_ = np.argsort(self.scores_, kind="stable")[-keep_count:]
        self.keep_indices_.sort()
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.keep_indices_]


def metric_row(y_true, probability, threshold: float = 0.5) -> dict:
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
