from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .io_utils import normalise_patient_ids, read_csv_compatible


GLOBAL_MRI_FEATURES = [
    "multifidus_3d_volume_asymmetry",
    "multifidus_mean_fip",
    "multifidus_mean_functional_csa",
    "erector_spinae_mean_fip",
    "erector_spinae_mean_functional_csa",
    "psoas_mean_fip",
    "psoas_mean_functional_csa",
]


@dataclass(frozen=True)
class ClinicalMRIConfig:
    label_file: str
    global_feature_file: str
    locked_feature_universe_file: str
    locked_selection_file: str
    output_dir: str
    repeats: int = 10
    outer_folds: int = 5
    inner_folds: int = 4
    base_seed: int = 20260730
    bootstrap_iterations: int = 3000


def _numeric_missing(series: pd.Series) -> pd.Series:
    cleaned = series.replace(
        {
            "#N/A": np.nan,
            "#NA": np.nan,
            "N/A": np.nan,
            "NA": np.nan,
            "": np.nan,
        }
    )
    values = pd.to_numeric(cleaned, errors="coerce")
    return values.where(values > 0)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": _safe_auc(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
    }


def _bootstrap_metric_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        if metric == "roc_auc":
            value = roc_auc_score(y[idx], p[idx])
        elif metric == "pr_auc":
            value = average_precision_score(y[idx], p[idx])
        elif metric == "brier":
            value = brier_score_loss(y[idx], p[idx])
        else:
            raise ValueError(metric)
        values.append(float(value))
    return tuple(np.percentile(values, [2.5, 97.5]).tolist())


def _paired_auc_ci(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = _safe_auc(y, p_a) - _safe_auc(y, p_b)
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(
            float(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    return observed, float(low), float(high)


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = logit(np.clip(p, 1e-5, 1 - 1e-5)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(x, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _load_clinical(label_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    raw = read_csv_compatible(label_file, dtype={"patient_id": "string"}, low_memory=False)
    required = [
        "patient_id",
        "instability_label",
        "age_years",
        "height_cm",
        "weight_kg",
        "bmi_kg_m2",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"PATIENT_LIST_FILE.csv missing required columns: {missing}")

    labels = pd.to_numeric(raw["instability_label"], errors="coerce")
    clinical = raw.loc[labels.isin([0, 1]), required].copy()
    clinical["patient_id"] = normalise_patient_ids(clinical["patient_id"])
    clinical["label"] = labels.loc[labels.isin([0, 1])].astype(int).to_numpy()
    if clinical["patient_id"].duplicated().any():
        duplicate_ids = clinical.loc[clinical["patient_id"].duplicated(False), "patient_id"]
        raise ValueError(f"Duplicate labeled patient_id values: {duplicate_ids.tolist()}")

    clinical["age_years"] = _numeric_missing(clinical["age_years"])
    clinical["height_m"] = _numeric_missing(clinical["height_cm"])
    clinical["weight_kg"] = _numeric_missing(clinical["weight_kg"])
    clinical["bmi_kg_m2"] = _numeric_missing(clinical["bmi_kg_m2"])
    clinical = clinical.drop(columns=["height_cm"]).sort_values("patient_id").reset_index(drop=True)

    audit: list[dict] = []
    for column in ["age_years", "height_m", "weight_kg", "bmi_kg_m2"]:
        series = clinical[column]
        audit.append(
            {
                "check": f"{column}_availability",
                "patients": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "minimum": float(series.min()) if series.notna().any() else np.nan,
                "median": float(series.median()) if series.notna().any() else np.nan,
                "maximum": float(series.max()) if series.notna().any() else np.nan,
                "detail": "0 and spreadsheet error strings are treated as missing",
            }
        )

    complete4 = clinical[
        ["age_years", "height_m", "weight_kg", "bmi_kg_m2"]
    ].notna().all(axis=1)
    age_bmi = clinical[["age_years", "bmi_kg_m2"]].notna().all(axis=1)
    audit.extend(
        [
            {
                "check": "complete_age_height_weight_bmi",
                "patients": int(complete4.sum()),
                "missing": int((~complete4).sum()),
                "minimum": np.nan,
                "median": np.nan,
                "maximum": np.nan,
                "detail": str(
                    clinical.loc[complete4, "label"].value_counts().sort_index().to_dict()
                ),
            },
            {
                "check": "complete_age_bmi",
                "patients": int(age_bmi.sum()),
                "missing": int((~age_bmi).sum()),
                "minimum": np.nan,
                "median": np.nan,
                "maximum": np.nan,
                "detail": str(
                    clinical.loc[age_bmi, "label"].value_counts().sort_index().to_dict()
                ),
            },
        ]
    )
    return clinical, raw, audit


def _bilateral_mean(frame: pd.DataFrame, muscle: str, metric: str) -> pd.Series:
    return frame[
        [f"{muscle}_left__{metric}", f"{muscle}_right__{metric}"]
    ].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _build_global_mri(global_feature_file: Path) -> pd.DataFrame:
    source = read_csv_compatible(
        global_feature_file, dtype={"patient_id": "string"}, low_memory=False
    )
    source["patient_id"] = normalise_patient_ids(source["patient_id"])
    if "muscle_name" in source.columns:
        required = {"patient_id", "muscle_name", "3D_Volume", "Mean_FIP", "Mean_Func_CSA"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"3D feature source is missing columns: {missing}")
        if source[["patient_id", "muscle_name"]].duplicated().any():
            raise ValueError("3D feature source contains duplicate patient/muscle rows")
        source = source.pivot(
            index="patient_id",
            columns="muscle_name",
            values=["3D_Volume", "Mean_FIP", "Mean_Func_CSA"],
        )
        source = source.swaplevel(0, 1, axis=1).sort_index(axis=1)
        source.columns = [f"{muscle}__{metric}" for muscle, metric in source.columns]
        source = source.reset_index()
    elif source["patient_id"].duplicated().any():
        raise ValueError("Global patient feature table contains duplicate patient_id values")

    left = pd.to_numeric(source["multifidus_left__3D_Volume"], errors="coerce")
    right = pd.to_numeric(source["multifidus_right__3D_Volume"], errors="coerce")
    denominator = left.abs() + right.abs()
    mri = pd.DataFrame({"patient_id": source["patient_id"]})
    mri["multifidus_3d_volume_asymmetry"] = (left - right).abs() / denominator.replace(0, np.nan)
    mri["multifidus_mean_fip"] = _bilateral_mean(source, "multifidus", "Mean_FIP")
    mri["multifidus_mean_functional_csa"] = _bilateral_mean(
        source, "multifidus", "Mean_Func_CSA"
    )
    mri["erector_spinae_mean_fip"] = _bilateral_mean(
        source, "erector_spinae", "Mean_FIP"
    )
    mri["erector_spinae_mean_functional_csa"] = _bilateral_mean(
        source, "erector_spinae", "Mean_Func_CSA"
    )
    mri["psoas_mean_fip"] = _bilateral_mean(source, "psoas", "Mean_FIP")
    mri["psoas_mean_functional_csa"] = _bilateral_mean(
        source, "psoas", "Mean_Func_CSA"
    )
    return mri


def _elasticnet_estimator(seed: int, inner_folds: int) -> GridSearchCV:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=seed,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 17)
    return GridSearchCV(
        pipe,
        {
            "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
            "model__l1_ratio": [0.0, 0.5, 1.0],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )


def _xgb_estimator(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=180,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def _locked_estimators(seed: int) -> dict[str, object]:
    return {
        "logistic_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", RobustScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.3,
                        penalty="l2",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                )
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", RobustScaler()),
                (
                    "model",
                    SVC(
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
            ]
        ),
        "xgboost_fixed": _xgb_estimator(seed),
    }


def _run_repeated_cv(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    cohort: str,
    model_family: str,
    config: ClinicalMRIConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    predictions: list[dict] = []
    folds: list[dict] = []
    tuning: list[dict] = []
    y = data["label"].astype(int).to_numpy()
    ids = data["patient_id"].astype(str).to_numpy()

    for repeat_index in range(config.repeats):
        seed = config.base_seed + repeat_index
        splitter = StratifiedKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=seed
        )
        for outer_fold, (train_idx, test_idx) in enumerate(splitter.split(ids, y)):
            for idx in test_idx:
                folds.append(
                    {
                        "cohort": cohort,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "patient_id": ids[idx],
                        "label": int(y[idx]),
                    }
                )
            for feature_set, columns in feature_sets.items():
                x_train = data.iloc[train_idx][columns].astype(float)
                x_test = data.iloc[test_idx][columns].astype(float)
                if model_family == "elasticnet":
                    estimator = _elasticnet_estimator(seed + outer_fold, config.inner_folds)
                    model_name = "elasticnet_logistic"
                elif model_family == "xgboost_native":
                    estimator = _xgb_estimator(seed + outer_fold)
                    model_name = "xgboost_native_missing"
                else:
                    raise ValueError(model_family)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(x_train, y[train_idx])
                probs = estimator.predict_proba(x_test)[:, 1]
                if isinstance(estimator, GridSearchCV):
                    tuning.append(
                        {
                            "cohort": cohort,
                            "repeat_index": repeat_index,
                            "outer_fold": outer_fold,
                            "feature_set": feature_set,
                            "model": model_name,
                            "best_inner_auc": float(estimator.best_score_),
                            "best_params": json.dumps(estimator.best_params_, sort_keys=True),
                        }
                    )
                for idx, probability in zip(test_idx, probs):
                    predictions.append(
                        {
                            "cohort": cohort,
                            "repeat_index": repeat_index,
                            "seed": seed,
                            "outer_fold": outer_fold,
                            "feature_set": feature_set,
                            "model": model_name,
                            "patient_id": ids[idx],
                            "label": int(y[idx]),
                            "probability": float(probability),
                        }
                    )
    return predictions, folds, tuning


def _run_locked7_cv(
    data: pd.DataFrame, features: list[str], cohort: str, config: ClinicalMRIConfig
) -> tuple[list[dict], list[dict]]:
    predictions: list[dict] = []
    folds: list[dict] = []
    y = data["label"].astype(int).to_numpy()
    ids = data["patient_id"].astype(str).to_numpy()
    for repeat_index in range(config.repeats):
        seed = config.base_seed + repeat_index
        splitter = StratifiedKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=seed
        )
        for outer_fold, (train_idx, test_idx) in enumerate(splitter.split(ids, y)):
            for idx in test_idx:
                folds.append(
                    {
                        "cohort": cohort,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "patient_id": ids[idx],
                        "label": int(y[idx]),
                    }
                )
            for model_name, estimator in _locked_estimators(seed + outer_fold).items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(data.iloc[train_idx][features], y[train_idx])
                probs = estimator.predict_proba(data.iloc[test_idx][features])[:, 1]
                for idx, probability in zip(test_idx, probs):
                    predictions.append(
                        {
                            "cohort": cohort,
                            "repeat_index": repeat_index,
                            "seed": seed,
                            "outer_fold": outer_fold,
                            "feature_set": "v7_label_selected_locked7",
                            "model": model_name,
                            "patient_id": ids[idx],
                            "label": int(y[idx]),
                            "probability": float(probability),
                        }
                    )
    return predictions, folds


def _summarise(
    predictions: pd.DataFrame, config: ClinicalMRIConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict] = []
    for keys, group in predictions.groupby(
        ["cohort", "feature_set", "model", "repeat_index"], sort=False
    ):
        cohort, feature_set, model, repeat_index = keys
        metrics = _metrics(
            group["label"].to_numpy(dtype=int),
            group["probability"].to_numpy(dtype=float),
        )
        repeat_rows.append(
            {
                "cohort": cohort,
                "feature_set": feature_set,
                "model": model,
                "repeat_index": repeat_index,
                **metrics,
            }
        )
    each_repeat = pd.DataFrame(repeat_rows)

    patient_mean = (
        predictions.groupby(
            ["cohort", "feature_set", "model", "patient_id", "label"], as_index=False
        )["probability"]
        .mean()
        .rename(columns={"probability": "mean_oof_probability"})
    )
    aggregate_rows: list[dict] = []
    for keys, group in patient_mean.groupby(["cohort", "feature_set", "model"], sort=False):
        cohort, feature_set, model = keys
        y = group["label"].to_numpy(dtype=int)
        p = group["mean_oof_probability"].to_numpy(dtype=float)
        metrics = _metrics(y, p)
        intercept, slope = _calibration(y, p)
        row = {
            "cohort": cohort,
            "patients": len(group),
            "label_0": int((y == 0).sum()),
            "label_1": int((y == 1).sum()),
            "feature_set": feature_set,
            "model": model,
            **metrics,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
        for metric in ["roc_auc", "pr_auc", "brier"]:
            low, high = _bootstrap_metric_ci(
                y, p, metric, config.bootstrap_iterations, config.base_seed + len(aggregate_rows)
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        repeats = each_repeat[
            (each_repeat["cohort"] == cohort)
            & (each_repeat["feature_set"] == feature_set)
            & (each_repeat["model"] == model)
        ]
        row["repeat_auc_mean"] = float(repeats["roc_auc"].mean())
        row["repeat_auc_sd"] = float(repeats["roc_auc"].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    paired_rows: list[dict] = []
    cohort_names = predictions["cohort"].drop_duplicates().tolist()
    comparisons = []
    for prefix, combined_name, mri_name in [
        ("complete4_n", "combined", "mri7_global_mechanistic"),
        ("age_bmi_n", "combined", "mri7_global_mechanistic"),
        ("all_native_missing_n", "combined_native_missing", "mri7_global_mechanistic"),
        (
            "locked7_complete4_overlap_n",
            "combined_locked7_clinical",
            "v7_label_selected_locked7",
        ),
    ]:
        matched = [name for name in cohort_names if name.startswith(prefix)]
        if matched:
            comparisons.append((matched[0], combined_name, mri_name))
    for cohort, combined_name, mri_name in comparisons:
        left = patient_mean[
            (patient_mean["cohort"] == cohort)
            & (patient_mean["feature_set"] == combined_name)
        ][["patient_id", "label", "mean_oof_probability"]].rename(
            columns={"mean_oof_probability": "combined_probability"}
        )
        right = patient_mean[
            (patient_mean["cohort"] == cohort)
            & (patient_mean["feature_set"] == mri_name)
        ][["patient_id", "mean_oof_probability"]].rename(
            columns={"mean_oof_probability": "mri_probability"}
        )
        merged = left.merge(right, on="patient_id", how="inner", validate="one_to_one")
        y = merged["label"].to_numpy(dtype=int)
        diff, low, high = _paired_auc_ci(
            y,
            merged["combined_probability"].to_numpy(dtype=float),
            merged["mri_probability"].to_numpy(dtype=float),
            config.bootstrap_iterations,
            config.base_seed + 700 + len(paired_rows),
        )
        paired_rows.append(
            {
                "cohort": cohort,
                "patients": len(merged),
                "comparison": f"{combined_name}_minus_{mri_name}",
                "combined_auc": _safe_auc(
                    y, merged["combined_probability"].to_numpy(dtype=float)
                ),
                "mri_auc": _safe_auc(y, merged["mri_probability"].to_numpy(dtype=float)),
                "paired_auc_difference": diff,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return each_repeat, patient_mean, aggregate, pd.DataFrame(paired_rows)


def _feature_definitions(locked7: Iterable[str]) -> pd.DataFrame:
    rows = [
        {
            "feature": "age_years",
            "group": "clinical",
            "definition": "Age in years from PATIENT_LIST_FILE.csv.",
        },
        {
            "feature": "height_m",
            "group": "clinical",
            "definition": "Height stored in source column height_cm but observed and interpreted as metres.",
        },
        {
            "feature": "weight_kg",
            "group": "clinical",
            "definition": "Weight in kilograms.",
        },
        {
            "feature": "bmi_kg_m2",
            "group": "clinical",
            "definition": "Recorded BMI; not recomputed from rounded height and weight.",
        },
        {
            "feature": "multifidus_3d_volume_asymmetry",
            "group": "mri7_global_mechanistic",
            "definition": "abs(left-right)/(abs(left)+abs(right)) for multifidus 3D volume.",
        },
    ]
    for muscle in ["multifidus", "erector_spinae", "psoas"]:
        rows.extend(
            [
                {
                    "feature": f"{muscle}_mean_fip",
                    "group": "mri7_global_mechanistic",
                    "definition": f"Mean of left and right {muscle} Mean_FIP.",
                },
                {
                    "feature": f"{muscle}_mean_functional_csa",
                    "group": "mri7_global_mechanistic",
                    "definition": f"Mean of left and right {muscle} Mean_Func_CSA.",
                },
            ]
        )
    rows.extend(
        {
            "feature": feature,
            "group": "v7_label_selected_locked7",
            "definition": "Selected with all 219 labels in canonical v10; reused unchanged in v11 and not independently validated.",
        }
        for feature in locked7
    )
    return pd.DataFrame(rows)


def _locked7_descriptive(
    data: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    y = data["label"].to_numpy(dtype=int)
    rows: list[dict] = []
    for feature in features:
        x = pd.to_numeric(data[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "missing": int(x.isna().sum()),
                "stable_median": float(x[y == 0].median()),
                "unstable_median": float(x[y == 1].median()),
                "pearson_r_with_label": float(x.corr(pd.Series(y, index=x.index))),
                "spearman_r_with_label": float(x.corr(pd.Series(y, index=x.index), method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def _write_summary(
    output_dir: Path,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    cohort_sizes = {
        row["cohort"]: int(row["patients"])
        for _, row in aggregate.drop_duplicates("cohort").iterrows()
    }
    complete_n = next(
        value for key, value in cohort_sizes.items() if key.startswith("complete4_n")
    )
    age_bmi_n = next(
        value for key, value in cohort_sizes.items() if key.startswith("age_bmi_n")
    )
    all_n = next(
        value for key, value in cohort_sizes.items() if key.startswith("all_native_missing_n")
    )
    locked_n = next(
        value for key, value in cohort_sizes.items() if key.startswith("locked7_overlap_n")
    )
    locked_clinical_n = next(
        value
        for key, value in cohort_sizes.items()
        if key.startswith("locked7_complete4_overlap_n")
    )
    complete_key = next(key for key in cohort_sizes if key.startswith("complete4_n"))
    age_bmi_key = next(key for key in cohort_sizes if key.startswith("age_bmi_n"))
    all_key = next(key for key in cohort_sizes if key.startswith("all_native_missing_n"))
    locked_key = next(key for key in cohort_sizes if key.startswith("locked7_overlap_n"))
    locked_clinical_key = next(
        key for key in cohort_sizes if key.startswith("locked7_complete4_overlap_n")
    )

    def auc_cell(cohort: str, feature_set: str, model: str | None = None) -> str:
        mask = aggregate["cohort"].eq(cohort) & aggregate["feature_set"].eq(feature_set)
        if model is not None:
            mask &= aggregate["model"].eq(model)
        row = aggregate.loc[mask]
        if len(row) != 1:
            raise ValueError(f"Expected one aggregate row for {cohort}/{feature_set}")
        item = row.iloc[0]
        return (
            f"{item['roc_auc']:.3f} "
            f"({item['roc_auc_ci_low']:.3f}–{item['roc_auc_ci_high']:.3f})"
        )

    def paired_cell(cohort: str) -> str:
        row = paired.loc[paired["cohort"].eq(cohort)]
        if len(row) != 1:
            raise ValueError(f"Expected one paired row for {cohort}")
        item = row.iloc[0]
        return (
            f"{item['paired_auc_difference']:+.3f} "
            f"({item['ci_low']:+.3f}–{item['ci_high']:+.3f})"
        )

    comparison_rows = [
        (
            complete_key,
            "clinical",
            "mri7_global_mechanistic",
            "combined",
        ),
        (
            age_bmi_key,
            "clinical",
            "mri7_global_mechanistic",
            "combined",
        ),
        (
            all_key,
            "clinical_native_missing",
            "mri7_global_mechanistic",
            "combined_native_missing",
        ),
        (
            locked_clinical_key,
            "clinical",
            "v7_label_selected_locked7",
            "combined_locked7_clinical",
        ),
    ]

    lines = [
        "# v11 clinical plus MRI experiment",
        "",
        "All performance values are based on repeated out-of-fold predictions.",
        "",
        "## Same-cohort model comparison",
        "",
        "| Cohort | Clinical AUC (95% CI) | MRI AUC (95% CI) | Combined AUC (95% CI) | Combined minus MRI (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort, clinical_set, mri_set, combined_set in comparison_rows:
        lines.append(
            f"| {cohort} | {auc_cell(cohort, clinical_set)} | "
            f"{auc_cell(cohort, mri_set)} | {auc_cell(cohort, combined_set)} | "
            f"{paired_cell(cohort)} |"
        )
    lines.extend(
        [
            "",
            f"The locked-panel L2 Logistic result on {locked_n} patients is "
            f"{auc_cell(locked_key, 'v7_label_selected_locked7', 'logistic_l2')}. It is a same-cohort "
            "reproducibility estimate, not external validation.",
            "",
            "## Interpretation constraints",
            "",
            "- Values 0 and spreadsheet error strings in clinical fields were converted to missing.",
            "- No clinical mean/median imputation was used; the full-cohort XGBoost uses native missing-value routing and indicators.",
            "- The global mechanistic MRI seven-variable panel was defined without current-label screening.",
            "- The canonical-v10 locked panel was selected using the same 219 labels. Its 219-patient result and the 68-patient combined result are not independent validation.",
            "- Slip segment was omitted from the full-cohort analysis.",
            "",
        ]
    )
    (output_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    cn = [
        "# v11 临床变量与MRI肌肉特征实验结果",
        "",
        "## 同队列完整模型结果",
        "",
        "| 队列 | 临床AUC（95%CI） | MRIAUC（95%CI） | 联合AUC（95%CI） | 联合−MRI（95%CI） |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort, clinical_set, mri_set, combined_set in comparison_rows:
        cn.append(
            f"| {cohort} | {auc_cell(cohort, clinical_set)} | "
            f"{auc_cell(cohort, mri_set)} | {auc_cell(cohort, combined_set)} | "
            f"{paired_cell(cohort)} |"
        )
    cn.extend(
        [
            "",
            f"旧七特征在{locked_n}例上的L2 Logistic结果为"
            f"{auc_cell(locked_key, 'v7_label_selected_locked7', 'logistic_l2')}，只能解释为同队列复现。",
            "",
            "## 0.708的解释",
            "",
            f"0.708来自{locked_clinical_n}例临床完整重叠队列的临床+旧MRI7模型。"
            "这七项曾利用同一219例标签筛选，因此不是独立验证；联合相对MRI的配对增益只有"
            f"{paired_cell(locked_clinical_key)}。",
            "",
            "## 解释边界",
            "",
            "- 完整病例分析不填补临床变量；全队列XGBoost原生接收NaN和缺失指示变量。",
            "- 全队列模型不使用或推断滑脱节段。",
            "- 机制型全局MRI7没有使用本轮标签筛选；旧MRI7与评估队列存在选择重叠。",
            "- 当前结果仍属于内部重复OOF验证，需要完全独立队列确认。",
            "",
        ]
    )
    (output_dir / "RESULTS_INTERPRETATION_CN.md").write_text(
        "\n".join(cn), encoding="utf-8-sig"
    )


def run_clinical_mri_experiment(config: ClinicalMRIConfig) -> None:
    started = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warning_rows: list[dict] = []

    clinical, _, clinical_audit = _load_clinical(Path(config.label_file))
    mri = _build_global_mri(Path(config.global_feature_file))
    full = clinical.merge(mri, on="patient_id", how="inner", validate="one_to_one")
    if len(full) != 312:
        raise ValueError(f"Expected 312 labeled patients with global MRI data, found {len(full)}")

    complete4_mask = full[
        ["age_years", "height_m", "weight_kg", "bmi_kg_m2"]
    ].notna().all(axis=1)
    age_bmi_mask = full[["age_years", "bmi_kg_m2"]].notna().all(axis=1)
    complete4 = full.loc[complete4_mask].reset_index(drop=True)
    age_bmi = full.loc[age_bmi_mask].reset_index(drop=True)
    complete4_cohort = f"complete4_n{len(complete4)}"
    age_bmi_cohort = f"age_bmi_n{len(age_bmi)}"
    all_missing_cohort = f"all_native_missing_n{len(full)}"

    clinical4 = ["age_years", "height_m", "weight_kg", "bmi_kg_m2"]
    clinical2 = ["age_years", "bmi_kg_m2"]
    full_missing = full.copy()
    missing_indicators: list[str] = []
    for column in clinical4:
        indicator = f"{column}__missing"
        full_missing[indicator] = full_missing[column].isna().astype(int)
        missing_indicators.append(indicator)

    all_predictions: list[dict] = []
    all_folds: list[dict] = []
    all_tuning: list[dict] = []

    pred, fold, tuning = _run_repeated_cv(
        complete4,
        {
            "clinical": clinical4,
            "mri7_global_mechanistic": GLOBAL_MRI_FEATURES,
            "combined": clinical4 + GLOBAL_MRI_FEATURES,
        },
        complete4_cohort,
        "elasticnet",
        config,
    )
    all_predictions.extend(pred)
    all_folds.extend(fold)
    all_tuning.extend(tuning)

    pred, fold, tuning = _run_repeated_cv(
        age_bmi,
        {
            "clinical": clinical2,
            "mri7_global_mechanistic": GLOBAL_MRI_FEATURES,
            "combined": clinical2 + GLOBAL_MRI_FEATURES,
        },
        age_bmi_cohort,
        "elasticnet",
        config,
    )
    all_predictions.extend(pred)
    all_folds.extend(fold)
    all_tuning.extend(tuning)

    pred, fold, tuning = _run_repeated_cv(
        full_missing,
        {
            "clinical_native_missing": clinical4 + missing_indicators,
            "mri7_global_mechanistic": GLOBAL_MRI_FEATURES,
            "combined_native_missing": clinical4 + missing_indicators + GLOBAL_MRI_FEATURES,
        },
        all_missing_cohort,
        "xgboost_native",
        config,
    )
    all_predictions.extend(pred)
    all_folds.extend(fold)
    all_tuning.extend(tuning)

    selection = read_csv_compatible(Path(config.locked_selection_file), low_memory=False)
    selected_mask = pd.to_numeric(
        selection["selected_final_max7"], errors="coerce"
    ).eq(1)
    locked7 = selection.loc[selected_mask, "feature"].astype(str).tolist()
    if len(locked7) != 7:
        raise ValueError(f"Expected exactly seven locked v7 features, found {len(locked7)}")
    locked_data = read_csv_compatible(
        Path(config.locked_feature_universe_file),
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    locked_data["patient_id"] = normalise_patient_ids(locked_data["patient_id"])
    missing_locked = [column for column in locked7 if column not in locked_data.columns]
    if missing_locked:
        raise ValueError(f"Locked v7 features unavailable: {missing_locked}")
    locked_cohort = f"locked7_overlap_n{len(locked_data)}"
    pred, fold = _run_locked7_cv(locked_data, locked7, locked_cohort, config)
    all_predictions.extend(pred)
    all_folds.extend(fold)

    locked_clinical = locked_data[["patient_id"] + locked7].merge(
        clinical[
            ["patient_id", "label", "age_years", "height_m", "weight_kg", "bmi_kg_m2"]
        ],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    locked_complete_mask = locked_clinical[clinical4].notna().all(axis=1)
    locked_complete = locked_clinical.loc[locked_complete_mask].reset_index(drop=True)
    locked_complete_cohort = f"locked7_complete4_overlap_n{len(locked_complete)}"
    pred, fold, tuning = _run_repeated_cv(
        locked_complete,
        {
            "clinical": clinical4,
            "v7_label_selected_locked7": locked7,
            "combined_locked7_clinical": clinical4 + locked7,
        },
        locked_complete_cohort,
        "elasticnet",
        config,
    )
    all_predictions.extend(pred)
    all_folds.extend(fold)
    all_tuning.extend(tuning)

    predictions = pd.DataFrame(all_predictions)
    folds = pd.DataFrame(all_folds).drop_duplicates()
    tuning = pd.DataFrame(all_tuning)
    each_repeat, patient_mean, aggregate, paired = _summarise(predictions, config)

    membership = full[
        ["patient_id", "label", "age_years", "height_m", "weight_kg", "bmi_kg_m2"]
    ].copy()
    membership[complete4_cohort] = complete4_mask.astype(int).to_numpy()
    membership[age_bmi_cohort] = age_bmi_mask.astype(int).to_numpy()
    membership[all_missing_cohort] = 1
    locked_ids = set(locked_data["patient_id"].astype(str))
    membership[locked_cohort] = membership["patient_id"].isin(locked_ids).astype(int)
    locked_complete_ids = set(locked_complete["patient_id"].astype(str))
    membership[locked_complete_cohort] = (
        membership["patient_id"].isin(locked_complete_ids).astype(int)
    )

    feature_audit = []
    for column in GLOBAL_MRI_FEATURES:
        feature_audit.append(
            {
                "check": f"{column}_availability",
                "patients": int(full[column].notna().sum()),
                "missing": int(full[column].isna().sum()),
                "minimum": float(full[column].min()),
                "median": float(full[column].median()),
                "maximum": float(full[column].max()),
                "detail": "global MRI mechanistic panel",
            }
        )
    audit = pd.DataFrame(clinical_audit + feature_audit)
    bmi_check = complete4["weight_kg"] / complete4["height_m"].pow(2)
    bmi_error = (complete4["bmi_kg_m2"] - bmi_check).abs()
    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {
                        "check": "recorded_bmi_vs_weight_height_squared",
                        "patients": len(complete4),
                        "missing": 0,
                        "minimum": float(bmi_error.min()),
                        "median": float(bmi_error.median()),
                        "maximum": float(bmi_error.max()),
                        "detail": f"fraction absolute difference <=2 BMI units: {(bmi_error <= 2).mean():.3f}",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    warning_rows.extend(
        [
            {
                "severity": "warning",
                "code": "COMPLETE4_COUNT_DIFFERS_FROM_EXPECTATION",
                "detail": f"Only {len(complete4)} patients have positive/non-error age, height, weight, and BMI values simultaneously; BMI is available for {len(age_bmi)}.",
            },
            {
                "severity": "warning",
                "code": "HEIGHT_HEADER_UNIT_MISMATCH",
                "detail": "Source header is height_cm, but observed values are 1.6 to 1.8 and were interpreted as metres without modifying the source file.",
            },
            {
                "severity": "warning",
                "code": "RECORDED_BMI_NOT_CONSISTENT_WITH_ROUNDED_HEIGHT_WEIGHT",
                "detail": f"Among complete cases, median absolute recorded-versus-calculated BMI difference is {bmi_error.median():.2f}; only {(bmi_error <= 2).mean():.1%} are within 2 BMI units. Recorded BMI was retained rather than recomputed.",
            },
            {
                "severity": "warning",
                "code": "CLINICAL_MISSINGNESS_NOT_RANDOM",
                "detail": "The 312-patient native-missing analysis can exploit missingness patterns; interpret it as secondary and audit data-source patterns before clinical claims.",
            },
            {
                "severity": "warning",
                "code": "LOCKED7_SELECTION_OVERLAP",
                "detail": "The exact seven canonical-v10 variables were selected with the same 219 labels. v11 reuse is not an independent validation even though the panel is fixed during its CV.",
            },
            {
                "severity": "info",
                "code": "NO_CLINICAL_IMPUTATION",
                "detail": "Clinical complete-case models contain no missing values; the full-cohort XGBoost receives NaN values and missingness indicators, with no mean or median substitution.",
            },
            {
                "severity": "info",
                "code": "SLIP_SEGMENT_OMITTED_FULL_COHORT",
                "detail": "The all-312 analysis uses global MRI features and omits slip-segment localization entirely.",
            },
        ]
    )

    predictions.to_csv(output_dir / "all_repeated_oof_predictions.csv", index=False)
    folds.to_csv(output_dir / "shared_fold_assignments.csv", index=False)
    tuning.to_csv(output_dir / "inner_tuning_choices.csv", index=False)
    each_repeat.to_csv(output_dir / "performance_each_repeat.csv", index=False)
    patient_mean.to_csv(output_dir / "mean_oof_predictions_by_patient.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_performance.csv", index=False)
    paired.to_csv(output_dir / "paired_auc_comparisons.csv", index=False)
    membership.to_csv(output_dir / "cohort_membership_and_clinical_values.csv", index=False)
    audit.to_csv(output_dir / "clinical_and_feature_data_audit.csv", index=False)
    _feature_definitions(locked7).to_csv(
        output_dir / "feature_definitions.csv", index=False
    )
    _locked7_descriptive(locked_data, locked7).to_csv(
        output_dir / "locked7_descriptive_associations.csv", index=False
    )
    pd.DataFrame(warning_rows).to_csv(
        output_dir / "warnings_and_bug_records.csv", index=False
    )

    configuration = {
        **asdict(config),
        "experiment_version": "v11_clinical_mri",
        "runtime_seconds": time.time() - started,
        "cohorts": {
            "complete4": len(complete4),
            "age_bmi": len(age_bmi),
            "all_labeled": len(full),
            "locked7_overlap": len(locked_data),
            "locked7_complete4_overlap": len(locked_complete),
        },
        "global_mri_features": GLOBAL_MRI_FEATURES,
        "locked_v7_features": locked7,
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "prefix": sys.prefix,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_summary(output_dir, aggregate, paired, audit)

    print("v11 completed")
    print(aggregate[["cohort", "feature_set", "model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high"]].to_string(index=False))
    print("\nPaired combined-minus-MRI comparisons")
    print(paired.to_string(index=False))
