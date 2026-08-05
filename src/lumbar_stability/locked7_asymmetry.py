"""v13 locked-seven ablation and asymmetry augmentation experiment.

This module deliberately separates four stages:

1. Seven leave-one-feature-out panels (six historical features each).
2. The complete historical locked-seven panel.
3. Exploratory screening of locked seven plus one non-overlapping asymmetry feature.
4. Selection-aware nested validation where the eighth feature is selected using
   outer-training data only.

The seven historical features were selected with the same 219 labels before
v13.  Therefore every absolute performance estimate in this module remains
exploratory.  Stage 4 removes leakage from selection of the *eighth* feature,
but it cannot undo the historical selection overlap of the base seven.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .io_utils import normalise_patient_ids, read_csv_compatible


@dataclass(frozen=True)
class Locked7AsymmetryConfig:
    label_file: str
    feature_universe_file: str
    locked_selection_file: str
    output_dir: str
    stages: tuple[int, ...] = (1, 2)
    repeats: int = 10
    outer_folds: int = 5
    inner_folds: int = 4
    stage3_repeats: int = 5
    stage4_repeats: int = 5
    stage4_inner_folds: int = 3
    base_seed: int = 20260803
    bootstrap_iterations: int = 3000
    screening_bootstrap_iterations: int = 1000
    permutation_iterations: int = 2000
    n_jobs: int = 1
    model_grids: dict[str, dict[str, list[Any]]] | None = None
    fixed_model_params: dict[str, dict[str, Any]] | None = None
    expected_locked_features: int = 7
    expected_asymmetry_features: int = 78
    expected_new_asymmetry_candidates: int = 75
    candidate_min_nonmissing_fraction: float = 0.80
    validate_only: bool = False
    resume: bool = True


@dataclass
class V13Data:
    table: pd.DataFrame
    locked_features: list[str]
    asymmetry_features: list[str]
    asymmetry_candidates: list[str]
    overlapping_asymmetry_features: list[str]
    feature_manifest: pd.DataFrame
    audit: pd.DataFrame
    warnings: pd.DataFrame


PREDICTION_COLUMNS = [
    "stage",
    "repeat",
    "seed",
    "fold",
    "panel",
    "model",
    "patient_id",
    "label",
    "probability",
]

TUNING_COLUMNS = [
    "stage",
    "repeat",
    "fold",
    "panel",
    "model",
    "best_inner_auc",
    "best_params",
]


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _make_logger(output_dir: Path) -> Callable[[str], None]:
    log_path = output_dir / "run_progress.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    return log


def _safe_panel_token(feature: str, index: int) -> str:
    short = feature.replace("__", "_").replace("/", "_").replace("\\", "_")
    short = "".join(character if character.isalnum() or character == "_" else "_" for character in short)
    # Keep checkpoint paths comfortably below legacy Windows path limits.
    return f"add_{index:03d}_{short[:48]}"


def _load_labels(path: Path) -> pd.DataFrame:
    raw = read_csv_compatible(path, dtype={"patient_id": "string"}, low_memory=False)
    required = {"patient_id", "instability_label"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Label file missing columns: {sorted(missing)}")
    labels = pd.to_numeric(raw["instability_label"], errors="coerce")
    cohort = raw.loc[labels.isin([0, 1]), ["patient_id"]].copy()
    cohort["patient_id"] = normalise_patient_ids(cohort["patient_id"])
    cohort["label"] = labels.loc[labels.isin([0, 1])].astype(int).to_numpy()
    if cohort["patient_id"].isna().any() or cohort["patient_id"].duplicated().any():
        raise ValueError("Labeled cohort must contain unique nonmissing patient_id values")
    return cohort


def load_v13_data(config: Locked7AsymmetryConfig) -> V13Data:
    labels = _load_labels(Path(config.label_file))
    universe = read_csv_compatible(
        Path(config.feature_universe_file),
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    if "patient_id" not in universe.columns:
        raise ValueError("Feature universe missing patient_id")
    universe["patient_id"] = normalise_patient_ids(universe["patient_id"])
    if universe["patient_id"].isna().any() or universe["patient_id"].duplicated().any():
        raise ValueError("Feature universe must contain unique nonmissing patient_id values")

    selection = read_csv_compatible(Path(config.locked_selection_file), low_memory=False)
    required_selection = {"feature", "selected_final_max7"}
    if not required_selection.issubset(selection.columns):
        raise ValueError(
            f"Locked selection file missing columns: {sorted(required_selection - set(selection.columns))}"
        )
    locked_mask = pd.to_numeric(selection["selected_final_max7"], errors="coerce").eq(1)
    locked_features = selection.loc[locked_mask, "feature"].astype(str).tolist()
    if len(locked_features) != config.expected_locked_features:
        raise ValueError(
            f"Expected {config.expected_locked_features} locked features, found {len(locked_features)}"
        )
    missing_locked = [feature for feature in locked_features if feature not in universe.columns]
    if missing_locked:
        raise ValueError(f"Locked features absent from universe: {missing_locked}")

    identifier_columns = {"patient_id", "label", "instability_label"}
    feature_columns = [column for column in universe.columns if column not in identifier_columns]
    asymmetry_features = [
        column for column in feature_columns if "asymmetry" in column.lower()
    ]
    overlapping = [feature for feature in asymmetry_features if feature in locked_features]
    raw_candidates = [feature for feature in asymmetry_features if feature not in locked_features]

    table = universe.drop(columns=[column for column in ("label", "instability_label") if column in universe.columns])
    table = labels.merge(table, on="patient_id", how="inner", validate="one_to_one")
    table = table.sort_values("patient_id").reset_index(drop=True)
    if len(table) != len(universe):
        raise ValueError(
            f"Feature universe/label mismatch: universe={len(universe)}, labeled_match={len(table)}"
        )

    excluded_candidates: list[dict[str, Any]] = []
    eligible_candidates: list[str] = []
    for feature in raw_candidates:
        numeric = pd.to_numeric(table[feature], errors="coerce")
        table[feature] = numeric
        nonmissing_fraction = float(numeric.notna().mean())
        finite_unique = int(numeric.replace([np.inf, -np.inf], np.nan).dropna().nunique())
        if nonmissing_fraction < config.candidate_min_nonmissing_fraction or finite_unique < 2:
            excluded_candidates.append(
                {
                    "feature": feature,
                    "nonmissing_fraction": nonmissing_fraction,
                    "finite_unique_values": finite_unique,
                }
            )
        else:
            eligible_candidates.append(feature)

    for feature in locked_features:
        table[feature] = pd.to_numeric(table[feature], errors="coerce")
        if table[feature].notna().sum() < 2 or table[feature].nunique(dropna=True) < 2:
            raise ValueError(f"Locked feature is unusable: {feature}")

    if len(asymmetry_features) != config.expected_asymmetry_features:
        warnings.warn(
            f"Expected {config.expected_asymmetry_features} asymmetry features, "
            f"found {len(asymmetry_features)}"
        )
    if len(eligible_candidates) != config.expected_new_asymmetry_candidates:
        warnings.warn(
            f"Expected {config.expected_new_asymmetry_candidates} eligible new asymmetry "
            f"candidates, found {len(eligible_candidates)}"
        )

    manifest_rows: list[dict[str, Any]] = []
    for index, feature in enumerate(locked_features, start=1):
        manifest_rows.append(
            {
                "feature": feature,
                "role": "historical_locked7",
                "locked_index": index,
                "is_asymmetry": feature in asymmetry_features,
                "already_in_locked7": True,
                "eligible_stage3_addition": False,
            }
        )
    for feature in asymmetry_features:
        if feature in locked_features:
            continue
        manifest_rows.append(
            {
                "feature": feature,
                "role": "asymmetry_candidate",
                "locked_index": "",
                "is_asymmetry": True,
                "already_in_locked7": feature in overlapping,
                "eligible_stage3_addition": feature in eligible_candidates,
            }
        )

    audit_rows = [
        {"check": "cohort", "value": len(table), "detail": str(table["label"].value_counts().sort_index().to_dict())},
        {"check": "locked_features", "value": len(locked_features), "detail": "historically selected with all 219 labels"},
        {"check": "asymmetry_features_total", "value": len(asymmetry_features), "detail": "feature names containing asymmetry"},
        {"check": "asymmetry_overlap_with_locked7", "value": len(overlapping), "detail": str(overlapping)},
        {"check": "new_asymmetry_candidates_eligible", "value": len(eligible_candidates), "detail": "locked7 overlaps excluded"},
        {"check": "new_asymmetry_candidates_excluded", "value": len(excluded_candidates), "detail": str(excluded_candidates)},
    ]
    warning_rows = [
        {
            "severity": "warning",
            "code": "LOCKED7_OUTCOME_SELECTION_OVERLAP",
            "detail": "The historical seven features were selected using the same 219 labels; every v13 absolute AUC remains exploratory.",
        },
        {
            "severity": "warning",
            "code": "STAGE3_MULTIPLE_COMPARISONS",
            "detail": "Stage 3 evaluates many add-one candidates; raw maximum AUC is not confirmatory and FDR correction is required.",
        },
        {
            "severity": "info",
            "code": "STAGE4_SCOPE",
            "detail": "Stage 4 nests selection of the eighth feature only; it cannot undo historical selection of the base seven.",
        },
        {
            "severity": "info",
            "code": "TRAIN_FOLD_PREPROCESSING",
            "detail": "Median imputation, scaling and tuning are fitted inside training folds only.",
        },
    ]
    return V13Data(
        table=table[["patient_id", "label"] + locked_features + eligible_candidates].copy(),
        locked_features=locked_features,
        asymmetry_features=asymmetry_features,
        asymmetry_candidates=eligible_candidates,
        overlapping_asymmetry_features=overlapping,
        feature_manifest=pd.DataFrame(manifest_rows),
        audit=pd.DataFrame(audit_rows),
        warnings=pd.DataFrame(warning_rows),
    )


def stage1_panels(data: V13Data) -> dict[str, list[str]]:
    return {
        f"loo_without_f{index:02d}": [
            feature for candidate_index, feature in enumerate(data.locked_features, start=1)
            if candidate_index != index
        ]
        for index in range(1, len(data.locked_features) + 1)
    }


def stage2_panels(data: V13Data) -> dict[str, list[str]]:
    return {"locked7_full": list(data.locked_features)}


def stage3_panels(data: V13Data) -> tuple[dict[str, list[str]], pd.DataFrame]:
    panels: dict[str, list[str]] = {"locked7_reference": list(data.locked_features)}
    rows: list[dict[str, str]] = [
        {"panel": "locked7_reference", "added_feature": "", "panel_role": "reference"}
    ]
    for index, feature in enumerate(data.asymmetry_candidates, start=1):
        panel = _safe_panel_token(feature, index)
        panels[panel] = data.locked_features + [feature]
        rows.append(
            {"panel": panel, "added_feature": feature, "panel_role": "add_one_candidate"}
        )
    return panels, pd.DataFrame(rows)


def _model_pipeline(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    labels: np.ndarray,
) -> Pipeline:
    if model_name == "l2_logistic":
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            **params,
        )
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
            **params,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", model),
        ]
    elif model_name == "xgboost":
        negatives = int((labels == 0).sum())
        positives = int((labels == 1).sum())
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=3.0,
            scale_pos_weight=negatives / max(positives, 1),
            random_state=seed,
            n_jobs=1,
            **params,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", model),
        ]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(steps)


def _grid_for_pipeline(grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {f"model__{key}": values for key, values in grid.items()}


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    labels = np.asarray(labels, dtype=int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def _bootstrap_metric_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        if metric == "roc_auc":
            value = roc_auc_score(sampled_labels, probabilities[index])
        elif metric == "pr_auc":
            value = average_precision_score(sampled_labels, probabilities[index])
        elif metric == "brier":
            value = brier_score_loss(sampled_labels, probabilities[index])
        else:  # pragma: no cover
            raise ValueError(metric)
        values.append(float(value))
    return tuple(np.percentile(values, [2.5, 97.5]).tolist())


def _paired_auc_ci(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(
        roc_auc_score(labels, probabilities_a) - roc_auc_score(labels, probabilities_b)
    )
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(
            float(
                roc_auc_score(sampled_labels, probabilities_a[index])
                - roc_auc_score(sampled_labels, probabilities_b[index])
            )
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return observed, float(low), float(high)


def _paired_auc_swap_pvalue(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    iterations: int,
    seed: int,
) -> float:
    observed = abs(
        float(
            roc_auc_score(labels, probabilities_a)
            - roc_auc_score(labels, probabilities_b)
        )
    )
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        swap = rng.random(len(labels)) < 0.5
        permuted_a = np.where(swap, probabilities_b, probabilities_a)
        permuted_b = np.where(swap, probabilities_a, probabilities_b)
        difference = abs(
            float(
                roc_auc_score(labels, permuted_a)
                - roc_auc_score(labels, permuted_b)
            )
        )
        exceed += int(difference >= observed)
    return float((exceed + 1) / (iterations + 1))


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    adjusted = np.full(len(values), np.nan, dtype=float)
    if not valid.any():
        return pd.Series(adjusted, index=p_values.index)
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(values[valid])]
    count = len(order)
    ranked = np.empty(count, dtype=float)
    running = 1.0
    for reverse_position in range(count - 1, -1, -1):
        index = order[reverse_position]
        rank = reverse_position + 1
        running = min(running, values[index] * count / rank)
        ranked[reverse_position] = min(running, 1.0)
    for position, index in enumerate(order):
        adjusted[index] = ranked[position]
    return pd.Series(adjusted, index=p_values.index)


def _fold_assignments(
    data: V13Data,
    repeats: int,
    outer_folds: int,
    base_seed: int,
) -> pd.DataFrame:
    labels = data.table["label"].to_numpy(dtype=int)
    patient_ids = data.table["patient_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        seed = base_seed + repeat
        splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        for fold, (_, test_index) in enumerate(splitter.split(patient_ids, labels)):
            for index in test_index:
                rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "patient_id": patient_ids[index],
                        "label": int(labels[index]),
                    }
                )
    return pd.DataFrame(rows)


def _task_signature(payload: dict[str, Any]) -> str:
    serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _load_task_checkpoint(
    task_dir: Path,
    signature: str,
    expected_test_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    marker_path = task_dir / "complete.json"
    prediction_path = task_dir / "predictions.csv"
    tuning_path = task_dir / "tuning.csv"
    if not all(path.is_file() for path in (marker_path, prediction_path, tuning_path)):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("signature") != signature:
            return None
        prediction_frame = read_csv_compatible(
            prediction_path, dtype={"patient_id": "string"}, low_memory=False
        )
        tuning_frame = read_csv_compatible(tuning_path, low_memory=False)
        actual_ids = prediction_frame["patient_id"].astype(str).tolist()
        if actual_ids != [str(patient_id) for patient_id in expected_test_ids]:
            return None
        if len(tuning_frame) != 1:
            return None
        return prediction_frame, tuning_frame
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return None


def _run_panel_stage(
    config: Locked7AsymmetryConfig,
    data: V13Data,
    stage: int,
    stage_dir: Path,
    panels: dict[str, list[str]],
    repeats: int,
    tune_models: bool,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.model_grids or not config.fixed_model_params:
        raise ValueError("model_grids and fixed_model_params must not be empty")
    model_names = tuple(config.model_grids)
    if set(model_names) != set(config.fixed_model_params):
        raise ValueError("model_grids and fixed_model_params must contain identical models")

    table = data.table
    labels = table["label"].to_numpy(dtype=int)
    patient_ids = table["patient_id"].astype(str).to_numpy()
    predictions: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    total_tasks = repeats * config.outer_folds * len(panels) * len(model_names)
    task_number = 0
    resumed = 0

    for repeat in range(repeats):
        seed = config.base_seed + repeat
        outer = StratifiedKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=seed
        )
        for fold, (train_index, test_index) in enumerate(
            outer.split(patient_ids, labels)
        ):
            y_train = labels[train_index]
            y_test = labels[test_index]
            log(
                f"STAGE {stage} OUTER repeat={repeat + 1}/{repeats} "
                f"fold={fold + 1}/{config.outer_folds}; "
                f"train={len(train_index)}, test={len(test_index)}"
            )
            for panel, features in panels.items():
                x_train = table.iloc[train_index][features]
                x_test = table.iloc[test_index][features]
                for model_name in model_names:
                    task_number += 1
                    task_started = time.time()
                    task_dir = (
                        stage_dir
                        / "checkpoints"
                        / f"repeat_{repeat:02d}"
                        / f"fold_{fold:02d}"
                        / panel
                        / model_name
                    )
                    signature = _task_signature(
                        {
                            "stage": stage,
                            "repeat": repeat,
                            "fold": fold,
                            "panel": panel,
                            "features": features,
                            "model": model_name,
                            "tune_models": tune_models,
                            "grid": config.model_grids[model_name],
                            "fixed_params": config.fixed_model_params[model_name],
                            "inner_folds": config.inner_folds,
                            "base_seed": config.base_seed,
                            "train_ids": patient_ids[train_index].tolist(),
                            "test_ids": patient_ids[test_index].tolist(),
                            "train_labels": y_train.tolist(),
                            "test_labels": y_test.tolist(),
                        }
                    )
                    if config.resume:
                        checkpoint = _load_task_checkpoint(
                            task_dir, signature, patient_ids[test_index]
                        )
                        if checkpoint is not None:
                            prediction_frame, tuning_frame = checkpoint
                            predictions.extend(prediction_frame.to_dict("records"))
                            tuning_rows.extend(tuning_frame.to_dict("records"))
                            resumed += 1
                            log(
                                f"STAGE {stage} RESUME task={task_number}/{total_tasks} "
                                f"panel={panel}, model={model_name}"
                            )
                            continue

                    log(
                        f"STAGE {stage} START task={task_number}/{total_tasks} "
                        f"panel={panel}, model={model_name}"
                    )
                    pipeline = _model_pipeline(
                        model_name,
                        {} if tune_models else config.fixed_model_params[model_name],
                        seed + fold,
                        y_train,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if tune_models:
                            inner = StratifiedKFold(
                                n_splits=config.inner_folds,
                                shuffle=True,
                                random_state=seed + fold + 1000,
                            )
                            search = GridSearchCV(
                                pipeline,
                                _grid_for_pipeline(config.model_grids[model_name]),
                                scoring="roc_auc",
                                cv=inner,
                                refit=True,
                                n_jobs=config.n_jobs,
                                error_score="raise",
                            )
                            search.fit(x_train, y_train)
                            fitted = search.best_estimator_
                            best_inner_auc = float(search.best_score_)
                            best_params = search.best_params_
                        else:
                            pipeline.fit(x_train, y_train)
                            fitted = pipeline
                            best_inner_auc = np.nan
                            best_params = {
                                f"model__{key}": value
                                for key, value in config.fixed_model_params[model_name].items()
                            }
                    probabilities = fitted.predict_proba(x_test)[:, 1]
                    task_predictions = pd.DataFrame(
                        [
                            {
                                "stage": stage,
                                "repeat": repeat,
                                "seed": seed,
                                "fold": fold,
                                "panel": panel,
                                "model": model_name,
                                "patient_id": patient_ids[index],
                                "label": int(labels[index]),
                                "probability": float(probability),
                            }
                            for index, probability in zip(test_index, probabilities)
                        ],
                        columns=PREDICTION_COLUMNS,
                    )
                    tuning_row = {
                        "stage": stage,
                        "repeat": repeat,
                        "fold": fold,
                        "panel": panel,
                        "model": model_name,
                        "best_inner_auc": best_inner_auc,
                        "best_params": json.dumps(
                            best_params, sort_keys=True, ensure_ascii=False
                        ),
                    }
                    task_tuning = pd.DataFrame([tuning_row], columns=TUNING_COLUMNS)
                    _write_csv_atomic(task_predictions, task_dir / "predictions.csv")
                    _write_csv_atomic(task_tuning, task_dir / "tuning.csv")
                    _write_json_atomic(
                        {
                            "signature": signature,
                            "completed_at": datetime.now().isoformat(timespec="seconds"),
                            "test_patients": len(test_index),
                            "test_auc": float(roc_auc_score(y_test, probabilities)),
                        },
                        task_dir / "complete.json",
                    )
                    predictions.extend(task_predictions.to_dict("records"))
                    tuning_rows.append(tuning_row)
                    inner_text = (
                        f"inner_auc={best_inner_auc:.3f}; "
                        if np.isfinite(best_inner_auc)
                        else "fixed_params; "
                    )
                    log(
                        f"STAGE {stage} DONE task={task_number}/{total_tasks} "
                        f"{inner_text}elapsed={time.time() - task_started:.1f}s"
                    )

    log(
        f"STAGE {stage} MODEL TASKS COMPLETE total={total_tasks}, "
        f"resumed={resumed}, newly_fitted={total_tasks - resumed}"
    )
    return (
        pd.DataFrame(predictions, columns=PREDICTION_COLUMNS),
        pd.DataFrame(tuning_rows, columns=TUNING_COLUMNS),
    )


def _summarize_predictions(
    predictions: pd.DataFrame,
    bootstrap_iterations: int,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["panel", "model", "repeat"], sort=False):
        panel, model, repeat = keys
        repeat_rows.append(
            {
                "panel": panel,
                "model": model,
                "repeat": int(repeat),
                **_metrics(
                    group["label"].to_numpy(dtype=int),
                    group["probability"].to_numpy(dtype=float),
                ),
            }
        )
    each_repeat = pd.DataFrame(repeat_rows)
    patient_mean = (
        predictions.groupby(
            ["panel", "model", "patient_id", "label"], as_index=False
        )["probability"]
        .mean()
        .rename(columns={"probability": "mean_oof_probability"})
    )
    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in patient_mean.groupby(["panel", "model"], sort=False):
        panel, model = keys
        labels = group["label"].to_numpy(dtype=int)
        probabilities = group["mean_oof_probability"].to_numpy(dtype=float)
        row = {
            "panel": panel,
            "model": model,
            "patients": len(group),
            "label_0": int((labels == 0).sum()),
            "label_1": int((labels == 1).sum()),
            **_metrics(labels, probabilities),
        }
        for metric in ("roc_auc", "pr_auc", "brier"):
            low, high = _bootstrap_metric_ci(
                labels,
                probabilities,
                metric,
                bootstrap_iterations,
                base_seed + len(aggregate_rows),
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        repeats = each_repeat[
            (each_repeat["panel"] == panel) & (each_repeat["model"] == model)
        ]
        row["repeat_auc_mean"] = float(repeats["roc_auc"].mean())
        row["repeat_auc_sd"] = float(repeats["roc_auc"].std(ddof=1))
        row["repeat_auc_min"] = float(repeats["roc_auc"].min())
        row["repeat_auc_max"] = float(repeats["roc_auc"].max())
        aggregate_rows.append(row)
    return each_repeat, patient_mean, pd.DataFrame(aggregate_rows)


def _save_stage_outputs(
    stage_dir: Path,
    predictions: pd.DataFrame,
    tuning: pd.DataFrame,
    bootstrap_iterations: int,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    each_repeat, patient_mean, aggregate = _summarize_predictions(
        predictions, bootstrap_iterations, base_seed
    )
    for frame, filename in [
        (predictions, "all_repeated_oof_predictions.csv"),
        (tuning, "inner_tuning_choices.csv"),
        (each_repeat, "performance_each_repeat.csv"),
        (patient_mean, "mean_oof_predictions_by_patient.csv"),
        (aggregate, "aggregate_performance.csv"),
    ]:
        _write_csv_atomic(frame, stage_dir / filename)
    return each_repeat, patient_mean, aggregate


def _stage1_ablation_comparison(
    stage1_repeat: pd.DataFrame,
    stage1_patient: pd.DataFrame,
    stage2_repeat: pd.DataFrame,
    stage2_patient: pd.DataFrame,
    data: V13Data,
    config: Locked7AsymmetryConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    panel_to_omitted = {
        f"loo_without_f{index:02d}": feature
        for index, feature in enumerate(data.locked_features, start=1)
    }
    for panel, omitted_feature in panel_to_omitted.items():
        for model in sorted(stage2_patient["model"].unique()):
            full = stage2_patient[
                (stage2_patient["panel"] == "locked7_full")
                & (stage2_patient["model"] == model)
            ][["patient_id", "label", "mean_oof_probability"]].rename(
                columns={"mean_oof_probability": "full_probability"}
            )
            reduced = stage1_patient[
                (stage1_patient["panel"] == panel)
                & (stage1_patient["model"] == model)
            ][["patient_id", "mean_oof_probability"]].rename(
                columns={"mean_oof_probability": "reduced_probability"}
            )
            merged = full.merge(reduced, on="patient_id", validate="one_to_one")
            labels = merged["label"].to_numpy(dtype=int)
            full_probability = merged["full_probability"].to_numpy(dtype=float)
            reduced_probability = merged["reduced_probability"].to_numpy(dtype=float)
            difference, low, high = _paired_auc_ci(
                labels,
                full_probability,
                reduced_probability,
                config.bootstrap_iterations,
                config.base_seed + len(rows),
            )
            repeat_differences: list[float] = []
            for repeat in sorted(stage2_repeat["repeat"].unique()):
                full_auc = stage2_repeat.loc[
                    (stage2_repeat["panel"] == "locked7_full")
                    & (stage2_repeat["model"] == model)
                    & (stage2_repeat["repeat"] == repeat),
                    "roc_auc",
                ].iloc[0]
                reduced_auc = stage1_repeat.loc[
                    (stage1_repeat["panel"] == panel)
                    & (stage1_repeat["model"] == model)
                    & (stage1_repeat["repeat"] == repeat),
                    "roc_auc",
                ].iloc[0]
                repeat_differences.append(float(full_auc - reduced_auc))
            rows.append(
                {
                    "omitted_feature": omitted_feature,
                    "loo_panel": panel,
                    "model": model,
                    "patients": len(merged),
                    "full7_auc": float(roc_auc_score(labels, full_probability)),
                    "loo6_auc": float(roc_auc_score(labels, reduced_probability)),
                    "delta_auc_full7_minus_loo6": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "positive_repeats": int(np.sum(np.asarray(repeat_differences) > 0)),
                    "repeat_count": len(repeat_differences),
                    "mean_repeat_delta_auc": float(np.mean(repeat_differences)),
                    "interpretation": "positive delta means omission reduced AUC",
                }
            )
    return pd.DataFrame(rows)


def _stage3_candidate_leaderboard(
    each_repeat: pd.DataFrame,
    patient_mean: pd.DataFrame,
    panel_manifest: pd.DataFrame,
    config: Locked7AsymmetryConfig,
    log: Callable[[str], None],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_manifest = panel_manifest[
        panel_manifest["panel_role"] == "add_one_candidate"
    ]
    models = sorted(patient_mean["model"].unique())
    total = len(candidate_manifest) * len(models)
    current = 0
    for _, manifest_row in candidate_manifest.iterrows():
        panel = str(manifest_row["panel"])
        feature = str(manifest_row["added_feature"])
        for model in models:
            current += 1
            if current == 1 or current % 15 == 0 or current == total:
                log(f"STAGE 3 COMPARE candidate_model={current}/{total}")
            reference = patient_mean[
                (patient_mean["panel"] == "locked7_reference")
                & (patient_mean["model"] == model)
            ][["patient_id", "label", "mean_oof_probability"]].rename(
                columns={"mean_oof_probability": "reference_probability"}
            )
            candidate = patient_mean[
                (patient_mean["panel"] == panel)
                & (patient_mean["model"] == model)
            ][["patient_id", "mean_oof_probability"]].rename(
                columns={"mean_oof_probability": "candidate_probability"}
            )
            merged = reference.merge(candidate, on="patient_id", validate="one_to_one")
            labels = merged["label"].to_numpy(dtype=int)
            reference_probability = merged["reference_probability"].to_numpy(dtype=float)
            candidate_probability = merged["candidate_probability"].to_numpy(dtype=float)
            difference, low, high = _paired_auc_ci(
                labels,
                candidate_probability,
                reference_probability,
                config.screening_bootstrap_iterations,
                config.base_seed + 10000 + current,
            )
            p_value = _paired_auc_swap_pvalue(
                labels,
                candidate_probability,
                reference_probability,
                config.permutation_iterations,
                config.base_seed + 20000 + current,
            )
            repeat_differences: list[float] = []
            for repeat in sorted(each_repeat["repeat"].unique()):
                reference_auc = each_repeat.loc[
                    (each_repeat["panel"] == "locked7_reference")
                    & (each_repeat["model"] == model)
                    & (each_repeat["repeat"] == repeat),
                    "roc_auc",
                ].iloc[0]
                candidate_auc = each_repeat.loc[
                    (each_repeat["panel"] == panel)
                    & (each_repeat["model"] == model)
                    & (each_repeat["repeat"] == repeat),
                    "roc_auc",
                ].iloc[0]
                repeat_differences.append(float(candidate_auc - reference_auc))
            rows.append(
                {
                    "added_feature": feature,
                    "panel": panel,
                    "model": model,
                    "patients": len(merged),
                    "reference_auc": float(
                        roc_auc_score(labels, reference_probability)
                    ),
                    "candidate_auc": float(
                        roc_auc_score(labels, candidate_probability)
                    ),
                    "delta_auc_candidate_minus_reference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "swap_permutation_p": p_value,
                    "positive_repeats": int(np.sum(np.asarray(repeat_differences) > 0)),
                    "repeat_count": len(repeat_differences),
                    "mean_repeat_delta_auc": float(np.mean(repeat_differences)),
                    "reference_brier": float(
                        brier_score_loss(labels, reference_probability)
                    ),
                    "candidate_brier": float(
                        brier_score_loss(labels, candidate_probability)
                    ),
                    "delta_brier_candidate_minus_reference": float(
                        brier_score_loss(labels, candidate_probability)
                        - brier_score_loss(labels, reference_probability)
                    ),
                }
            )
    leaderboard = pd.DataFrame(rows)
    leaderboard["fdr_q_within_model"] = np.nan
    for model, indices in leaderboard.groupby("model").groups.items():
        leaderboard.loc[indices, "fdr_q_within_model"] = _benjamini_hochberg(
            leaderboard.loc[indices, "swap_permutation_p"]
        ).to_numpy()
    required_positive_repeats = np.ceil(
        0.8 * leaderboard["repeat_count"].to_numpy(dtype=float)
    )
    leaderboard["passes_effect_size"] = (
        leaderboard["delta_auc_candidate_minus_reference"] >= 0.02
    )
    leaderboard["passes_repeat_consistency"] = (
        leaderboard["positive_repeats"].to_numpy() >= required_positive_repeats
    )
    leaderboard["passes_fdr"] = leaderboard["fdr_q_within_model"] <= 0.10
    leaderboard["passes_brier"] = (
        leaderboard["delta_brier_candidate_minus_reference"] <= 0.005
    )
    leaderboard["passes_all_within_model"] = leaderboard[
        ["passes_effect_size", "passes_repeat_consistency", "passes_fdr", "passes_brier"]
    ].all(axis=1)
    return leaderboard.sort_values(
        ["model", "delta_auc_candidate_minus_reference"],
        ascending=[True, False],
    ).reset_index(drop=True)


def _stage3_cross_model_consensus(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, group in leaderboard.groupby("added_feature", sort=False):
        deltas = group["delta_auc_candidate_minus_reference"].to_numpy(dtype=float)
        rows.append(
            {
                "added_feature": feature,
                "models": len(group),
                "models_positive_delta": int((deltas > 0).sum()),
                "models_delta_ge_0_02": int((deltas >= 0.02).sum()),
                "models_passing_all_rules": int(group["passes_all_within_model"].sum()),
                "median_delta_auc_across_models": float(np.median(deltas)),
                "minimum_delta_auc_across_models": float(np.min(deltas)),
                "maximum_delta_auc_across_models": float(np.max(deltas)),
                "robust_candidate": bool(
                    ((deltas > 0).sum() >= 2)
                    and ((deltas >= 0.02).sum() >= 2)
                    and (group["passes_all_within_model"].sum() >= 2)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["robust_candidate", "median_delta_auc_across_models"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _load_stage4_checkpoint(
    task_dir: Path,
    signature: str,
    expected_test_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    marker_path = task_dir / "complete.json"
    prediction_path = task_dir / "predictions.csv"
    score_path = task_dir / "candidate_scores.csv"
    selection_path = task_dir / "selection.csv"
    if not all(
        path.is_file()
        for path in (marker_path, prediction_path, score_path, selection_path)
    ):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("signature") != signature:
            return None
        prediction = read_csv_compatible(
            prediction_path, dtype={"patient_id": "string"}, low_memory=False
        )
        score = read_csv_compatible(score_path, low_memory=False)
        selection = read_csv_compatible(selection_path, low_memory=False)
        reference = prediction[prediction["panel"] == "locked7_reference"]
        actual_ids = reference[reference["model"] == reference["model"].iloc[0]][
            "patient_id"
        ].astype(str).tolist()
        if actual_ids != [str(patient_id) for patient_id in expected_test_ids]:
            return None
        if len(selection) != 1:
            return None
        return prediction, score, selection
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError, pd.errors.ParserError):
        return None


def _run_stage4_nested_selection(
    config: Locked7AsymmetryConfig,
    data: V13Data,
    stage_dir: Path,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not config.fixed_model_params:
        raise ValueError("fixed_model_params must not be empty")
    model_names = tuple(config.fixed_model_params)
    table = data.table
    labels = table["label"].to_numpy(dtype=int)
    patient_ids = table["patient_id"].astype(str).to_numpy()
    all_predictions: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    total_tasks = config.stage4_repeats * config.outer_folds
    task_number = 0
    resumed = 0

    for repeat in range(config.stage4_repeats):
        seed = config.base_seed + repeat
        outer = StratifiedKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=seed
        )
        for fold, (train_index, test_index) in enumerate(
            outer.split(patient_ids, labels)
        ):
            task_number += 1
            task_started = time.time()
            y_train = labels[train_index]
            y_test = labels[test_index]
            task_dir = (
                stage_dir
                / "checkpoints"
                / f"repeat_{repeat:02d}"
                / f"fold_{fold:02d}"
            )
            signature = _task_signature(
                {
                    "stage": 4,
                    "repeat": repeat,
                    "fold": fold,
                    "locked_features": data.locked_features,
                    "candidate_features": data.asymmetry_candidates,
                    "models": config.fixed_model_params,
                    "inner_folds": config.stage4_inner_folds,
                    "base_seed": config.base_seed,
                    "train_ids": patient_ids[train_index].tolist(),
                    "test_ids": patient_ids[test_index].tolist(),
                    "train_labels": y_train.tolist(),
                    "test_labels": y_test.tolist(),
                }
            )
            if config.resume:
                checkpoint = _load_stage4_checkpoint(
                    task_dir, signature, patient_ids[test_index]
                )
                if checkpoint is not None:
                    prediction_frame, score_frame, selection_frame = checkpoint
                    all_predictions.extend(prediction_frame.to_dict("records"))
                    all_scores.extend(score_frame.to_dict("records"))
                    all_selections.extend(selection_frame.to_dict("records"))
                    resumed += 1
                    log(
                        f"STAGE 4 RESUME outer_task={task_number}/{total_tasks} "
                        f"repeat={repeat + 1}, fold={fold + 1}"
                    )
                    continue

            log(
                f"STAGE 4 START outer_task={task_number}/{total_tasks} "
                f"repeat={repeat + 1}/{config.stage4_repeats}, "
                f"fold={fold + 1}/{config.outer_folds}; evaluate "
                f"{len(data.asymmetry_candidates)} candidates inside outer training only"
            )
            inner = StratifiedKFold(
                n_splits=config.stage4_inner_folds,
                shuffle=True,
                random_state=seed + fold + 3000,
            )
            x_base = table.iloc[train_index][data.locked_features]
            base_inner_auc: dict[str, float] = {}
            for model_name in model_names:
                pipeline = _model_pipeline(
                    model_name,
                    config.fixed_model_params[model_name],
                    seed + fold,
                    y_train,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    probabilities = cross_val_predict(
                        pipeline,
                        x_base,
                        y_train,
                        cv=inner,
                        method="predict_proba",
                        n_jobs=config.n_jobs,
                    )[:, 1]
                base_inner_auc[model_name] = float(
                    roc_auc_score(y_train, probabilities)
                )

            score_rows: list[dict[str, Any]] = []
            candidate_total = len(data.asymmetry_candidates)
            for candidate_index, candidate in enumerate(
                data.asymmetry_candidates, start=1
            ):
                if candidate_index == 1 or candidate_index % 10 == 0 or candidate_index == candidate_total:
                    log(
                        f"STAGE 4 outer_task={task_number}/{total_tasks} "
                        f"inner_candidate={candidate_index}/{candidate_total}"
                    )
                features = data.locked_features + [candidate]
                x_candidate = table.iloc[train_index][features]
                candidate_deltas: list[float] = []
                per_model_rows: list[dict[str, Any]] = []
                for model_name in model_names:
                    pipeline = _model_pipeline(
                        model_name,
                        config.fixed_model_params[model_name],
                        seed + fold,
                        y_train,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        probabilities = cross_val_predict(
                            pipeline,
                            x_candidate,
                            y_train,
                            cv=inner,
                            method="predict_proba",
                            n_jobs=config.n_jobs,
                        )[:, 1]
                    candidate_auc = float(roc_auc_score(y_train, probabilities))
                    delta = candidate_auc - base_inner_auc[model_name]
                    candidate_deltas.append(delta)
                    per_model_rows.append(
                        {
                            "stage": 4,
                            "repeat": repeat,
                            "fold": fold,
                            "candidate_feature": candidate,
                            "model": model_name,
                            "base_inner_auc": base_inner_auc[model_name],
                            "candidate_inner_auc": candidate_auc,
                            "delta_inner_auc": delta,
                        }
                    )
                consensus_delta = float(np.median(candidate_deltas))
                for row in per_model_rows:
                    row["median_delta_auc_across_models"] = consensus_delta
                    score_rows.append(row)

            score_frame = pd.DataFrame(score_rows)
            candidate_consensus = (
                score_frame.groupby("candidate_feature", as_index=False)[
                    "median_delta_auc_across_models"
                ]
                .first()
                .sort_values(
                    ["median_delta_auc_across_models", "candidate_feature"],
                    ascending=[False, True],
                )
            )
            selected_feature = str(candidate_consensus.iloc[0]["candidate_feature"])
            selected_score = float(
                candidate_consensus.iloc[0]["median_delta_auc_across_models"]
            )
            selection_row = {
                "stage": 4,
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "selected_feature": selected_feature,
                "median_delta_auc_across_models": selected_score,
                "selection_scope": "outer_training_inner_cv_only",
            }
            prediction_rows: list[dict[str, Any]] = []
            for model_name in model_names:
                for panel, features in (
                    ("locked7_reference", data.locked_features),
                    ("nested_selected_add_one", data.locked_features + [selected_feature]),
                ):
                    pipeline = _model_pipeline(
                        model_name,
                        config.fixed_model_params[model_name],
                        seed + fold,
                        y_train,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipeline.fit(table.iloc[train_index][features], y_train)
                    probabilities = pipeline.predict_proba(
                        table.iloc[test_index][features]
                    )[:, 1]
                    for index, probability in zip(test_index, probabilities):
                        prediction_rows.append(
                            {
                                "stage": 4,
                                "repeat": repeat,
                                "seed": seed,
                                "fold": fold,
                                "panel": panel,
                                "model": model_name,
                                "patient_id": patient_ids[index],
                                "label": int(labels[index]),
                                "probability": float(probability),
                                "selected_feature": selected_feature,
                            }
                        )
            prediction_frame = pd.DataFrame(prediction_rows)
            selection_frame = pd.DataFrame([selection_row])
            _write_csv_atomic(prediction_frame, task_dir / "predictions.csv")
            _write_csv_atomic(score_frame, task_dir / "candidate_scores.csv")
            _write_csv_atomic(selection_frame, task_dir / "selection.csv")
            _write_json_atomic(
                {
                    "signature": signature,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "selected_feature": selected_feature,
                    "test_patients": len(test_index),
                },
                task_dir / "complete.json",
            )
            all_predictions.extend(prediction_rows)
            all_scores.extend(score_rows)
            all_selections.append(selection_row)
            log(
                f"STAGE 4 DONE outer_task={task_number}/{total_tasks}; "
                f"selected={selected_feature}; inner_consensus_delta={selected_score:.3f}; "
                f"elapsed={time.time() - task_started:.1f}s"
            )

    log(
        f"STAGE 4 OUTER TASKS COMPLETE total={total_tasks}, resumed={resumed}, "
        f"newly_fitted={total_tasks - resumed}"
    )
    return (
        pd.DataFrame(all_predictions),
        pd.DataFrame(all_scores),
        pd.DataFrame(all_selections),
    )


def _two_panel_comparison(
    each_repeat: pd.DataFrame,
    patient_mean: pd.DataFrame,
    reference_panel: str,
    candidate_panel: str,
    config: Locked7AsymmetryConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(patient_mean["model"].unique()):
        reference = patient_mean[
            (patient_mean["panel"] == reference_panel)
            & (patient_mean["model"] == model)
        ][["patient_id", "label", "mean_oof_probability"]].rename(
            columns={"mean_oof_probability": "reference_probability"}
        )
        candidate = patient_mean[
            (patient_mean["panel"] == candidate_panel)
            & (patient_mean["model"] == model)
        ][["patient_id", "mean_oof_probability"]].rename(
            columns={"mean_oof_probability": "candidate_probability"}
        )
        merged = reference.merge(candidate, on="patient_id", validate="one_to_one")
        labels = merged["label"].to_numpy(dtype=int)
        reference_probability = merged["reference_probability"].to_numpy(dtype=float)
        candidate_probability = merged["candidate_probability"].to_numpy(dtype=float)
        difference, low, high = _paired_auc_ci(
            labels,
            candidate_probability,
            reference_probability,
            config.bootstrap_iterations,
            config.base_seed + 40000 + len(rows),
        )
        repeat_deltas: list[float] = []
        for repeat in sorted(each_repeat["repeat"].unique()):
            reference_auc = each_repeat.loc[
                (each_repeat["panel"] == reference_panel)
                & (each_repeat["model"] == model)
                & (each_repeat["repeat"] == repeat),
                "roc_auc",
            ].iloc[0]
            candidate_auc = each_repeat.loc[
                (each_repeat["panel"] == candidate_panel)
                & (each_repeat["model"] == model)
                & (each_repeat["repeat"] == repeat),
                "roc_auc",
            ].iloc[0]
            repeat_deltas.append(float(candidate_auc - reference_auc))
        rows.append(
            {
                "model": model,
                "reference_panel": reference_panel,
                "candidate_panel": candidate_panel,
                "patients": len(merged),
                "reference_auc": float(roc_auc_score(labels, reference_probability)),
                "candidate_auc": float(roc_auc_score(labels, candidate_probability)),
                "delta_auc_candidate_minus_reference": difference,
                "ci_low": low,
                "ci_high": high,
                "positive_repeats": int(np.sum(np.asarray(repeat_deltas) > 0)),
                "repeat_count": len(repeat_deltas),
                "mean_repeat_delta_auc": float(np.mean(repeat_deltas)),
            }
        )
    return pd.DataFrame(rows)


def _write_root_summary(
    output_dir: Path,
    data: V13Data,
    completed_stages: list[int],
) -> None:
    lines = [
        "# v13 历史7项消融与左右不对称增量实验",
        "",
        f"- 患者：{len(data.table)}例；标签：{data.table['label'].value_counts().sort_index().to_dict()}。",
        f"- 历史锁定特征：{len(data.locked_features)}项。",
        f"- 左右不对称特征：{len(data.asymmetry_features)}项；排除与7项重叠后新增候选：{len(data.asymmetry_candidates)}项。",
        f"- 本次已完成阶段：{sorted(completed_stages)}。",
        "",
        "## 证据边界",
        "",
        "- 历史7项曾使用同一219例标签筛选，因此所有绝对AUC均属于探索性内部结果。",
        "- 阶段3是多候选筛查，不能把最高AUC直接当作验证结果。",
        "- 阶段4只对第8项特征的选择过程进行训练折内嵌套，不能消除基础7项的历史选择重叠。",
        "",
    ]
    for stage, directory in [
        (1, "stage_01_leave_one_out"),
        (2, "stage_02_full_locked7"),
        (3, "stage_03_add_one_screen"),
        (4, "stage_04_nested_selection"),
    ]:
        aggregate_path = output_dir / directory / "aggregate_performance.csv"
        if not aggregate_path.is_file():
            continue
        aggregate = read_csv_compatible(aggregate_path, low_memory=False)
        lines.extend([f"## 阶段{stage}结果索引", ""])
        if stage in (2, 4):
            lines.extend(
                [
                    "| 面板 | 模型 | AUC | 95%CI | PR-AUC | Brier |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for _, row in aggregate.sort_values(["panel", "model"]).iterrows():
                lines.append(
                    f"| {row['panel']} | {row['model']} | {row['roc_auc']:.3f} | "
                    f"{row['roc_auc_ci_low']:.3f}–{row['roc_auc_ci_high']:.3f} | "
                    f"{row['pr_auc']:.3f} | {row['brier']:.3f} |"
                )
        else:
            lines.append(
                f"该阶段包含{aggregate['panel'].nunique()}个面板和"
                f"{aggregate['model'].nunique()}个模型；详细结果见阶段目录CSV。"
            )
        lines.append("")
    (output_dir / "RESULTS_SUMMARY_CN.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def _detect_completed_stages(output_dir: Path) -> list[int]:
    stage_directories = {
        1: "stage_01_leave_one_out",
        2: "stage_02_full_locked7",
        3: "stage_03_add_one_screen",
        4: "stage_04_nested_selection",
    }
    return [
        stage
        for stage, directory in stage_directories.items()
        if (output_dir / directory / "aggregate_performance.csv").is_file()
    ]


def run_locked7_asymmetry_experiment(config: Locked7AsymmetryConfig) -> None:
    invalid_stages = sorted(set(config.stages) - {1, 2, 3, 4})
    if invalid_stages or not config.stages:
        raise ValueError(f"stages must be a nonempty subset of 1,2,3,4; invalid={invalid_stages}")
    started = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(output_dir)
    run_state_path = output_dir / "run_state.json"
    completed_stages: list[int] = []
    log(
        f"START v13 locked7-asymmetry experiment; executable={sys.executable}; "
        f"stages={list(config.stages)}; resume={config.resume}; "
        f"validate_only={config.validate_only}"
    )
    _write_json_atomic(
        {
            "status": "running",
            "started_at": started_at,
            "requested_stages": list(config.stages),
            "completed_stages": completed_stages,
            "resume": config.resume,
        },
        run_state_path,
    )
    try:
        log("LOAD labels, historical locked7 and asymmetry feature universe")
        data = load_v13_data(config)
        _write_csv_atomic(data.feature_manifest, output_dir / "feature_manifest.csv")
        _write_csv_atomic(data.audit, output_dir / "data_quality_report.csv")
        _write_csv_atomic(data.warnings, output_dir / "warnings_and_bug_records.csv")
        _write_csv_atomic(
            data.table[["patient_id", "label"] + data.locked_features],
            output_dir / "locked7_patient_table.csv",
        )
        log(
            f"LOAD COMPLETE patients={len(data.table)}, "
            f"labels={data.table['label'].value_counts().sort_index().to_dict()}, "
            f"locked7={len(data.locked_features)}, asymmetry_total={len(data.asymmetry_features)}, "
            f"new_candidates={len(data.asymmetry_candidates)}"
        )

        max_repeats = max(config.repeats, config.stage3_repeats, config.stage4_repeats)
        shared_folds = _fold_assignments(
            data, max_repeats, config.outer_folds, config.base_seed
        )
        _write_csv_atomic(shared_folds, output_dir / "shared_outer_fold_assignments.csv")

        configuration = {
            **asdict(config),
            "environment": {
                "python_executable": sys.executable,
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            "cohort": {
                "patients": len(data.table),
                "label_counts": data.table["label"].value_counts().sort_index().to_dict(),
            },
            "locked_features": data.locked_features,
            "asymmetry_features_total": len(data.asymmetry_features),
            "overlapping_asymmetry_features": data.overlapping_asymmetry_features,
            "new_asymmetry_candidates": data.asymmetry_candidates,
            "evidence_boundary": (
                "Exploratory: base seven were historically selected with all 219 labels"
            ),
        }
        _write_json_atomic(configuration, output_dir / "experiment_config.json")

        if config.validate_only:
            _write_root_summary(output_dir, data, completed_stages)
            _write_json_atomic(
                {
                    "status": "validation_only_completed",
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "requested_stages": list(config.stages),
                    "completed_stages": [],
                    "runtime_seconds": time.time() - started,
                },
                run_state_path,
            )
            log("END validation-only; models_fitted=0")
            return

        stage1_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
        stage2_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None

        if 1 in config.stages:
            stage_dir = output_dir / "stage_01_leave_one_out"
            panels = stage1_panels(data)
            _write_csv_atomic(
                pd.DataFrame(
                    [
                        {
                            "panel": panel,
                            "feature_count": len(features),
                            "features": json.dumps(features, ensure_ascii=False),
                        }
                        for panel, features in panels.items()
                    ]
                ),
                stage_dir / "panel_manifest.csv",
            )
            log(
                f"STAGE 1 BEGIN panels={len(panels)}, models={len(config.model_grids or {})}, "
                f"repeats={config.repeats}, folds={config.outer_folds}; nested tuning enabled"
            )
            predictions, tuning = _run_panel_stage(
                config,
                data,
                1,
                stage_dir,
                panels,
                config.repeats,
                True,
                log,
            )
            stage1_outputs = _save_stage_outputs(
                stage_dir,
                predictions,
                tuning,
                config.bootstrap_iterations,
                config.base_seed + 100,
            )
            completed_stages.append(1)
            log("STAGE 1 END")

        if 2 in config.stages:
            stage_dir = output_dir / "stage_02_full_locked7"
            panels = stage2_panels(data)
            _write_csv_atomic(
                pd.DataFrame(
                    [
                        {
                            "panel": "locked7_full",
                            "feature_count": len(data.locked_features),
                            "features": json.dumps(data.locked_features, ensure_ascii=False),
                        }
                    ]
                ),
                stage_dir / "panel_manifest.csv",
            )
            log(
                f"STAGE 2 BEGIN full_locked7; models={len(config.model_grids or {})}, "
                f"repeats={config.repeats}, folds={config.outer_folds}; nested tuning enabled"
            )
            predictions, tuning = _run_panel_stage(
                config,
                data,
                2,
                stage_dir,
                panels,
                config.repeats,
                True,
                log,
            )
            stage2_outputs = _save_stage_outputs(
                stage_dir,
                predictions,
                tuning,
                config.bootstrap_iterations,
                config.base_seed + 200,
            )
            completed_stages.append(2)
            log("STAGE 2 END")

        stage1_dir = output_dir / "stage_01_leave_one_out"
        stage2_dir = output_dir / "stage_02_full_locked7"
        if stage1_outputs is None and all(
            (stage1_dir / filename).is_file()
            for filename in ("performance_each_repeat.csv", "mean_oof_predictions_by_patient.csv")
        ):
            stage1_outputs = (
                read_csv_compatible(stage1_dir / "performance_each_repeat.csv", low_memory=False),
                read_csv_compatible(stage1_dir / "mean_oof_predictions_by_patient.csv", dtype={"patient_id": "string"}, low_memory=False),
                read_csv_compatible(stage1_dir / "aggregate_performance.csv", low_memory=False),
            )
        if stage2_outputs is None and all(
            (stage2_dir / filename).is_file()
            for filename in ("performance_each_repeat.csv", "mean_oof_predictions_by_patient.csv")
        ):
            stage2_outputs = (
                read_csv_compatible(stage2_dir / "performance_each_repeat.csv", low_memory=False),
                read_csv_compatible(stage2_dir / "mean_oof_predictions_by_patient.csv", dtype={"patient_id": "string"}, low_memory=False),
                read_csv_compatible(stage2_dir / "aggregate_performance.csv", low_memory=False),
            )
        if stage1_outputs is not None and stage2_outputs is not None:
            ablation = _stage1_ablation_comparison(
                stage1_outputs[0],
                stage1_outputs[1],
                stage2_outputs[0],
                stage2_outputs[1],
                data,
                config,
            )
            _write_csv_atomic(
                ablation, output_dir / "stage_01_02_paired_ablation_comparison.csv"
            )

        if 3 in config.stages:
            stage_dir = output_dir / "stage_03_add_one_screen"
            panels, panel_manifest = stage3_panels(data)
            _write_csv_atomic(panel_manifest, stage_dir / "panel_manifest.csv")
            log(
                f"STAGE 3 BEGIN panels={len(panels)} (1 reference + "
                f"{len(data.asymmetry_candidates)} candidates), "
                f"models={len(config.fixed_model_params or {})}, "
                f"repeats={config.stage3_repeats}, folds={config.outer_folds}; fixed parameters"
            )
            predictions, tuning = _run_panel_stage(
                config,
                data,
                3,
                stage_dir,
                panels,
                config.stage3_repeats,
                False,
                log,
            )
            each_repeat, patient_mean, _ = _save_stage_outputs(
                stage_dir,
                predictions,
                tuning,
                config.screening_bootstrap_iterations,
                config.base_seed + 300,
            )
            leaderboard = _stage3_candidate_leaderboard(
                each_repeat, patient_mean, panel_manifest, config, log
            )
            consensus = _stage3_cross_model_consensus(leaderboard)
            _write_csv_atomic(leaderboard, stage_dir / "candidate_leaderboard_by_model.csv")
            _write_csv_atomic(consensus, stage_dir / "candidate_cross_model_consensus.csv")
            completed_stages.append(3)
            log("STAGE 3 END")

        if 4 in config.stages:
            stage_dir = output_dir / "stage_04_nested_selection"
            log(
                f"STAGE 4 BEGIN candidates={len(data.asymmetry_candidates)}, "
                f"outer_repeats={config.stage4_repeats}, outer_folds={config.outer_folds}, "
                f"inner_folds={config.stage4_inner_folds}; eighth-feature selection nested"
            )
            predictions, candidate_scores, selections = _run_stage4_nested_selection(
                config, data, stage_dir, log
            )
            empty_tuning = pd.DataFrame(columns=TUNING_COLUMNS)
            each_repeat, patient_mean, _ = _save_stage_outputs(
                stage_dir,
                predictions,
                empty_tuning,
                config.bootstrap_iterations,
                config.base_seed + 400,
            )
            _write_csv_atomic(candidate_scores, stage_dir / "all_inner_candidate_scores.csv")
            _write_csv_atomic(selections, stage_dir / "outer_fold_selected_features.csv")
            selection_frequency = (
                selections.groupby("selected_feature", as_index=False)
                .size()
                .rename(columns={"size": "selected_outer_folds"})
                .sort_values("selected_outer_folds", ascending=False)
            )
            selection_frequency["selection_fraction"] = (
                selection_frequency["selected_outer_folds"] / len(selections)
            )
            _write_csv_atomic(
                selection_frequency, stage_dir / "selected_feature_frequency.csv"
            )
            comparison = _two_panel_comparison(
                each_repeat,
                patient_mean,
                "locked7_reference",
                "nested_selected_add_one",
                config,
            )
            _write_csv_atomic(
                comparison, stage_dir / "paired_nested_selection_increment.csv"
            )
            completed_stages.append(4)
            log("STAGE 4 END")

        all_completed_stages = sorted(
            set(_detect_completed_stages(output_dir)) | set(completed_stages)
        )
        configuration["runtime_seconds_last_invocation"] = time.time() - started
        configuration["completed_stages_last_invocation"] = completed_stages
        configuration["all_completed_stages"] = all_completed_stages
        _write_json_atomic(configuration, output_dir / "experiment_config.json")
        _write_root_summary(output_dir, data, all_completed_stages)
        _write_json_atomic(
            {
                "status": "completed",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "requested_stages": list(config.stages),
                "completed_stages": all_completed_stages,
                "runtime_seconds": time.time() - started,
            },
            run_state_path,
        )
        log(
            f"END completed requested_stages={completed_stages}, "
            f"all_completed_stages={all_completed_stages}; "
            f"runtime={time.time() - started:.1f}s"
        )
    except Exception as exc:
        _write_json_atomic(
            {
                "status": "failed",
                "started_at": started_at,
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "requested_stages": list(config.stages),
                "completed_stages": completed_stages,
                "runtime_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            run_state_path,
        )
        log(f"ERROR {type(exc).__name__}: {exc}")
        raise
