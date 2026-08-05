"""Prior-knowledge target-level muscle features and tree-model validation.

The primary panel is label independent. The historical locked seven-feature
panel is retained as an explicitly exploratory sensitivity analysis because it
was selected with the same 219 outcome labels in an earlier experiment.
"""

from __future__ import annotations

import json
import hashlib
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .io_utils import normalise_patient_ids, read_csv_compatible


MUSCLE_PAIRS = {
    "multifidus": ("multifidus_left", "multifidus_right"),
    "erector_spinae": ("erector_spinae_left", "erector_spinae_right"),
    "psoas": ("psoas_left", "psoas_right"),
}
SLIP_TO_DISC = {
    "L1": "L1_L2_DISC",
    "L2": "L2_L3_DISC",
    "L3": "L3_L4_DISC",
    "L4": "L4_L5_DISC",
    "L5": "L5_S1_DISC",
}


@dataclass(frozen=True)
class PriorMLConfig:
    label_file: str
    annotation_file: str
    feature_2d_file: str
    locked_feature_universe_file: str
    locked_selection_file: str
    output_dir: str
    csa_column: str = "Area"
    mean_gray_column: str = "Mean_Intensity_Muscle"
    include_target_slip_segment: bool = True
    feature_sets: tuple[str, ...] = ("prior_core", "prior_plus_locked7")
    repeats: int = 10
    outer_folds: int = 5
    inner_folds: int = 4
    base_seed: int = 20260802
    bootstrap_iterations: int = 3000
    permutation_repeats: int = 10
    n_jobs: int = 1
    model_grids: dict[str, dict[str, list[Any]]] | None = None
    validate_only: bool = False
    resume: bool = True


@dataclass
class FeatureBuildResult:
    table: pd.DataFrame
    core_features: list[str]
    locked_features: list[str]
    categorical_features: list[str]
    dictionary: pd.DataFrame
    audit: pd.DataFrame
    warnings: list[dict[str, str]]


IMPORTANCE_COLUMNS = [
    "repeat",
    "fold",
    "feature_set",
    "model",
    "feature",
    "permutation_auc_drop_mean",
    "permutation_auc_drop_sd",
]


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV completely before replacing its visible destination."""
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


def _asymmetry(left: pd.Series, right: pd.Series) -> pd.Series:
    present = left.notna() & right.notna()
    value = 2.0 * (left - right).abs() / (left.abs() + right.abs() + 1e-8)
    return value.where(present)


def _load_labeled_cohort(path: Path) -> pd.DataFrame:
    raw = read_csv_compatible(path, dtype={"patient_id": "string"}, low_memory=False)
    required = {"patient_id", "instability_label", "target_slip_segment"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Label file missing columns: {sorted(missing)}")
    labels = pd.to_numeric(raw["instability_label"], errors="coerce")
    cohort = raw.loc[
        labels.isin([0, 1]), ["patient_id", "target_slip_segment"]
    ].copy()
    cohort["patient_id"] = normalise_patient_ids(cohort["patient_id"])
    cohort["label"] = labels.loc[labels.isin([0, 1])].astype(int).to_numpy()
    cohort["target_slip_segment"] = (
        cohort["target_slip_segment"].astype("string").str.strip().str.upper()
    )
    cohort["target_level"] = cohort["target_slip_segment"].map(SLIP_TO_DISC)
    if cohort["patient_id"].isna().any() or cohort["patient_id"].duplicated().any():
        raise ValueError("Labeled cohort must contain unique nonmissing patient_id values")
    if cohort["target_level"].isna().any():
        invalid = cohort.loc[cohort["target_level"].isna(), "target_slip_segment"].unique()
        raise ValueError(f"Unsupported target_slip_segment values: {invalid.tolist()}")
    return cohort


def build_prior_feature_table(config: PriorMLConfig) -> FeatureBuildResult:
    labels = _load_labeled_cohort(Path(config.label_file))
    annotations = read_csv_compatible(
        Path(config.annotation_file), dtype={"patient_id": "string"}, low_memory=False
    )
    two_d = read_csv_compatible(
        Path(config.feature_2d_file), dtype={"patient_id": "string"}, low_memory=False
    )
    required_annotation = {"patient_id", "slice_index", "anatomical_level"}
    required_2d = {
        "patient_id",
        "slice_index",
        "muscle_name",
        config.csa_column,
        config.mean_gray_column,
    }
    if not required_annotation.issubset(annotations.columns):
        raise ValueError(
            f"Annotation file missing columns: {sorted(required_annotation - set(annotations.columns))}"
        )
    if not required_2d.issubset(two_d.columns):
        raise ValueError(f"2D file missing columns: {sorted(required_2d - set(two_d.columns))}")
    annotations["patient_id"] = normalise_patient_ids(annotations["patient_id"])
    two_d["patient_id"] = normalise_patient_ids(two_d["patient_id"])
    annotations["slice_index"] = pd.to_numeric(annotations["slice_index"], errors="raise").astype(int)
    two_d["slice_index"] = pd.to_numeric(two_d["slice_index"], errors="raise").astype(int)
    if annotations.duplicated(["patient_id", "slice_index"]).any():
        raise ValueError("Annotation has duplicate patient_id/slice_index rows")

    annotation_ids = annotations["patient_id"].dropna().drop_duplicates().tolist()
    cohort = labels[labels["patient_id"].isin(annotation_ids)].copy()
    if len(cohort) != len(annotation_ids):
        raise ValueError(
            f"Annotation/label mismatch: annotations={len(annotation_ids)}, labeled match={len(cohort)}"
        )
    cohort = cohort.sort_values("patient_id").reset_index(drop=True)
    if len(cohort) != 219:
        warnings.warn(f"Expected historical 219-patient cohort, found {len(cohort)}")

    two_d[config.csa_column] = pd.to_numeric(two_d[config.csa_column], errors="coerce")
    two_d[config.mean_gray_column] = pd.to_numeric(
        two_d[config.mean_gray_column], errors="coerce"
    )
    selected = two_d[two_d["patient_id"].isin(annotation_ids)].merge(
        annotations[["patient_id", "slice_index", "anatomical_level"]],
        on=["patient_id", "slice_index"],
        how="inner",
        validate="many_to_one",
    )
    selected = selected.merge(
        cohort[["patient_id", "target_level"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    target = selected[selected["anatomical_level"] == selected["target_level"]].copy()
    expected_muscles = [muscle for pair in MUSCLE_PAIRS.values() for muscle in pair]
    unknown_muscles = sorted(set(target["muscle_name"].dropna()) - set(expected_muscles))
    if unknown_muscles:
        raise ValueError(f"Unexpected muscle names: {unknown_muscles}")

    medians = (
        target.groupby(["patient_id", "muscle_name"])[
            [config.csa_column, config.mean_gray_column]
        ]
        .median()
        .unstack("muscle_name")
        .reindex(index=cohort["patient_id"], columns=pd.MultiIndex.from_product(
            [[config.csa_column, config.mean_gray_column], expected_muscles]
        ))
    )
    table = cohort[["patient_id", "target_slip_segment", "label"]].copy()
    core_features: list[str] = []
    dictionary_rows: list[dict[str, str]] = []
    for group, (left, right) in MUSCLE_PAIRS.items():
        left_csa = f"target__{left}__CSA_mm2"
        right_csa = f"target__{right}__CSA_mm2"
        left_gray = f"target__{left}__Mean_Gray"
        right_gray = f"target__{right}__Mean_Gray"
        csa_asym = f"target__{group}__CSA_asymmetry"
        gray_asym = f"target__{group}__Mean_Gray_asymmetry"
        table[left_csa] = medians[(config.csa_column, left)].to_numpy()
        table[right_csa] = medians[(config.csa_column, right)].to_numpy()
        table[left_gray] = medians[(config.mean_gray_column, left)].to_numpy()
        table[right_gray] = medians[(config.mean_gray_column, right)].to_numpy()
        table[csa_asym] = _asymmetry(table[left_csa], table[right_csa])
        table[gray_asym] = _asymmetry(table[left_gray], table[right_gray])
        for name, source, relation, definition in [
            (left_csa, config.csa_column, left, "Median target-level CSA across annotated slices"),
            (right_csa, config.csa_column, right, "Median target-level CSA across annotated slices"),
            (left_gray, config.mean_gray_column, left, "Median target-level mean MRI intensity"),
            (right_gray, config.mean_gray_column, right, "Median target-level mean MRI intensity"),
            (csa_asym, config.csa_column, group, "2*abs(left-right)/(abs(left)+abs(right))"),
            (gray_asym, config.mean_gray_column, group, "2*abs(left-right)/(abs(left)+abs(right))"),
        ]:
            dictionary_rows.append(
                {
                    "feature": name,
                    "panel": "prior_core",
                    "source": Path(config.feature_2d_file).name,
                    "base_column": source,
                    "muscle_or_group": relation,
                    "definition": definition,
                    "outcome_selected": "no",
                }
            )
        core_features.extend([left_csa, right_csa, left_gray, right_gray, csa_asym, gray_asym])

    selection = read_csv_compatible(Path(config.locked_selection_file), low_memory=False)
    selected_mask = pd.to_numeric(selection["selected_final_max7"], errors="coerce").eq(1)
    locked_features = selection.loc[selected_mask, "feature"].astype(str).tolist()
    if len(locked_features) != 7:
        raise ValueError(f"Expected seven locked historical features, found {len(locked_features)}")
    universe = read_csv_compatible(
        Path(config.locked_feature_universe_file),
        dtype={"patient_id": "string"},
        low_memory=False,
    )
    universe["patient_id"] = normalise_patient_ids(universe["patient_id"])
    missing_locked = [feature for feature in locked_features if feature not in universe.columns]
    if missing_locked:
        raise ValueError(f"Locked features missing from universe: {missing_locked}")
    if universe["patient_id"].duplicated().any():
        raise ValueError("Locked feature universe has duplicate patient_id values")
    table = table.merge(
        universe[["patient_id"] + locked_features],
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    if table[locked_features].isna().all(axis=1).any():
        raise ValueError("At least one cohort patient is absent from the locked feature universe")
    dictionary_rows.extend(
        {
            "feature": feature,
            "panel": "locked7_exploratory",
            "source": Path(config.locked_feature_universe_file).name,
            "base_column": feature,
            "muscle_or_group": "see historical feature dictionary",
            "definition": "Historical locked feature; selected using all 219 labels before v12",
            "outcome_selected": "yes",
        }
        for feature in locked_features
    )
    categorical_features = ["target_slip_segment"] if config.include_target_slip_segment else []
    audit_rows = [
        {
            "check": "cohort",
            "value": len(table),
            "detail": str(table["label"].value_counts().sort_index().to_dict()),
        },
        {
            "check": "target_slip_segment_counts",
            "value": table["target_slip_segment"].nunique(),
            "detail": str(table["target_slip_segment"].value_counts().to_dict()),
        },
        {
            "check": "target_rows_after_annotation_join",
            "value": len(target),
            "detail": f"expected={len(table) * 4 * len(expected_muscles)}",
        },
        {
            "check": "patients_with_any_prior_core_missing",
            "value": int(table[core_features].isna().any(axis=1).sum()),
            "detail": str(table.loc[table[core_features].isna().any(axis=1), "patient_id"].tolist()),
        },
        {
            "check": "prior_core_feature_count",
            "value": len(core_features),
            "detail": "fixed before v12 outcome modeling",
        },
        {
            "check": "locked_feature_count",
            "value": len(locked_features),
            "detail": "historically outcome selected; exploratory only",
        },
    ]
    warning_rows = [
        {
            "severity": "warning",
            "code": "SEGMENT_NEAR_CONSTANT",
            "detail": f"target_slip_segment counts={table['target_slip_segment'].value_counts().to_dict()}; segment provides almost no estimable population-level information.",
        },
        {
            "severity": "warning",
            "code": "LOCKED7_OUTCOME_SELECTION_OVERLAP",
            "detail": "The historical seven features were selected using the same 219 labels; prior_plus_locked7 is exploratory and optimistically biased.",
        },
        {
            "severity": "warning",
            "code": "ANNOTATION_PROTOCOL_INFERRED",
            "detail": "The 219 slice-level anatomical mapping is protocol-derived rather than individually reviewed anatomy.",
        },
        {
            "severity": "info",
            "code": "TRAIN_FOLD_IMPUTATION_ONLY",
            "detail": "Missing numeric values are retained in the table and median-imputed inside each training fold only.",
        },
        {
            "severity": "info",
            "code": "MRI_GRAY_NOT_DIRECT_FAT_FRACTION",
            "detail": "Mean MRI gray intensity is sequence- and normalization-dependent and must not be described as a quantitative fat fraction without sequence-specific validation.",
        },
    ]
    return FeatureBuildResult(
        table=table,
        core_features=core_features,
        locked_features=locked_features,
        categorical_features=categorical_features,
        dictionary=pd.DataFrame(dictionary_rows),
        audit=pd.DataFrame(audit_rows),
        warnings=warning_rows,
    )


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    labels = np.asarray(labels, dtype=int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def _bootstrap_ci(
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
        if len(np.unique(labels[index])) < 2:
            continue
        if metric == "roc_auc":
            value = roc_auc_score(labels[index], probabilities[index])
        elif metric == "pr_auc":
            value = average_precision_score(labels[index], probabilities[index])
        elif metric == "brier":
            value = brier_score_loss(labels[index], probabilities[index])
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
        if len(np.unique(labels[index])) < 2:
            continue
        values.append(
            float(
                roc_auc_score(labels[index], probabilities_a[index])
                - roc_auc_score(labels[index], probabilities_b[index])
            )
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return observed, float(low), float(high)


def _preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ]
            ),
            numeric_features,
        )
    ]
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def _model_and_grid(
    model_name: str,
    grid: dict[str, list[Any]],
    seed: int,
    labels: np.ndarray,
) -> tuple[Any, dict[str, list[Any]]]:
    if model_name == "xgboost":
        negatives = int((labels == 0).sum())
        positives = int((labels == 1).sum())
        estimator = XGBClassifier(
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
        )
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=3.0,
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=1,
        )
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    parameter_grid = {f"model__{key}": values for key, values in grid.items()}
    return estimator, parameter_grid


def _feature_sets(build: FeatureBuildResult, requested: tuple[str, ...]) -> dict[str, list[str]]:
    available = {
        "prior_core": build.core_features,
        "prior_plus_locked7": build.core_features + build.locked_features,
    }
    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise ValueError(f"Unsupported feature sets: {invalid}")
    return {name: available[name] for name in requested}


def _run_models(
    config: PriorMLConfig,
    build: FeatureBuildResult,
    output_dir: Path,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not config.model_grids:
        raise ValueError("model_grids must not be empty")
    table = build.table
    labels = table["label"].astype(int).to_numpy()
    patient_ids = table["patient_id"].astype(str).to_numpy()
    feature_sets = _feature_sets(build, config.feature_sets)
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    checkpoint_root = output_dir / "checkpoints"
    total_tasks = (
        config.repeats
        * config.outer_folds
        * len(feature_sets)
        * len(config.model_grids)
    )
    task_number = 0
    resumed_tasks = 0

    def task_signature(
        repeat: int,
        fold: int,
        feature_set: str,
        model_name: str,
        numeric_features: list[str],
        grid: dict[str, list[Any]],
        train_index: np.ndarray,
        test_index: np.ndarray,
    ) -> str:
        payload = {
            "repeat": repeat,
            "fold": fold,
            "feature_set": feature_set,
            "model": model_name,
            "numeric_features": numeric_features,
            "categorical_features": build.categorical_features,
            "grid": grid,
            "inner_folds": config.inner_folds,
            "permutation_repeats": config.permutation_repeats,
            "base_seed": config.base_seed,
            "train_patient_ids": patient_ids[train_index].tolist(),
            "test_patient_ids": patient_ids[test_index].tolist(),
            "train_labels": labels[train_index].tolist(),
            "test_labels": labels[test_index].tolist(),
        }
        serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def load_checkpoint(
        task_dir: Path,
        signature: str,
        expected_test_ids: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        marker_path = task_dir / "complete.json"
        prediction_path = task_dir / "predictions.csv"
        tuning_path = task_dir / "tuning.csv"
        importance_path = task_dir / "permutation_importance.csv"
        if not all(
            path.is_file()
            for path in (marker_path, prediction_path, tuning_path, importance_path)
        ):
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("signature") != signature:
                return None
            prediction_frame = read_csv_compatible(
                prediction_path, dtype={"patient_id": "string"}, low_memory=False
            )
            tuning_frame = read_csv_compatible(tuning_path, low_memory=False)
            importance_frame = read_csv_compatible(importance_path, low_memory=False)
            actual_ids = prediction_frame["patient_id"].astype(str).tolist()
            if actual_ids != [str(item) for item in expected_test_ids]:
                return None
            if len(tuning_frame) != 1:
                return None
            return prediction_frame, tuning_frame, importance_frame
        except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
            return None

    for repeat in range(config.repeats):
        seed = config.base_seed + repeat
        outer = StratifiedKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=seed
        )
        for fold, (train_index, test_index) in enumerate(outer.split(patient_ids, labels)):
            log(
                f"OUTER repeat={repeat + 1}/{config.repeats} "
                f"fold={fold + 1}/{config.outer_folds}; "
                f"train={len(train_index)}, test={len(test_index)}"
            )
            y_train = labels[train_index]
            y_test = labels[test_index]
            for index in test_index:
                folds.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "patient_id": patient_ids[index],
                        "label": int(labels[index]),
                    }
                )
            _write_csv_atomic(
                pd.DataFrame(folds).drop_duplicates(),
                output_dir / "outer_fold_assignments.csv",
            )
            for feature_set, numeric_features in feature_sets.items():
                input_features = numeric_features + build.categorical_features
                x_train = table.iloc[train_index][input_features]
                x_test = table.iloc[test_index][input_features]
                for model_name, grid in config.model_grids.items():
                    task_number += 1
                    task_started = time.time()
                    task_dir = (
                        checkpoint_root
                        / f"repeat_{repeat:02d}"
                        / f"fold_{fold:02d}"
                        / f"{feature_set}__{model_name}"
                    )
                    signature = task_signature(
                        repeat,
                        fold,
                        feature_set,
                        model_name,
                        numeric_features,
                        grid,
                        train_index,
                        test_index,
                    )
                    if config.resume:
                        checkpoint = load_checkpoint(
                            task_dir, signature, patient_ids[test_index]
                        )
                        if checkpoint is not None:
                            prediction_frame, tuning_frame, importance_frame = checkpoint
                            predictions.extend(prediction_frame.to_dict("records"))
                            tuning.extend(tuning_frame.to_dict("records"))
                            importances.extend(importance_frame.to_dict("records"))
                            resumed_tasks += 1
                            log(
                                f"RESUME task={task_number}/{total_tasks} "
                                f"feature_set={feature_set}, model={model_name}"
                            )
                            continue
                    log(
                        f"START task={task_number}/{total_tasks} "
                        f"feature_set={feature_set}, model={model_name}"
                    )
                    model, parameter_grid = _model_and_grid(
                        model_name, grid, seed + fold, y_train
                    )
                    pipeline = Pipeline(
                        [
                            (
                                "preprocess",
                                _preprocessor(numeric_features, build.categorical_features),
                            ),
                            ("model", model),
                        ]
                    )
                    inner = StratifiedKFold(
                        n_splits=config.inner_folds,
                        shuffle=True,
                        random_state=seed + fold + 1000,
                    )
                    search = GridSearchCV(
                        pipeline,
                        parameter_grid,
                        scoring="roc_auc",
                        cv=inner,
                        refit=True,
                        n_jobs=config.n_jobs,
                        error_score="raise",
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        search.fit(x_train, y_train)
                    probabilities = search.predict_proba(x_test)[:, 1]
                    tuning_row = {
                        "repeat": repeat,
                        "fold": fold,
                        "feature_set": feature_set,
                        "model": model_name,
                        "best_inner_auc": float(search.best_score_),
                        "best_params": json.dumps(
                            search.best_params_, sort_keys=True, ensure_ascii=False
                        ),
                    }
                    tuning.append(tuning_row)
                    task_predictions: list[dict[str, Any]] = []
                    for index, probability in zip(test_index, probabilities):
                        row = {
                            "repeat": repeat,
                            "seed": seed,
                            "fold": fold,
                            "feature_set": feature_set,
                            "model": model_name,
                            "patient_id": patient_ids[index],
                            "label": int(labels[index]),
                            "probability": float(probability),
                        }
                        task_predictions.append(row)
                        predictions.append(row)
                    task_importances: list[dict[str, Any]] = []
                    if config.permutation_repeats > 0:
                        result = permutation_importance(
                            search.best_estimator_,
                            x_test,
                            y_test,
                            scoring="roc_auc",
                            n_repeats=config.permutation_repeats,
                            random_state=seed + fold + 2000,
                            n_jobs=1,
                        )
                        for feature, mean, std in zip(
                            input_features, result.importances_mean, result.importances_std
                        ):
                            row = {
                                "repeat": repeat,
                                "fold": fold,
                                "feature_set": feature_set,
                                "model": model_name,
                                "feature": feature,
                                "permutation_auc_drop_mean": float(mean),
                                "permutation_auc_drop_sd": float(std),
                            }
                            task_importances.append(row)
                            importances.append(row)

                    # The completion marker is deliberately written last.  An
                    # interrupted fit therefore cannot be mistaken for a valid
                    # checkpoint on the next PyCharm run.
                    _write_csv_atomic(
                        pd.DataFrame(task_predictions), task_dir / "predictions.csv"
                    )
                    _write_csv_atomic(
                        pd.DataFrame([tuning_row]), task_dir / "tuning.csv"
                    )
                    _write_csv_atomic(
                        pd.DataFrame(task_importances, columns=IMPORTANCE_COLUMNS),
                        task_dir / "permutation_importance.csv",
                    )
                    _write_json_atomic(
                        {
                            "signature": signature,
                            "completed_at": datetime.now().isoformat(timespec="seconds"),
                            "test_patients": len(test_index),
                        },
                        task_dir / "complete.json",
                    )
                    log(
                        f"DONE task={task_number}/{total_tasks} "
                        f"inner_auc={search.best_score_:.3f}; "
                        f"elapsed={time.time() - task_started:.1f}s"
                    )
    log(
        f"MODEL TASKS COMPLETE total={total_tasks}, resumed={resumed_tasks}, "
        f"newly_fitted={total_tasks - resumed_tasks}"
    )
    return (
        pd.DataFrame(predictions),
        pd.DataFrame(folds).drop_duplicates(),
        pd.DataFrame(tuning),
        pd.DataFrame(importances, columns=IMPORTANCE_COLUMNS),
    )


def _summarize(
    predictions: pd.DataFrame,
    config: PriorMLConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["feature_set", "model", "repeat"], sort=False
    ):
        feature_set, model, repeat = keys
        repeat_rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "repeat": int(repeat),
                **_metrics(group["label"].to_numpy(), group["probability"].to_numpy()),
            }
        )
    each_repeat = pd.DataFrame(repeat_rows)
    patient_mean = (
        predictions.groupby(
            ["feature_set", "model", "patient_id", "label"], as_index=False
        )["probability"]
        .mean()
        .rename(columns={"probability": "mean_oof_probability"})
    )
    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in patient_mean.groupby(["feature_set", "model"], sort=False):
        feature_set, model = keys
        labels = group["label"].to_numpy(dtype=int)
        probabilities = group["mean_oof_probability"].to_numpy(dtype=float)
        row = {
            "feature_set": feature_set,
            "model": model,
            "patients": len(group),
            "label_0": int((labels == 0).sum()),
            "label_1": int((labels == 1).sum()),
            **_metrics(labels, probabilities),
        }
        for metric in ("roc_auc", "pr_auc", "brier"):
            low, high = _bootstrap_ci(
                labels,
                probabilities,
                metric,
                config.bootstrap_iterations,
                config.base_seed + len(aggregate_rows),
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        repeats = each_repeat[
            (each_repeat["feature_set"] == feature_set)
            & (each_repeat["model"] == model)
        ]
        row["repeat_auc_mean"] = float(repeats["roc_auc"].mean())
        row["repeat_auc_sd"] = float(repeats["roc_auc"].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    paired_rows: list[dict[str, Any]] = []
    if {"prior_core", "prior_plus_locked7"}.issubset(set(patient_mean["feature_set"])):
        for model in sorted(patient_mean["model"].unique()):
            core = patient_mean[
                (patient_mean["feature_set"] == "prior_core")
                & (patient_mean["model"] == model)
            ][["patient_id", "label", "mean_oof_probability"]].rename(
                columns={"mean_oof_probability": "core_probability"}
            )
            combined = patient_mean[
                (patient_mean["feature_set"] == "prior_plus_locked7")
                & (patient_mean["model"] == model)
            ][["patient_id", "mean_oof_probability"]].rename(
                columns={"mean_oof_probability": "combined_probability"}
            )
            merged = core.merge(combined, on="patient_id", validate="one_to_one")
            labels = merged["label"].to_numpy(dtype=int)
            difference, low, high = _paired_auc_ci(
                labels,
                merged["combined_probability"].to_numpy(),
                merged["core_probability"].to_numpy(),
                config.bootstrap_iterations,
                config.base_seed + 500 + len(paired_rows),
            )
            paired_rows.append(
                {
                    "model": model,
                    "comparison": "prior_plus_locked7_minus_prior_core",
                    "patients": len(merged),
                    "prior_core_auc": float(
                        roc_auc_score(labels, merged["core_probability"])
                    ),
                    "prior_plus_locked7_auc": float(
                        roc_auc_score(labels, merged["combined_probability"])
                    ),
                    "paired_auc_difference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "interpretation": "exploratory only because locked7 used all 219 labels",
                }
            )
    return each_repeat, patient_mean, aggregate, pd.DataFrame(paired_rows)


def _write_summary(output_dir: Path, build: FeatureBuildResult, aggregate: pd.DataFrame) -> None:
    lines = [
        "# v12 先验肌肉特征传统机器学习实验",
        "",
        f"- 患者：{len(build.table)}例；标签分布：{build.table['label'].value_counts().sort_index().to_dict()}。",
        f"- 先验核心数值特征：{len(build.core_features)}项。",
        f"- 既往锁定特征：{len(build.locked_features)}项，仅作探索性敏感性分析。",
        f"- 滑脱节段：{build.table['target_slip_segment'].value_counts().to_dict()}。",
        "",
        "## 解释边界",
        "",
        "- prior_core是按医学先验固定的主分析面板。",
        "- prior_plus_locked7使用过同一219例标签进行历史筛选，不是独立验证。",
        "- 平均MRI灰度受序列、扫描设备和归一化影响，不能直接等同于定量脂肪分数。",
        "- target_slip_segment几乎恒定，无法可靠估计不同滑脱节段的效应。",
        "- 解剖节段表来自统一协议映射，不等同于逐患者医生复核。",
        "",
    ]
    if len(aggregate):
        lines.extend(
            [
                "## 重复OOF集成结果",
                "",
                "| 特征集 | 模型 | AUC（95%CI） | PR-AUC | Brier |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for _, row in aggregate.sort_values(["feature_set", "model"]).iterrows():
            lines.append(
                f"| {row['feature_set']} | {row['model']} | {row['roc_auc']:.3f} "
                f"（{row['roc_auc_ci_low']:.3f}–{row['roc_auc_ci_high']:.3f}） | "
                f"{row['pr_auc']:.3f} | {row['brier']:.3f} |"
            )
    else:
        lines.extend(["## 当前状态", "", "仅完成特征表审计；validate-only未拟合模型。", ""])
    (output_dir / "RESULTS_SUMMARY_CN.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def run_prior_ml_experiment(config: PriorMLConfig) -> None:
    started = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(output_dir)
    run_state_path = output_dir / "run_state.json"
    log(
        f"START v12 prior-feature experiment; executable={sys.executable}; "
        f"resume={config.resume}; validate_only={config.validate_only}"
    )
    _write_json_atomic(
        {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "resume": config.resume,
        },
        run_state_path,
    )
    try:
        log("BUILD patient-level prior feature table")
        build = build_prior_feature_table(config)
        _write_csv_atomic(
            build.table, output_dir / "patient_level_prior_feature_table.csv"
        )
        _write_csv_atomic(build.dictionary, output_dir / "feature_dictionary.csv")
        _write_csv_atomic(build.audit, output_dir / "data_quality_report.csv")
        _write_csv_atomic(
            pd.DataFrame(build.warnings), output_dir / "warnings_and_bug_records.csv"
        )
        log(
            f"BUILD COMPLETE patients={len(build.table)}, "
            f"labels={build.table['label'].value_counts().sort_index().to_dict()}, "
            f"core_features={len(build.core_features)}, locked_features={len(build.locked_features)}"
        )

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
                "lightgbm": lightgbm.__version__,
            },
            "cohort": {
                "patients": len(build.table),
                "label_counts": build.table["label"].value_counts().sort_index().to_dict(),
                "segment_counts": build.table["target_slip_segment"].value_counts().to_dict(),
            },
            "core_features": build.core_features,
            "locked_features": build.locked_features,
        }
        # Save the configuration before fitting, so an interrupted run remains
        # fully auditable rather than looking like a mysterious empty folder.
        _write_json_atomic(configuration, output_dir / "experiment_config.json")

        if config.validate_only:
            configuration["runtime_seconds"] = time.time() - started
            _write_json_atomic(configuration, output_dir / "experiment_config.json")
            _write_summary(output_dir, build, pd.DataFrame())
            _write_json_atomic(
                {
                    "status": "validation_only_completed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "runtime_seconds": time.time() - started,
                },
                run_state_path,
            )
            log("END validation-only; models_fitted=0")
            return

        total_tasks = (
            config.repeats
            * config.outer_folds
            * len(config.feature_sets)
            * len(config.model_grids or {})
        )
        log(
            f"MODEL PHASE tasks={total_tasks} "
            f"({config.repeats} repeats x {config.outer_folds} folds x "
            f"{len(config.feature_sets)} feature sets x {len(config.model_grids or {})} models)"
        )
        predictions, folds, tuning, importances = _run_models(
            config, build, output_dir, log
        )
        log("SUMMARIZE repeated OOF predictions and bootstrap confidence intervals")
        each_repeat, patient_mean, aggregate, paired = _summarize(predictions, config)
        for frame, filename in [
            (predictions, "all_repeated_oof_predictions.csv"),
            (folds, "outer_fold_assignments.csv"),
            (tuning, "inner_cv_best_parameters.csv"),
            (importances, "outer_test_permutation_importance.csv"),
            (each_repeat, "performance_each_repeat.csv"),
            (patient_mean, "mean_oof_predictions_by_patient.csv"),
            (aggregate, "aggregate_performance.csv"),
            (paired, "paired_locked7_increment.csv"),
        ]:
            _write_csv_atomic(frame, output_dir / filename)
        configuration["runtime_seconds"] = time.time() - started
        _write_json_atomic(configuration, output_dir / "experiment_config.json")
        _write_summary(output_dir, build, aggregate)
        _write_json_atomic(
            {
                "status": "completed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "runtime_seconds": time.time() - started,
                "model_tasks": total_tasks,
            },
            run_state_path,
        )
        log(f"END completed; runtime={time.time() - started:.1f}s")
        print(aggregate.to_string(index=False), flush=True)
    except Exception as exc:
        _write_json_atomic(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "runtime_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            run_state_path,
        )
        log(f"ERROR {type(exc).__name__}: {exc}")
        raise
