"""Audit whether a fixed patient-level Pearson threshold leaves usable features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def audit_threshold(feature_table: Path, assignments_file: Path, threshold: float) -> pd.DataFrame:
    table = pd.read_csv(feature_table)
    assignments = pd.read_csv(assignments_file)
    feature_names = [name for name in table.columns if name not in {"patient_id", "label"}]
    rows: list[dict] = []
    for (seed, fold), group in assignments.groupby(["repeat_seed", "outer_fold"]):
        test_ids = set(group["patient_id"].astype(str))
        train = table[~table["patient_id"].astype(str).isin(test_ids)]
        values = train[feature_names].apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(values.median())
        correlations = values.corrwith(train["label"].astype(float)).abs().fillna(0.0)
        rows.append({
            "repeat_seed": int(seed),
            "outer_fold": int(fold),
            "training_patients": int(len(train)),
            "threshold": float(threshold),
            "selected_features": int((correlations >= threshold).sum()),
            "maximum_abs_pearson": float(correlations.max()),
        })
    return pd.DataFrame(rows)


def audit_nested_inner_threshold(
    feature_table: Path,
    assignments_file: Path,
    threshold: float,
    inner_folds: int = 4,
) -> pd.DataFrame:
    """Audit every inner-training partition used by the repeated nested CV."""
    table = pd.read_csv(feature_table)
    assignments = pd.read_csv(assignments_file)
    feature_names = [name for name in table.columns if name not in {"patient_id", "label"}]
    rows: list[dict] = []
    for (seed, outer_fold), group in assignments.groupby(["repeat_seed", "outer_fold"]):
        outer_test_ids = set(group["patient_id"].astype(str))
        outer_train = table[~table["patient_id"].astype(str).isin(outer_test_ids)].reset_index(drop=True)
        y = outer_train["label"].astype(int).to_numpy()
        splitter = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=int(seed) + int(outer_fold),
        )
        for inner_fold, (inner_train_index, _) in enumerate(
            splitter.split(np.zeros((len(y), 1)), y), start=1
        ):
            inner_train = outer_train.iloc[inner_train_index]
            values = inner_train[feature_names].apply(pd.to_numeric, errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).fillna(values.median())
            correlations = values.corrwith(inner_train["label"].astype(float)).abs().fillna(0.0)
            rows.append({
                "repeat_seed": int(seed),
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "training_patients": int(len(inner_train)),
                "threshold": float(threshold),
                "selected_features": int((correlations >= threshold).sum()),
                "maximum_abs_pearson": float(correlations.max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-inner", action="store_true")
    args = parser.parse_args()
    result = audit_threshold(args.feature_table, args.assignments, args.threshold)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"splits={len(result)}")
    print(
        "selected_features min/median/max="
        f"{result['selected_features'].min()}/"
        f"{result['selected_features'].median():.1f}/"
        f"{result['selected_features'].max()}"
    )
    print(f"zero_feature_splits={(result['selected_features'] == 0).sum()}")
    print(
        "maximum_abs_pearson min/median/max="
        f"{result['maximum_abs_pearson'].min():.6f}/"
        f"{result['maximum_abs_pearson'].median():.6f}/"
        f"{result['maximum_abs_pearson'].max():.6f}"
    )
    print("selected_count_frequency=" + str(result["selected_features"].value_counts().sort_index().to_dict()))
    if args.audit_inner:
        inner = audit_nested_inner_threshold(args.feature_table, args.assignments, args.threshold)
        inner_path = args.output.with_name(args.output.stem + "_inner.csv") if args.output else None
        if inner_path:
            inner.to_csv(inner_path, index=False, encoding="utf-8-sig")
        print(f"inner_splits={len(inner)}")
        print(
            "inner_selected_features min/median/max="
            f"{inner['selected_features'].min()}/"
            f"{inner['selected_features'].median():.1f}/"
            f"{inner['selected_features'].max()}"
        )
        print(f"inner_zero_feature_splits={(inner['selected_features'] == 0).sum()}")


if __name__ == "__main__":
    main()
