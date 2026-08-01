"""Editable PyCharm entry point for the v7 Stability LASSO experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiment.stability_lasso_replication import run_stability_lasso_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_FILE = PROJECT_ROOT / "slice_level_annotations_219_v2.csv"
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
FEATURE_DIR = PROJECT_ROOT / "features_312_20260722"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v7_stability_lasso_replication"

CORRELATION_THRESHOLD = 0.90
UNIVARIATE_LOGISTIC_P_THRESHOLD = 0.10
LASSO_CS = (0.03, 0.10, 0.30, 1.0, 3.0)
STABILITY_SUBSAMPLES = 100
STABILITY_TRAIN_FRACTION = 0.80
STABILITY_FREQUENCY_THRESHOLD = 0.60
MAX_FEATURES = 7

REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_ITERATIONS = 3000
BASE_SEED = 20260730


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    log("START v7 Stability LASSO <=7 feature experiment")
    run_stability_lasso_experiment(
        annotation_file=ANNOTATION_FILE,
        label_file=LABEL_FILE,
        feature_dir=FEATURE_DIR,
        output_dir=OUTPUT_DIR,
        repeats=REPEATS,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        correlation_threshold=CORRELATION_THRESHOLD,
        univariate_p_threshold=UNIVARIATE_LOGISTIC_P_THRESHOLD,
        lasso_cs=LASSO_CS,
        stability_subsamples=STABILITY_SUBSAMPLES,
        stability_train_fraction=STABILITY_TRAIN_FRACTION,
        stability_frequency_threshold=STABILITY_FREQUENCY_THRESHOLD,
        max_features=MAX_FEATURES,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        base_seed=BASE_SEED,
        log=log,
    )
    log(f"END results saved to {OUTPUT_DIR}")

