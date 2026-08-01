"""Editable PyCharm entry point for the v9 Pearson-factor replication."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiment.pearson_factor_replication import run_pearson_factor_replication


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_FILE = PROJECT_ROOT / "slice_level_annotations_219_v2.csv"
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
FEATURE_DIR = PROJECT_ROOT / "features_312_20260722"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v9_pearson_factor_replication"

# The colleague's exact threshold is audited first.  It retains zero features
# in this cohort, so the formal executable fallback remains the previously
# approved 0.15 threshold.
EXACT_REQUESTED_PEARSON_THRESHOLD = 0.25
FORMAL_PEARSON_THRESHOLD = 0.15
FACTOR_COUNT = 6

REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_ITERATIONS = 3000
BASE_SEED = 20260730


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    log("START v9 Pearson-to-six-factor replication and leakage audit")
    run_pearson_factor_replication(
        annotation_file=ANNOTATION_FILE,
        label_file=LABEL_FILE,
        feature_dir=FEATURE_DIR,
        output_dir=OUTPUT_DIR,
        pearson_threshold=FORMAL_PEARSON_THRESHOLD,
        exact_requested_threshold=EXACT_REQUESTED_PEARSON_THRESHOLD,
        factor_count=FACTOR_COUNT,
        repeats=REPEATS,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        base_seed=BASE_SEED,
        log=log,
    )
    log(f"END results saved to {OUTPUT_DIR}")
