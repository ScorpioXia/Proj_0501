"""Editable PyCharm entry point for the v8 nested feature-discovery experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiment.feature_discovery import run_feature_discovery


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_FILE = PROJECT_ROOT / "slice_level_annotations_219_v2.csv"
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
FEATURE_DIR = PROJECT_ROOT / "features_312_20260722"
PILOT_ANNOTATION_FILE = PROJECT_ROOT / "slice_level_annotations_test30_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v8_nested_feature_discovery"

REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 4
CANDIDATES_PER_SIZE = 40
STABILITY_SUBSAMPLES = 20
BOOTSTRAP_ITERATIONS = 3000
PERMUTATION_ITERATIONS = 20
BASE_SEED = 20260729


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    log("START v8 nested feature-discovery experiment")
    result = run_feature_discovery(
        annotation_file=ANNOTATION_FILE,
        label_file=LABEL_FILE,
        feature_dir=FEATURE_DIR,
        pilot_annotation_file=PILOT_ANNOTATION_FILE,
        output_dir=OUTPUT_DIR,
        repeats=REPEATS,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        candidates_per_size=CANDIDATES_PER_SIZE,
        stability_subsamples=STABILITY_SUBSAMPLES,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        permutation_iterations=PERMUTATION_ITERATIONS,
        base_seed=BASE_SEED,
        log=log,
    )
    log(f"Completed; results saved to {OUTPUT_DIR}")
