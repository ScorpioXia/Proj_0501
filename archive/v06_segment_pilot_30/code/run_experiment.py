"""Editable PyCharm entry point for the v6 30-patient segment pilot."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiment.segment_pilot import run_segment_pilot


# Basic paths: edit these lines for a new pilot cohort.
PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_FILE = PROJECT_ROOT / "slice_level_annotations_test30_v1.csv"
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
FEATURE_FILE = (
    PROJECT_ROOT
    / "features_312_20260722"
    / "muscle_features_2d_v7_test30patient.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "v6_segment_pilot_test30"

# Validation settings. Three outer folds are used because the current cohort
# contains only seven unstable patients.
REPEATS = 30
OUTER_FOLDS = 3
INNER_FOLDS = 2
BOOTSTRAP_ITERATIONS = 5000
BASE_SEED = 20260725


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    log("START v6 segment-localisation feasibility experiment")
    result = run_segment_pilot(
        annotation_file=ANNOTATION_FILE,
        label_file=LABEL_FILE,
        feature_file=FEATURE_FILE,
        output_dir=OUTPUT_DIR,
        repeats=REPEATS,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        base_seed=BASE_SEED,
        log=log,
    )
    log(result["conclusion"])
    log(f"Results saved to {OUTPUT_DIR}")
