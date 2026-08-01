"""Editable PyCharm entry point for the v7 compact 219-patient experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiment.segment_validation_219 import run_segment_validation_219


PROJECT_ROOT = Path(__file__).resolve().parent
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
SOURCE_ANNOTATION_FILE = PROJECT_ROOT / "slice_level_annotations_test30_v1.csv"
ANNOTATION_FILE = PROJECT_ROOT / "slice_level_annotations_219_v2.csv"
FEATURE_FILE = PROJECT_ROOT / "features_312_20260722" / "muscle_features_2d_v7.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v7_segment_validation_219"

REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_ITERATIONS = 3000
BASE_SEED = 20260726


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    log("START v7 compact 219-patient segment validation")
    result = run_segment_validation_219(
        label_file=LABEL_FILE,
        source_annotation_file=SOURCE_ANNOTATION_FILE,
        annotation_file=ANNOTATION_FILE,
        feature_file=FEATURE_FILE,
        output_dir=OUTPUT_DIR,
        repeats=REPEATS,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        base_seed=BASE_SEED,
        log=log,
    )
    log(f"Completed; results saved to {OUTPUT_DIR}")
