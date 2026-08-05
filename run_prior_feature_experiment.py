"""PyCharm entry point for the v12 prior-feature machine-learning experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumbar_stability.prior_feature_ml import PriorMLConfig, run_prior_ml_experiment


# ---------------------------------------------------------------------------
# Parameters most likely to be edited in PyCharm.
# ---------------------------------------------------------------------------
CONFIG_FILE = PROJECT_ROOT / "configs" / "v12_prior_feature_ml.json"
OUTPUT_DIR_OVERRIDE: str | None = None
VALIDATE_ONLY = False  # True: build/audit the feature table, fit no model.
RESUME_EXISTING_OUTPUT = True  # Reuse completed fold/model checkpoints safely.
OVERWRITE_EXISTING_OUTPUT = False


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v12 prior-knowledge features with XGBoost, LightGBM and Random Forest"
    )
    parser.add_argument("--config", default=str(CONFIG_FILE))
    parser.add_argument("--output-dir", default=OUTPUT_DIR_OVERRIDE)
    parser.add_argument("--validate-only", action="store_true", default=VALIDATE_ONLY)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=RESUME_EXISTING_OUTPUT,
        help="Resume compatible completed fold/model checkpoints in a non-empty output directory",
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_EXISTING_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = json.loads(_resolve(args.config).read_text(encoding="utf-8-sig"))
    paths = raw["paths"]
    validation = raw["validation"]
    output_dir = _resolve(args.output_dir or paths["output_dir"])
    if (
        output_dir.is_dir()
        and any(output_dir.iterdir())
        and not args.overwrite
        and not args.resume
    ):
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory: {output_dir}\n"
            "Choose a new OUTPUT_DIR_OVERRIDE, enable --resume, or pass --overwrite explicitly."
        )
    config = PriorMLConfig(
        label_file=str(_resolve(paths["label_file"])),
        annotation_file=str(_resolve(paths["annotation_file"])),
        feature_2d_file=str(_resolve(paths["feature_2d_file"])),
        locked_feature_universe_file=str(_resolve(paths["locked_feature_universe_file"])),
        locked_selection_file=str(_resolve(paths["locked_selection_file"])),
        output_dir=str(output_dir),
        csa_column=str(raw["features"]["csa_column"]),
        mean_gray_column=str(raw["features"]["mean_gray_column"]),
        include_target_slip_segment=bool(raw["features"]["include_target_slip_segment"]),
        feature_sets=tuple(raw["features"]["feature_sets"]),
        repeats=int(validation["repeats"]),
        outer_folds=int(validation["outer_folds"]),
        inner_folds=int(validation["inner_folds"]),
        base_seed=int(validation["base_seed"]),
        bootstrap_iterations=int(validation["bootstrap_iterations"]),
        permutation_repeats=int(validation["permutation_repeats"]),
        n_jobs=int(validation["n_jobs"]),
        model_grids=raw["models"],
        validate_only=bool(args.validate_only),
        # --overwrite means recompute all tasks in the same directory.  The
        # default direct PyCharm run instead resumes compatible checkpoints.
        resume=bool(args.resume and not args.overwrite),
    )
    run_prior_ml_experiment(config)


if __name__ == "__main__":
    main()
