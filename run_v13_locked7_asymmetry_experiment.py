"""PyCharm entry point for the v13 four-stage experiment.

The default direct run executes stages 1 and 2 together.  After those results
have been reviewed, change STAGES_TO_RUN to (3,) and later to (4,) as advised.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumbar_stability.locked7_asymmetry import (  # noqa: E402
    Locked7AsymmetryConfig,
    run_locked7_asymmetry_experiment,
)


# ---------------------------------------------------------------------------
# Parameters intended for direct editing in PyCharm.
# ---------------------------------------------------------------------------
CONFIG_FILE = PROJECT_ROOT / "configs" / "v13_locked7_asymmetry.json"
OUTPUT_DIR_OVERRIDE: str | None = None

# Default first run: stage 1 (seven leave-one-out panels) and stage 2 (full7).
# Do not enable stage 3 or 4 until stages 1 and 2 have been reviewed.
STAGES_TO_RUN = (3,)

VALIDATE_ONLY = False
RESUME_EXISTING_OUTPUT = True
OVERWRITE_EXISTING_OUTPUT = False


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_stages(value: str) -> tuple[int, ...]:
    try:
        stages = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Stages must be comma-separated integers, e.g. 1,2") from exc
    if not stages or any(stage not in (1, 2, 3, 4) for stage in stages):
        raise argparse.ArgumentTypeError("Stages must be a nonempty subset of 1,2,3,4")
    return stages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v13 locked7 ablation and asymmetry augmentation experiments"
    )
    parser.add_argument("--config", default=str(CONFIG_FILE))
    parser.add_argument("--output-dir", default=OUTPUT_DIR_OVERRIDE)
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=tuple(STAGES_TO_RUN),
        help="Comma-separated stages. Default PyCharm setting is 1,2",
    )
    parser.add_argument("--validate-only", action="store_true", default=VALIDATE_ONLY)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=RESUME_EXISTING_OUTPUT,
        help="Resume compatible completed checkpoints in a non-empty output directory",
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_EXISTING_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = json.loads(_resolve(args.config).read_text(encoding="utf-8-sig"))
    paths = raw["paths"]
    validation = raw["validation"]
    contract = raw["feature_contract"]
    output_dir = _resolve(args.output_dir or paths["output_dir"])
    if (
        output_dir.is_dir()
        and any(output_dir.iterdir())
        and not args.overwrite
        and not args.resume
    ):
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory: {output_dir}\n"
            "Choose a new OUTPUT_DIR_OVERRIDE, enable resume, or pass --overwrite explicitly."
        )
    config = Locked7AsymmetryConfig(
        label_file=str(_resolve(paths["label_file"])),
        feature_universe_file=str(_resolve(paths["feature_universe_file"])),
        locked_selection_file=str(_resolve(paths["locked_selection_file"])),
        output_dir=str(output_dir),
        stages=tuple(args.stages),
        repeats=int(validation["repeats"]),
        outer_folds=int(validation["outer_folds"]),
        inner_folds=int(validation["inner_folds"]),
        stage3_repeats=int(validation["stage3_repeats"]),
        stage4_repeats=int(validation["stage4_repeats"]),
        stage4_inner_folds=int(validation["stage4_inner_folds"]),
        base_seed=int(validation["base_seed"]),
        bootstrap_iterations=int(validation["bootstrap_iterations"]),
        screening_bootstrap_iterations=int(validation["screening_bootstrap_iterations"]),
        permutation_iterations=int(validation["permutation_iterations"]),
        n_jobs=int(validation["n_jobs"]),
        model_grids=raw["model_grids_stage1_stage2"],
        fixed_model_params=raw["fixed_model_params_stage3_stage4"],
        expected_locked_features=int(contract["expected_locked_features"]),
        expected_asymmetry_features=int(contract["expected_asymmetry_features"]),
        expected_new_asymmetry_candidates=int(
            contract["expected_new_asymmetry_candidates"]
        ),
        candidate_min_nonmissing_fraction=float(
            contract["candidate_min_nonmissing_fraction"]
        ),
        validate_only=bool(args.validate_only),
        resume=bool(args.resume and not args.overwrite),
    )
    run_locked7_asymmetry_experiment(config)


if __name__ == "__main__":
    main()
