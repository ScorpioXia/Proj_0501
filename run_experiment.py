"""Canonical v11 entry point: clinical variables plus MRI muscle features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumbar_stability import ClinicalMRIConfig, run_clinical_mri_experiment


DEFAULT_CONFIG_FILE = PROJECT_ROOT / "configs" / "v11_clinical_mri.json"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical v11 clinical plus MRI experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="JSON configuration file")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory; relative paths are resolved from the project root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_file = _resolve(args.config)
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    paths = raw["paths"]
    validation = raw["validation"]
    output_dir = _resolve(args.output_dir or paths["output_dir"])

    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite non-empty result directory: {output_dir}\n"
            "Use --output-dir for a new run, or pass --overwrite explicitly."
        )

    config = ClinicalMRIConfig(
        label_file=str(_resolve(paths["label_file"])),
        global_feature_file=str(_resolve(paths["global_3d_feature_file"])),
        locked_feature_universe_file=str(_resolve(paths["locked_feature_universe_file"])),
        locked_selection_file=str(_resolve(paths["locked_selection_file"])),
        output_dir=str(output_dir),
        repeats=int(validation["repeats"]),
        outer_folds=int(validation["outer_folds"]),
        inner_folds=int(validation["inner_folds"]),
        base_seed=int(validation["base_seed"]),
        bootstrap_iterations=int(validation["bootstrap_iterations"]),
    )
    run_clinical_mri_experiment(config)


if __name__ == "__main__":
    main()
