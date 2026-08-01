"""v5 PyCharm entry point: Pearson-screened rotated factor models."""

from pathlib import Path

from experiment.factor_runner import run_factor_experiment


PROJECT_ROOT = Path(__file__).resolve().parent

# ========================= v5 USER CONFIGURATION =========================
FEATURE_DIR = PROJECT_ROOT / "features_312_20260722"
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v5_factor_analysis"
FEATURE_VERSION = "v7"
CONFIG_FILE = PROJECT_ROOT / "experiment" / "config.json"

# Ten seeds; every seed runs five outer folds and four inner folds.
REPEAT_SEEDS = list(range(2026, 2036))

# The agreed threshold is locked here for visibility. Factor counts are 3/5/8/10.
CONFIG_OVERRIDES = {"factor_pearson_threshold": 0.15}

REQUIRE_NNUNET_ENVIRONMENT = True

# Load completed seed checkpoints after interruption instead of refitting them.
RESUME = True
# ========================================================================


if __name__ == "__main__":
    run_factor_experiment(
        project_dir=PROJECT_ROOT,
        feature_dir=FEATURE_DIR,
        label_file=LABEL_FILE,
        output_dir=OUTPUT_DIR,
        config_file=CONFIG_FILE,
        feature_version=FEATURE_VERSION,
        repeat_seeds=REPEAT_SEEDS,
        config_overrides=CONFIG_OVERRIDES,
        require_nnunet_environment=REQUIRE_NNUNET_ENVIRONMENT,
        resume=RESUME,
    )
