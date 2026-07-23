"""v4 PyCharm entry point: 10-seed repeated nested cross-validation."""

from pathlib import Path

from experiment.repeated_runner import run_repeated_experiment


PROJECT_ROOT = Path(__file__).resolve().parent

# ========================= v4 USER CONFIGURATION =========================
FEATURE_DIR = PROJECT_ROOT / "features_312_20260722"
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v4_repeated_nested_cv"
FEATURE_VERSION = "v7"
CONFIG_FILE = PROJECT_ROOT / "experiment" / "config.json"

# Three prespecified v4 candidates; no additional model-by-feature search.
LOCKED_EXPERIMENTS = [
    ("lightgbm", "E2_2d"),
    ("xgboost", "E3_combined"),
    ("elastic_net", "E1_3d_level3"),
]

# Ten seeds; every seed runs five outer folds and four inner folds.
REPEAT_SEEDS = list(range(2026, 2036))

# Optional config overrides. Keep empty for the locked formal v4 protocol.
CONFIG_OVERRIDES = {}

REQUIRE_NNUNET_ENVIRONMENT = True

# Load completed seed checkpoints after interruption instead of refitting them.
RESUME = True
# ========================================================================


if __name__ == "__main__":
    run_repeated_experiment(
        project_dir=PROJECT_ROOT,
        feature_dir=FEATURE_DIR,
        label_file=LABEL_FILE,
        output_dir=OUTPUT_DIR,
        config_file=CONFIG_FILE,
        feature_version=FEATURE_VERSION,
        experiments=LOCKED_EXPERIMENTS,
        repeat_seeds=REPEAT_SEEDS,
        config_overrides=CONFIG_OVERRIDES,
        require_nnunet_environment=REQUIRE_NNUNET_ENVIRONMENT,
        resume=RESUME,
    )
