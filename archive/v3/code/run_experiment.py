"""PyCharm-friendly experiment entry point.

Edit the variables in the USER CONFIGURATION section, select the
``nnUNet-master`` interpreter in PyCharm, and run this file directly.
"""

from pathlib import Path

from experiment.runner import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent

# ========================= USER CONFIGURATION =========================
FEATURE_DIR = PROJECT_ROOT / "features_312_20260722"
LABEL_FILE = PROJECT_ROOT / "PATIENT_LIST_FILE.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "formal_v7_run_01"
FEATURE_VERSION = "v7"
CONFIG_FILE = PROJECT_ROOT / "experiment" / "config.json"

# 第一轮正式实验推荐模型
MODELS = ["elastic_net", "xgboost", "lightgbm"]

# 同时比较3D、2D和联合特征
FEATURE_SETS = ["E1_3d_level3", "E2_2d", "E3_combined"]

# 保持默认5折外层、4折内层和2000次bootstrap
CONFIG_OVERRIDES = {}

# 复用刚才验证阶段生成的患者折分
REUSE_OUTER_FOLDS = (
    PROJECT_ROOT / "results" / "next_run" / "outer_fold_assignments.csv"
)

# 关闭仅验证模式，开始训练模型
VALIDATE_ONLY = False

REQUIRE_NNUNET_ENVIRONMENT = True
ALLOW_EXISTING_OUTPUT = False
# ======================================================================


if __name__ == "__main__":
    run_experiment(
        project_dir=PROJECT_ROOT,
        feature_dir=FEATURE_DIR,
        label_file=LABEL_FILE,
        output_dir=OUTPUT_DIR,
        config_file=CONFIG_FILE,
        feature_version=FEATURE_VERSION,
        models=MODELS,
        feature_sets=FEATURE_SETS,
        config_overrides=CONFIG_OVERRIDES,
        reuse_outer_folds=REUSE_OUTER_FOLDS,
        validate_only=VALIDATE_ONLY,
        require_nnunet_environment=REQUIRE_NNUNET_ENVIRONMENT,
        allow_existing_output=ALLOW_EXISTING_OUTPUT,
    )
