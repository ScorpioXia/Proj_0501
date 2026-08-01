# 2026-07-22 代码重构与目录清理记录

## 删除

以下根目录早期脚本依赖已经删除的旧宽表、硬编码历史路径或包含不适用于正式验证的
流程，且功能已由当前嵌套交叉验证代码覆盖，因此删除：

- `04_classification_model.py`
- `04_classification_simple.py`
- `04b_classification_lasso_lr.py`
- `04c_classification_ebm.py`
- `05_feature_engineering.py`
- `05_feature_engineering_v2.py`
- `05_validation_exploration.py`
- `06_weight_optimization.py`
- `07_validate_features.py`
- `explore_data.py`
- 所有 `__pycache__`、`.pyc` 和本次验证临时输出。

## 归档移动

- `experiment_v1/` → `archive/v1/code/`
- `experiment_results_v1/` → `archive/v1/results/`
- `features_311/` → `archive/v1/features/`
- `experiment_v2/` → `archive/v2/code/`
- `experiment_results_v2/` → `archive/v2/results/formal_v6/`
- `experiment_results_v2_sensitivity/` → `archive/v2/results/sensitivity_v6/`
- `experiment_results_v2_newdata/` → `results/pre_refactor_v7_20260722/`

## 当前保留

- `PATIENT_LIST_FILE.csv`：唯一标签/患者 ID 数据源。
- `features_312_20260722/`：当前 v7 四个特征 CSV。
- `experiment/`：重构后的当前实验包。
- `run_experiment.py`：PyCharm 唯一运行入口。
- `results/`：当前版本结果总目录。
- `archive/`：按版本组织的历史代码、结果和 v6 特征。
- `requirements.txt`、`README.md`、`特征名称及含义解释.xlsx` 和 IDE/Git 配置。

## 本次验证

- 使用 `C:\ProgramData\anaconda3\envs\nnUNet-master\python.exe`。
- validation-only：312 人，label 0=190、label 1=122；特征审计 0 error、9 warning；
  E1/E2/E3 分别生成 154/447/601 个候选特征。
- 确认预测变量中不存在 `patient_id`、`csf_value`、像素间距、层厚或绝对峰值层号。
- 极小两折冒烟测试成功跑通 ElasticNet 和校准 RBF-SVM；该测试不是正式实验结果。

### validation-only 警告明细

本次 9 项警告均已记录为可审查事项，没有 error 级数学/队列错误：

- 1 个患者-肌肉记录少于 3 个 2D 切片，IQR/P90 聚合可能不稳定。
- 3 组 GLRLM/GLSZM 字段仍为逐行完全重复，已从建模候选中排除。
- `Symmetry_Index_Area_Psoas` 有 1 个绝对值大于 10 的原始值。
- `Symmetry_Index_FIP_MF`、`Symmetry_Index_FIP_ES`、`Symmetry_Index_FIP_Psoas`
  分别有 3、3、8 个绝对值大于 10 的原始值。
- `Rat_FIP_MF_Psoas` 有 172 个绝对值大于 10 的原始值。

当前代码不直接使用上述分母敏感原始比值，而使用 signed-log1p 或有界不对称变换；
每次运行仍会在输出目录重新生成完整 `feature_accuracy_audit.csv`、
`bug_records.json`、`outlier_records.csv`、`runtime_warnings.csv` 和 `fatal_errors.json`。
