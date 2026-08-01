# 腰椎稳定性预测项目

本项目使用腰椎椎旁肌 MRI 提取特征预测腰椎稳定性。当前主线代码仅使用
`PATIENT_LIST_FILE.csv` 中的 `patient_id` 和 `label`：不再用姓名或汉语拼音识别患者，
有标签患者数量由程序在每次运行时动态确定。

## 当前目录

```text
Proj_0501/
├─ run_experiment.py                 # PyCharm 直接运行的唯一入口
├─ experiment/                       # 当前建模、审计和配置代码
├─ features_312_20260722/            # 当前 v7 原始特征 CSV
├─ PATIENT_LIST_FILE.csv             # 患者主表；只有 label=0/1 的行进入实验
├─ results/                           # 当前版本实验结果；按单次运行分子目录
├─ archive/                           # 旧版代码、结果和 v6 特征
├─ requirements.txt                  # nnUNet-master 已验证的软件版本
└─ 特征名称及含义解释.xlsx
```

`archive/` 仅用于追溯旧实验，不被当前代码导入。根目录早期 `04*.py`～`07*.py`
依赖已删除数据且存在旧路径/泄漏风险，已移除。

## 在 PyCharm 中运行

1. 将项目解释器设置为：
   `C:\ProgramData\anaconda3\envs\nnUNet-master\python.exe`。
2. 打开 `run_experiment.py`，修改“USER CONFIGURATION”区域：
   特征目录、标签文件、输出目录、特征版本、模型、特征集和参数覆盖项。
3. 首次保持 `VALIDATE_ONLY = True`，直接运行。它会完成标签、队列、特征审计、
   患者级聚合和折分验证，但不会拟合模型。
4. 检查输出中的 `FEATURE_ACCURACY_AUDIT.md`、`data_quality_report.csv` 和
   `outer_fold_assignments.csv` 后，将 `VALIDATE_ONLY = False` 开始正式实验。

默认输出目录为 `results/next_run`。为防止覆盖真实结果，如果目录非空程序会停止；
请改用新目录，或明确将 `ALLOW_EXISTING_OUTPUT` 设为 `True`。

## 当前方法学保护

- 有标签队列动态读取，不固定为 311 或 312 人；特征文件可以包含额外无标签患者，
  但不能缺少任何有标签患者。
- 所有患者关联仅使用规范化后的 `patient_id`。
- 缺失值填补、1%/99% 截断、方差过滤、Spearman 去冗余、点二列相关 top-k
  筛选、缩放和调参均在训练折内完成。
- `patient_id`、`csf_value`、像素间距、层厚、脂肪阈值和绝对峰值层号不作为预测变量。
- v7 已修正的 GLCM、`SA_V`、`3D_Shape_Index` 已纳入候选；仍完全重复的
  GLRLM/GLSZM 映射继续排除。
- 分母敏感的多肌肉比值使用 signed-log1p 或有界不对称变换。
- 外层折以 `patient_id` 保存；比较 v6/v7 时可通过 `REUSE_OUTER_FOLDS`
  复用完全相同的患者折分。
- RBF-SVM 使用 `CalibratedClassifierCV(SVC(probability=False))`，不再依赖已弃用的
  `SVC(probability=True)`。

## 结果说明

`results/pre_refactor_v7_20260722/` 是代码重构前完成的 v7 结果，仅作为历史参考：
它使用了不同环境/折分，并混入旧队列参考，不能作为当前代码的正式验证结果。
本次重构只进行了 validation-only 和极小两折冒烟测试，没有伪造或生成新的正式研究结果。
