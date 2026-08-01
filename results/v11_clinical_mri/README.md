# v11_clinical_mri 结果文件说明

本目录是当前正式结果快照。实验使用 10 次重复、5 折外层 OOF；每位患者在每次重复中恰有一次外层预测。现有结果最初以内部名 `v8_clinical_mri` 运行，项目整理后统一命名为 v11；数值未因改名而重新计算。

## 最重要的结论

| n=68 同队列模型 | 重复 OOF 集成 AUC | 95%CI |
|---|---:|---:|
| 四项临床变量 | 0.432 | 0.287–0.584 |
| 旧标签筛选的 7 项 MRI | 0.686 | 0.539–0.823 |
| 四项临床+旧 7 项 MRI | **0.708** | **0.564–0.833** |

0.708 没有丢失；它位于 `aggregate_performance.csv` 的 `locked7_complete4_overlap_n68 / combined_locked7_clinical` 行。它此前只出现在配对比较表而未在模型总览中突出，是摘要展示不完整，不是结果文件缺失。

必须同时报告：联合−MRI 的配对 AUC 差值是 +0.023（95%CI −0.058–0.104），且这 7 项特征曾使用同一 219 例标签筛选，所以结果不是独立验证。

## 文件清单

| 文件 | 内容 |
|---|---|
| `aggregate_performance.csv` | 每个队列/特征集/模型的 AUC、PR-AUC、Brier、校准及 CI |
| `performance_each_repeat.csv` | 每个重复的性能，反映随机折波动 |
| `all_repeated_oof_predictions.csv` | 全部重复 OOF 概率，共 24,720 行 |
| `mean_oof_predictions_by_patient.csv` | 每位患者按模型汇总的重复 OOF 均值 |
| `paired_auc_comparisons.csv` | 联合模型−MRI 的同患者配对 AUC 差值及 bootstrap CI |
| `shared_fold_assignments.csv` | 共享外层折，便于复核配对设计 |
| `inner_tuning_choices.csv` | 内层调参选择记录 |
| `cohort_membership_and_clinical_values.csv` | 队列成员与临床可用性审计 |
| `clinical_and_feature_data_audit.csv` | 缺失、范围、队列规模和特征审计 |
| `feature_definitions.csv` | 当前特征含义和来源 |
| `locked7_descriptive_associations.csv` | 旧七特征的描述性关联，不是独立验证 |
| `warnings_and_bug_records.csv` | 已知警告和解释边界 |
| `experiment_config.json` | 原始运行环境、内部版本名和路径；用于来源追踪 |
| `run_stdout.log` / `run_stderr.log` | 原始运行日志 |
| `RESULTS_SUMMARY.md` | 简明英文结果摘要 |
| `RESULTS_INTERPRETATION_CN.md` | 中文研究解释和报告建议 |

## 复核状态

- 已由 OOF 概率重新计算并核对 `aggregate_performance.csv`，AUC 与汇总表一致。
- 每位患者、每个模型均有 10 个 OOF 概率；未发现重复的患者/重复/模型预测键。
- 当前原始 3D 特征构建的预设 MRI7 已与旧宽表逐值比较，在1×10⁻¹²容差内完全一致（最大绝对差2.27×10⁻¹³）。
