# 正式结果目录

根目录下只保存当前主线 `v11_clinical_mri/`。历史 v01–v10 的结果已经随各版代码一并冻结到 `archive/<版本>/results/`。

`v11_clinical_mri/` 包含汇总指标、逐重复指标、OOF 预测、共享折分配、配对 AUC 比较、审计记录、运行环境和中英文解释。优先阅读：

1. `v11_clinical_mri/README.md`：文件清单和证据等级；
2. `v11_clinical_mri/RESULTS_INTERPRETATION_CN.md`：中文结果解释；
3. `v11_clinical_mri/aggregate_performance.csv`：全部模型的聚合性能；
4. `v11_clinical_mri/paired_auc_comparisons.csv`：联合模型相对 MRI 的同队列配对差值。

患者级 OOF、队列成员表与折分文件可能包含研究标识符，不应直接发布到公共仓库。
