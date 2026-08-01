# v11 冻结派生输入

版本：`v11_clinical_mri`

- `patient_feature_universe_raw.csv`：从统一目录 v10（原内部 v7 Stability LASSO）复制的 219 例患者级候选特征宇宙。
- `optimistic_global_selection_NOT_VALID.csv`：旧版在全数据标签上得到的七特征清单，仅用于锁定面板复现。

这些文件不是独立、无标签选择的输入。使用它们得到的 219 例 AUC 0.665 和 68 例联合 AUC 0.708 均须标记为“特征选择队列重叠、非独立验证”。文件名中的 `NOT_VALID` 用于提醒：全局选择过程本身不能当作有效 OOF 评估。
