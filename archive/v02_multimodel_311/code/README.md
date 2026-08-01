# 第二阶段：多模型嵌套交叉验证

本阶段固定使用 311 名患者以及第一阶段完全相同的 5 折外层划分。所有缺失值填补、1%/99% 截断、方差过滤、Spearman 去冗余、Pearson top-K 筛选都只在训练折内学习。

## 为什么加入这些模型

- XGBoost：主实验；检验浅层、强正则化的非线性与特征交互是否优于 ElasticNet。
- Random Forest：作为袋装树对照，对尺度不敏感，能够捕获非线性，但小样本高维时容易方差过大。
- RBF-SVM：适合中小样本的平滑非线性边界，必须在训练折内标准化。
- LightGBM：与 XGBoost 不同的叶生长策略；本实验限制叶子数和深度，降低小样本过拟合。
- EBM：可解释的加性非线性模型；不加入交互，先判断单特征的非线性响应能否提供增益。

参数候选集预先写在 `config.json`，不是看过外层测试结果后调整。主指标为外层 OOF ROC-AUC，同时报告 PR-AUC、Brier、敏感度、特异度和 95% bootstrap CI。

Windows 下参数搜索使用单进程。首次冒烟测试已确认 loky 多进程会触发 DLL 级崩溃，详见 `BUG_FIX_LOG.md`；流水线缓存用于弥补串行搜索的额外耗时。

## 在 nnUNet-master 环境中运行

```powershell
conda activate nnUNet-master
python experiment_v2/run_experiment.py
```

如只需要快速重跑部分模型：

```powershell
python experiment_v2/run_experiment.py --models xgboost lightgbm --feature-sets E1_3d_level3 E3_combined --output-dir experiment_results_v2_subset
```

正式结果目录为 `experiment_results_v2/`。`nested_cv_predictions.csv` 是逐患者真实 OOF 预测；`outer_fold_performance.csv` 保存折级结果和最优参数；`feature_accuracy_audit.csv` 是当前保留 CSV 能支持的数值准确性检查。
