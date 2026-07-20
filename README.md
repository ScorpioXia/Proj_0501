# 311 人腰椎稳定性预测项目

本项目仅包含 311 名具有稳定性标签的患者数据、建模代码和真实实验结果。`label=0` 为稳定（189 人），`label=1` 为不稳定（122 人）。

## 数据

- `features_311/muscle_features_2d_v6.csv`：40,146 行切片/肌肉级 2D 特征。
- `features_311/muscle_features_3d_v6.csv`：1,866 行患者/肌肉级 3D 特征（311×6）。
- `features_311/muscle_features_level3_cross_v6.csv`：1,866 行跨层特征。
- `features_311/muscle_features_level3_multi_v6.csv`：311 行患者级多肌肉关系特征。
- `patient_stable_311.xlsx`：311 人稳定性标签。
- `特征名称及含义解释.xlsx`：特征名称与含义。

四个特征文件与标签表的患者集合完全一致。当前项目不包含原始 MRI、分割掩膜或特征提取代码，因此只能做数值一致性审计，不能替代原图/掩膜可视化复核。

## 实验

- `experiment_v1/`、`experiment_results_v1/`：第一阶段 ElasticNet 嵌套交叉验证代码与结果。
- `experiment_v2/`：第二阶段 XGBoost、随机森林、RBF-SVM、LightGBM、EBM 代码。
- `experiment_results_v2/`：第二阶段 15 组模型×特征集正式结果。
- `experiment_results_v2_sensitivity/`：删除 Solidity/Deep_Fat_Ratio 后的事后敏感性分析。

第二阶段必须在 `nnUNet-master` Conda 环境运行：

```powershell
conda activate nnUNet-master
python experiment_v2/run_experiment.py
python experiment_v2/run_sensitivity.py
```

所有缺失值填补、异常值截断、方差过滤、Spearman 去冗余、Pearson top-K 筛选与调参均在训练折内完成。核心结果和限制见 `experiment_results_v2/RESULTS_SUMMARY.md`。

旧的 `04*.py`～`07*.py` 是早期思路代码，部分仍依赖已删除的旧标准化宽表；正式复现请使用 `experiment_v1/` 和 `experiment_v2/`。
