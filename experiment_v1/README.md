# 腰椎稳定性预测实验 v1

本目录实现311人现有标签条件下的第一阶段可复现实验：

1. 审计并清理已知全缺失、常量、重复和技术字段；
2. 将2D切片特征按患者—肌肉聚合为中位数、IQR和P90；
3. 构建E1（3D+跨层+多肌肉）、E2（2D）、E3（组合）三组未标准化宽表；
4. 输出全队列Pearson点二列相关、FDR、单变量AUC和SMD，仅作探索描述；
5. 使用5折外层、4折内层的嵌套分层交叉验证；
6. 中位数填充、训练折1%/99%缩尾、方差过滤、Spearman冗余过滤、Pearson Top-K、标准化和Elastic-net调参全部在训练折内完成；
7. 使用每个外层训练折的标签患病率生成不使用特征的E0基线；
8. 保存每名患者的真实外层OOF预测、逐折参数、系数、选择频率和Bootstrap置信区间。

## 环境

推荐使用项目验证过的Conda环境依赖，见`requirements.txt`。

```powershell
C:\ProgramData\anaconda3\python.exe experiment_v1\run_experiment.py
```

也可在自己的Conda环境中运行：

```powershell
conda create -n lumbar_stability_v1 python=3.11
conda activate lumbar_stability_v1
pip install -r experiment_v1\requirements.txt
python experiment_v1\run_experiment.py
```

默认输出目录为项目根目录下的`experiment_results_v1`。如需指定位置：

```powershell
python experiment_v1\run_experiment.py --output-dir E:\path\to\results
```

## 重要说明

- 本实验不使用每个2D切片作为独立样本。
- 不会预先对全部311人进行标准化或监督筛选。
- `univariate_feature_association.csv`是全队列探索性关联表，不等同于无偏模型性能。
- 正式性能来自`nested_cv_predictions.csv`的外层OOF预测。
- 异常值不直接删除；记录在`outlier_records.csv`，模型在每个训练折内学习缩尾边界。
- XGBoost不是本轮必做实验，未安装也不影响完整主实验。
