# v12 先验肌肉特征传统机器学习方案

## 运行入口

在PyCharm中使用以下解释器：

```text
C:\ProgramData\anaconda3\envs\nnUNet-master\python.exe
```

直接运行：

```text
run_prior_feature_experiment.py
```

入口顶部可修改：

- `CONFIG_FILE`：配置文件；
- `OUTPUT_DIR_OVERRIDE`：新一轮结果路径；
- `VALIDATE_ONLY`：仅构建和审计特征表，不拟合模型；
- `RESUME_EXISTING_OUTPUT=True`：默认安全读取已完成的折/模型检查点；
- `OVERWRITE_EXISTING_OUTPUT=True`：忽略旧检查点并在同一目录重新计算。

模型、折数、种子和网格在`configs/v12_prior_feature_ml.json`中修改。

运行时，PyCharm控制台会实时显示时间、重复次数、外层折、特征集、模型和
`task=当前任务/总任务`。相同内容同步追加到输出目录的`run_progress.log`。
每个任务完成后写入`checkpoints/`，只有预测、调参和重要性文件全部保存成功后，
才写入`complete.json`完成标记。程序中断后直接再次运行即可续跑；不会把半成品
误认为已完成任务。`run_state.json`用于区分`running`、`failed`和`completed`状态。

## 患者与节段

- 使用`slice_level_annotations_219_v2.csv`覆盖的219例患者；
- 标签分布：稳定132例、不稳定87例；
- 目标滑脱节段：L4 218例、L5 1例；
- L4映射到L4/L5椎间盘区域，L5映射到L5/S1；
- 每个目标节段原则上4张轴位切片，每块肌肉在4张切片内取中位数。

## 主分析：prior_core

三组肌肉：多裂肌、竖脊肌、腰大肌。每组6项，共18项：

1. 左侧CSA；
2. 右侧CSA；
3. 左侧肌肉区域平均灰度；
4. 右侧肌肉区域平均灰度；
5. CSA左右不对称；
6. 平均灰度左右不对称。

不对称定义：

```text
2 * abs(left - right) / (abs(left) + abs(right) + 1e-8)
```

滑脱节段以类别变量拼接，并在每个训练折内部独热编码。由于219例中仅1例不是L4，该变量几乎没有可估计信息，必须在结果中如实报告。

## 探索分析：prior_plus_locked7

在18项主特征基础上追加既往7项：

1. 目标节段腰大肌Aspect Ratio左右平均；
2. 多裂肌目标节段减头侧节段的瘦肌平均信号；
3. 竖脊肌目标节段减头侧节段的瘦肌平均信号；
4. 目标节段竖脊肌GLCM Contrast左右不对称；
5. 目标节段竖脊肌灰度标准差左右不对称；
6. 多裂肌3D体积左右不对称；
7. 竖脊肌Z轴质心漂移左右平均。

这7项曾使用同一219例标签筛选，因此`prior_plus_locked7`结果存在选择重叠，只能作为探索性结果。程序会在结果警告文件和总结中自动记录。

## 模型和验证

- XGBoost；
- LightGBM；
- Random Forest；
- 10个随机种子×5折患者级外层交叉验证；
- 每个外层训练折中使用4折内层CV选择参数；
- 三个模型和两个特征集共享完全相同的外层患者折；
- 数值缺失仅在训练折内使用中位数处理；
- 类别变量只在训练折内拟合独热编码器；
- 外层测试折计算置换重要性；
- 输出重复OOF AUC、PR-AUC、Brier和Bootstrap 95%CI。

## 建议的首次运行

先在配置中临时设置：

```json
"repeats": 1,
"permutation_repeats": 2
```

并将输出改为新的测试目录。确认运行和输出正常后，再恢复10次重复进行正式实验。

## 灰度值的解释限制

MRI平均灰度可能与脂肪浸润有关，但它还受扫描序列、设备、线圈、归一化和重建参数影响。除非使用经过验证的定量脂肪序列，否则不能把普通T1/T2平均灰度直接表述为“脂肪浸润率”。
