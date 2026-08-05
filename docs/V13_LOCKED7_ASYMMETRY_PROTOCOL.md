# v13 历史7项消融与左右不对称增量实验

## 默认运行方式

PyCharm解释器：

```text
C:\ProgramData\anaconda3\envs\nnUNet-master\python.exe
```

直接运行：

```text
run_v13_locked7_asymmetry_experiment.py
```

入口默认设置：

```python
STAGES_TO_RUN = (1, 2)
VALIDATE_ONLY = False
RESUME_EXISTING_OUTPUT = True
OVERWRITE_EXISTING_OUTPUT = False
```

第一次正式运行不要启用阶段3和4。阶段1、2完成并分析后，再根据建议依次改为：

```python
STAGES_TO_RUN = (3,)
```

以及：

```python
STAGES_TO_RUN = (4,)
```

所有阶段写入同一个`results/v13_locked7_asymmetry`目录，但分别保存在独立子目录中。
直接重新运行会读取兼容检查点，不会重复拟合已完成任务。

## 四个阶段

### 阶段1：历史7项留一消融

构建7个面板。每个面板删除一项历史特征并保留其余6项，使用L2 Logistic、
Random Forest和XGBoost。采用10个随机种子、5折患者级外层CV、4折内层调参。

主要比较为：

```text
完整7项AUC - 删除某项后的6项AUC
```

正差值表示删除该特征后性能下降。

### 阶段2：完整历史7项

使用与阶段1完全相同的患者折和调参方法运行完整7项，作为消融参照。
阶段1和2完成后自动生成`stage_01_02_paired_ablation_comparison.csv`。

### 阶段3：7项加单个左右不对称特征

患者级特征表中共有78项`asymmetry`特征，其中3项已经属于历史7项，排除重复后
剩余75项。阶段3包含1个完整7项参考面板和75个加一面板。

为控制CPU时间，阶段3使用预先固定的保守参数、5个随机种子×5折，不在每个
候选面板内重新进行网格调参。三个模型共享完全相同的患者折。

输出包括配对AUC差值、Bootstrap区间、交换置换P值、模型内BH-FDR q值、重复方向
一致性、Brier变化以及跨模型共识。最高AUC本身不能作为候选通过的依据。

### 阶段4：第8项特征的选择感知嵌套验证

在每个外层训练折中，用3折内层CV评估全部75个候选。每个候选分别由三个模型
计算相对完整7项的AUC增量，再以三模型增量中位数选择一个共同候选。锁定该候选
后，才在外层测试折上比较完整7项和“7项+所选第8项”。

外层测试患者不参与第8项的选择。阶段4能够验证第8项选择流程，但不能消除基础
7项曾使用同一219例标签筛选的历史重叠。

## 特征与模型

标签始终从`data/labels/PATIENT_LIST_FILE.csv`按`patient_id`读取。特征来自
`data/derived/v11_clinical_mri/patient_feature_universe_raw.csv`。

模型：

- L2 Logistic：训练折内中位数插补和标准化；
- Random Forest：训练折内中位数插补；
- XGBoost：训练折内中位数插补、类别权重仅由当前训练折计算。

阶段1和2进行训练折内网格调参。阶段3和4使用配置文件中预先固定的参数，避免
75项候选带来不可接受的嵌套网格计算量。

## 输出目录

```text
results/v13_locked7_asymmetry/
├── experiment_config.json
├── run_progress.log
├── run_state.json
├── feature_manifest.csv
├── data_quality_report.csv
├── shared_outer_fold_assignments.csv
├── stage_01_leave_one_out/
├── stage_02_full_locked7/
├── stage_03_add_one_screen/
└── stage_04_nested_selection/
```

每个模型任务或阶段4外层折完成后才写入`complete.json`。任务中断时不会把半成品
当作完成结果。

## 解释限制

历史7项曾使用同一219例患者标签进行筛选。因此：

- 阶段1和2是对历史面板内部结构的诊断；
- 阶段3是多重比较控制后的候选发现；
- 阶段4是第8项选择流程的内部验证；
- 四个阶段都不能替代未参与历史筛选患者上的独立验证。
