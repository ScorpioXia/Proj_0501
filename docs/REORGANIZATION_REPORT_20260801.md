# 项目整理报告（2026-08-01）

## 整理目标

将原先混合在根目录的多代入口、模块、数据和结果拆分为“当前主线、冻结历史、源数据、派生数据、正式结果、文档”六类，并保留全部可追溯材料。

## 完成的结构调整

- 原 `archive/v1` 至 `archive/v7` 按真实实验顺序统一为 `v01`–`v05`、`v09`、`v10`。
- 原根目录节段试验、特征发现代码和结果补齐为 `v06`–`v08` 归档。
- 原内部名 `v8_clinical_mri` 提升为统一主线 `v11_clinical_mri`。
- 当前代码迁入 `src/lumbar_stability/`，根目录只保留一个正式入口 `run_experiment.py`。
- 当前路径、随机种子、折数与 bootstrap 次数集中到 `configs/v11_clinical_mri.json`。
- 标签、MRI 特征、节段标注和 v11 派生输入分别迁入 `data/labels`、`data/features`、`data/annotations` 和 `data/derived`。
- 历史结果随对应版本冻结；根 `results/` 只保留 v11。
- 确认逐文件哈希一致的旧重复结果移入 `archive/legacy/verified_duplicate_results/`，没有删除。
- 原多版本工作包、旧根入口、开发产物和 Python 缓存移入 `archive/legacy/`，不参与当前运行。
- 新的像素级深度学习试验迁出为同级独立项目 `E:\code\Proj_0801`；本项目仅保留权威标签源，不保留其运行时代码、缓存或结果。

## 最新版本保存状态

- 入口：`run_experiment.py`
- 配置：`configs/v11_clinical_mri.json`
- 当前包：`src/lumbar_stability/`
- 依赖：`pyproject.toml`、`requirements.txt`
- 正式结果：`results/v11_clinical_mri/`
- 当前说明：根 `README.md` 与 `results/v11_clinical_mri/README.md`

现有结果快照最初以内部名 v8 运行。整理只改变规范目录和当前重跑入口，不重新计算或美化数值；原始绝对路径、内部版本和日志仍保存在结果目录以供追溯。

## 兼容与风险说明

- 历史版本是冻结快照，不保证能在当前依赖环境直接运行。
- 旧 v05 根入口要求 `label` 字段，而当前患者表使用 `instability_label`；该问题保留在历史快照，当前 v11 已使用正确字段。
- 原始结果 CSV 中 `v7_label_selected_locked7` 是历史内部名称，统一目录来源实际为 v10；未批量修改以免改变既有结果。
- 患者级数据和 OOF 文件包含研究标识符，`.gitignore` 默认不发布这些文件；聚合结果与说明文件可以纳入版本控制。

## 完成后的只读验证

- 10个历史版本均存在版本级 `README.md`、`code/` 和 `results/`。
- 当前四项配置输入均存在。
- 从原始3D长表直接构建的312×7 MRI面板，与完成实验时使用的旧患者宽表逐值一致；最大绝对浮点差为2.27×10⁻¹³，在1×10⁻¹²容差内完全一致。
- 24,720行重复OOF记录不存在重复预测键；每位患者、每种模型均有10个OOF概率。
- 从患者平均OOF概率重新计算的全部ROC-AUC与 `aggregate_performance.csv` 一致。
- 结果摘要生成器已单独测试，英文和中文输出均明确包含0.708。
- `python -m unittest discover -s tests -v` 与 `python scripts/validate_project.py` 均通过。
