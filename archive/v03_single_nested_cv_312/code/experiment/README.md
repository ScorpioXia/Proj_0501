# 当前实验代码

- `build_features.py`：从 `PATIENT_LIST_FILE.csv` 动态读取有标签 `patient_id`，
  对 v7 2D/3D/跨层/多肌肉特征进行确定性患者级整理。
- `feature_audit.py`：动态队列覆盖、数学恒等式、取值范围、重复纹理和极端比值审计。
- `preprocessing.py`：只在训练折拟合的预处理与点二列相关筛选。
- `modeling.py`：ElasticNet、XGBoost、LightGBM、随机森林、校准 RBF-SVM、EBM
  的嵌套交叉验证及真实 OOF 结果。
- `runner.py`：统一运行、异常记录、环境记录和结果保存。
- `config.json`：默认折数、随机种子、bootstrap、候选超参数和模型清单。

正式入口是项目根目录的 `run_experiment.py`。当前包不依赖 `archive/` 中的任何代码。

模型筛选的默认主指标为 ROC-AUC，同时输出 PR-AUC、Brier、敏感度、特异度、
混淆矩阵指标、bootstrap 置信区间、折内过拟合差和特征选择稳定性。阈值 0.5
仅用于描述性分类指标，不参与调参。
