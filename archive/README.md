# 历史版本归档

本目录冻结 v01–v10。每个规范版本目录均包含：

- `README.md`：带版本号的中文说明、实验设计、主要结果和限制；
- `code/`：该版入口、模块、配置及依赖快照；
- `results/`：该版已有输出，未伪造或补写缺失实验结果；
- `features/`：仅在该版需要冻结特征输入时存在。

## 统一版本表

| 统一版本 | 内容 | 核心结论 |
|---|---|---|
| `v01_elasticnet_311` | 311 例、3D 特征、ElasticNet | AUC 约 0.566 |
| `v02_multimodel_311` | 多模型与 2D/3D 组合 | 最好约 0.583；可疑特征去除后约 0.527 |
| `v03_single_nested_cv_312` | 312 例单次嵌套 CV | LightGBM+2D AUC 约 0.596 |
| `v04_repeated_nested_cv_312` | 10×5 折重复嵌套 CV | 最好 ElasticNet+3D，AUC 约 0.572 |
| `v05_factor_analysis_312` | Pearson 筛选与因子建模 | 最好 AUC 约 0.570 |
| `v06_segment_pilot_30` | 30 例滑脱节段可行性试验 | target 方案 AUC 约 0.615，探索性 |
| `v07_segment_validation_219` | 节段定位确认与敏感性分析 | 主要 n=189，gradient AUC 约 0.577 |
| `v08_feature_discovery_219` | 嵌套特征发现与负对照 | locked baseline 0.518；新子集未提升 |
| `v09_pearson_factor_219` | Pearson→六因子严格复现 | 严格 Logistic AUC 约 0.512 |
| `v10_stability_lasso_219` | Stability LASSO、最多 7 特征 | 严格最好约 0.442；全局筛选的 0.63–0.66 有泄漏 |

最新版 v11 不在归档中；其代码与结果位于项目根目录。详细的旧编号映射和提升分析见 [实验版本沿革](../docs/EXPERIMENT_LINEAGE.md)。

`legacy/` 保存整理过程中确认的重复结果、旧工作包和生成缓存，仅用于追溯；它不代表额外实验版本，也不被当前代码读取。
