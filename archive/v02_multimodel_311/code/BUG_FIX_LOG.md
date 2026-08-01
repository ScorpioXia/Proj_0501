# 第二阶段异常与 Bug 记录

## 2026-07-19：Windows 多进程参数搜索崩溃

- 发生位置：XGBoost + E1 的第一次完整冒烟测试，外层第 1 折。
- 表现：`GridSearchCV(n_jobs=2)` 的 loky 子进程在 NumPy `corrcoef/cov` 中发生 Windows fatal exception `0xc06d007f`，主进程报 `TerminatedWorkerError`；子进程清理时另有 `psutil.AccessDenied`。
- 判断：Windows 子进程加载科学计算 DLL/权限的兼容性问题；发生在模型拟合前的训练折 Pearson 计算，不是数据结论。
- 处理：参数搜索改为 `n_jobs=1`，所有模型内部线程也固定为 1；增加每个外层折的流水线缓存，减少串行条件下的重复预处理。
- 验证要求：重新运行完整 XGBoost+E1 冒烟测试，必须得到 311 条唯一 OOF 预测且 5 折均完成后，才运行全模型矩阵。

### 进一步定位

- 单进程重试仍在 `numpy.corrcoef -> numpy.cov` 原生代码路径发生相同 `0xc06d007f`，而独立 XGBoost 拟合正常，因此排除了 XGBoost 安装问题。
- 处理：第二阶段使用中心化点积直接计算点二列相关（与 Pearson/point-biserial 公式等价），绕开该 NumPy Windows 原生崩溃路径；训练折内筛选原则不变。

## 2026-07-19：包安装镜像 SSL 失败

- 表现：环境默认清华 PyPI 镜像在获取 XGBoost 时出现 `SSLEOFError`，未安装任何目标包。
- 处理：改用 PyPI 官方源，成功安装并导入 XGBoost 3.2.0、LightGBM 4.7.0、interpret 0.7.8。

## 2026-07-19：nnUNet-master 的 NumPy/MKL 原生崩溃

- 最小复现：仅执行 186×154 矩阵乘法 `x.T @ y` 即出现 Windows fatal exception `0xc06d007f`；与项目数据和 XGBoost 拟合无关。
- 环境状态：Conda NumPy 2.2.6 链接 MKL 2025.3.0，SciPy 1.15.3 来自 pip，数值栈来源混合。
- 首次修复尝试：清华 Conda 镜像 SSL 失败；直接使用官方 URL 又被用户级 `.condarc` 的 `custom_channels` 重写回清华镜像。
- 尝试方案：不修改用户全局 `.condarc`，曾用项目级临时配置指向官方频道；但 Conda 23.7 仍合并用户频道并访问失效镜像，因此该方案未执行安装，临时配置未保留。

### 最终修复

- Conda 23.7 仍会合并用户级频道列表并读取失效镜像，故未修改全局配置，也未执行半完成的 Conda 事务。
- 最终从 PyPI 官方源强制重装同版本 NumPy 2.2.6 与 SciPy 1.15.3 wheel。验证：1000×421 矩阵乘法、SciPy Pearson、scikit-learn 流水线、XGBoost/LightGBM/EBM 导入全部成功，`pip check` 无损坏依赖；运行库由损坏的 MKL 切换为 wheel 自带 OpenBLAS。

## 2026-07-19：敏感性实验缓存路径过长

- 表现：模型完成但捕获 1,250 条 joblib `CacheWarning`，均为 `output.pkl` 临时路径不存在。
- 原因：Windows 传统路径长度限制；敏感性实验名称较长，叠加 joblib 子目录与哈希临时文件后超过限制。
- 影响：仅缓存写盘失败，流水线自动重新计算，模型和 OOF 预测未中断。
- 处理：缓存目录改用实验 ID 的 10 位 SHA-1 短键，重跑敏感性实验并要求运行警告为 0。
