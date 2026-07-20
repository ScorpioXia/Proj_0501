import pandas as pd
import numpy as np
from pathlib import Path

# 加载数据
data_dir = Path(__file__).resolve().parent
feature_file = data_dir / 'patient_level_features_cleaned.csv'
label_file = data_dir / 'patient_stable_311.xlsx'

df_features = pd.read_csv(feature_file)
df_labels = pd.read_excel(label_file)

print("=" * 60)
print("特征数据概况")
print("=" * 60)
print(f"样本数: {df_features.shape[0]}, 特征数: {df_features.shape[1]}")
print(f"\n前5行:\n{df_features.head()}")
print(f"\n特征列名:\n{df_features.columns.tolist()}")

print("\n" + "=" * 60)
print("标签数据概况")
print("=" * 60)
print(f"样本数: {df_labels.shape[0]}, 列数: {df_labels.shape[1]}")
print(f"\n列名: {df_labels.columns.tolist()}")
print(f"\n前5行:\n{df_labels.head()}")

# 转换patient_id类型并合并
df_labels['patient_id'] = df_labels['patient_id'].astype(str)
df_features['patient_id'] = df_features['patient_id'].astype(str)

df_merged = pd.merge(df_features, df_labels, on='patient_id', how='inner')
print("\n" + "=" * 60)
print("合并后数据概况")
print("=" * 60)
print(f"合并后样本数: {df_merged.shape[0]}")
print(f"\n标签分布:\n{df_merged['label'].value_counts()}")
print(f"\n标签占比:\n{df_merged['label'].value_counts(normalize=True)}")

# 检查缺失值
X = df_merged.drop(columns=['patient_id', 'label'])
print("\n" + "=" * 60)
print("缺失值分析")
print("=" * 60)
missing_count = X.isnull().sum()
missing_percent = (X.isnull().sum() / len(X)) * 100
missing_df = pd.DataFrame({'缺失数': missing_count, '缺失率(%)': missing_percent})
print(f"有缺失值的特征数: {(missing_count > 0).sum()}")
print(f"\n缺失值前10个特征:\n{missing_df[missing_count > 0].head(10)}")

# 检查特征方差（识别常数或低方差特征）
print("\n" + "=" * 60)
print("特征方差分析")
print("=" * 60)
variances = X.var()
print(f"方差为0的特征数: {(variances == 0).sum()}")
print(f"方差小于0.01的特征数: {(variances < 0.01).sum()}")
print(f"\n最小方差的10个特征:\n{variances.sort_values().head(10)}")

# 检查特征类型
print("\n" + "=" * 60)
print("特征类型分析")
print("=" * 60)
print(f"数值型特征数: {X.select_dtypes(include=[np.number]).shape[1]}")
print(f"非数值型特征数: {X.select_dtypes(exclude=[np.number]).shape[1]}")

# 检查类别不平衡
print("\n" + "=" * 60)
print("类别不平衡分析")
print("=" * 60)
class_counts = df_merged['label'].value_counts()
print(f"类别0数量: {class_counts.get(0, 0)}")
print(f"类别1数量: {class_counts.get(1, 0)}")
print(f"类别比例 (0:1): {class_counts.get(0, 1)}:{class_counts.get(1, 1)}")
if class_counts.get(0, 1) > 0 and class_counts.get(1, 1) > 0:
    print(f"不平衡比率: {max(class_counts) / min(class_counts):.2f}")
