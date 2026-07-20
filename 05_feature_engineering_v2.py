"""
特征工程优化脚本 - 简化版
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def create_clinical_features(df):
    """创建临床组合特征"""
    print("创建临床组合特征...")

    # 多裂肌与竖脊肌功能比
    df['MF_ES_Func_Ratio'] = (
        df['multifidus_left_3D_Func_Volume'] + df['multifidus_right_3D_Func_Volume']
    ) / (df['erector_spinae_left_3D_Func_Volume'] + df['erector_spinae_right_3D_Func_Volume'] + 1e-8)

    # 多裂肌平均FIP
    df['MF_Avg_FIP'] = (df['multifidus_left_3D_FIP'] + df['multifidus_right_3D_FIP']) / 2

    # 竖脊肌平均FIP
    df['ES_Avg_FIP'] = (df['erector_spinae_left_3D_FIP'] + df['erector_spinae_right_3D_FIP']) / 2

    # 腰大肌平均FIP
    df['Psoas_Avg_FIP'] = (df['psoas_left_3D_FIP'] + df['psoas_right_3D_FIP']) / 2

    # FIP不对称指数
    df['MF_FIP_Asymmetry'] = abs(df['multifidus_left_3D_FIP'] - df['multifidus_right_3D_FIP'])
    df['ES_FIP_Asymmetry'] = abs(df['erector_spinae_left_3D_FIP'] - df['erector_spinae_right_3D_FIP'])

    # 综合萎缩指数
    df['Composite_Atrophy'] = 1 - (df['MF_Avg_FIP'] + df['ES_Avg_FIP'] + df['Psoas_Avg_FIP']) / 3

    # 后群总功能体积
    df['Posterior_Func_Volume'] = (
        df['erector_spinae_left_3D_Func_Volume'] + df['erector_spinae_right_3D_Func_Volume'] +
        df['multifidus_left_3D_Func_Volume'] + df['multifidus_right_3D_Func_Volume']
    )

    # 总功能体积
    df['Total_Func_Volume'] = (
        df['Posterior_Func_Volume'] +
        df['psoas_left_3D_Func_Volume'] + df['psoas_right_3D_Func_Volume']
    )

    # 功能体积比例
    df['Func_Volume_Ratio'] = df['Posterior_Func_Volume'] / (df['Total_Func_Volume'] + 1e-8)

    # 多裂肌与腰大肌FIP差异
    df['MF_Psoas_FIP_Diff'] = df['MF_Avg_FIP'] - df['Psoas_Avg_FIP']

    # 跨肌肉FIP协同性
    df['MF_ES_FIP_Synergy'] = df['MF_Avg_FIP'] * df['ES_Avg_FIP']

    return df


def main():
    print("="*60)
    print("特征工程优化")
    print("="*60)

    # 加载数据
    data_dir = str(Path(__file__).resolve().parent)
    df = pd.read_csv(f"{data_dir}/patient_level_features_cleaned.csv")
    labels = pd.read_excel(f"{data_dir}/patient_stable_311.xlsx")
    labels.columns = ['patient_id', 'label']
    labels['patient_id'] = labels['patient_id'].astype(str)
    df['patient_id'] = df['patient_id'].astype(str)
    df = pd.merge(df, labels, on='patient_id', how='inner')

    # 移除图像采集参数
    pixel_cols = [c for c in df.columns if 'pixel_spacing' in c]
    df = df.drop(columns=pixel_cols)

    # 创建临床特征
    df = create_clinical_features(df)

    y = df['label']
    print(f"\n样本数: {len(df)}, 特征数: {df.shape[1]-2}")

    # 测试不同特征集
    results = []

    # 1. 原始特征
    X1 = df.drop(columns=['patient_id', 'label'])
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    X1_train, X1_test, y_train, y_test = train_test_split(X1_scaled, y, test_size=0.2, random_state=42, stratify=y)

    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    lr.fit(X1_train, y_train)
    y_prob = lr.predict_proba(X1_test)[:, 1]
    auc1 = roc_auc_score(y_test, y_prob)
    print(f"\n【1】原始特征 ({X1.shape[1]}个): AUC = {auc1:.4f}")

    # 2. 仅临床新特征
    clinical_features = ['MF_ES_Func_Ratio', 'MF_Avg_FIP', 'ES_Avg_FIP', 'Psoas_Avg_FIP',
                        'MF_FIP_Asymmetry', 'ES_FIP_Asymmetry', 'Composite_Atrophy',
                        'Posterior_Func_Volume', 'Total_Func_Volume', 'Func_Volume_Ratio',
                        'MF_Psoas_FIP_Diff', 'MF_ES_FIP_Synergy']
    X2 = df[clinical_features]
    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)
    X2_train, X2_test, _, _ = train_test_split(X2_scaled, y, test_size=0.2, random_state=42, stratify=y)

    lr2 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    lr2.fit(X2_train, y_train)
    y_prob2 = lr2.predict_proba(X2_test)[:, 1]
    auc2 = roc_auc_score(y_test, y_prob2)
    print(f"【2】临床新特征 ({len(clinical_features)}个): AUC = {auc2:.4f}")

    # 3. 组合特征
    combined_features = clinical_features + [
        'multifidus_left_3D_FIP', 'multifidus_right_3D_FIP',
        'erector_spinae_left_3D_FIP', 'erector_spinae_right_3D_FIP',
        'psoas_left_3D_FIP', 'psoas_right_3D_FIP',
        'multifidus_left_3D_Func_Volume', 'multifidus_right_3D_Func_Volume',
        'Symmetry_Index_Area_MF', 'Symmetry_Index_Area_ES',
        'Psoas_Posterior_Ratio'
    ]
    combined_features = [c for c in combined_features if c in df.columns]
    X3 = df[combined_features]
    scaler3 = StandardScaler()
    X3_scaled = scaler3.fit_transform(X3)
    X3_train, X3_test, _, _ = train_test_split(X3_scaled, y, test_size=0.2, random_state=42, stratify=y)

    lr3 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    lr3.fit(X3_train, y_train)
    y_prob3 = lr3.predict_proba(X3_test)[:, 1]
    auc3 = roc_auc_score(y_test, y_prob3)
    print(f"【3】组合特征 ({len(combined_features)}个): AUC = {auc3:.4f}")

    # 4. XGBoost + 组合特征
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=1.5, random_state=42)
    X4 = df[combined_features]
    X4_train, X4_test, _, _ = train_test_split(X4, y, test_size=0.2, random_state=42, stratify=y)
    xgb_model.fit(X4_train, y_train)
    y_prob4 = xgb_model.predict_proba(X4_test)[:, 1]
    auc4 = roc_auc_score(y_test, y_prob4)
    print(f"【4】XGBoost + 组合特征: AUC = {auc4:.4f}")

    # 5. 精选特征（多裂肌核心）
    mf_core = ['MF_Avg_FIP', 'MF_FIP_Asymmetry', 'MF_ES_Func_Ratio',
               'multifidus_left_3D_FIP', 'multifidus_right_3D_FIP',
               'multifidus_left_3D_Func_Volume', 'multifidus_right_3D_Func_Volume',
               'Symmetry_Index_Area_MF', 'Composite_Atrophy']
    mf_core = [c for c in mf_core if c in df.columns]
    X5 = df[mf_core]
    scaler5 = StandardScaler()
    X5_scaled = scaler5.fit_transform(X5)
    X5_train, X5_test, _, _ = train_test_split(X5_scaled, y, test_size=0.2, random_state=42, stratify=y)

    lr5 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)
    lr5.fit(X5_train, y_train)
    y_prob5 = lr5.predict_proba(X5_test)[:, 1]
    auc5 = roc_auc_score(y_test, y_prob5)
    print(f"【5】多裂肌核心特征 ({len(mf_core)}个): AUC = {auc5:.4f}")

    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)

    summary = pd.DataFrame({
        '特征集': ['原始特征', '临床新特征', '组合特征', 'XGBoost+组合', '多裂肌核心'],
        'AUC': [auc1, auc2, auc3, auc4, auc5]
    })
    print(summary.to_string(index=False))

    # 保存结果
    df.to_csv(f"{data_dir}/classification_results/patient_level_features_engineered.csv", index=False)
    summary.to_csv(f"{data_dir}/classification_results/feature_engineering_summary.csv", index=False)
    print("\n结果已保存!")


if __name__ == '__main__':
    main()
