"""
验证05脚本中12个临床特征的真实性能（5折交叉验证）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_data():
    """加载数据"""
    data_dir = str(Path(__file__).resolve().parent)
    df = pd.read_csv(f"{data_dir}/patient_level_features_cleaned.csv")
    labels = pd.read_excel(f"{data_dir}/patient_stable_311.xlsx")
    labels.columns = ['patient_id', 'label']
    labels['patient_id'] = labels['patient_id'].astype(str)
    df['patient_id'] = df['patient_id'].astype(str)
    df = pd.merge(df, labels, on='patient_id', how='inner')
    return df


def create_clinical_features(df):
    """创建05脚本中的12个临床特征"""
    df['MF_ES_Func_Ratio'] = (
        df['multifidus_left_3D_Func_Volume'] + df['multifidus_right_3D_Func_Volume']
    ) / (df['erector_spinae_left_3D_Func_Volume'] + df['erector_spinae_right_3D_Func_Volume'] + 1e-8)

    df['MF_Avg_FIP'] = (df['multifidus_left_3D_FIP'] + df['multifidus_right_3D_FIP']) / 2
    df['ES_Avg_FIP'] = (df['erector_spinae_left_3D_FIP'] + df['erector_spinae_right_3D_FIP']) / 2
    df['Psoas_Avg_FIP'] = (df['psoas_left_3D_FIP'] + df['psoas_right_3D_FIP']) / 2

    df['MF_FIP_Asymmetry'] = abs(df['multifidus_left_3D_FIP'] - df['multifidus_right_3D_FIP'])
    df['ES_FIP_Asymmetry'] = abs(df['erector_spinae_left_3D_FIP'] - df['erector_spinae_right_3D_FIP'])

    df['Composite_Atrophy'] = 1 - (df['MF_Avg_FIP'] + df['ES_Avg_FIP'] + df['Psoas_Avg_FIP']) / 3

    df['Posterior_Func_Volume'] = (
        df['erector_spinae_left_3D_Func_Volume'] + df['erector_spinae_right_3D_Func_Volume'] +
        df['multifidus_left_3D_Func_Volume'] + df['multifidus_right_3D_Func_Volume']
    )

    df['Total_Func_Volume'] = (
        df['Posterior_Func_Volume'] +
        df['psoas_left_3D_Func_Volume'] + df['psoas_right_3D_Func_Volume']
    )

    df['Func_Volume_Ratio'] = df['Posterior_Func_Volume'] / (df['Total_Func_Volume'] + 1e-8)
    df['MF_Psoas_FIP_Diff'] = df['MF_Avg_FIP'] - df['Psoas_Avg_FIP']
    df['MF_ES_FIP_Synergy'] = df['MF_Avg_FIP'] * df['ES_Avg_FIP']

    clinical_features = ['MF_ES_Func_Ratio', 'MF_Avg_FIP', 'ES_Avg_FIP', 'Psoas_Avg_FIP',
                        'MF_FIP_Asymmetry', 'ES_FIP_Asymmetry', 'Composite_Atrophy',
                        'Posterior_Func_Volume', 'Total_Func_Volume', 'Func_Volume_Ratio',
                        'MF_Psoas_FIP_Diff', 'MF_ES_FIP_Synergy']
    return df, clinical_features


def evaluate_with_cv(X, y, name):
    """5折交叉验证评估"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr.fit(X_train_scaled, y_train)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]
        scores.append(roc_auc_score(y_test, y_prob))
    
    print(f"【{name}】")
    print(f"  CV-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores), np.std(scores)


def main():
    print("="*60)
    print("验证临床特征的真实性能（5折交叉验证）")
    print("="*60)
    
    df = load_data()
    y = df['label']
    
    # 移除图像采集参数
    pixel_cols = [c for c in df.columns if 'pixel_spacing' in c]
    df = df.drop(columns=pixel_cols)
    
    # 创建临床特征
    df, clinical_features = create_clinical_features(df)
    
    # 评估不同特征集
    print(f"\n样本数: {len(df)}, 原始特征数: {df.shape[1]-2}")
    
    # 1. 原始特征
    X_all = df.drop(columns=['patient_id', 'label'])
    evaluate_with_cv(X_all, y, "原始特征 (166个)")
    
    # 2. 12个临床新特征
    X_clinical = df[clinical_features]
    evaluate_with_cv(X_clinical, y, "临床新特征 (12个)")
    
    # 3. Top 5核心特征
    top5_features = ['MF_Avg_FIP', 'Composite_Atrophy', 'MF_ES_Func_Ratio',
                    'Posterior_Func_Volume', 'MF_FIP_Asymmetry']
    X_top5 = df[top5_features]
    evaluate_with_cv(X_top5, y, "Top 5核心特征")
    
    # 4. Top 3核心特征
    top3_features = ['MF_Avg_FIP', 'Composite_Atrophy', 'MF_ES_Func_Ratio']
    X_top3 = df[top3_features]
    evaluate_with_cv(X_top3, y, "Top 3核心特征")
    
    # 5. 仅多裂肌特征
    mf_only = ['MF_Avg_FIP', 'MF_FIP_Asymmetry', 'multifidus_left_3D_FIP', 'multifidus_right_3D_FIP']
    X_mf = df[mf_only]
    evaluate_with_cv(X_mf, y, "仅多裂肌特征")
    
    # 6. 检查标签分布
    print("\n" + "="*60)
    print("标签分布")
    print("="*60)
    print(f"类别0: {y.value_counts()[0]} ({y.value_counts()[0]/len(y)*100:.1f}%)")
    print(f"类别1: {y.value_counts()[1]} ({y.value_counts()[1]/len(y)*100:.1f}%)")
    print(f"不平衡比: {y.value_counts()[0]/y.value_counts()[1]:.2f}")
    
    # 7. 特征与标签相关性分析
    print("\n" + "="*60)
    print("特征与标签相关性分析")
    print("="*60)
    for feature in clinical_features:
        corr = df[feature].corr(y)
        print(f"  {feature}: r = {corr:.4f}")


if __name__ == '__main__':
    main()
