"""
特征工程优化脚本
基于166个特征的临床含义，创建有区分度的组合特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


def load_and_clean_data():
    """加载并清理数据"""
    data_dir = str(Path(__file__).resolve().parent)

    df_features = pd.read_csv(f"{data_dir}/patient_level_features_cleaned.csv")
    df_labels = pd.read_excel(f"{data_dir}/patient_stable_311.xlsx")
    df_labels.columns = ['patient_id', 'label']

    df_labels['patient_id'] = df_labels['patient_id'].astype(str)
    df_features['patient_id'] = df_features['patient_id'].astype(str)

    df_merged = pd.merge(df_features, df_labels, on='patient_id', how='inner')

    return df_merged


def remove_non_biological_features(df):
    """移除非生物学意义的特征（图像采集参数）"""
    cols_to_remove = [col for col in df.columns if 'pixel_spacing' in col]
    print(f"移除 {len(cols_to_remove)} 个图像采集参数特征: {cols_to_remove[:4]}...")
    df = df.drop(columns=cols_to_remove, errors='ignore')
    return df


def create_clinical_features(df):
    """基于临床意义创建组合特征"""
    print("\n" + "="*60)
    print("创建临床组合特征")
    print("="*60)

    # 1. 后群肌肉总功能体积（脊柱稳定的关键）
    df['Posterior_Total_Func_Volume'] = (
        df['erector_spinae_left_3D_Func_Volume'] + df['erector_spinae_right_3D_Func_Volume'] +
        df['multifidus_left_3D_Func_Volume'] + df['multifidus_right_3D_Func_Volume']
    )

    # 2. 后群肌肉总体积
    df['Posterior_Total_Volume'] = (
        df['erector_spinae_left_3D_Volume'] + df['erector_spinae_right_3D_Volume'] +
        df['multifidus_left_3D_Volume'] + df['multifidus_right_3D_Volume']
    )

    # 3. 后群功能比例
    df['Posterior_Func_Ratio'] = df['Posterior_Total_Func_Volume'] / (df['Posterior_Total_Volume'] + 1e-8)

    # 4. 多裂肌与竖脊肌功能比（多裂肌是脊柱节段稳定的关键）
    df['MF_ES_Func_Ratio'] = (
        df['multifidus_left_3D_Func_Volume'] + df['multifidus_right_3D_Func_Volume']
    ) / (df['erector_spinae_left_3D_Func_Volume'] + df['erector_spinae_right_3D_Func_Volume'] + 1e-8)

    # 5. 多裂肌功能质量（左右平均FIP）
    df['MF_Avg_FIP'] = (df['multifidus_left_3D_FIP'] + df['multifidus_right_3D_FIP']) / 2
    df['MF_FIP_Asymmetry'] = abs(df['multifidus_left_3D_FIP'] - df['multifidus_right_3D_FIP'])

    # 6. 竖脊肌功能质量（左右平均FIP）
    df['ES_Avg_FIP'] = (df['erector_spinae_left_3D_FIP'] + df['erector_spinae_right_3D_FIP']) / 2
    df['ES_FIP_Asymmetry'] = abs(df['erector_spinae_left_3D_FIP'] - df['erector_spinae_right_3D_FIP'])

    # 7. 腰大肌功能质量（左右平均FIP）
    df['Psoas_Avg_FIP'] = (df['psoas_left_3D_FIP'] + df['psoas_right_3D_FIP']) / 2
    df['Psoas_FIP_Asymmetry'] = abs(df['psoas_left_3D_FIP'] - df['psoas_right_3D_FIP'])

    # 8. 总功能体积（前群+后群）
    df['Total_Func_Volume'] = (
        df['Posterior_Total_Func_Volume'] +
        df['psoas_left_3D_Func_Volume'] + df['psoas_right_3D_Func_Volume']
    )

    # 9. 总肌肉体积
    df['Total_Volume'] = (
        df['Posterior_Total_Volume'] +
        df['psoas_left_3D_Volume'] + df['psoas_right_3D_Volume']
    )

    # 10. 前后肌肉功能比例
    df['Anterior_Posterior_Func_Ratio'] = (
        df['psoas_left_3D_Func_Volume'] + df['psoas_right_3D_Func_Volume']
    ) / (df['Posterior_Total_Func_Volume'] + 1e-8)

    # 11. 多裂肌萎缩指数（功能体积/总体积）
    df['MF_Atrophy_Index'] = 1 - df['MF_Avg_FIP']

    # 12. 竖脊肌萎缩指数
    df['ES_Atrophy_Index'] = 1 - df['ES_Avg_FIP']

    # 13. 腰大肌萎缩指数
    df['Psoas_Atrophy_Index'] = 1 - df['Psoas_Avg_FIP']

    # 14. 综合萎缩指数（加权平均）
    df['Composite_Atrophy_Index'] = (
        df['MF_Atrophy_Index'] * 0.4 +  # 多裂肌权重更高
        df['ES_Atrophy_Index'] * 0.3 +
        df['Psoas_Atrophy_Index'] * 0.3
    )

    # 15. 左右总不对称指数
    df['Total_Volume_Asymmetry'] = abs(
        (df['erector_spinae_left_3D_Volume'] + df['multifidus_left_3D_Volume'] + df['psoas_left_3D_Volume']) -
        (df['erector_spinae_right_3D_Volume'] + df['multifidus_right_3D_Volume'] + df['psoas_right_3D_Volume'])
    ) / df['Total_Volume']

    # 16. FIP梯度（后群到前群）
    df['FIP_Gradient_Posterior_to_Anterior'] = df['Psoas_Avg_FIP'] - df['MF_Avg_FIP']

    # 17. 多裂肌与腰大肌FIP差异
    df['MF_Psoas_FIP_Diff'] = df['MF_Avg_FIP'] - df['Psoas_Avg_FIP']

    # 18. 多裂肌分布均匀性（平均CV）
    df['MF_Distribution_Uniformity'] = 1 / (
        (abs(df['multifidus_left_CV_Area_Z']) + abs(df['multifidus_right_CV_Area_Z'])) / 2 + 1e-8
    )

    # 19. 竖脊肌分布均匀性
    df['ES_Distribution_Uniformity'] = 1 / (
        (abs(df['erector_spinae_left_CV_Area_Z']) + abs(df['erector_spinae_right_CV_Area_Z'])) / 2 + 1e-8
    )

    # 20. 跨肌肉FIP协同性
    df['MF_ES_FIP_Synergy'] = df['MF_Avg_FIP'] * df['ES_Avg_FIP']
    df['MF_Psoas_FIP_Synergy'] = df['MF_Avg_FIP'] * df['Psoas_Avg_FIP']

    # 21. 多裂肌峰值位置一致性
    df['MF_Peak_Position_Consistency'] = abs(
        df['multifidus_left_Peak_Area_Slice_Index'] - df['multifidus_right_Peak_Area_Slice_Index']
    )

    # 22. 竖脊肌峰值位置一致性
    df['ES_Peak_Position_Consistency'] = abs(
        df['erector_spinae_left_Peak_Area_Slice_Index'] - df['erector_spinae_right_Peak_Area_Slice_Index']
    )

    # 23. 功能体积集中度（后群功能体积/总功能体积）
    df['Func_Volume_Concentration'] = df['Posterior_Total_Func_Volume'] / (df['Total_Func_Volume'] + 1e-8)

    # 24. 肌肉内FIP一致性（标准差的倒数）
    df['MF_FIP_Consistency'] = 1 / (
        (df['multifidus_left_Std_FIP'] + df['multifidus_right_Std_FIP']) / 2 + 1e-8
    )
    df['ES_FIP_Consistency'] = 1 / (
        (df['erector_spinae_left_Std_FIP'] + df['erector_spinae_right_Std_FIP']) / 2 + 1e-8
    )

    # 25. 跨层变化率（多裂肌）
    df['MF_Cross_Layer_Variability'] = (
        df['multifidus_left_cross_FIP_Slope'] + df['multifidus_right_cross_FIP_Slope']
    ) / 2

    # 26. 跨层变化率（竖脊肌）
    df['ES_Cross_Layer_Variability'] = (
        df['erector_spinae_left_cross_FIP_Slope'] + df['erector_spinae_right_cross_FIP_Slope']
    ) / 2

    # 27. 跨层变化率（腰大肌）
    df['Psoas_Cross_Layer_Variability'] = (
        df['psoas_left_cross_FIP_Slope'] + df['psoas_right_cross_FIP_Slope']
    ) / 2

    # 28. 总体跨层变化率
    df['Total_Cross_Layer_Variability'] = (
        df['MF_Cross_Layer_Variability'] + df['ES_Cross_Layer_Variability'] + df['Psoas_Cross_Layer_Variability']
    ) / 3

    print(f"新增 {30} 个临床组合特征")

    return df


def select_key_features(df, y, n_features=30):
    """特征选择：结合临床意义和统计显著性"""
    print("\n" + "="*60)
    print("特征选择")
    print("="*60)

    X = df.drop(columns=['patient_id', 'label'])

    # 临床关键特征列表（优先保留）
    clinical_key_features = [
        'MF_Avg_FIP', 'ES_Avg_FIP', 'Psoas_Avg_FIP',
        'Posterior_Total_Func_Volume', 'Total_Func_Volume',
        'MF_FIP_Asymmetry', 'ES_FIP_Asymmetry', 'Psoas_FIP_Asymmetry',
        'Composite_Atrophy_Index',
        'Psoas_Posterior_Ratio',
        'Symmetry_Index_Area_MF', 'Symmetry_Index_Area_ES',
        'MF_ES_Func_Ratio',
        'MF_Psoas_FIP_Diff',
        'Total_Volume_Asymmetry',
        'MF_Distribution_Uniformity',
        'MF_ES_FIP_Synergy',
        'Func_Volume_Concentration',
        'Total_Cross_Layer_Variability'
    ]

    # 过滤存在的特征
    clinical_key_features = [f for f in clinical_key_features if f in X.columns]
    print(f"临床关键特征数: {len(clinical_key_features)}")

    # 统计特征选择
    selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    # 获取统计显著的特征
    stat_features = X.columns[selector.get_support()].tolist()
    print(f"统计显著特征数: {len(stat_features)}")

    # 合并特征列表（临床优先）
    final_features = list(dict.fromkeys(clinical_key_features + stat_features))[:n_features]

    # 确保选择的特征都在数据中
    final_features = [f for f in final_features if f in X.columns]

    print(f"最终选择特征数: {len(final_features)}")
    print(f"最终选择特征:\n{final_features}")

    return final_features


def test_model(X, y, feature_name):
    """测试模型性能"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'LR': LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'XGB': xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=1.5, random_state=42),
        'LGB': lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, is_unbalance=True, random_state=42),
    }

    results = []
    for name, model in models.items():
        if name == 'LR':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if name == 'LR':
            cv_scores = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        results.append({
            'model': name,
            'auc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std()
        })

        print(f"  {name}: AUC={auc:.4f}, CV-AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    return pd.DataFrame(results)


def analyze_feature_importance(X, y, feature_names):
    """分析特征重要性"""
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)

    # 使用随机森林评估特征重要性
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 重要特征:")
    for i, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return importance_df


def main():
    """主函数"""
    print("="*60)
    print("脊柱稳定性特征工程优化")
    print("="*60)

    # 加载数据
    df = load_and_clean_data()
    y = df['label']

    print(f"\n原始数据: {df.shape[0]} 样本, {df.shape[1]-2} 特征")

    # 移除非生物学特征
    df = remove_non_biological_features(df)
    print(f"移除后: {df.shape[1]-2} 特征")

    # 创建临床组合特征
    df = create_clinical_features(df)
    print(f"创建组合特征后: {df.shape[1]-2} 特征")

    # 测试不同特征集
    print("\n" + "="*60)
    print("测试不同特征集的性能")
    print("="*60)

    X_all = df.drop(columns=['patient_id', 'label'])

    # 测试1: 原始特征（去除pixel_spacing后）
    print("\n【测试1】原始特征 (去除pixel_spacing)")
    results1 = test_model(X_all, y, "原始特征")

    # 测试2: 仅临床组合特征
    clinical_new_features = [
        'Posterior_Total_Func_Volume', 'Posterior_Total_Volume', 'Posterior_Func_Ratio',
        'MF_ES_Func_Ratio', 'MF_Avg_FIP', 'ES_Avg_FIP', 'Psoas_Avg_FIP',
        'MF_FIP_Asymmetry', 'ES_FIP_Asymmetry', 'Psoas_FIP_Asymmetry',
        'Total_Func_Volume', 'Total_Volume', 'Anterior_Posterior_Func_Ratio',
        'MF_Atrophy_Index', 'ES_Atrophy_Index', 'Psoas_Atrophy_Index', 'Composite_Atrophy_Index',
        'Total_Volume_Asymmetry', 'FIP_Gradient_Posterior_to_Anterior', 'MF_Psoas_FIP_Diff',
        'MF_Distribution_Uniformity', 'ES_Distribution_Uniformity',
        'MF_ES_FIP_Synergy', 'MF_Psoas_FIP_Synergy',
        'MF_Peak_Position_Consistency', 'ES_Peak_Position_Consistency',
        'Func_Volume_Concentration',
        'MF_FIP_Consistency', 'ES_FIP_Consistency',
        'MF_Cross_Layer_Variability', 'ES_Cross_Layer_Variability', 'Psoas_Cross_Layer_Variability',
        'Total_Cross_Layer_Variability'
    ]
    clinical_new_features = [f for f in clinical_new_features if f in X_all.columns]
    print(f"\n【测试2】仅临床组合特征 ({len(clinical_new_features)}个)")
    results2 = test_model(X_all[clinical_new_features], y, "临床组合特征")

    # 测试3: 临床组合特征 + 关键原始特征
    key_original_features = [
        'multifidus_left_3D_FIP', 'multifidus_right_3D_FIP',
        'erector_spinae_left_3D_FIP', 'erector_spinae_right_3D_FIP',
        'psoas_left_3D_FIP', 'psoas_right_3D_FIP',
        'multifidus_left_3D_Func_Volume', 'multifidus_right_3D_Func_Volume',
        'Symmetry_Index_Area_MF', 'Symmetry_Index_Area_ES', 'Symmetry_Index_Area_Psoas',
        'Psoas_Posterior_Ratio', 'Posterior_Func_Area_Total'
    ]
    key_original_features = [f for f in key_original_features if f in X_all.columns]
    combined_features = clinical_new_features + key_original_features
    print(f"\n【测试3】临床组合特征 + 关键原始特征 ({len(combined_features)}个)")
    results3 = test_model(X_all[combined_features], y, "组合特征")

    # 测试4: 自动选择的最优特征
    selected_features = select_key_features(df, y, n_features=30)
    print(f"\n【测试4】自动选择的最优特征 ({len(selected_features)}个)")
    results4 = test_model(X_all[selected_features], y, "自动选择特征")

    # 分析特征重要性
    importance_df = analyze_feature_importance(X_all[combined_features], y, combined_features)

    # 保存结果
    output_dir = Path(__file__).resolve().parent / 'classification_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存增强后的数据
    df.to_csv(output_dir / 'patient_level_features_engineered.csv', index=False)
    print(f"\n增强后的数据已保存到: {output_dir / 'patient_level_features_engineered.csv'}")

    # 保存特征重要性
    importance_df.to_csv(output_dir / 'feature_importance_engineered.csv', index=False)

    # 汇总结果
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)

    best_results = pd.DataFrame({
        '特征集': ['原始特征', '临床组合特征', '组合特征', '自动选择特征'],
        '最佳AUC': [
            results1['auc'].max(),
            results2['auc'].max(),
            results3['auc'].max(),
            results4['auc'].max()
        ],
        '最佳CV-AUC': [
            results1['cv_auc_mean'].max(),
            results2['cv_auc_mean'].max(),
            results3['cv_auc_mean'].max(),
            results4['cv_auc_mean'].max()
        ]
    })
    print(best_results.to_string(index=False))

    best_results.to_csv(output_dir / 'feature_engineering_results.csv', index=False)

    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == '__main__':
    main()
