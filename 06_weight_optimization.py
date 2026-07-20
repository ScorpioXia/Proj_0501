"""
临床特征权重优化脚本
基于临床意义调整特征权重，使用网格搜索和交叉验证
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def load_data():
    """加载数据"""
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
    
    return df


def create_weighted_clinical_features(df, weights=None):
    """
    创建加权临床特征
    weights: {MF: 多裂肌权重, ES: 竖脊肌权重, Psoas: 腰大肌权重}
    """
    if weights is None:
        weights = {'MF': 0.5, 'ES': 0.3, 'Psoas': 0.2}  # 默认权重
    
    print(f"使用权重: MF={weights['MF']}, ES={weights['ES']}, Psoas={weights['Psoas']}")
    
    # 基础特征
    df['MF_Avg_FIP'] = (df['multifidus_left_3D_FIP'] + df['multifidus_right_3D_FIP']) / 2
    df['ES_Avg_FIP'] = (df['erector_spinae_left_3D_FIP'] + df['erector_spinae_right_3D_FIP']) / 2
    df['Psoas_Avg_FIP'] = (df['psoas_left_3D_FIP'] + df['psoas_right_3D_FIP']) / 2
    
    # 加权萎缩指数（多裂肌权重最高）
    df['Weighted_Atrophy_Index'] = (
        (1 - df['MF_Avg_FIP']) * weights['MF'] +
        (1 - df['ES_Avg_FIP']) * weights['ES'] +
        (1 - df['Psoas_Avg_FIP']) * weights['Psoas']
    )
    
    # 加权功能体积比
    df['MF_Func_Volume'] = df['multifidus_left_3D_Func_Volume'] + df['multifidus_right_3D_Func_Volume']
    df['ES_Func_Volume'] = df['erector_spinae_left_3D_Func_Volume'] + df['erector_spinae_right_3D_Func_Volume']
    df['Psoas_Func_Volume'] = df['psoas_left_3D_Func_Volume'] + df['psoas_right_3D_Func_Volume']
    
    df['Weighted_Func_Volume'] = (
        df['MF_Func_Volume'] * weights['MF'] +
        df['ES_Func_Volume'] * weights['ES'] +
        df['Psoas_Func_Volume'] * weights['Psoas']
    )
    
    # 多裂肌相关特征（核心稳定肌）
    df['MF_FIP_Asymmetry'] = abs(df['multifidus_left_3D_FIP'] - df['multifidus_right_3D_FIP'])
    df['MF_Volume_Asymmetry'] = abs(
        df['multifidus_left_3D_Volume'] - df['multifidus_right_3D_Volume']
    ) / (df['MF_Func_Volume'] + 1e-8)
    
    # 多裂肌与竖脊肌功能比（强调多裂肌）
    df['MF_ES_Func_Ratio_Weighted'] = (
        df['MF_Func_Volume'] * weights['MF']
    ) / (
        df['ES_Func_Volume'] * weights['ES'] + 1e-8
    )
    
    # 多裂肌质量指数（综合FIP和体积）
    df['MF_Quality_Index'] = df['MF_Avg_FIP'] * np.log1p(df['MF_Func_Volume'])
    
    # 竖脊肌质量指数
    df['ES_Quality_Index'] = df['ES_Avg_FIP'] * np.log1p(df['ES_Func_Volume'])
    
    # 腰大肌质量指数
    df['Psoas_Quality_Index'] = df['Psoas_Avg_FIP'] * np.log1p(df['Psoas_Func_Volume'])
    
    # 加权质量指数
    df['Weighted_Quality_Index'] = (
        df['MF_Quality_Index'] * weights['MF'] +
        df['ES_Quality_Index'] * weights['ES'] +
        df['Psoas_Quality_Index'] * weights['Psoas']
    )
    
    # 多裂肌退化指数（与脊柱稳定性最相关）
    df['MF_Degeneration_Score'] = (
        (1 - df['MF_Avg_FIP']) * 0.4 +
        df['MF_FIP_Asymmetry'] * 0.3 +
        (1 - df['multifidus_left_Std_FIP'] / df['MF_Avg_FIP']) * 0.3
    )
    
    # 跨肌肉协同性（加权）
    df['Weighted_Synergy'] = (
        df['MF_Avg_FIP'] * df['ES_Avg_FIP'] * weights['MF'] * weights['ES'] +
        df['MF_Avg_FIP'] * df['Psoas_Avg_FIP'] * weights['MF'] * weights['Psoas'] +
        df['ES_Avg_FIP'] * df['Psoas_Avg_FIP'] * weights['ES'] * weights['Psoas']
    ) / (weights['MF'] * weights['ES'] + weights['MF'] * weights['Psoas'] + weights['ES'] * weights['Psoas'])
    
    # 前后功能平衡（后群权重更高）
    posterior_weight = weights['MF'] + weights['ES']
    anterior_weight = weights['Psoas']
    df['Anterior_Posterior_Balance'] = (
        (df['MF_Func_Volume'] + df['ES_Func_Volume']) * posterior_weight
    ) / (
        df['Psoas_Func_Volume'] * anterior_weight + 1e-8
    )
    
    # 多裂肌分布均匀性（逆向指标，越小越均匀）
    df['MF_Distribution_Score'] = (
        df['multifidus_left_CV_Area_Z'] + df['multifidus_right_CV_Area_Z']
    ) / 2
    
    # 竖脊肌分布均匀性
    df['ES_Distribution_Score'] = (
        df['erector_spinae_left_CV_Area_Z'] + df['erector_spinae_right_CV_Area_Z']
    ) / 2
    
    # 综合分布评分（加权）
    df['Weighted_Distribution_Score'] = (
        df['MF_Distribution_Score'] * weights['MF'] +
        df['ES_Distribution_Score'] * weights['ES']
    )
    
    # 多裂肌跨层变化率（反映退化程度）
    df['MF_Cross_Layer_Change'] = abs(
        df['multifidus_left_cross_FIP_Slope'] + df['multifidus_right_cross_FIP_Slope']
    ) / 2
    
    return df


def get_feature_sets(df, weights):
    """定义不同特征集"""
    
    # 创建加权特征
    df = create_weighted_clinical_features(df, weights)
    
    # 特征集1: 加权核心特征
    weighted_core = [
        'Weighted_Atrophy_Index',
        'Weighted_Func_Volume',
        'Weighted_Quality_Index',
        'Weighted_Synergy',
        'Anterior_Posterior_Balance',
        'Weighted_Distribution_Score',
        'MF_Degeneration_Score',
        'MF_Quality_Index',
        'MF_FIP_Asymmetry',
        'MF_ES_Func_Ratio_Weighted'
    ]
    
    # 特征集2: 多裂肌专项特征（最关键）
    mf_specific = [
        'MF_Avg_FIP',
        'MF_Func_Volume',
        'MF_FIP_Asymmetry',
        'MF_Volume_Asymmetry',
        'MF_Quality_Index',
        'MF_Degeneration_Score',
        'MF_Distribution_Score',
        'MF_Cross_Layer_Change',
        'multifidus_left_3D_FIP',
        'multifidus_right_3D_FIP',
        'multifidus_left_3D_Func_Volume',
        'multifidus_right_3D_Func_Volume',
        'Symmetry_Index_Area_MF',
        'Symmetry_Index_FIP_MF'
    ]
    
    # 特征集3: 加权综合特征
    weighted_comprehensive = weighted_core + [
        'ES_Avg_FIP',
        'Psoas_Avg_FIP',
        'ES_Quality_Index',
        'Psoas_Quality_Index',
        'Psoas_Posterior_Ratio',
        'Symmetry_Index_Area_ES'
    ]
    
    return df, weighted_core, mf_specific, weighted_comprehensive


def test_with_cv(X, y, name):
    """使用交叉验证测试"""
    # 处理缺失值
    X = X.fillna(0)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic Regression
    lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, 
                                   scale_pos_weight=1.55, random_state=42)
    xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')
    
    print(f"\n【{name}】")
    print(f"  LR:   CV-AUC = {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    print(f"  RF:   CV-AUC = {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"  XGB:  CV-AUC = {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
    
    return {
        'name': name,
        'n_features': X.shape[1],
        'LR_auc': lr_scores.mean(),
        'LR_std': lr_scores.std(),
        'RF_auc': rf_scores.mean(),
        'RF_std': rf_scores.std(),
        'XGB_auc': xgb_scores.mean(),
        'XGB_std': xgb_scores.std(),
        'best_auc': max(lr_scores.mean(), rf_scores.mean(), xgb_scores.mean())
    }


def grid_search_weights(df, y):
    """网格搜索最优权重"""
    print("\n" + "="*60)
    print("网格搜索最优权重组合")
    print("="*60)
    
    weight_combinations = [
        {'MF': 0.5, 'ES': 0.3, 'Psoas': 0.2},  # 默认（多裂肌优先）
        {'MF': 0.6, 'ES': 0.25, 'Psoas': 0.15}, # 强调多裂肌
        {'MF': 0.7, 'ES': 0.2, 'Psoas': 0.1},  # 极强调多裂肌
        {'MF': 0.4, 'ES': 0.4, 'Psoas': 0.2},  # MF与ES平衡
        {'MF': 0.45, 'ES': 0.35, 'Psoas': 0.2}, # 中等MF优先
        {'MF': 0.55, 'ES': 0.3, 'Psoas': 0.15}, # 略强MF
    ]
    
    results = []
    
    for weights in weight_combinations:
        df_temp = df.copy()
        df_temp = create_weighted_clinical_features(df_temp, weights)
        
        weighted_core = [
            'Weighted_Atrophy_Index',
            'Weighted_Func_Volume',
            'Weighted_Quality_Index',
            'Weighted_Synergy',
            'Anterior_Posterior_Balance',
            'MF_Degeneration_Score',
            'MF_Quality_Index'
        ]
        weighted_core = [f for f in weighted_core if f in df_temp.columns]
        
        X = df_temp[weighted_core].fillna(0)  # 处理缺失值
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
        
        result = {
            'MF': weights['MF'],
            'ES': weights['ES'],
            'Psoas': weights['Psoas'],
            'CV_AUC': scores.mean(),
            'Std': scores.std()
        }
        results.append(result)
        
        print(f"权重(MF={weights['MF']:.2f}, ES={weights['ES']:.2f}, Psoas={weights['Psoas']:.2f}): "
              f"CV-AUC={scores.mean():.4f} ± {scores.std():.4f}")
    
    results_df = pd.DataFrame(results)
    best = results_df.sort_values('CV_AUC', ascending=False).iloc[0]
    
    print(f"\n最优权重组合:")
    print(f"  MF={best['MF']:.2f}, ES={best['ES']:.2f}, Psoas={best['Psoas']:.2f}")
    print(f"  CV-AUC={best['CV_AUC']:.4f} ± {best['Std']:.4f}")
    
    return best, results_df


def optimize_regularization(X, y):
    """优化正则化参数"""
    print("\n" + "="*60)
    print("优化正则化参数")
    print("="*60)
    
    X = X.fillna(0)  # 处理缺失值
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_scaled, y)
    
    print(f"最优参数: C={grid.best_params_['C']}, "
          f"penalty={grid.best_params_['penalty']}, "
          f"solver={grid.best_params_['solver']}")
    print(f"最优CV-AUC: {grid.best_score_:.4f}")
    
    return grid.best_params_, grid.best_score_


def main():
    print("="*60)
    print("临床特征权重优化")
    print("="*60)
    
    # 加载数据
    df = load_data()
    y = df['label']
    
    print(f"样本数: {len(df)}")
    
    # 网格搜索最优权重
    best_weights, weight_results = grid_search_weights(df, y)
    
    # 使用最优权重创建特征
    optimal_weights = {'MF': best_weights['MF'], 'ES': best_weights['ES'], 'Psoas': best_weights['Psoas']}
    df, weighted_core, mf_specific, weighted_comprehensive = get_feature_sets(df, optimal_weights)
    
    # 测试不同特征集
    print("\n" + "="*60)
    print("测试不同特征集（最优权重）")
    print("="*60)
    
    all_results = []
    
    # 测试加权核心特征
    weighted_core = [f for f in weighted_core if f in df.columns]
    result1 = test_with_cv(df[weighted_core], y, "加权核心特征")
    all_results.append(result1)
    
    # 测试多裂肌专项特征
    mf_specific = [f for f in mf_specific if f in df.columns]
    result2 = test_with_cv(df[mf_specific], y, "多裂肌专项特征")
    all_results.append(result2)
    
    # 测试加权综合特征
    weighted_comprehensive = [f for f in weighted_comprehensive if f in df.columns]
    result3 = test_with_cv(df[weighted_comprehensive], y, "加权综合特征")
    all_results.append(result3)
    
    # 优化正则化参数
    print("\n" + "="*60)
    print("优化正则化参数")
    print("="*60)
    
    best_features = df[weighted_core]
    best_params, best_score = optimize_regularization(best_features, y)
    
    # 最终模型评估
    print("\n" + "="*60)
    print("最终模型评估")
    print("="*60)
    
    best_features = best_features.fillna(0)  # 处理缺失值
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(best_features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    final_model = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    )
    
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    
    final_auc = roc_auc_score(y_test, y_prob)
    final_f1 = f1_score(y_test, y_pred)
    final_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n最终模型性能:")
    print(f"  AUC: {final_auc:.4f}")
    print(f"  F1: {final_f1:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    
    # 特征重要性（系数绝对值）
    feature_importance = pd.DataFrame({
        'feature': weighted_core,
        'coefficient': final_model.coef_[0],
        'abs_coefficient': abs(final_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\n特征重要性（Logistic回归系数）:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    # 保存结果
    output_dir = Path(__file__).resolve().parent / 'classification_results'
    
    weight_results.to_csv(output_dir / 'weight_optimization_results.csv', index=False)
    
    results_summary = pd.DataFrame(all_results)
    results_summary.to_csv(output_dir / 'feature_set_comparison.csv', index=False)
    
    feature_importance.to_csv(output_dir / 'feature_importance_final.csv', index=False)
    
    # 保存最优配置
    optimal_config = {
        'optimal_weights': optimal_weights,
        'optimal_params': best_params,
        'optimal_features': weighted_core,
        'cv_auc': best_score,
        'test_auc': final_auc
    }
    
    pd.DataFrame([optimal_config]).to_csv(output_dir / 'optimal_configuration.csv', index=False)
    
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    
    summary = pd.DataFrame(all_results)
    print(summary[['name', 'n_features', 'LR_auc', 'RF_auc', 'XGB_auc', 'best_auc']].to_string(index=False))
    
    print(f"\n最优配置:")
    print(f"  权重: MF={optimal_weights['MF']:.2f}, ES={optimal_weights['ES']:.2f}, Psoas={optimal_weights['Psoas']:.2f}")
    print(f"  正则化: C={best_params['C']}, penalty={best_params['penalty']}")
    print(f"  CV-AUC: {best_score:.4f}")
    print(f"  Test-AUC: {final_auc:.4f}")
    
    print("\n结果已保存到 classification_results 目录")


if __name__ == '__main__':
    main()
