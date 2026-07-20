"""
简化版脊柱稳定性分类模型
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


def main():
    data_dir = str(Path(__file__).resolve().parent)
    
    df_features = pd.read_csv(data_dir + '/patient_level_features_cleaned.csv')
    df_labels = pd.read_excel(data_dir + '/patient_stable_311.xlsx')
    df_labels.columns = ['patient_id', 'label']
    
    df_labels['patient_id'] = df_labels['patient_id'].astype(str)
    df_features['patient_id'] = df_features['patient_id'].astype(str)
    
    df_merged = pd.merge(df_features, df_labels, on='patient_id', how='inner')
    X = df_merged.drop(columns=['patient_id', 'label'])
    y = df_merged['label']
    
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"标签分布:\n{y.value_counts()}\n")
    
    results = []
    
    for n_features in [10, 20, 30]:
        print(f"特征数: {n_features}")
        
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'LR': LogisticRegression(C=1.0, class_weight='balanced', random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
            'GB': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
            'XGB': xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=(y==0).sum()/(y==1).sum(), random_state=42),
            'LGB': lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, is_unbalance=True, random_state=42),
        }
        
        for name, model in models.items():
            if name == 'LR':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                cv_score = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc', n_jobs=-1).mean()
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                cv_score = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc', n_jobs=-1).mean()
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            print(f"  {name}: AUC={auc:.4f}, CV-AUC={cv_score:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
            results.append({'n': n_features, 'model': name, 'auc': auc, 'cv_auc': cv_score, 'f1': f1})
        
        print()
    
    print("="*60)
    print("最佳结果:")
    results_df = pd.DataFrame(results)
    best = results_df.sort_values('cv_auc', ascending=False).iloc[0]
    print(f"模型: {best['model']}, 特征数: {best['n']}")
    print(f"AUC: {best['auc']:.4f}, CV-AUC: {best['cv_auc']:.4f}, F1: {best['f1']:.4f}")
    
    results_df.to_csv(data_dir + '/classification_results/model_results_simple.csv', index=False)
    print(f"\n结果已保存到 model_results_simple.csv")


if __name__ == '__main__':
    main()
