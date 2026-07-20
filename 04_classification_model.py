"""
脊柱稳定性分类模型
使用 XGBoost/LightGBM + SHAP 进行二分类训练和可解释性分析
优化版本：添加特征选择、超参数调优、类别不平衡处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap
import statsmodels.api as sm


class SpinalStabilityClassifier:
    def __init__(self, data_dir=None, label_file=None):
        self.data_dir = data_dir or str(Path(__file__).resolve().parent)
        self.label_file = label_file or str(Path(self.data_dir) / 'patient_stable_311.xlsx')
        self.model = None
        self.model_type = None
        self.expected_labels = None
        self.shap_explainer = None
        self.X_test = None
        self.y_test = None
        self.selected_features = None

    def load_data(self):
        """加载特征数据和标签数据"""
        print("=" * 60)
        print("加载数据")
        print("=" * 60)

        feature_file = Path(self.data_dir) / 'patient_level_features_cleaned.csv'
        if not feature_file.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_file}")

        df_features = pd.read_csv(feature_file)
        print(f"特征数据: {df_features.shape[0]} 行, {df_features.shape[1]} 列")

        df_labels = pd.read_excel(self.label_file)
        print(f"标签数据: {df_labels.shape[0]} 行, {df_labels.shape[1]} 列")
        print(f"标签列: {df_labels.columns.tolist()}")

        df_labels.columns = ['patient_id', 'label']
        df_labels['patient_id'] = df_labels['patient_id'].astype(str)
        df_features['patient_id'] = df_features['patient_id'].astype(str)

        df_merged = pd.merge(df_features, df_labels, on='patient_id', how='inner')
        print(f"\n合并后: {df_merged.shape[0]} 行")

        self.expected_labels = df_merged['label'].unique()
        print(f"标签类别: {self.expected_labels}")
        print(f"标签分布:\n{df_merged['label'].value_counts()}")

        X = df_merged.drop(columns=['patient_id', 'label'])
        y = df_merged['label']

        print(f"\n特征数量: {X.shape[1]}")
        return X, y

    def calculate_vif(self, X):
        """计算VIF值"""
        vif_values = []
        for i in range(X.shape[1]):
            try:
                r_squared = sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1)).fit().rsquared
                vif = 1 / (1 - r_squared) if r_squared < 1 else 1000
            except:
                vif = 1000
            vif_values.append(vif)
        vif_data = pd.DataFrame({
            'feature': X.columns,
            'VIF': vif_values
        })
        return vif_data

    def remove_collinear_features(self, X, vif_threshold=10):
        """去除高共线性特征"""
        print("\n" + "=" * 60)
        print(f"去除共线性特征 (VIF阈值: {vif_threshold})")
        print("=" * 60)

        X_copy = X.copy()
        vif_data = self.calculate_vif(X_copy)
        high_vif_features = vif_data[vif_data['VIF'] > vif_threshold]['feature'].tolist()
        print(f"原始特征数: {len(X_copy.columns)}")
        print(f"高VIF特征数 (> {vif_threshold}): {len(high_vif_features)}")

        iteration = 0
        max_iterations = 20
        while len(high_vif_features) > 0 and iteration < max_iterations:
            vif_data = self.calculate_vif(X_copy)
            vif_data_sorted = vif_data.sort_values('VIF', ascending=False)
            feature_to_remove = vif_data_sorted.iloc[0]['feature']
            X_copy = X_copy.drop(columns=[feature_to_remove])
            iteration += 1
            vif_data = self.calculate_vif(X_copy)
            high_vif_features = vif_data[vif_data['VIF'] > vif_threshold]['feature'].tolist()

        print(f"去除后特征数: {len(X_copy.columns)}")
        print(f"剩余特征:\n{X_copy.columns.tolist()}")

        return X_copy

    def feature_selection(self, X, y, method='tree_importance', n_features=30, remove_collinear=True):
        """特征选择"""
        print("\n" + "=" * 60)
        print(f"特征选择 - 方法: {method}, 目标特征数: {n_features}, 去除共线性: {remove_collinear}")
        print("=" * 60)

        if remove_collinear:
            X = self.remove_collinear_features(X, vif_threshold=10)

        if method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            selected_mask = selector.get_support()

        elif method == 'kbest':
            selector = SelectKBest(f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()

        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()

        elif method == 'tree_importance':
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:n_features]
            selected_mask = np.zeros(len(X.columns), dtype=bool)
            selected_mask[indices] = True
            X_selected = X.iloc[:, selected_mask]

        else:
            X_selected = X
            selected_mask = np.ones(len(X.columns), dtype=bool)

        self.selected_features = X.columns[selected_mask].tolist()
        print(f"选择后特征数量: {len(self.selected_features)}")
        print(f"选择的特征:\n{self.selected_features}")

        return pd.DataFrame(X_selected, columns=self.selected_features)

    def train_xgboost(self, X_train, y_train, X_val, y_val, scale_pos_weight=1.0):
        """训练 XGBoost 模型（带超参数优化）"""
        print("\n" + "=" * 60)
        print("训练 XGBoost 模型")
        print("=" * 60)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        print("XGBoost 训练完成")

        return model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """训练 LightGBM 模型（带超参数优化）"""
        print("\n" + "=" * 60)
        print("训练 LightGBM 模型")
        print("=" * 60)

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
            reg_alpha=0.01,
            reg_lambda=1.0,
            is_unbalance=True,
            objective='binary',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        print("LightGBM 训练完成")

        return model

    def hyperparameter_tuning(self, model, param_grid, X, y, cv=5):
        """超参数调优"""
        print("\n" + "=" * 60)
        print("超参数调优")
        print("=" * 60)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X, y)
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test, model_name):
        """评估模型性能"""
        print("\n" + "=" * 60)
        print(f"{model_name} 模型评估")
        print("=" * 60)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
        print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")

        return metrics, y_prob

    def cross_validate(self, model, X, y, cv=5):
        """交叉验证"""
        print("\n" + "=" * 60)
        print("交叉验证")
        print("=" * 60)

        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"ROC-AUC 交叉验证分数: {cv_scores}")
        print(f"平均 ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return cv_scores

    def plot_roc_curves(self, models_dict, X_test, y_test, output_dir):
        """绘制 ROC 曲线"""
        plt.figure(figsize=(10, 8))

        for name, (model, y_prob) in models_dict.items():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = Path(output_dir) / 'roc_curves.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nROC曲线已保存到: {output_path}")

    def plot_feature_importance(self, model, feature_names, output_dir, top_n=30):
        """绘制特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:top_n]

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(indices)), importance[indices[::-1]])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()

            output_path = Path(output_dir) / 'feature_importance.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"特征重要性图已保存到: {output_path}")

    def explain_with_shap(self, X_train, X_test, model, model_type, output_dir):
        """使用 SHAP 进行模型解释"""
        print("\n" + "=" * 60)
        print("SHAP 可解释性分析")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test)

        print("计算 SHAP 值完成")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist(),
                         show=False, max_display=30)
        plt.title('SHAP Summary Plot - Global Feature Importance')
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SHAP Summary Plot 已保存到: {output_path / 'shap_summary.png'}")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist(),
                         plot_type='bar', show=False, max_display=30)
        plt.title('SHAP Feature Importance (Mean |SHAP|)')
        plt.tight_layout()
        plt.savefig(output_path / 'shap_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SHAP Bar Plot 已保存到: {output_path / 'shap_bar.png'}")

        self.shap_explainer = explainer
        return shap_values

    def explain_single_patient(self, patient_idx, X_test, shap_values, model, output_dir):
        """单个病人的 SHAP 局部解释"""
        print("\n" + "=" * 60)
        print(f"病人 {patient_idx} 的局部解释")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        patient_shap = shap_values[patient_idx]
        patient_features = X_test.iloc[patient_idx]

        plt.figure(figsize=(12, 6))
        shap.force_plot(
            self.shap_explainer.expected_value,
            patient_shap,
            patient_features,
            feature_names=X_test.columns.tolist(),
            show=False,
            matplotlib=True
        )
        plt.tight_layout()
        plt.savefig(output_path / f'patient_{patient_idx}_force_plot.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"病人 {patient_idx} 的 Force Plot 已保存到: {output_path / f'patient_{patient_idx}_force_plot.png'}")

        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(values=patient_shap,
                           base_values=self.shap_explainer.expected_value,
                           data=patient_features.values,
                           feature_names=X_test.columns.tolist()),
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path / f'patient_{patient_idx}_waterfall.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"病人 {patient_idx} 的 Waterfall Plot 已保存到: {output_path / f'patient_{patient_idx}_waterfall.png'}")

    def run(self, test_size=0.2, random_state=42, cv_folds=5, n_features=50):
        """运行完整的分类流程"""
        output_dir = Path(self.data_dir) / 'classification_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        X, y = self.load_data()

        X = self.feature_selection(X, y, method='tree_importance', n_features=n_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )
        print(f"\n训练集: {X_train.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        print(f"特征数: {X_train.shape[1]}")

        self.X_test = X_test
        self.y_test = y_test

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
        print(f"\n类别权重 (scale_pos_weight): {scale_pos_weight:.2f}")

        xgb_model = self.train_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight)
        lgb_model = self.train_lightgbm(X_train, y_train, X_test, y_test)

        xgb_metrics, xgb_prob = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        lgb_metrics, lgb_prob = self.evaluate_model(lgb_model, X_test, y_test, "LightGBM")

        print("\n" + "=" * 60)
        print("XGBoost 交叉验证")
        print("=" * 60)
        xgb_cv_scores = self.cross_validate(xgb_model, X, y, cv=cv_folds)

        print("\n" + "=" * 60)
        print("LightGBM 交叉验证")
        print("=" * 60)
        lgb_cv_scores = self.cross_validate(lgb_model, X, y, cv=cv_folds)

        best_model_name = "XGBoost" if xgb_metrics['ROC-AUC'] > lgb_metrics['ROC-AUC'] else "LightGBM"
        best_model = xgb_model if best_model_name == "XGBoost" else lgb_model
        self.model = best_model
        self.model_type = 'xgboost' if best_model_name == "XGBoost" else 'lightgbm'

        print(f"\n最佳模型: {best_model_name}")

        self.plot_roc_curves(
            {"XGBoost": (xgb_model, xgb_prob), "LightGBM": (lgb_model, lgb_prob)},
            X_test, y_test, output_dir
        )

        self.plot_feature_importance(best_model, X.columns.tolist(), output_dir)

        shap_values = self.explain_with_shap(X_train, X_test, best_model,
                                            self.model_type, output_dir)

        print("\n" + "=" * 60)
        print("生成单个病人解释示例（选取测试集前3个病人）")
        print("=" * 60)
        for i in range(min(3, len(X_test))):
            self.explain_single_patient(i, X_test, shap_values, best_model, output_dir)

        results_df = pd.DataFrame({
            'Model': ['XGBoost', 'LightGBM'],
            'Accuracy': [xgb_metrics['Accuracy'], lgb_metrics['Accuracy']],
            'Precision': [xgb_metrics['Precision'], lgb_metrics['Precision']],
            'Recall': [xgb_metrics['Recall'], lgb_metrics['Recall']],
            'F1-Score': [xgb_metrics['F1-Score'], lgb_metrics['F1-Score']],
            'ROC-AUC': [xgb_metrics['ROC-AUC'], lgb_metrics['ROC-AUC']],
            'CV-AUC': [xgb_cv_scores.mean(), lgb_cv_scores.mean()]
        })
        results_df.to_csv(output_dir / 'model_results.csv', index=False)
        print(f"\n模型结果已保存到: {output_dir / 'model_results.csv'}")

        results = {
            'xgb_metrics': xgb_metrics,
            'lgb_metrics': lgb_metrics,
            'best_model': best_model_name,
            'output_dir': str(output_dir)
        }

        print("\n" + "=" * 60)
        print("分类任务完成！")
        print(f"结果保存在: {output_dir}")
        print("=" * 60)

        return results


if __name__ == '__main__':
    classifier = SpinalStabilityClassifier()
    results = classifier.run(n_features=50)
