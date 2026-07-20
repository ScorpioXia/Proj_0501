"""
脊柱稳定性分类模型 - 方案三
可解释性提升机 (EBM, Explainable Boosting Machine)

数据来源：
- 特征文件：patient_level_features_cleaned.csv（清洗标准化后的特征）
- 标签文件：patient_stable_311.xlsx

方案特点：
- "玻璃盒"模型，精度接近 XGBoost，但完全透明
- 每个特征的独立贡献可视化（Shape Functions）
- 支持全局和局部解释
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

# 尝试导入 interpret 库（用于 EBM），如果未安装则降级使用 GradientBoosting + SHAP
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret import show
    HAS_INTERPRET = True
except ImportError:
    HAS_INTERPRET = False
    print("警告: interpret 库未安装，将使用 GradientBoostingClassifier + SHAP 作为替代方案")
    print("安装命令: pip install interpret")


class SpinalStabilityEBM:
    def __init__(self, data_dir=None, label_file=None):
        self.data_dir = data_dir or str(Path(__file__).resolve().parent)
        self.label_file = label_file or str(Path(self.data_dir) / 'patient_stable_311.xlsx')
        self.model = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.use_ebm = HAS_INTERPRET

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

        print(f"标签分布:\n{df_merged['label'].value_counts()}")

        X = df_merged.drop(columns=['patient_id', 'label'])
        y = df_merged['label']

        self.feature_names = X.columns.tolist()
        print(f"\n特征数量: {X.shape[1]}")

        return X, y

    def train_ebm(self, X_train, y_train):
        """训练 EBM 模型（或降级使用 GradientBoosting）"""
        print("\n" + "=" * 60)
        
        if HAS_INTERPRET:
            print("训练 Explainable Boosting Machine (EBM)")
            print("=" * 60)
            model = ExplainableBoostingClassifier(
                n_estimators=200,
                max_bins=128,
                max_interaction_bins=32,
                learning_rate=0.1,
                interaction_rate='auto',
                random_state=42,
                n_jobs=-1,
                validation_size=0.15,
                early_stopping=True,
                early_stopping_rounds=50
            )
            model.fit(X_train, y_train)
            print("EBM 模型训练完成")
        else:
            print("训练 GradientBoostingClassifier（EBM 降级方案）")
            print("=" * 60)
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                validation_fraction=0.15,
                n_iter_no_change=50
            )
            model.fit(X_train, y_train)
            print("GradientBoostingClassifier 训练完成")
            print("\n提示: 安装 interpret 库可获得更好的可解释性")
            print("      pip install interpret")

        return model

    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        print("\n" + "=" * 60)
        print("EBM 模型评估")
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

    def cross_validate(self, X, y, cv_folds=5):
        """交叉验证"""
        print("\n" + "=" * 60)
        print("交叉验证")
        print("=" * 60)

        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='roc_auc')
        print(f"ROC-AUC 交叉验证分数: {cv_scores}")
        print(f"平均 ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return cv_scores

    def plot_roc_curve(self, y_test, y_prob, output_dir):
        """绘制 ROC 曲线"""
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Explainable Boosting Machine')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        output_path = Path(output_dir) / 'ebm_roc_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nROC曲线已保存到: {output_path}")

        return auc

    def global_explanation(self, output_dir):
        """全局解释：特征重要性和形状函数"""
        print("\n" + "=" * 60)
        
        if HAS_INTERPRET:
            print("EBM 全局解释")
            print("=" * 60)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            ebm_global = self.model.explain_global()

            plt.figure(figsize=(14, 10))
            ebm_global.plot()
            plt.title('EBM Global Feature Importance')
            plt.tight_layout()
            plt.savefig(output_path / 'ebm_global_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"EBM 全局特征重要性已保存到: {output_path / 'ebm_global_importance.png'}")

            print("\n保存所有特征的形状函数...")
            for i, feat_name in enumerate(self.feature_names[:20]):
                try:
                    plt.figure(figsize=(10, 6))
                    ebm_global.plot(i)
                    plt.title(f'Shape Function: {feat_name}')
                    plt.tight_layout()
                    safe_name = feat_name.replace('/', '_').replace('\\', '_')
                    plt.savefig(output_path / f'ebm_shape_{safe_name}.png',
                               dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"  跳过 {feat_name}: {str(e)[:50]}")

            print(f"形状函数图已保存到: {output_path / 'ebm_shape_*.png'}")
            return ebm_global
        else:
            print("GradientBoosting 全局解释（特征重要性）")
            print("=" * 60)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            importances = self.model.feature_importances_
            feat_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            plt.barh(range(min(20, len(feat_importance))), 
                     feat_importance['importance'][:20],
                     color='skyblue')
            plt.yticks(range(min(20, len(feat_importance))), 
                       feat_importance['feature'][:20])
            plt.xlabel('Feature Importance')
            plt.title('GradientBoosting Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"特征重要性图已保存到: {output_path / 'feature_importance.png'}")
            
            print("\n前20个重要特征:")
            print(feat_importance.head(20).to_string(index=False))
            return feat_importance

    def local_explanation(self, X_test, y_test, output_dir, n_samples=5):
        """局部解释：单个病人的预测解释"""
        print("\n" + "=" * 60)
        
        if HAS_INTERPRET:
            print("EBM 局部解释")
            print("=" * 60)
            output_path = Path(output_dir)

            ebm_local = self.model.explain_local(X_test[:n_samples], y_test[:n_samples])

            plt.figure(figsize=(14, 10))
            ebm_local.plot()
            plt.title('EBM Local Explanation (First 5 Patients)')
            plt.tight_layout()
            plt.savefig(output_path / 'ebm_local_explanation.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"EBM 局部解释已保存到: {output_path / 'ebm_local_explanation.png'}")

            for i in range(min(n_samples, len(X_test))):
                patient_local = self.model.explain_local(X_test.iloc[[i]], y_test.iloc[[i]])

                plt.figure(figsize=(14, 10))
                patient_local.plot()
                plt.title(f'Patient {i} Local Explanation')
                plt.tight_layout()
                plt.savefig(output_path / f'ebm_patient_{i}_local.png',
                           dpi=150, bbox_inches='tight')
                plt.close()
                print(f"病人 {i} 局部解释已保存到: {output_path / f'ebm_patient_{i}_local.png'}")

            return ebm_local
        else:
            print("GradientBoosting 局部解释（使用 SHAP）")
            print("=" * 60)
            output_path = Path(output_dir)
            
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test[:n_samples])
                
                plt.figure(figsize=(14, 10))
                shap.summary_plot(shap_values, X_test[:n_samples], 
                                  feature_names=self.feature_names, show=False)
                plt.title('SHAP Summary Plot')
                plt.tight_layout()
                plt.savefig(output_path / 'shap_summary.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"SHAP 汇总图已保存到: {output_path / 'shap_summary.png'}")
                
                # 单个病人的force plot
                for i in range(min(n_samples, len(X_test))):
                    plt.figure(figsize=(14, 6))
                    shap.force_plot(explainer.expected_value,
                                   shap_values[i],
                                   X_test.iloc[i],
                                   feature_names=self.feature_names,
                                   matplotlib=True,
                                   show=False)
                    plt.title(f'Patient {i} SHAP Force Plot')
                    plt.tight_layout()
                    plt.savefig(output_path / f'shap_patient_{i}_force.png',
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"病人 {i} SHAP 力图已保存到: {output_path / f'shap_patient_{i}_force.png'}")
                
                return shap_values
            except ImportError:
                print("SHAP 未安装，跳过局部解释")
                print("安装命令: pip install shap")
                return None

    def plot_feature_pairs(self, output_dir, top_n=10):
        """绘制特征交互对（EBM 自动学习的高阶交互）"""
        print("\n" + "=" * 60)
        
        if HAS_INTERPRET:
            print("EBM 特征交互分析")
            print("=" * 60)
            output_path = Path(output_dir)

            try:
                ebm_global = self.model.explain_global()

                if hasattr(ebm_global, 'data') and 'interaction' in str(type(ebm_global)):
                    print("检测到交互作用数据")

                    interactions = getattr(self.model, 'interactions_', [])

                    if len(interactions) > 0:
                        print(f"\n发现 {len(interactions)} 个特征交互:")
                        for i, (f1, f2) in enumerate(interactions[:10]):
                            print(f"  {i+1}. {self.feature_names[f1]} x {self.feature_names[f2]}")

                        for i, (f1, f2) in enumerate(interactions[:5]):
                            try:
                                plt.figure(figsize=(10, 8))
                                ebm_global.plot(i)
                                plt.title(f'Interaction: {self.feature_names[f1]} x {self.feature_names[f2]}')
                                plt.tight_layout()
                                plt.savefig(output_path / f'ebm_interaction_{i}.png',
                                           dpi=150, bbox_inches='tight')
                                plt.close()
                                print(f"交互图 {i+1} 已保存")
                            except Exception as e:
                                print(f"  跳过交互 {i+1}: {str(e)[:50]}")
            except Exception as e:
                print(f"交互分析跳过: {str(e)[:100]}")
        else:
            print("GradientBoosting 特征交互分析（跳过，使用 EBM 可获得更好的交互分析）")
            print("=" * 60)
            print("提示: 安装 interpret 库可获得自动特征交互检测")
            print("      pip install interpret")

    def compare_with_other_models(self, X_train, X_test, y_train, y_test, output_dir):
        """与其他模型对比"""
        print("\n" + "=" * 60)
        print("与其他模型对比")
        print("=" * 60)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_name = 'EBM' if HAS_INTERPRET else 'GradientBoosting'
        models = {
            model_name: (self.model, X_test),
            'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_test),
            'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), X_test_scaled)
        }

        results = {}
        plt.figure(figsize=(10, 8))

        for name, (model, X_test_use) in models.items():
            if name != model_name:
                model.fit(X_train_scaled if name == 'Logistic Regression' else X_train, y_train)

            y_prob = model.predict_proba(X_test_use)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            results[name] = auc

            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
            print(f"{name}: ROC-AUC = {auc:.4f}")

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Model Comparison: {model_name} vs Other Models')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = Path(output_dir) / 'model_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n模型对比图已保存到: {output_path}")

        return results

    def run(self, cv_folds=5):
        """运行完整的分类流程"""
        output_dir = Path(self.data_dir) / 'classification_results_ebm'
        output_dir.mkdir(parents=True, exist_ok=True)

        X, y = self.load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"\n训练集: {X_train.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")

        self.model = self.train_ebm(X_train, y_train)

        metrics, y_prob = self.evaluate_model(self.model, X_test, y_test)

        self.cross_validate(X, y, cv_folds=cv_folds)

        auc = self.plot_roc_curve(y_test, y_prob, output_dir)

        self.global_explanation(output_dir)

        self.local_explanation(X_test, y_test, output_dir)

        self.plot_feature_pairs(output_dir)

        self.compare_with_other_models(X_train, X_test, y_train, y_test, output_dir)

        results = {
            'metrics': metrics,
            'auc': auc,
            'output_dir': str(output_dir)
        }

        print("\n" + "=" * 60)
        if HAS_INTERPRET:
            print("EBM 分类任务完成！")
            print("\n提示: EBM 支持交互式可视化，可以运行:")
            print("  show(self.model.explain_global())")
            print("  show(self.model.explain_local(X_test, y_test))")
        else:
            print("GradientBoosting 分类任务完成！")
            print("\n提示: 安装 interpret 库可获得更好的可解释性")
            print("      pip install interpret")
        print(f"结果保存在: {output_dir}")
        print(f"ROC-AUC: {auc:.4f}")
        print("=" * 60)

        return results


if __name__ == '__main__':
    classifier = SpinalStabilityEBM()
    results = classifier.run()
