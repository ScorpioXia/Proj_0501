"""
验证与特征探索脚本
完成以下任务：
1. 5折/10折 Stratified K-Fold 交叉验证
2. VIF（方差膨胀因子）共线性检查
3. 特征子集对比实验：基础特征 vs 基础+Level3特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureExplorationValidator:
    def __init__(self, data_dir=None, label_file=None):
        self.data_dir = data_dir or str(Path(__file__).resolve().parent)
        self.label_file = label_file or str(Path(self.data_dir) / 'patient_stable_311.xlsx')
        self.df_features = None
        self.df_labels = None
        self.X = None
        self.y = None

    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("加载数据")
        print("=" * 60)

        feature_file = Path(self.data_dir) / 'patient_level_features_cleaned.csv'
        if not feature_file.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_file}")

        self.df_features = pd.read_csv(feature_file)
        print(f"特征数据: {self.df_features.shape[0]} 行, {self.df_features.shape[1]} 列")

        self.df_labels = pd.read_excel(self.label_file)
        self.df_labels.columns = ['patient_id', 'label']
        self.df_labels['patient_id'] = self.df_labels['patient_id'].astype(str)
        self.df_features['patient_id'] = self.df_features['patient_id'].astype(str)

        df_merged = pd.merge(self.df_features, self.df_labels, on='patient_id', how='inner')
        print(f"合并后: {df_merged.shape[0]} 行")

        print(f"\n标签分布:\n{df_merged['label'].value_counts()}")

        self.X = df_merged.drop(columns=['patient_id', 'label'])
        self.y = df_merged['label']

        print(f"\n总特征数量: {self.X.shape[1]}")

        return self.X, self.y

    def identify_feature_groups(self):
        """识别特征所属的层级/类别"""
        all_features = self.X.columns.tolist()

        level3_cross_features = [f for f in all_features if '_cross_' in f.lower()]
        level3_multi_features = [f for f in all_features if f.lower().startswith(('ratio_', 'asymmetry_', 'balance_', 'total_', 'combined_'))]

        basic_3d_features = [f for f in all_features
                           if any(x in f.lower() for x in ['_3d_', 'volume', 'csa', 'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity'])
                           and f not in level3_cross_features
                           and f not in level3_multi_features]

        basic_2d_agg_features = [f for f in all_features if f.startswith('2d_agg_')]

        feature_groups = {
            '基础3D特征': basic_3d_features,
            '基础2D聚合特征': basic_2d_agg_features,
            'Level3_跨层梯度特征': level3_cross_features,
            'Level3_多肌肉协同特征': level3_multi_features
        }

        print("\n特征层级分组:")
        for group, features in feature_groups.items():
            print(f"  {group}: {len(features)} 个")

        return feature_groups

    def calculate_vif(self, X, max_vif=10.0):
        """计算方差膨胀因子，检查多重共线性"""
        print("\n" + "=" * 60)
        print("VIF 共线性检查")
        print("=" * 60)

        X_scaled = StandardScaler().fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        vif_data = []
        for i, col in enumerate(X.columns):
            try:
                vif_value = variance_inflation_factor(X_scaled.values, i)
                vif_data.append({'Feature': col, 'VIF': vif_value})
            except:
                vif_data.append({'Feature': col, 'VIF': np.nan})

        vif_df = pd.DataFrame(vif_data)
        vif_df = vif_df.sort_values('VIF', ascending=False)

        high_vif_features = vif_df[vif_df['VIF'] > max_vif]

        print(f"\nVIF > {max_vif} 的特征 ({len(high_vif_features)} 个):")
        if len(high_vif_features) > 0:
            print(vif_df.head(20).to_string(index=False))
        else:
            print("  无高共线性特征")

        severe_vif = vif_df[vif_df['VIF'] > 100]
        if len(severe_vif) > 0:
            print(f"\n严重共线性 (VIF > 100):")
            for _, row in severe_vif.iterrows():
                print(f"  {row['Feature']}: {row['VIF']:.2f}")

        return vif_df, high_vif_features['Feature'].tolist()

    def stratified_cv_evaluation(self, X, y, model_name='LogisticRegression', cv=5):
        """Stratified K-Fold 交叉验证"""
        print("\n" + "=" * 60)
        print(f"{cv}折 Stratified 交叉验证 ({model_name})")
        print("=" * 60)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        cv_results = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'ROC-AUC': []
        }

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            cv_results['Accuracy'].append(accuracy_score(y_val, y_pred))
            cv_results['Precision'].append(precision_score(y_val, y_pred, zero_division=0))
            cv_results['Recall'].append(recall_score(y_val, y_pred, zero_division=0))
            cv_results['F1'].append(f1_score(y_val, y_pred, zero_division=0))
            cv_results['ROC-AUC'].append(roc_auc_score(y_val, y_prob))

        cv_df = pd.DataFrame(cv_results)
        cv_df.loc['Mean'] = cv_df.mean()
        cv_df.loc['Std'] = cv_df.std()

        print(f"\n{cv}折交叉验证结果:")
        print(cv_df.round(4).to_string())

        return cv_df

    def feature_subset_experiment(self, feature_groups):
        """特征子集对比实验"""
        print("\n" + "=" * 60)
        print("特征子集对比实验")
        print("=" * 60)

        X = self.X
        y = self.y

        experiment_results = {}

        baseline_features = feature_groups.get('基础3D特征', []) + feature_groups.get('基础2D聚合特征', [])
        level3_features = feature_groups.get('Level3_跨层梯度特征', []) + feature_groups.get('Level3_多肌肉协同特征', [])

        if len(baseline_features) == 0:
            baseline_features = [c for c in X.columns if not any(x in c.lower() for x in ['_cross_', 'ratio_', 'asymmetry_', 'balance_', 'total_', 'combined_'])]

        subsets = {
            '基础特征': baseline_features,
            'Level3特征': level3_features,
            '基础+Level3': baseline_features + level3_features,
            '全部特征': X.columns.tolist()
        }

        for subset_name, features in subsets.items():
            if len(features) == 0:
                print(f"\n{subset_name}: 无特征，跳过")
                continue

            valid_features = [f for f in features if f in X.columns]
            if len(valid_features) == 0:
                print(f"\n{subset_name}: 无有效特征，跳过")
                continue

            print(f"\n--- {subset_name} ({len(valid_features)} 特征) ---")

            X_subset = X[valid_features]

            cv_df = self.stratified_cv_evaluation(X_subset, y, model_name='LogisticRegression', cv=5)

            experiment_results[subset_name] = {
                'n_features': len(valid_features),
                'mean_roc_auc': cv_df.loc['Mean', 'ROC-AUC'],
                'mean_f1': cv_df.loc['Mean', 'F1'],
                'cv_results': cv_df
            }

        print("\n" + "=" * 60)
        print("特征子集对比汇总")
        print("=" * 60)

        summary_data = []
        for name, results in experiment_results.items():
            summary_data.append({
                '特征集': name,
                '特征数': results['n_features'],
                'Mean ROC-AUC': results['mean_roc_auc'],
                'Mean F1': results['mean_f1']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean ROC-AUC', ascending=False)
        print(summary_df.to_string(index=False))

        return experiment_results, summary_df

    def plot_experiment_results(self, experiment_results, output_dir):
        """可视化对比实验结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_data = []
        for name, results in experiment_results.items():
            summary_data.append({
                'Feature Set': name,
                'ROC-AUC': results['mean_roc_auc'],
                'F1': results['mean_f1']
            })

        summary_df = pd.DataFrame(summary_data)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(summary_df)))
        bars = ax1.bar(summary_df['Feature Set'], summary_df['ROC-AUC'], color=colors)
        ax1.set_ylabel('Mean ROC-AUC')
        ax1.set_title('ROC-AUC by Feature Set')
        ax1.set_ylim(0, 1)
        for bar, val in zip(bars, summary_df['ROC-AUC']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax1.tick_params(axis='x', rotation=15)

        ax2 = axes[1]
        bars = ax2.bar(summary_df['Feature Set'], summary_df['F1'], color=colors)
        ax2.set_ylabel('Mean F1-Score')
        ax2.set_title('F1-Score by Feature Set')
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars, summary_df['F1']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax2.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plt.savefig(output_path / 'feature_subset_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n特征子集对比图已保存到: {output_path / 'feature_subset_comparison.png'}")

        baseline_auc = experiment_results.get('基础特征', {}).get('mean_roc_auc', 0)
        combined_auc = experiment_results.get('基础+Level3', {}).get('mean_roc_auc', 0)
        improvement = ((combined_auc - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0

        print(f"\n关键发现:")
        print(f"  基础特征 ROC-AUC: {baseline_auc:.4f}")
        print(f"  基础+Level3 ROC-AUC: {combined_auc:.4f}")
        print(f"  提升: {improvement:+.2f}%")

        if improvement > 5:
            print(f"  结论: Level3 特征显著提升了模型性能!")
        elif improvement > 0:
            print(f"  结论: Level3 特征对模型有一定提升")
        else:
            print(f"  结论: Level3 特征对模型无显著提升")

        return summary_df

    def run(self, cv_folds=5):
        """运行完整的验证与特征探索流程"""
        output_dir = Path(self.data_dir) / 'validation_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        self.load_data()

        feature_groups = self.identify_feature_groups()

        vif_df, high_vif_features = self.calculate_vif(self.X)

        print("\n" + "=" * 60)
        print(f"{cv_folds}折交叉验证（全部特征）")
        print("=" * 60)
        self.stratified_cv_evaluation(self.X, self.y, model_name='LogisticRegression', cv=cv_folds)

        experiment_results, summary_df = self.feature_subset_experiment(feature_groups)

        summary_df = self.plot_experiment_results(experiment_results, output_dir)

        vif_df.to_csv(output_dir / 'vif_results.csv', index=False)
        print(f"\nVIF结果已保存到: {output_dir / 'vif_results.csv'}")

        summary_df.to_csv(output_dir / 'feature_subset_results.csv', index=False)
        print(f"特征子集对比结果已保存到: {output_dir / 'feature_subset_results.csv'}")

        print("\n" + "=" * 60)
        print("验证与特征探索完成！")
        print(f"结果保存在: {output_dir}")
        print("=" * 60)

        return {
            'vif_df': vif_df,
            'high_vif_features': high_vif_features,
            'experiment_results': experiment_results,
            'summary_df': summary_df,
            'output_dir': str(output_dir)
        }


if __name__ == '__main__':
    validator = FeatureExplorationValidator()
    results = validator.run()
