"""
脊柱稳定性分类模型 - 方案二
LASSO 特征筛选 + 逻辑回归 + 列线图 (Nomogram)

数据来源：
- 特征文件：patient_level_features_cleaned.csv（清洗标准化后的特征）
- 标签文件：patient_stable_311.xlsx

方案特点：
- 白盒模型，临床可解释性强
- LASSO 正则化筛选核心特征
- Nomogram 将模型可视化为临床可用的打分系统
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.linear_model import LassoCV
except ImportError:
    from sklearn.feature_selection import LassoCV
import statsmodels.api as sm

# 尝试导入regtabulator库（用于创建列线图），如果未安装则降级使用手动实现
try:
    import regtabulator
    HAS_REGTABULATOR = True
except ImportError:
    HAS_REGTABULATOR = False


class SpinalStabilityLassoLR:
    def __init__(self, data_dir=None, label_file=None):
        self.data_dir = data_dir or str(Path(__file__).resolve().parent)
        self.label_file = label_file or str(Path(self.data_dir) / 'patient_stable_311.xlsx')
        self.model = None
        self.selected_features = None
        self.scaler = StandardScaler()
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None

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

        print(f"\n特征数量: {X.shape[1]}")
        
        # 记录少数类样本数，用于后续特征数量限制
        self.minority_count = y.value_counts().min()
        print(f"少数类样本数: {self.minority_count}")
        
        return X, y

    def calculate_vif(self, X):
        """计算方差膨胀因子（VIF），检测多重共线性"""
        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [1 / (1 - sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1)).fit().rsquared)
                          for i in range(X.shape[1])]
        return vif_data

    def lasso_feature_selection(self, X, y, cv_folds=5, n_alphas=100):
        """使用 LASSO 进行特征筛选"""
        print("\n" + "=" * 60)
        print("LASSO 特征筛选")
        print("=" * 60)

        X_scaled = self.scaler.fit_transform(X)

        alphas = np.logspace(-4, 0, n_alphas)
        lasso_cv = LassoCV(alphas=alphas, cv=cv_folds, random_state=42, n_jobs=-1, max_iter=10000)
        lasso_cv.fit(X_scaled, y)

        print(f"最优 alpha: {lasso_cv.alpha_:.6f}")

        coef = pd.Series(lasso_cv.coef_, index=X.columns)
        selected = coef[coef != 0].sort_values(key=abs, ascending=False)

        self.selected_features = selected.index.tolist()

        print(f"\nLASSO筛选后特征数量: {len(self.selected_features)}/{X.shape[1]}")
        
        # 限制特征数量：最多为少数类样本数的1/5（经验法则）
        max_features = max(2, self.minority_count // 5)
        if len(self.selected_features) > max_features:
            print(f"特征过多，限制为 {max_features} 个（少数类样本数的1/5）")
            self.selected_features = self.selected_features[:max_features]
            selected = selected.head(max_features)
        
        print(f"\n筛选后特征数量: {len(self.selected_features)}")
        print("\n入选特征及其系数:")
        for feat, c in selected.items():
            print(f"  {feat}: {c:.4f}")

        return selected

    def train_logistic_regression(self, X, y, selected_features):
        """训练逻辑回归模型"""
        print("\n" + "=" * 60)
        print("逻辑回归模型训练")
        print("=" * 60)

        X_selected = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        lr_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)

        self.model = lr_model

        print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        print(f"测试集: {X_test.shape[0]} 样本")

        return lr_model, X_train, X_test, y_train, y_test

    def build_statsmodels_lr(self, X_train, y_train, selected_features):
        """使用 statsmodels 构建详细的逻辑回归模型（用于获取 p 值等统计量）"""
        print("\n" + "=" * 60)
        print("Statsmodels 逻辑回归详情")
        print("=" * 60)
        
        print(f"用于statsmodels的特征数量: {len(selected_features)}")
        print(f"训练样本数: {len(y_train)}, 少数类样本数: {y_train.sum()}")

        if len(selected_features) == 0:
            print("没有特征可用于statsmodels拟合")
            return None

        X_train_sm = sm.add_constant(X_train)
        lr_sm = sm.Logit(y_train, X_train_sm)
        
        # 尝试多种拟合方法，优先使用正则化
        methods = [
            ('正则化(L2)', lambda: lr_sm.fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)),
            ('正则化(L1+L2)', lambda: lr_sm.fit_regularized(alpha=0.01, L1_wt=0.5, disp=0)),
            ('bfgs', lambda: lr_sm.fit(method='bfgs', maxiter=5000, disp=0)),
            ('nm', lambda: lr_sm.fit(method='nm', maxiter=5000, disp=0)),
            ('powell', lambda: lr_sm.fit(method='powell', maxiter=5000, disp=0)),
        ]
        
        result = None
        last_error = None
        
        for method_name, fit_func in methods:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    warnings.filterwarnings('ignore', category=HessianInversionWarning)
                    result = fit_func()
                print(f"使用{method_name}求解器成功")
                break
            except Exception as e:
                last_error = e
                print(f"{method_name}求解器失败: {type(e).__name__}")
        
        if result is None:
            print(f"\n所有statsmodels求解器均失败: {last_error}")
            print("跳过statsmodels详情输出，将使用sklearn的模型结果")
            return None

        print(result.summary())

        return result

    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        print("\n" + "=" * 60)
        print("模型评估")
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

        X_scaled = self.scaler.fit_transform(X)
        lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=1000, random_state=42, class_weight='balanced')

        cv_scores = cross_val_score(lr, X_scaled, y, cv=cv_folds, scoring='roc_auc')
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
        plt.title('ROC Curve - Logistic Regression')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        output_path = Path(output_dir) / 'lr_roc_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nROC曲线已保存到: {output_path}")

        return auc

    def plot_lasso_coef(self, coef_series, output_dir):
        """绘制 LASSO 特征系数"""
        plt.figure(figsize=(10, 8))
        coef_sorted = coef_series.sort_values(key=abs, ascending=True)
        colors = ['red' if c < 0 else 'blue' for c in coef_sorted]
        plt.barh(range(len(coef_sorted)), coef_sorted.values, color=colors)
        plt.yticks(range(len(coef_sorted)), coef_sorted.index)
        plt.xlabel('LASSO Coefficient')
        plt.title('LASSO Feature Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        output_path = Path(output_dir) / 'lasso_coefficients.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"LASSO系数图已保存到: {output_path}")

    def plot_calibration_curve(self, y_test, y_prob, output_dir, n_bins=10):
        """绘制校准曲线"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        observed = np.zeros(n_bins)
        count = np.zeros(n_bins)

        for i, (obs, idx) in enumerate(zip(y_test.values, bin_indices)):
            observed[idx] += obs
            count[idx] += 1

        observed_prob = np.divide(observed, count, where=count > 0, out=np.zeros(n_bins))

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_centers, observed_prob, 'bo-', label='Logistic Regression')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Observed Positive Rate')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = Path(output_dir) / 'calibration_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"校准曲线已保存到: {output_path}")

    def create_nomogram(self, lr_model, X_train, y_train, selected_features, output_dir):
        """创建列线图（Nomogram）"""
        print("\n" + "=" * 60)
        print("创建列线图 (Nomogram)")
        print("=" * 60)

        if HAS_REGTABULATOR:
            self._create_nomogram_regtabulator(lr_model, selected_features, output_dir)
        else:
            self._create_nomogram_manual(lr_model, X_train, y_train, selected_features, output_dir)

    def _create_nomogram_manual(self, lr_model, X_train, y_train, selected_features, output_dir):
        """手动创建简化的列线图"""
        coef = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]

        coef_df = pd.DataFrame({
            'feature': selected_features,
            'coefficient': coef
        })
        coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coef', ascending=False)

        max_score = 100
        max_coef = coef_df['abs_coef'].max()

        coef_df['points'] = (coef_df['abs_coef'] / max_coef) * max_score

        print("\n特征打分表:")
        print(coef_df[['feature', 'coefficient', 'points']].to_string(index=False))

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, len(selected_features) + 2)
        ax.axis('off')

        y_positions = np.arange(len(selected_features), 0, -1)

        for i, (feat, row) in enumerate(coef_df.iterrows()):
            feature_name = row['feature']
            points = row['points']
            coef_val = row['coefficient']

            ax.plot([0, points], [y_positions[i], y_positions[i]], 'b-', linewidth=2)
            ax.plot(points, y_positions[i], 'ro', markersize=8)

            direction = "↑" if coef_val > 0 else "↓"
            ax.text(-5, y_positions[i], f"{feature_name}\n({direction})",
                   ha='right', va='center', fontsize=10)
            ax.text(points + 3, y_positions[i], f"{points:.1f}",
                   ha='left', va='center', fontsize=10)

        total_points_pos = len(selected_features) + 1.5
        ax.plot([0, max_score], [total_points_pos, total_points_pos], 'k-', linewidth=3)
        ax.text(max_score / 2, total_points_pos + 0.3, 'Total Points',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

        risk_positions = [0, 25, 50, 75, 100]
        risk_values = [0.1, 0.25, 0.5, 0.75, 0.9]

        ax.text(max_score + 10, y_positions[0], 'Risk',
               ha='left', va='center', fontsize=11, fontweight='bold')

        for rp, rv in zip(risk_positions, risk_values):
            ax.plot([rp, rp], [y_positions[-1] - 0.5, y_positions[-1] + 0.5], 'k-')
            ax.text(rp, y_positions[-1] - 1, f'{rv:.1f}',
                   ha='center', va='top', fontsize=9)

        ax.plot([0, 100], [y_positions[-1] - 1.5, y_positions[-1] - 1.5], 'k-', linewidth=2)

        ax.text(-30, total_points_pos, 'Feature → Score',
               ha='center', va='center', fontsize=11, rotation=90)

        plt.title('Spinal Stability Prediction Nomogram\n(LASSO + Logistic Regression)',
                 fontsize=14, fontweight='bold', pad=20)

        output_path = Path(output_dir) / 'nomogram.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"列线图已保存到: {output_path}")

        coef_df.to_csv(Path(output_dir) / 'nomogram_feature_scores.csv', index=False)
        print(f"特征打分表已保存到: {Path(output_dir) / 'nomogram_feature_scores.csv'}")

    def _create_nomogram_regtabulator(self, lr_model, selected_features, output_dir):
        """使用 regtabulator 库创建列线图"""
        coef = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]

        df = pd.DataFrame({
            'Variable': selected_features,
            'Coefficient': coef
        })

        nomogram = regtabulator.Nomogram(df, precision=2)
        nomogram.plot()
        plt.savefig(Path(output_dir) / 'nomogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"列线图已保存到: {Path(output_dir) / 'nomogram.png'}")

    def run(self, cv_folds=5, n_lasso_alphas=100):
        """运行完整的分类流程"""
        output_dir = Path(self.data_dir) / 'classification_results_lasso_lr'
        output_dir.mkdir(parents=True, exist_ok=True)

        X, y = self.load_data()

        selected = self.lasso_feature_selection(X, y, cv_folds=cv_folds, n_alphas=n_lasso_alphas)

        if len(self.selected_features) == 0:
            print("警告: LASSO 未筛选出任何特征，使用全部特征")
            self.selected_features = X.columns.tolist()

        lr_model, X_train, X_test, y_train, y_test = self.train_logistic_regression(
            X, y, self.selected_features
        )
        
        # VIF过滤：移除共线性强的特征
        if len(self.selected_features) > 1:
            vif_data = self.calculate_vif(X_train)
            print(f"\nVIF检查结果:")
            print(vif_data.to_string(index=False))
            
            # 迭代移除VIF > 10的特征
            high_vif_features = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
            if high_vif_features:
                print(f"\n移除高VIF特征（VIF > 10）: {high_vif_features}")
                self.selected_features = [f for f in self.selected_features 
                                         if f not in high_vif_features]
                
                # 如果移除后特征太少，保留前5个最重要的
                if len(self.selected_features) < 2:
                    self.selected_features = [f for f in selected.index.tolist() 
                                             if f not in high_vif_features][:5]
                
                # 重新训练逻辑回归（使用VIF过滤后的特征）
                print(f"\n使用VIF过滤后的特征重新训练模型: {len(self.selected_features)} 个")
                lr_model, X_train, X_test, y_train, y_test = self.train_logistic_regression(
                    X, y, self.selected_features
                )

        stats_result = self.build_statsmodels_lr(X_train, y_train, self.selected_features)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        metrics, y_prob = self.evaluate_model(lr_model, X_test_scaled, y_test)

        self.cross_validate(X[self.selected_features], y, cv_folds=cv_folds)

        auc = self.plot_roc_curve(y_test, y_prob, output_dir)

        self.plot_lasso_coef(selected, output_dir)

        self.plot_calibration_curve(y_test, y_prob, output_dir)

        self.create_nomogram(lr_model, X_train, y_train, self.selected_features, output_dir)

        results = {
            'metrics': metrics,
            'selected_features': self.selected_features,
            'coefficients': dict(zip(self.selected_features, lr_model.coef_[0])),
            'auc': auc,
            'output_dir': str(output_dir)
        }

        print("\n" + "=" * 60)
        print("分类任务完成！")
        print(f"结果保存在: {output_dir}")
        print(f"入选特征数量: {len(self.selected_features)}")
        print(f"ROC-AUC: {auc:.4f}")
        print("=" * 60)

        return results


if __name__ == '__main__':
    classifier = SpinalStabilityLassoLR()
    results = classifier.run()
