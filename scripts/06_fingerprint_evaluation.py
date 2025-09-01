#!/usr/bin/env python3
"""
脚本 06: 指纹区分能力全面评估
目标：彻底评估现有指纹特征对noise_gaussian和adversarial_pgd的区分能力
如果现有特征无法区分，则提供寻找新特征的指导
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import yaml
import sys
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FingerprintEvaluator:
    """指纹评估器类"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.evaluation_results = {}
        self.data_quality_report = {}
        
    def comprehensive_evaluation(self, data, label_encoder):
        """全面评估指纹特征"""
        print("=== 开始全面指纹评估 ===")
        
        # 1. 数据质量检查
        print("\n1. 数据质量检查...")
        self._check_data_quality(data, label_encoder)
        
        # 2. 数据清理
        print("\n2. 数据清理...")
        data_cleaned = self._clean_data(data)
        
        # 3. 特征区分度分析
        print("\n3. 特征区分度分析...")
        feature_discrimination = self._analyze_feature_discrimination(data_cleaned, label_encoder)
        
        # 4. 类别间差异深度分析
        print("\n4. 类别间差异深度分析...")
        class_differences = self._deep_class_difference_analysis(data_cleaned, label_encoder)
        
        # 5. 特征组合分析
        print("\n5. 特征组合分析...")
        feature_combinations = self._analyze_feature_combinations(data_cleaned, label_encoder)
        
        # 6. 机器学习模型验证
        print("\n6. 机器学习模型验证...")
        ml_validation = self._validate_with_ml_models(data_cleaned, label_encoder)
        
        # 7. 生成评估报告
        print("\n7. 生成评估报告...")
        self._generate_comprehensive_report(feature_discrimination, class_differences, 
                                         feature_combinations, ml_validation)
        
        return feature_discrimination, class_differences, feature_combinations, ml_validation
    
    def _clean_data(self, data):
        """清理数据中的无穷大值、NaN值和异常值"""
        print("清理数据...")
        
        data_cleaned = data.copy()
        
        # 获取数值列
        numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
        if 'vulnerability_type' in numeric_columns:
            numeric_columns = numeric_columns.drop('vulnerability_type')
        
        print(f"处理 {len(numeric_columns)} 个数值列...")
        
        for col in numeric_columns:
            # 1. 替换无穷大值
            data_cleaned[col] = data_cleaned[col].replace([np.inf, -np.inf], np.nan)
            
            # 2. 计算分位数来识别异常值
            Q1 = data_cleaned[col].quantile(0.01)  # 1%分位数
            Q3 = data_cleaned[col].quantile(0.99)  # 99%分位数
            IQR = Q3 - Q1
            
            # 将异常值替换为分位数边界值
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data_cleaned[col] = data_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 3. 用中位数填充NaN值
            if data_cleaned[col].isna().sum() > 0:
                median_val = data_cleaned[col].median()
                data_cleaned[col].fillna(median_val, inplace=True)
        
        # 最终检查
        remaining_nan = data_cleaned.isnull().sum().sum()
        remaining_inf = np.isinf(data_cleaned.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"清理完成: 剩余NaN值 {remaining_nan}, 剩余无穷大值 {remaining_inf}")
        
        return data_cleaned
        
    def _check_data_quality(self, data, label_encoder):
        """检查数据质量"""
        print("检查数据质量...")
        
        quality_report = {}
        
        # 1. 标签分布检查
        label_counts = data['vulnerability_type'].value_counts()
        quality_report['label_distribution'] = label_counts.to_dict()
        
        print(f"标签分布: {label_counts.to_dict()}")
        
        # 2. 重复样本检查
        duplicate_samples = data.duplicated().sum()
        quality_report['duplicate_samples'] = duplicate_samples
        
        print(f"重复样本数量: {duplicate_samples}")
        
        # 3. 缺失值检查
        missing_values = data.isnull().sum()
        quality_report['missing_values'] = missing_values.to_dict()
        
        print(f"缺失值统计: {missing_values.sum()} 个总缺失值")
        
        # 4. 特征值范围检查
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if 'vulnerability_type' in numeric_cols:
            numeric_cols = numeric_cols.drop('vulnerability_type')
        
        feature_ranges = {}
        for col in numeric_cols:
            col_data = data[col]
            feature_ranges[col] = {
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'has_inf': np.isinf(col_data).any(),
                'has_nan': col_data.isna().any()
            }
        
        quality_report['feature_ranges'] = feature_ranges
        
        # 5. 异常值检查
        outlier_report = {}
        for col in numeric_cols:
            col_data = data[col]
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            outlier_report[col] = {
                'outlier_count': outliers,
                'outlier_percentage': outliers / len(col_data) * 100
            }
        
        quality_report['outlier_report'] = outlier_report
        
        # 保存质量报告
        quality_df = pd.DataFrame(quality_report['feature_ranges']).T
        quality_df.to_csv(os.path.join(self.results_dir, 'data_quality_report.csv'))
        
        self.data_quality_report = quality_report
        
        print("✅ 数据质量检查完成")
        return quality_report
    
    def _analyze_feature_discrimination(self, data, label_encoder):
        """分析特征区分能力"""
        print("分析特征区分能力...")
        
        # 准备数据
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 最终检查数据是否干净
        if X.isnull().any().any() or np.isinf(X.values).any():
            print("警告：数据中仍包含NaN或无穷大值，进行最终清理...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        print(f"数据形状: {X.shape}")
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        
        # 1. F统计量分析（ANOVA）
        f_scores, p_values = f_classif(X, y)
        
        # 2. 互信息分析
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 3. 创建特征区分度DataFrame
        feature_discrimination = pd.DataFrame({
            'Feature_Name': X.columns,
            'F_Score': f_scores,
            'P_Value': p_values,
            'Mutual_Info': mi_scores,
            'Significant_F': p_values < 0.05,
            'F_Rank': f_scores.argsort()[::-1].argsort() + 1,
            'MI_Rank': mi_scores.argsort()[::-1].argsort() + 1
        }).sort_values('F_Score', ascending=False)
        
        # 4. 保存结果
        feature_discrimination.to_csv(os.path.join(self.results_dir, 'feature_discrimination_analysis.csv'), index=False)
        
        # 5. 显示结果
        print(f"\n�� 特征区分能力排行榜 (Top 20, 按F统计量):")
        for i, (_, row) in enumerate(feature_discrimination.head(20).iterrows()):
            significance = "***" if row['Significant_F'] else ""
            print(f"   {i+1:2d}. {row['Feature_Name']:<25} : F={row['F_Score']:.4f}, "
                  f"p={row['P_Value']:.6f}, MI={row['Mutual_Info']:.4f} {significance}")
        
        # 6. 创建可视化
        self._create_discrimination_visualization(feature_discrimination)
        
        return feature_discrimination
    
    def _create_discrimination_visualization(self, feature_discrimination):
        """创建区分能力可视化"""
        # 1. F统计量 vs 互信息散点图
        plt.figure(figsize=(12, 8))
        
        plt.scatter(feature_discrimination['F_Score'], feature_discrimination['Mutual_Info'], 
                   alpha=0.7, s=50)
        
        # 标注前10个特征
        top_10 = feature_discrimination.head(10)
        for _, row in top_10.iterrows():
            plt.annotate(row['Feature_Name'], 
                        (row['F_Score'], row['Mutual_Info']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.xlabel('F统计量', fontsize=12, fontweight='bold')
        plt.ylabel('互信息', fontsize=12, fontweight='bold')
        plt.title('特征区分能力对比: F统计量 vs 互信息', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_dir, 'feature_discrimination_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top 20特征F统计量条形图
        plt.figure(figsize=(16, 10))
        
        top_20 = feature_discrimination.head(20)
        bars = plt.barh(range(len(top_20)), top_20['F_Score'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_20))))
        
        plt.yticks(range(len(top_20)), top_20['Feature_Name'], fontsize=10)
        
        # 在条形图上添加数值标签
        for i, (bar, f_score) in enumerate(zip(bars, top_20['F_Score'])):
            plt.text(f_score + 0.001, i, f'{f_score:.3f}', 
                    va='center', fontsize=9, fontweight='bold')
        
        plt.xlabel('F统计量', fontsize=12, fontweight='bold')
        plt.title('Top 20特征区分能力 (F统计量)', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_dir, 'top_features_f_score.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _deep_class_difference_analysis(self, data, label_encoder):
        """深度分析类别间的特征差异"""
        print("深度分析类别间差异...")
        
        classes = label_encoder.classes_
        target_classes = ['noise_gaussian', 'adversarial_pgd']
        
        # 分析每个特征在两个类别上的分布
        feature_analysis = {}
        
        for feature in data.columns:
            if feature == 'vulnerability_type':
                continue
                
            if data[feature].dtype in ['float64', 'int64']:
                class_stats = {}
                
                for class_name in target_classes:
                    if class_name in classes:
                        class_data = data[data['vulnerability_type'] == class_name][feature]
                        if len(class_data) > 0:
                            class_stats[class_name] = {
                                'mean': class_data.mean(),
                                'std': class_data.std(),
                                'median': class_data.median(),
                                'q25': class_data.quantile(0.25),
                                'q75': class_data.quantile(0.75),
                                'min': class_data.min(),
                                'max': class_data.max(),
                                'skewness': class_data.skew(),
                                'kurtosis': class_data.kurtosis()
                            }
                
                # 计算类别间差异
                if len(class_stats) == 2:
                    # Cohen's d效应量
                    mean_diff = abs(class_stats[target_classes[0]]['mean'] - class_stats[target_classes[1]]['mean'])
                    pooled_std = np.sqrt((class_stats[target_classes[0]]['std']**2 + class_stats[target_classes[1]]['std']**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # 重叠度（Overlap）
                    overlap = self._calculate_overlap(class_stats[target_classes[0]], class_stats[target_classes[1]])
                    
                    # 进行t检验
                    class1_data = data[data['vulnerability_type'] == target_classes[0]][feature]
                    class2_data = data[data['vulnerability_type'] == target_classes[1]][feature]
                    
                    # 初始化统计变量
                    t_stat, p_value, mannwhitney_u, mw_p_value = 0, 1, 0, 1
                    
                    if len(class1_data) > 0 and len(class2_data) > 0:
                        try:
                            t_stat, p_value = stats.ttest_ind(class1_data, class2_data)
                            mannwhitney_u, mw_p_value = stats.mannwhitneyu(class1_data, class2_data, alternative='two-sided')
                        except Exception as e:
                            print(f"警告：特征 {feature} 的统计检验失败: {e}")
                            # 使用默认值
                            t_stat, p_value, mannwhitney_u, mw_p_value = 0, 1, 0, 1
                    
                    feature_analysis[feature] = {
                        'cohens_d': cohens_d,
                        'overlap': overlap,
                        't_statistic': t_stat,
                        't_p_value': p_value,
                        'mannwhitney_u': mannwhitney_u,
                        'mw_p_value': mw_p_value,
                        'class_stats': class_stats,
                        'effect_size': self._classify_effect_size(cohens_d)
                    }
        
        # 按效应量排序
        sorted_features = sorted(feature_analysis.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
        
        print(f"\n📊 特征区分度排序 (按Cohen's d效应量):")
        for i, (feature, stats) in enumerate(sorted_features[:20]):
            print(f"   {i+1:2d}. {feature:<25} : d={stats['cohens_d']:.4f} ({stats['effect_size']}), "
                  f"重叠度={stats['overlap']:.2%}, t检验p={stats['t_p_value']:.6f}")
        
        # 保存结果
        analysis_df = pd.DataFrame([
            {
                'Feature_Name': feature,
                'Cohens_d': stats['cohens_d'],
                'Effect_Size': stats['effect_size'],
                'Overlap': stats['overlap'],
                'T_Statistic': stats['t_statistic'],
                'T_P_Value': stats['t_p_value'],
                'MannWhitney_U': stats['mannwhitney_u'],
                'MW_P_Value': stats['mw_p_value']
            }
            for feature, stats in feature_analysis.items()
        ]).sort_values('Cohens_d', ascending=False)
        
        analysis_df.to_csv(os.path.join(self.results_dir, 'class_difference_analysis.csv'), index=False)
        
        # 创建可视化
        self._create_class_difference_visualization(analysis_df)
        
        return feature_analysis
    
    def _calculate_overlap(self, stats1, stats2):
        """计算两个分布的重叠度"""
        # 使用四分位数范围计算重叠度
        range1 = (stats1['q25'], stats1['q75'])
        range2 = (stats2['q25'], stats2['q75'])
        
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        total_length = (range1[1] - range1[0]) + (range2[1] - range2[0])
        
        return overlap_length / total_length if total_length > 0 else 0.0
    
    def _classify_effect_size(self, cohens_d):
        """分类效应量大小"""
        if cohens_d < 0.2:
            return "微小"
        elif cohens_d < 0.5:
            return "小"
        elif cohens_d < 0.8:
            return "中等"
        else:
            return "大"
    
    def _create_class_difference_visualization(self, analysis_df):
        """创建类别差异可视化"""
        # 1. Cohen's d效应量分布
        plt.figure(figsize=(15, 10))
        
        # 子图1：效应量分布直方图
        ax1 = plt.subplot(2, 2, 1)
        plt.hist(analysis_df['Cohens_d'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel("Cohen's d效应量", fontsize=12, fontweight='bold')
        plt.ylabel('特征数量', fontsize=12, fontweight='bold')
        plt.title("效应量分布", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 子图2：重叠度分布
        ax2 = plt.subplot(2, 2, 2)
        plt.hist(analysis_df['Overlap'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('重叠度', fontsize=12, fontweight='bold')
        plt.ylabel('特征数量', fontsize=12, fontweight='bold')
        plt.title('类别重叠度分布', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 子图3：效应量 vs 重叠度散点图
        ax3 = plt.subplot(2, 2, 3)
        plt.scatter(analysis_df['Cohens_d'], analysis_df['Overlap'], alpha=0.6, s=30)
        plt.xlabel("Cohen's d效应量", fontsize=12, fontweight='bold')
        plt.ylabel('重叠度', fontsize=12, fontweight='bold')
        plt.title('效应量 vs 重叠度', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 子图4：Top 15特征效应量
        ax4 = plt.subplot(2, 2, 4)
        top_15 = analysis_df.head(15)
        bars = plt.barh(range(len(top_15)), top_15['Cohens_d'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_15))))
        
        plt.yticks(range(len(top_15)), top_15['Feature_Name'], fontsize=8)
        plt.xlabel("Cohen's d效应量", fontsize=12, fontweight='bold')
        plt.title('Top 15特征效应量', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'class_difference_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_feature_combinations(self, data, label_encoder):
        """分析特征组合的区分能力"""
        print("分析特征组合...")
        
        # 获取Top特征
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 使用F统计量选择Top特征
        selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"选择的Top {len(selected_features)} 特征: {selected_features}")
        
        # 分析特征组合
        combination_results = []
        
        # 2特征组合
        print("分析2特征组合...")
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                feat1, feat2 = selected_features[i], selected_features[j]
                
                # 创建组合特征
                X_combined = X[[feat1, feat2]]
                
                # 使用随机森林评估组合特征
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(rf, X_combined, y, cv=5, scoring='accuracy')
                
                combination_results.append({
                    'Feature1': feat1,
                    'Feature2': feat2,
                    'Mean_Accuracy': cv_scores.mean(),
                    'Std_Accuracy': cv_scores.std(),
                    'Combination_Type': '2_features'
                })
        
        # 3特征组合（选择Top 10个2特征组合）
        print("分析3特征组合...")
        top_2_combinations = sorted(combination_results, key=lambda x: x['Mean_Accuracy'], reverse=True)[:10]
        
        for combo in top_2_combinations:
            feat1, feat2 = combo['Feature1'], combo['Feature2']
            
            # 为每个2特征组合添加一个额外的特征
            for feat3 in selected_features:
                if feat3 not in [feat1, feat2]:
                    X_combined = X[[feat1, feat2, feat3]]
                    
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    cv_scores = cross_val_score(rf, X_combined, y, cv=5, scoring='accuracy')
                    
                    combination_results.append({
                        'Feature1': feat1,
                        'Feature2': feat2,
                        'Feature3': feat3,
                        'Mean_Accuracy': cv_scores.mean(),
                        'Std_Accuracy': cv_scores.std(),
                        'Combination_Type': '3_features'
                    })
        
        # 保存结果
        combination_df = pd.DataFrame(combination_results)
        combination_df.to_csv(os.path.join(self.results_dir, 'feature_combination_analysis.csv'), index=False)
        
        # 显示最佳组合
        print(f"\n📊 最佳特征组合 (Top 10):")
        best_combinations = combination_df.sort_values('Mean_Accuracy', ascending=False).head(10)
        for i, (_, row) in enumerate(best_combinations.iterrows()):
            if row['Combination_Type'] == '2_features':
                print(f"   {i+1:2d}. {row['Feature1']} + {row['Feature2']} : "
                      f"准确率={row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
            else:
                print(f"   {i+1:2d}. {row['Feature1']} + {row['Feature2']} + {row['Feature3']} : "
                      f"准确率={row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
        
        return combination_results
    
    def _validate_with_ml_models(self, data, label_encoder):
        """使用机器学习模型验证特征区分能力"""
        print("使用机器学习模型验证...")
        
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 最终检查数据
        if X.isnull().any().any() or np.isinf(X.values).any():
            print("警告：模型验证前进行最终数据清理...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用多个模型验证
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        
        validation_results = {}
        
        for model_name, model in models.items():
            print(f"验证 {model_name}...")
            
            # 交叉验证
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # 准确率
            accuracy_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            
            # ROC AUC（二分类问题）
            if len(np.unique(y)) == 2:
                roc_auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            else:
                roc_auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc_ovr')
            
            # F1分数
            f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
            
            validation_results[model_name] = {
                'Accuracy_Mean': accuracy_scores.mean(),
                'Accuracy_Std': accuracy_scores.std(),
                'ROC_AUC_Mean': roc_auc_scores.mean(),
                'ROC_AUC_Std': roc_auc_scores.std(),
                'F1_Mean': f1_scores.mean(),
                'F1_Std': f1_scores.std()
            }
        
        # 保存验证结果
        validation_df = pd.DataFrame(validation_results).T
        validation_df.to_csv(os.path.join(self.results_dir, 'ml_validation_results.csv'))
        
        # 显示结果
        print(f"\n📊 机器学习模型验证结果:")
        for model_name, results in validation_results.items():
            print(f"   {model_name}:")
            print(f"     准确率: {results['Accuracy_Mean']:.4f} ± {results['Accuracy_Std']:.4f}")
            print(f"     ROC AUC: {results['ROC_AUC_Mean']:.4f} ± {results['ROC_AUC_Std']:.4f}")
            print(f"     F1分数: {results['F1_Mean']:.4f} ± {results['F1_Std']:.4f}")
        
        return validation_results
    
    def _generate_comprehensive_report(self, feature_discrimination, class_differences, 
                                     feature_combinations, ml_validation):
        """生成综合评估报告"""
        print("生成综合评估报告...")
        
        report_path = os.path.join(self.results_dir, 'fingerprint_evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("指纹特征区分能力全面评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("评估时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            # 数据质量总结
            f.write("1. 数据质量总结\n")
            f.write("-" * 40 + "\n")
            if self.data_quality_report:
                f.write(f"总样本数: {sum(self.data_quality_report['label_distribution'].values())}\n")
                f.write(f"标签分布: {self.data_quality_report['label_distribution']}\n")
                f.write(f"重复样本: {self.data_quality_report['duplicate_samples']}\n")
                f.write(f"缺失值总数: {sum(self.data_quality_report['missing_values'].values())}\n\n")
            
            # 特征区分能力总结
            f.write("2. 特征区分能力总结\n")
            f.write("-" * 40 + "\n")
            if not feature_discrimination.empty:
                significant_features = feature_discrimination[feature_discrimination['Significant_F'] == True]
                f.write(f"总特征数: {len(feature_discrimination)}\n")
                f.write(f"统计显著特征数: {len(significant_features)}\n")
                f.write(f"显著特征比例: {len(significant_features)/len(feature_discrimination)*100:.2f}%\n\n")
                
                f.write("Top 10特征 (按F统计量):\n")
                for i, (_, row) in enumerate(feature_discrimination.head(10).iterrows()):
                    f.write(f"   {i+1:2d}. {row['Feature_Name']:<25} : F={row['F_Score']:.4f}, "
                           f"p={row['P_Value']:.6f}\n")
                f.write("\n")
            
            # 类别差异总结
            f.write("3. 类别差异总结\n")
            f.write("-" * 40 + "\n")
            if class_differences:
                large_effect_features = [f for f, stats in class_differences.items() 
                                       if stats['effect_size'] in ['大', '中等']]
                f.write(f"大/中等效应量特征数: {len(large_effect_features)}\n")
                f.write(f"微小/小效应量特征数: {len(class_differences) - len(large_effect_features)}\n\n")
                
                f.write("Top 10特征 (按Cohen's d效应量):\n")
                sorted_features = sorted(class_differences.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
                for i, (feature, stats) in enumerate(sorted_features[:10]):
                    f.write(f"   {i+1:2d}. {feature:<25} : d={stats['cohens_d']:.4f} ({stats['effect_size']})\n")
                f.write("\n")
            
            # 特征组合总结
            f.write("4. 特征组合总结\n")
            f.write("-" * 40 + "\n")
            if feature_combinations:
                best_2_feature = max([c for c in feature_combinations if c['Combination_Type'] == '2_features'], 
                                   key=lambda x: x['Mean_Accuracy'])
                best_3_feature = max([c for c in feature_combinations if c['Combination_Type'] == '3_features'], 
                                   key=lambda x: x['Mean_Accuracy'])
                
                f.write(f"最佳2特征组合: {best_2_feature['Feature1']} + {best_2_feature['Feature2']}\n")
                f.write(f"准确率: {best_2_feature['Mean_Accuracy']:.4f} ± {best_2_feature['Std_Accuracy']:.4f}\n\n")
                
                f.write(f"最佳3特征组合: {best_3_feature['Feature1']} + {best_3_feature['Feature2']} + {best_3_feature['Feature3']}\n")
                f.write(f"准确率: {best_3_feature['Mean_Accuracy']:.4f} ± {best_3_feature['Std_Accuracy']:.4f}\n\n")
            
            # 机器学习验证总结
            f.write("5. 机器学习验证总结\n")
            f.write("-" * 40 + "\n")
            if ml_validation:
                for model_name, results in ml_validation.items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  准确率: {results['Accuracy_Mean']:.4f} ± {results['Accuracy_Std']:.4f}\n")
                    f.write(f"  ROC AUC: {results['ROC_AUC_Mean']:.4f} ± {results['ROC_AUC_Std']:.4f}\n")
                    f.write(f"  F1分数: {results['F1_Mean']:.4f} ± {results['F1_Std']:.4f}\n\n")
            
            # 结论和建议
            f.write("6. 结论和建议\n")
            f.write("-" * 40 + "\n")
            
            # 基于结果给出建议
            if feature_discrimination.empty or class_differences is None:
                f.write("数据不足，无法给出有效结论。\n")
            else:
                significant_count = len(feature_discrimination[feature_discrimination['Significant_F'] == True])
                large_effect_count = len([f for f, stats in class_differences.items() 
                                        if stats['effect_size'] in ['大', '中等']])
                
                if significant_count > len(feature_discrimination) * 0.3 and large_effect_count > 5:
                    f.write("✅ 现有特征集合具有较好的区分能力，建议:\n")
                    f.write("   - 使用Top特征进行模型训练\n")
                    f.write("   - 优化特征组合\n")
                    f.write("   - 调整模型参数\n")
                elif significant_count > 0 and large_effect_count > 0:
                    f.write("⚠️  现有特征集合具有有限的区分能力，建议:\n")
                    f.write("   - 使用最佳特征组合\n")
                    f.write("   - 考虑特征工程优化\n")
                    f.write("   - 寻找新的区分特征\n")
                else:
                    f.write("❌ 现有特征集合区分能力不足，建议:\n")
                    f.write("   - 重新审视数据标注\n")
                    f.write("   - 寻找新的特征来源\n")
                    f.write("   - 考虑不同的分类策略\n")
            
            f.write("\n生成的文件:\n")
            f.write("-" * 40 + "\n")
            for file in os.listdir(self.results_dir):
                if file.endswith(('.png', '.csv', '.txt')):
                    f.write(f"  {file}\n")
        
        print(f"✅ 综合评估报告已保存到: {report_path}")


def main():
    """主函数"""
    print("=== 脚本 06: 指纹区分能力全面评估 ===")
    
    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    try:
        # 1. 加载数据
        print("加载数据...")
        runs_dir = config['output_paths']['runs_directory']
        candidate_csvs = glob.glob(os.path.join(runs_dir, '*/vulnerability_fingerprints.csv'))
        
        if not candidate_csvs:
            print("\n错误：找不到 vulnerability_fingerprints.csv 文件。")
            sys.exit(1)
        
        fingerprint_file_path = max(candidate_csvs, key=os.path.getctime)
        print(f"使用最新的数据文件: {fingerprint_file_path}")
        
        data = pd.read_csv(fingerprint_file_path)
        print(f"成功加载 {len(data)} 个样本")
        
        # 2. 创建结果目录
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(runs_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        results_dir = os.path.join(output_dir, f"fingerprint_evaluation_{current_time}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"创建结果目录: {results_dir}")
        
        # 3. 创建标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(data['vulnerability_type'])
        
        # 4. 创建指纹评估器
        evaluator = FingerprintEvaluator(results_dir)
        
        # 5. 执行全面评估
        print("\n开始全面指纹评估...")
        
        feature_discrimination, class_differences, feature_combinations, ml_validation = \
            evaluator.comprehensive_evaluation(data, label_encoder)
        
        print(f"\n✅ 指纹评估完成！结果保存在: {results_dir}")
        print("\n📊 主要评估内容:")
        print("   - 数据质量检查")
        print("   - 数据清理")
        print("   - 特征区分能力分析")
        print("   - 类别间差异深度分析")
        print("   - 特征组合分析")
        print("   - 机器学习模型验证")
        print("   - 综合评估报告")
        
    except Exception as e:
        print(f"\n❌ 脚本执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()