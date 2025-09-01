#!/usr/bin/env python3
"""
脚本 07: 特征区分度深度测评
目标：专门测评feature_extractor.py中定义的特征的区分能力
帮助决定是否放弃现有特征，还是可以基于它们优化
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
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FeatureDiscriminativeAnalyzer:
    """特征区分度分析器"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.analysis_results = {}
        
    def comprehensive_feature_analysis(self, data, label_encoder):
        """全面分析特征区分度"""
        print("=== 开始特征区分度深度测评 ===")
        
        # 1. 数据质量检查
        print("\n1. 数据质量检查...")
        self._check_data_quality(data, label_encoder)
        
        # 2. 数据清理
        print("\n2. 数据清理...")
        data_cleaned = self._clean_data(data)
        
        # 3. 特征分类分析
        print("\n3. 特征分类分析...")
        feature_categories = self._categorize_features(data_cleaned)
        
        # 4. 单特征区分度分析
        print("\n4. 单特征区分度分析...")
        single_feature_analysis = self._analyze_single_features(data_cleaned, label_encoder)
        
        # 5. 特征组合分析
        print("\n5. 特征组合分析...")
        feature_combinations = self._analyze_feature_combinations(data_cleaned, label_encoder)
        
        # 6. 特征重要性分析
        print("\n6. 特征重要性分析...")
        feature_importance = self._analyze_feature_importance(data_cleaned, label_encoder)
        
        # 7. 生成测评报告
        print("\n7. 生成测评报告...")
        self._generate_discriminative_report(feature_categories, single_feature_analysis, 
                                          feature_combinations, feature_importance)
        
        return single_feature_analysis, feature_combinations, feature_importance
    
    def _clean_data(self, data):
        """清理数据"""
        print("清理数据...")
        
        data_cleaned = data.copy()
        
        # 获取数值列
        numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
        if 'vulnerability_type' in numeric_columns:
            numeric_columns = numeric_columns.drop('vulnerability_type')
        
        print(f"处理 {len(numeric_columns)} 个数值列...")
        
        for col in numeric_columns:
            # 替换无穷大值
            data_cleaned[col] = data_cleaned[col].replace([np.inf, -np.inf], np.nan)
            
            # 处理异常值
            Q1 = data_cleaned[col].quantile(0.01)
            Q3 = data_cleaned[col].quantile(0.99)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data_cleaned[col] = data_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 填充NaN值
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
        
        # 标签分布
        label_counts = data['vulnerability_type'].value_counts()
        quality_report['label_distribution'] = label_counts.to_dict()
        
        print(f"标签分布: {label_counts.to_dict()}")
        
        # 缺失值
        missing_values = data.isnull().sum()
        quality_report['missing_values'] = missing_values.to_dict()
        
        print(f"缺失值统计: {missing_values.sum()} 个总缺失值")
        
        # 无穷大值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if 'vulnerability_type' in numeric_cols:
            numeric_cols = numeric_cols.drop('vulnerability_type')
        
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(data[col]).sum()
            inf_counts[col] = inf_count
        
        quality_report['infinity_values'] = inf_counts
        total_inf = sum(inf_counts.values())
        print(f"无穷大值统计: {total_inf} 个总无穷大值")
        
        # 保存质量报告
        quality_df = pd.DataFrame(quality_report['infinity_values'], index=['infinity_count']).T
        quality_df.to_csv(os.path.join(self.results_dir, 'data_quality_report.csv'))
        
        self.data_quality_report = quality_report
        
        return quality_report
    
    def _categorize_features(self, data):
        """对特征进行分类"""
        print("对特征进行分类...")
        
        # 基于feature_extractor.py中的特征定义进行分类
        feature_categories = {
            'KL散度特征': ['kl_divergence_pos', 'kl_divergence_neg'],
            '相似度特征': ['cosine_similarity'],
            '纹理特征': ['contrast', 'homogeneity', 'energy', 'correlation'],
            '统计特征': ['std_dev_diff', 'kurtosis_diff'],
            '频域特征': ['high_freq_ratio'],
            '小波特征': ['dynamic_wavelet_ratio_change', 'll_distortion'],
            '其他特征': ['ratio_zscore', 'super_fingerprint']
        }
        
        # 检查哪些特征实际存在于数据中
        existing_categories = {}
        for category, features in feature_categories.items():
            existing_features = [f for f in features if f in data.columns]
            if existing_features:
                existing_categories[category] = existing_features
        
        print(f"特征分类结果:")
        for category, features in existing_categories.items():
            print(f"  {category}: {features}")
        
        # 保存特征分类
        category_df = pd.DataFrame([
            {'Category': cat, 'Features': ', '.join(feats), 'Count': len(feats)}
            for cat, feats in existing_categories.items()
        ])
        category_df.to_csv(os.path.join(self.results_dir, 'feature_categories.csv'), index=False)
        
        return existing_categories
    
    def _analyze_single_features(self, data, label_encoder):
        """分析单特征的区分度"""
        print("分析单特征区分度...")
        
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 最终检查数据
        if X.isnull().any().any() or np.isinf(X.values).any():
            print("警告：单特征分析前进行最终数据清理...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        # 1. F统计量分析
        f_scores, p_values = f_classif(X, y)
        
        # 2. 互信息分析
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 3. 类别间差异分析
        class_differences = self._calculate_class_differences(data, label_encoder)
        
        # 4. 创建综合分析DataFrame
        feature_analysis = pd.DataFrame({
            'Feature_Name': X.columns,
            'F_Score': f_scores,
            'P_Value': p_values,
            'Mutual_Info': mi_scores,
            'Significant_F': p_values < 0.05,
            'F_Rank': f_scores.argsort()[::-1].argsort() + 1,
            'MI_Rank': mi_scores.argsort()[::-1].argsort() + 1
        })
        
        # 添加类别差异信息
        for feature in X.columns:
            if feature in class_differences:
                feature_analysis.loc[feature_analysis['Feature_Name'] == feature, 'Cohens_d'] = \
                    class_differences[feature]['cohens_d']
                feature_analysis.loc[feature_analysis['Feature_Name'] == feature, 'Effect_Size'] = \
                    class_differences[feature]['effect_size']
                feature_analysis.loc[feature_analysis['Feature_Name'] == feature, 'Overlap'] = \
                    class_differences[feature]['overlap']
        
        # 按F统计量排序
        feature_analysis = feature_analysis.sort_values('F_Score', ascending=False)
        
        # 保存结果
        feature_analysis.to_csv(os.path.join(self.results_dir, 'single_feature_analysis.csv'), index=False)
        
        # 显示结果
        print(f"\n�� 单特征区分度排行榜 (Top 20, 按F统计量):")
        for i, (_, row) in enumerate(feature_analysis.head(20).iterrows()):
            significance = "***" if row['Significant_F'] else ""
            cohens_d_info = f", d={row.get('Cohens_d', 'N/A'):.4f}" if 'Cohens_d' in row and not pd.isna(row['Cohens_d']) else ""
            print(f"   {i+1:2d}. {row['Feature_Name']:<25} : F={row['F_Score']:.4f}, "
                  f"p={row['P_Value']:.6f}, MI={row['Mutual_Info']:.4f}{cohens_d_info} {significance}")
        
        # 创建可视化
        self._create_single_feature_visualization(feature_analysis)
        
        return feature_analysis
    
    def _calculate_class_differences(self, data, label_encoder):
        """计算类别间差异"""
        classes = label_encoder.classes_
        target_classes = ['noise_gaussian', 'adversarial_pgd']
        
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
                                'q75': class_data.quantile(0.75)
                            }
                
                # 计算类别间差异
                if len(class_stats) == 2:
                    # Cohen's d效应量
                    mean_diff = abs(class_stats[target_classes[0]]['mean'] - class_stats[target_classes[1]]['mean'])
                    pooled_std = np.sqrt((class_stats[target_classes[0]]['std']**2 + class_stats[target_classes[1]]['std']**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # 重叠度
                    overlap = self._calculate_overlap(class_stats[target_classes[0]], class_stats[target_classes[1]])
                    
                    feature_analysis[feature] = {
                        'cohens_d': cohens_d,
                        'overlap': overlap,
                        'effect_size': self._classify_effect_size(cohens_d)
                    }
        
        return feature_analysis
    
    def _calculate_overlap(self, stats1, stats2):
        """计算重叠度"""
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
        """分类效应量"""
        if cohens_d < 0.2:
            return "微小"
        elif cohens_d < 0.5:
            return "小"
        elif cohens_d < 0.8:
            return "中等"
        else:
            return "大"
    
    def _create_single_feature_visualization(self, feature_analysis):
        """创建单特征分析可视化"""
        # 1. F统计量 vs 互信息散点图
        plt.figure(figsize=(12, 8))
        
        plt.scatter(feature_analysis['F_Score'], feature_analysis['Mutual_Info'], 
                   alpha=0.7, s=50)
        
        # 标注前10个特征
        top_10 = feature_analysis.head(10)
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
        
        plt.savefig(os.path.join(self.results_dir, 'single_feature_discrimination.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top 20特征F统计量条形图
        plt.figure(figsize=(16, 10))
        
        top_20 = feature_analysis.head(20)
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
    
    def _analyze_feature_combinations(self, data, label_encoder):
        """分析特征组合"""
        print("分析特征组合...")
        
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 使用F统计量选择Top特征
        selector = SelectKBest(f_classif, k=min(15, X.shape[1]))
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
                
                X_combined = X[[feat1, feat2]]
                
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(rf, X_combined, y, cv=5, scoring='accuracy')
                
                combination_results.append({
                    'Feature1': feat1,
                    'Feature2': feat2,
                    'Mean_Accuracy': cv_scores.mean(),
                    'Std_Accuracy': cv_scores.std(),
                    'Combination_Type': '2_features'
                })
        
        # 3特征组合
        print("分析3特征组合...")
        top_2_combinations = sorted(combination_results, key=lambda x: x['Mean_Accuracy'], reverse=True)[:10]
        
        for combo in top_2_combinations:
            feat1, feat2 = combo['Feature1'], combo['Feature2']
            
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
    
    def _analyze_feature_importance(self, data, label_encoder):
        """分析特征重要性"""
        print("分析特征重要性...")
        
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 最终检查数据
        if X.isnull().any().any() or np.isinf(X.values).any():
            print("警告：特征重要性分析前进行最终数据清理...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'Feature_Name': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # 保存特征重要性
        feature_importance.to_csv(os.path.join(self.results_dir, 'feature_importance_analysis.csv'), index=False)
        
        # 显示前20个最重要的特征
        print("\n�� 特征重要性排行榜 (Top 20):")
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows()):
            print(f"   {i+1:2d}. {row['Feature_Name']:<25} : {row['Importance']:.4f}")
        
        # 创建可视化
        self._create_feature_importance_visualization(feature_importance)
        
        return feature_importance
    
    def _create_feature_importance_visualization(self, feature_importance):
        """创建特征重要性可视化"""
        plt.figure(figsize=(16, 10))
        
        # 选择前25个最重要的特征
        top_25 = feature_importance.head(25)
        
        # 创建水平条形图
        bars = plt.barh(range(len(top_25)), top_25['Importance'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_25))))
        
        plt.yticks(range(len(top_25)), top_25['Feature_Name'], fontsize=10)
        
        # 在条形图上添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, top_25['Importance'])):
            plt.text(importance + 0.001, i, f'{importance:.4f}', 
                    va='center', fontsize=9, fontweight='bold')
        
        plt.xlabel('特征重要性得分', fontsize=12, fontweight='bold')
        plt.title('特征重要性排行榜 - 基于随机森林', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, 'feature_importance_ranking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_discriminative_report(self, feature_categories, single_feature_analysis, 
                                      feature_combinations, feature_importance):
        """生成区分度测评报告"""
        print("生成区分度测评报告...")
        
        report_path = os.path.join(self.results_dir, 'feature_discriminative_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("特征区分度深度测评报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("测评时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            # 特征分类总结
            f.write("1. 特征分类总结\n")
            f.write("-" * 40 + "\n")
            for category, features in feature_categories.items():
                f.write(f"{category}: {len(features)} 个特征\n")
                f.write(f"  特征列表: {', '.join(features)}\n\n")
            
            # 单特征区分度总结
            f.write("2. 单特征区分度总结\n")
            f.write("-" * 40 + "\n")
            if not single_feature_analysis.empty:
                significant_features = single_feature_analysis[single_feature_analysis['Significant_F'] == True]
                f.write(f"总特征数: {len(single_feature_analysis)}\n")
                f.write(f"统计显著特征数: {len(significant_features)}\n")
                f.write(f"显著特征比例: {len(significant_features)/len(single_feature_analysis)*100:.2f}%\n\n")
                
                f.write("Top 10特征 (按F统计量):\n")
                for i, (_, row) in enumerate(single_feature_analysis.head(10).iterrows()):
                    f.write(f"   {i+1:2d}. {row['Feature_Name']:<25} : F={row['F_Score']:.4f}, "
                           f"p={row['P_Value']:.6f}\n")
                f.write("\n")
            
            # 特征组合总结
            f.write("3. 特征组合总结\n")
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
            
            # 特征重要性总结
            f.write("4. 特征重要性总结\n")
            f.write("-" * 40 + "\n")
            if not feature_importance.empty:
                f.write("Top 10特征 (按重要性):\n")
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                    f.write(f"   {i+1:2d}. {row['Feature_Name']:<25} : {row['Importance']:.4f}\n")
                f.write("\n")
            
            # 结论和建议
            f.write("5. 结论和建议\n")
            f.write("-" * 40 + "\n")
            
            # 基于结果给出建议
            if single_feature_analysis.empty:
                f.write("数据不足，无法给出有效结论。\n")
            else:
                significant_count = len(single_feature_analysis[single_feature_analysis['Significant_F'] == True])
                total_features = len(single_feature_analysis)
                
                # 检查是否有大效应量特征
                large_effect_features = 0
                if 'Cohens_d' in single_feature_analysis.columns:
                    large_effect_features = len(single_feature_analysis[
                        single_feature_analysis['Cohens_d'] > 0.8
                    ])
                
                if significant_count > total_features * 0.8 and large_effect_features > 3:
                    f.write("✅ 现有特征集合具有较好的区分能力，建议:\n")
                    f.write("   - 保留现有特征，进行优化组合\n")
                    f.write("   - 使用最佳特征组合进行模型训练\n")
                    f.write("   - 调整模型参数和阈值\n")
                    f.write("   - 考虑特征工程优化\n")
                elif significant_count > total_features * 0.5 and large_effect_features > 0:
                    f.write("⚠️  现有特征集合具有有限的区分能力，建议:\n")
                    f.write("   - 保留有区分能力的特征\n")
                    f.write("   - 寻找新的补充特征\n")
                    f.write("   - 优化特征组合策略\n")
                    f.write("   - 考虑特征转换和组合\n")
                else:
                    f.write("❌ 现有特征集合区分能力不足，建议:\n")
                    f.write("   - 重新审视特征设计\n")
                    f.write("   - 寻找新的特征来源\n")
                    f.write("   - 考虑不同的分类策略\n")
                    f.write("   - 可能需要重新设计整个特征提取流程\n")
            
            f.write("\n生成的文件:\n")
            f.write("-" * 40 + "\n")
            for file in os.listdir(self.results_dir):
                if file.endswith(('.png', '.csv', '.txt')):
                    f.write(f"  {file}\n")
        
        print(f"✅ 区分度测评报告已保存到: {report_path}")


def main():
    """主函数"""
    print("=== 脚本 07: 特征区分度深度测评 ===")
    
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
        
        results_dir = os.path.join(output_dir, f"feature_discriminative_analysis_{current_time}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"创建结果目录: {results_dir}")
        
        # 3. 创建标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(data['vulnerability_type'])
        
        # 4. 创建特征区分度分析器
        analyzer = FeatureDiscriminativeAnalyzer(results_dir)
        
        # 5. 执行全面分析
        print("\n开始特征区分度深度测评...")
        
        single_feature_analysis, feature_combinations, feature_importance = \
            analyzer.comprehensive_feature_analysis(data, label_encoder)
        
        print(f"\n✅ 特征区分度测评完成！结果保存在: {results_dir}")
        print("\n📊 主要测评内容:")
        print("   - 数据质量检查")
        print("   - 特征分类分析")
        print("   - 单特征区分度分析")
        print("   - 特征组合分析")
        print("   - 特征重要性分析")
        print("   - 区分度测评报告")
        
    except Exception as e:
        print(f"\n❌ 脚本执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()