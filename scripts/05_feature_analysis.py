#!/usr/bin/env python3
"""
脚本 05: 深度特征分析 - 寻找noise_gaussian和adversarial_pgd的区分特征
目标：分析关键特征的分布差异，发现新的特征组合，提升分类性能
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
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FeatureAnalyzer:
    """特征分析器类"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.analysis_results = {}
        
    def clean_data(self, data):
        """清理数据中的无穷大值和异常值"""
        print("清理数据中的无穷大值和异常值...")
        
        # 获取数值列
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if 'vulnerability_type' in numeric_columns:
            numeric_columns = numeric_columns.drop('vulnerability_type')
        
        # 处理无穷大值
        data_cleaned = data.copy()
        for col in numeric_columns:
            # 替换无穷大值
            data_cleaned[col] = data_cleaned[col].replace([np.inf, -np.inf], np.nan)
            
            # 计算分位数来识别异常值
            Q1 = data_cleaned[col].quantile(0.01)  # 1%分位数
            Q3 = data_cleaned[col].quantile(0.99)  # 99%分位数
            IQR = Q3 - Q1
            
            # 将异常值替换为分位数边界值
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data_cleaned[col] = data_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 用中位数填充NaN值
            if data_cleaned[col].isna().sum() > 0:
                median_val = data_cleaned[col].median()
                data_cleaned[col].fillna(median_val, inplace=True)
        
        print(f"✅ 数据清理完成，处理了 {len(numeric_columns)} 个数值列")
        return data_cleaned
        
    def analyze_feature_distributions(self, data, label_encoder):
        """分析关键特征在不同类别上的分布差异"""
        print("=== 分析关键特征分布差异 ===")
        
        # 获取关键特征（基于之前的重要性分析）
        key_features = ['ratio_zscore', 'high_freq_ratio', 'std_dev_diff', 
                       'correlation', 'energy', 'homogeneity', 'contrast', 'll_distortion']
        
        # 过滤出存在的特征
        existing_features = [f for f in key_features if f in data.columns]
        print(f"分析的特征: {existing_features}")
        
        # 创建分布对比图
        self._create_distribution_comparison(data, existing_features, label_encoder)
        
        # 计算统计差异
        self._calculate_statistical_differences(data, existing_features, label_encoder)
        
        return existing_features
    
    def _create_distribution_comparison(self, data, features, label_encoder):
        """创建特征分布对比图"""
        print("创建特征分布对比图...")
        
        # 获取类别标签
        classes = label_encoder.classes_
        target_classes = ['noise_gaussian', 'adversarial_pgd']
        
        # 为每个特征创建分布对比图
        for feature in features:
            if feature not in data.columns:
                continue
                
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 子图1：箱线图对比
            plot_data = []
            labels = []
            colors = []
            
            for i, class_name in enumerate(target_classes):
                if class_name in classes:
                    class_idx = list(classes).index(class_name)
                    class_data = data[data['vulnerability_type'] == class_name][feature]
                    if len(class_data) > 0:
                        plot_data.append(class_data)
                        labels.append(class_name)
                        colors.append(['#FF6B6B', '#4ECDC4'][i])
            
            if plot_data:
                bp = ax1.boxplot(plot_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax1.set_title(f'{feature} - 箱线图对比', fontsize=14, fontweight='bold')
                ax1.set_ylabel('特征值', fontsize=12)
                ax1.grid(True, alpha=0.3)
            
            # 子图2：密度图对比
            for i, class_name in enumerate(target_classes):
                if class_name in classes:
                    class_data = data[data['vulnerability_type'] == class_name][feature]
                    if len(class_data) > 0:
                        ax2.hist(class_data, bins=30, alpha=0.7, density=True, 
                                label=class_name, color=['#FF6B6B', '#4ECDC4'][i])
            
            ax2.set_title(f'{feature} - 密度分布对比', fontsize=14, fontweight='bold')
            ax2.set_xlabel('特征值', fontsize=12)
            ax2.set_ylabel('密度', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'distribution_{feature}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ 分布对比图已保存到: {self.results_dir}")
    
    def _calculate_statistical_differences(self, data, features, label_encoder):
        """计算统计差异指标"""
        print("计算统计差异指标...")
        
        classes = label_encoder.classes_
        target_classes = ['noise_gaussian', 'adversarial_pgd']
        
        stats_results = []
        
        for feature in features:
            if feature not in data.columns:
                continue
                
            feature_stats = {'feature': feature}
            
            for class_name in target_classes:
                if class_name in classes:
                    class_data = data[data['vulnerability_type'] == class_name][feature]
                    if len(class_data) > 0:
                        feature_stats[f'{class_name}_mean'] = class_data.mean()
                        feature_stats[f'{class_name}_std'] = class_data.std()
                        feature_stats[f'{class_name}_median'] = class_data.median()
                        feature_stats[f'{class_name}_q25'] = class_data.quantile(0.25)
                        feature_stats[f'{class_name}_q75'] = class_data.quantile(0.75)
            
            # 计算两个类别之间的差异
            if (f'noise_gaussian_mean' in feature_stats and 
                f'adversarial_pgd_mean' in feature_stats):
                
                # 均值差异
                mean_diff = feature_stats['adversarial_pgd_mean'] - feature_stats['noise_gaussian_mean']
                feature_stats['mean_difference'] = mean_diff
                
                # 相对差异
                if feature_stats['noise_gaussian_mean'] != 0:
                    relative_diff = abs(mean_diff) / abs(feature_stats['noise_gaussian_mean'])
                    feature_stats['relative_difference'] = relative_diff
                
                # 进行t检验
                noise_data = data[data['vulnerability_type'] == 'noise_gaussian'][feature]
                adv_data = data[data['vulnerability_type'] == 'adversarial_pgd'][feature]
                
                if len(noise_data) > 0 and len(adv_data) > 0:
                    t_stat, p_value = stats.ttest_ind(noise_data, adv_data)
                    feature_stats['t_statistic'] = t_stat
                    feature_stats['p_value'] = p_value
                    feature_stats['significant'] = p_value < 0.05
            
            stats_results.append(feature_stats)
        
        # 保存统计结果
        stats_df = pd.DataFrame(stats_results)
        stats_df.to_csv(os.path.join(self.results_dir, 'feature_statistical_analysis.csv'), index=False)
        
        # 显示显著差异的特征
        significant_features = stats_df[stats_df.get('significant', False) == True]
        print(f"\n📊 统计显著差异的特征 ({len(significant_features)} 个):")
        for _, row in significant_features.iterrows():
            print(f"   {row['feature']}: p值={row['p_value']:.6f}, t统计量={row['t_statistic']:.3f}")
        
        self.analysis_results['statistical_analysis'] = stats_df
        return stats_df
    
    def create_new_features(self, data):
        """创建新的特征组合"""
        print("\n=== 创建新特征 ===")
        
        new_features = {}
        
        # 1. 比率特征
        print("创建比率特征...")
        if 'ratio_zscore' in data.columns and 'std_dev_diff' in data.columns:
            new_features['ratio_std_ratio'] = data['ratio_zscore'] / (data['std_dev_diff'] + 1e-8)
        
        if 'high_freq_ratio' in data.columns and 'energy' in data.columns:
            new_features['freq_energy_ratio'] = data['high_freq_ratio'] / (data['energy'] + 1e-8)
        
        # 2. 差异特征
        print("创建差异特征...")
        if 'ratio_zscore' in data.columns and 'high_freq_ratio' in data.columns:
            new_features['ratio_freq_diff'] = data['ratio_zscore'] - data['high_freq_ratio']
        
        if 'std_dev_diff' in data.columns and 'correlation' in data.columns:
            new_features['std_corr_diff'] = data['std_dev_diff'] - data['correlation']
        
        # 3. 乘积特征
        print("创建乘积特征...")
        if 'ratio_zscore' in data.columns and 'std_dev_diff' in data.columns:
            new_features['ratio_std_product'] = data['ratio_zscore'] * data['std_dev_diff']
        
        if 'high_freq_ratio' in data.columns and 'energy' in data.columns:
            new_features['freq_energy_product'] = data['high_freq_ratio'] * data['energy']
        
        # 4. 多项式特征
        print("创建多项式特征...")
        if 'ratio_zscore' in data.columns:
            new_features['ratio_zscore_squared'] = data['ratio_zscore'] ** 2
        
        if 'high_freq_ratio' in data.columns:
            new_features['high_freq_ratio_squared'] = data['high_freq_ratio'] ** 2
        
        # 5. 统计特征
        print("创建统计特征...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'vulnerability_type']
        
        if len(numeric_columns) > 0:
            # 行均值
            new_features['row_mean'] = data[numeric_columns].mean(axis=1)
            # 行标准差
            new_features['row_std'] = data[numeric_columns].std(axis=1)
            # 行最大值
            new_features['row_max'] = data[numeric_columns].max(axis=1)
            # 行最小值
            new_features['row_min'] = data[numeric_columns].min(axis=1)
            # 行范围
            new_features['row_range'] = new_features['row_max'] - new_features['row_min']
        
        print(f"✅ 创建了 {len(new_features)} 个新特征")
        
        # 将新特征添加到数据中
        for feature_name, feature_values in new_features.items():
            data[feature_name] = feature_values
        
        return data, new_features
    
    def evaluate_new_features(self, data, label_encoder):
        """评估新特征的重要性"""
        print("\n=== 评估新特征重要性 ===")
        
        # 准备数据
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 再次检查数据是否包含无穷大值
        print("检查数据中的无穷大值...")
        if np.any(np.isinf(X.values)) or np.any(np.isnan(X.values)):
            print("警告：数据中仍包含无穷大值或NaN值，进行最终清理...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        print(f"数据形状: {X.shape}")
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        
        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'Feature_Name': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # 保存特征重要性
        feature_importance.to_csv(os.path.join(self.results_dir, 'new_feature_importance.csv'), index=False)
        
        # 显示前20个最重要的特征
        print("\n📊 新特征重要性排行榜 (Top 20):")
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows()):
            print(f"   {i+1:2d}. {row['Feature_Name']:<25} : {row['Importance']:.4f}")
        
        # 创建新特征重要性可视化
        self._create_new_feature_importance_plot(feature_importance)
        
        return feature_importance
    
    def _create_new_feature_importance_plot(self, feature_importance):
        """创建新特征重要性可视化"""
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
        plt.title('新特征重要性排行榜 - 包含新创建的特征', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, 'new_feature_importance_ranking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_correlation_analysis(self, data):
        """创建特征相关性分析"""
        print("\n=== 特征相关性分析 ===")
        
        # 选择数值特征
        numeric_data = data.select_dtypes(include=[np.number])
        if 'vulnerability_type' in numeric_data.columns:
            numeric_data = numeric_data.drop('vulnerability_type', axis=1)
        
        # 计算相关性矩阵
        correlation_matrix = numeric_data.corr()
        
        # 创建热力图
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdBu_r', center=0, square=True, linewidths=0.5)
        
        plt.title('特征相关性热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, 'feature_correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存相关性矩阵
        correlation_matrix.to_csv(os.path.join(self.results_dir, 'feature_correlation_matrix.csv'))
        
        # 找出高相关性的特征对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # 高相关性阈值
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            print(f"\n🔍 发现 {len(high_corr_pairs)} 对高相关性特征:")
            for pair in high_corr_pairs:
                print(f"   {pair['feature1']} <-> {pair['feature2']} : {pair['correlation']:.3f}")
        
        return correlation_matrix
    
    def create_dimensionality_reduction_analysis(self, data, label_encoder):
        """创建降维分析"""
        print("\n=== 降维分析 ===")
        
        # 准备数据
        X = data.drop('vulnerability_type', axis=1)
        y = label_encoder.transform(data['vulnerability_type'])
        
        # 再次检查数据
        if np.any(np.isinf(X.values)) or np.any(np.isnan(X.values)):
            print("警告：降维分析前进行数据清理...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA分析
        print("进行PCA分析...")
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # 创建PCA可视化
        self._create_pca_visualization(X_pca, y, label_encoder, pca)
        
        # t-SNE分析
        print("进行t-SNE分析...")
        if X_scaled.shape[1] > 50:  # 如果特征太多，先降维
            X_reduced = PCA(n_components=50).fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_reduced)//4))
        X_tsne = tsne.fit_transform(X_reduced)
        
        # 创建t-SNE可视化
        self._create_tsne_visualization(X_tsne, y, label_encoder)
        
        return pca, tsne
    
    def _create_pca_visualization(self, X_pca, y, label_encoder, pca):
        """创建PCA可视化"""
        plt.figure(figsize=(15, 6))
        
        # 子图1：前两个主成分的散点图
        ax1 = plt.subplot(1, 2, 1)
        classes = label_encoder.classes_
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, class_name in enumerate(classes):
            if i < len(colors):
                mask = y == i
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=colors[i], label=class_name, alpha=0.7, s=30)
        
        ax1.set_xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.3f})')
        ax1.set_ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.3f})')
        ax1.set_title('PCA降维结果 - 前两个主成分')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：解释方差比例
        ax2 = plt.subplot(1, 2, 2)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        ax2.plot(range(1, len(explained_variance_ratio) + 1), 
                cumulative_variance_ratio, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('主成分数量')
        ax2.set_ylabel('累积解释方差比例')
        ax2.set_title('PCA解释方差累积比例')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%方差')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'pca_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_tsne_visualization(self, X_tsne, y, label_encoder):
        """创建t-SNE可视化"""
        plt.figure(figsize=(10, 8))
        
        classes = label_encoder.classes_
# 在t-SNE可视化中使用不同颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红色、青色、蓝色
# 确保每个类别都有独特的颜色
        
        for i, class_name in enumerate(classes):
            if i < len(colors):
                mask = y == i
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=colors[i], label=class_name, alpha=0.7, s=30)
        
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.title('t-SNE降维结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'tsne_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")
        
        report_path = os.path.join(self.results_dir, 'feature_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("特征分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("分析时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            if 'statistical_analysis' in self.analysis_results:
                f.write("统计分析结果:\n")
                f.write("-" * 30 + "\n")
                stats_df = self.analysis_results['statistical_analysis']
                significant_features = stats_df[stats_df.get('significant', False) == True]
                f.write(f"显著差异特征数量: {len(significant_features)}\n")
                for _, row in significant_features.iterrows():
                    f.write(f"  {row['feature']}: p值={row['p_value']:.6f}\n")
                f.write("\n")
            
            f.write("生成的文件:\n")
            f.write("-" * 30 + "\n")
            for file in os.listdir(self.results_dir):
                if file.endswith(('.png', '.csv')):
                    f.write(f"  {file}\n")
        
        print(f"✅ 分析报告已保存到: {report_path}")


def main():
    """主函数"""
    print("=== 脚本 05: 深度特征分析 ===")
    
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
        
        # 2. 创建结果目录 - 修改为runs/output目录
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(runs_dir, 'output')  # 新增output子目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建新的特征分析目录，名称与之前区分
        results_dir = os.path.join(output_dir, f"feature_analysis_{current_time}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"创建结果目录: {results_dir}")
        
        # 3. 创建标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(data['vulnerability_type'])
        
        # 4. 创建特征分析器
        analyzer = FeatureAnalyzer(results_dir)
        
        # 5. 数据清理
        print("\n开始数据清理...")
        data_cleaned = analyzer.clean_data(data)
        
        # 6. 执行分析
        print("\n开始深度特征分析...")
        
        # 分析现有特征分布
        existing_features = analyzer.analyze_feature_distributions(data_cleaned, label_encoder)
        
        # 创建新特征
        data_with_new_features, new_features = analyzer.create_new_features(data_cleaned)
        
        # 评估新特征
        feature_importance = analyzer.evaluate_new_features(data_with_new_features, label_encoder)
        
        # 相关性分析
        correlation_matrix = analyzer.create_feature_correlation_analysis(data_with_new_features)
        
        # 降维分析
        pca, tsne = analyzer.create_dimensionality_reduction_analysis(data_with_new_features, label_encoder)
        
        # 生成报告
        analyzer.generate_analysis_report()
        
        print(f"\n✅ 特征分析完成！结果保存在: {results_dir}")
        print("\n📊 主要发现:")
        print("   - 特征分布差异分析")
        print("   - 新特征创建和评估")
        print("   - 特征相关性分析")
        print("   - 降维分析结果")
        
    except Exception as e:
        print(f"\n❌ 脚本执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()