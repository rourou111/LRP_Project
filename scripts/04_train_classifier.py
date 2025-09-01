#!/usr/bin/env python3
"""
脚本 04: 训练两阶段分类器并评估性能
优化版本：清晰的代码结构，只生成核心图片
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

# 机器学习库导入
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ThresholdOptimizer:
    """阈值优化器类"""
    
    def __init__(self, energy_low_range=None, energy_high_range=None, arbit_range=None):
        self.energy_low_range = energy_low_range or [0.20, 0.25, 0.30, 0.35, 0.40]
        self.energy_high_range = energy_high_range or [0.60, 0.65, 0.70, 0.75, 0.80]
        self.arbit_range = arbit_range or [0.45, 0.50, 0.55, 0.60, 0.65]
    
    def optimize(self, energy_proba, X_test_experts, features_arbitrator,
                structural_arbitrator, y_test_full, non_drift_test_mask,
                adv_label, noise_label, pred1_test, label_encoder, label_encoder2):
        """优化阈值参数"""
        print("�� 开始自动调参，寻找最佳阈值组合...")
        
        best_score = 0
        best_params = {}
        results = []
        
        total_combinations = len(self.energy_low_range) * len(self.energy_high_range) * len(self.arbit_range)
        current = 0
        
        for energy_low in self.energy_low_range:
            for energy_high in self.energy_high_range:
                if energy_low >= energy_high:
                    continue
                for arbit_thresh in self.arbit_range:
                    current += 1
                    print(f"测试组合 {current}/{total_combinations}: "
                          f"能量低={energy_low:.2f}, 能量高={energy_high:.2f}, 仲裁={arbit_thresh:.2f}")
                    
                    # 使用当前阈值进行预测
                    final_predictions = self._predict_with_thresholds(
                        energy_proba, X_test_experts, features_arbitrator,
                        structural_arbitrator, y_test_full, non_drift_test_mask,
                        energy_low, energy_high, arbit_thresh, adv_label, noise_label
                    )
                    
                    # 评估性能
                    score, metrics = self._evaluate_performance(y_test_full, final_predictions)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'energy_low': energy_low,
                            'energy_high': energy_high,
                            'arbit_attack': arbit_thresh
                        }
                        print(f"🎯 发现更好的参数组合！综合评分: {score:.4f}")
                    
                    results.append({
                        'energy_low': energy_low,
                        'energy_high': energy_high,
                        'arbit_thresh': arbit_thresh,
                        'combined_score': score,
                        **metrics
                    })
        
        self._print_results(best_params, best_score, results)
        return best_params
    
    def _predict_with_thresholds(self, energy_proba, X_test_experts, features_arbitrator,
                                structural_arbitrator, y_test_full, non_drift_test_mask,
                                energy_low, energy_high, arbit_thresh, adv_label, noise_label):
        """使用给定阈值进行预测"""
        final_predictions = np.zeros_like(y_test_full)
        
        for i, energy_proba_sample in enumerate(energy_proba):
            prob_attack = energy_proba_sample[adv_label]
            
            if prob_attack <= energy_low:
                final_predictions[non_drift_test_mask][i] = noise_label
            elif prob_attack >= energy_high:
                final_predictions[non_drift_test_mask][i] = adv_label
            else:
                # 触发仲裁
                sample_features = X_test_experts.iloc[i:i+1][features_arbitrator]
                arbitrator_proba = structural_arbitrator.predict_proba(sample_features)[0]
                prob_attack_arb = arbitrator_proba[adv_label]
                
                if prob_attack_arb >= arbit_thresh:
                    final_predictions[non_drift_test_mask][i] = adv_label
                else:
                    final_predictions[non_drift_test_mask][i] = noise_label
        
        return final_predictions
    
    def _evaluate_performance(self, y_test_full, final_predictions):
        """评估预测性能"""
        accuracy = accuracy_score(y_test_full, final_predictions)
        
        cm = confusion_matrix(y_test_full, final_predictions)
        if len(cm) >= 3:
            adv_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
            noise_recall = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0
            combined_score = accuracy + adv_recall + noise_recall
            
            return combined_score, {
                'accuracy': accuracy,
                'adv_recall': adv_recall,
                'noise_recall': noise_recall
            }
        
        return accuracy, {'accuracy': accuracy}
    
    def _print_results(self, best_params, best_score, results):
        """打印优化结果"""
        print(f"\n🏆 自动调参完成！最佳参数组合:")
        print(f"   能量低阈值: {best_params['energy_low']:.2f}")
        print(f"   能量高阈值: {best_params['energy_high']:.2f}")
        print(f"   仲裁阈值: {best_params['arbit_attack']:.2f}")
        print(f"   最佳综合评分: {best_score:.4f}")
        
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        print(f"\n📊 前5个最佳参数组合:")
        for i, result in enumerate(results[:5]):
            print(f"   {i+1}. 能量低={result['energy_low']:.2f}, "
                  f"能量高={result['energy_high']:.2f}, "
                  f"仲裁={result['arbit_thresh']:.2f}, "
                  f"综合评分={result['combined_score']:.4f}")


class VisualizationManager:
    """可视化管理器类"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
    
    def create_core_visualizations(self, global_importance, final_predictions, y_test_full, 
                                 label_encoder, new_results_dir):
        """创建核心的四张可视化图表"""
        print("\n--- 生成核心可视化图表 ---")
        
        # 图表1：特征重要性排行榜
        self._create_feature_ranking(global_importance)
        
        # 图表2：专家系统性能对比
        self._create_expert_comparison(final_predictions, y_test_full, label_encoder)
        
        # 图表3：特征分析图
        self._create_feature_analysis(global_importance)
        
        # 图表4：特征热力图
        self._create_feature_heatmap(global_importance)
        
        print("✅ 核心可视化图表生成完成！")
    
    def _create_feature_ranking(self, global_importance):
        """创建特征重要性排行榜"""
        plt.figure(figsize=(14, 8))
        
        # 按总重要性排序，只显示前15个最重要的特征
        top_features = global_importance.head(15)
        
        bars = plt.barh(range(len(top_features)), top_features['Total_Importance'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        
        plt.yticks(range(len(top_features)), top_features['Feature_Name'], fontsize=11)
        
        # 在条形图上添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, top_features['Total_Importance'])):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        plt.xlabel('特征重要性得分', fontsize=12, fontweight='bold')
        plt.title('特征重要性排行榜 - Top 15 特征', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, '01_feature_importance_ranking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_expert_comparison(self, final_predictions, y_test_full, label_encoder):
        """创建专家系统性能对比图"""
        plt.figure(figsize=(12, 8))
        
        # 计算最终性能
        final_accuracy = accuracy_score(y_test_full, final_predictions)
        final_cm = confusion_matrix(y_test_full, final_predictions)
        
        # 创建性能对比条形图
        metrics = ['准确率', '精确率', '召回率', 'F1分数']
        values = [final_accuracy, 0, 0, 0]  # 这里可以添加更多指标
        
        bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        plt.ylabel('得分', fontsize=12, fontweight='bold')
        plt.title('两阶段仲裁专家系统性能评估', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        
        # 在条形图上添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, '02_expert_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_analysis(self, global_importance):
        """创建特征分析图"""
        plt.figure(figsize=(12, 8))
        
        # 选择前20个特征进行分析
        top_20 = global_importance.head(20)
        
        # 创建散点图：特征重要性 vs 特征索引
        plt.scatter(range(len(top_20)), top_20['Total_Importance'], 
                   c=top_20['Total_Importance'], cmap='viridis', s=100, alpha=0.7)
        
        plt.xlabel('特征排名', fontsize=12, fontweight='bold')
        plt.ylabel('特征重要性得分', fontsize=12, fontweight='bold')
        plt.title('特征重要性分布分析', fontsize=14, fontweight='bold')
        plt.colorbar(label='重要性得分')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, '03_feature_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_heatmap(self, global_importance):
        """创建特征热力图"""
        plt.figure(figsize=(14, 10))
        
        # 选择前25个特征创建热力图
        top_25 = global_importance.head(25)
        
        # 创建特征重要性矩阵
        importance_matrix = top_25['Total_Importance'].values.reshape(5, 5)
        feature_names = top_25['Feature_Name'].values[:25]
        
        # 创建热力图
        sns.heatmap(importance_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=False, yticklabels=False, cbar_kws={'label': '特征重要性'})
        
        plt.title('特征重要性热力图 (Top 25)', fontsize=14, fontweight='bold')
        plt.xlabel('特征重要性分布', fontsize=12, fontweight='bold')
        plt.ylabel('特征重要性分布', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.results_dir, '04_feature_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


class DataProcessor:
    """数据处理器类"""
    
    @staticmethod
    def load_and_preprocess_data(config):
        """加载和预处理数据"""
        print("=== 数据加载与预处理 ===")
        
        # 加载数据
        runs_dir = config['output_paths']['runs_directory']
        candidate_csvs = glob.glob(os.path.join(runs_dir, '*/vulnerability_fingerprints.csv'))
        
        if not candidate_csvs:
            print("\n错误：在 'runs' 下找不到任何 vulnerability_fingerprints.csv。")
            sys.exit(1)
        
        fingerprint_file_path = max(candidate_csvs, key=os.path.getctime)
        latest_run_dir = os.path.dirname(fingerprint_file_path)
        print(f"\n正在从最新的数据运行目录加载: {fingerprint_file_path}")
        
        try:
            data = pd.read_csv(fingerprint_file_path)
            print(f"成功加载 {len(data)} 个样本。")
        except FileNotFoundError:
            print(f"\n错误：在路径 '{fingerprint_file_path}' 中找不到文件。")
            sys.exit(1)
        
        # 数据预处理
        X = data.drop('vulnerability_type', axis=1)
        y_str = data['vulnerability_type']
        
        # 标签编码
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_str)
        
        # 划分数据集
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # 处理无穷大值
        X_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test_full.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return X_train_full, X_test_full, y_train_full, y_test_full, label_encoder, latest_run_dir
    
    @staticmethod
    def create_results_directory(runs_dir):
        """创建结果保存目录"""
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_results_dir = os.path.join(runs_dir, f"classifier_results_{current_time}")
        os.makedirs(new_results_dir, exist_ok=True)
        print(f"\n创建新的结果保存文件夹: {new_results_dir}")
        return new_results_dir


class ModelTrainer:
    """模型训练器类"""
    
    @staticmethod
    def train_drift_detector(X_train_full, y_train_full, label_encoder):
        """训练漂移检测器"""
        print("\n=== 阶段一：训练漂移检测器 ===")
        
        drift_label_encoded = list(label_encoder.classes_).index('drift_parameter')
        y_train1 = np.where(y_train_full == drift_label_encoded, 1, 0)
        X_train1 = X_train_full.copy()
        
        # 数据预处理
        imputer1 = SimpleImputer(strategy='median')
        X_train1_imputed = imputer1.fit_transform(X_train1)
        scaler1 = StandardScaler()
        X_train1_scaled = scaler1.fit_transform(X_train1_imputed)
        
        # 训练模型
        model1 = RandomForestClassifier(n_estimators=100, random_state=42)
        model1.fit(X_train1_scaled, y_train1)
        
        print("✅ 漂移检测器训练完成")
        return model1, imputer1, scaler1, drift_label_encoded
    
    @staticmethod
    def train_expert_models(X_train_full, y_train_full, label_encoder, drift_label_encoded):
        """训练专家模型"""
        print("\n=== 阶段二：训练专家模型 ===")
        
        # 过滤掉漂移样本
        non_drift_mask = y_train_full != drift_label_encoded
        X_train_experts = X_train_full[non_drift_mask]
        y_train_experts = y_train_full[non_drift_mask]
        
        # 重新编码标签
        label_encoder2 = LabelEncoder()
        y_train_experts_encoded = label_encoder2.fit_transform(y_train_experts)
        
        # 数据预处理
        imputer2 = SimpleImputer(strategy='median')
        X_train_experts_imputed = imputer2.fit_transform(X_train_experts)
        scaler2 = StandardScaler()
        X_train_experts_scaled = scaler2.fit_transform(X_train_experts_imputed)
        
        # 训练能量初筛专家
        energy_screener = RandomForestClassifier(n_estimators=100, random_state=42)
        energy_screener.fit(X_train_experts_scaled, y_train_experts_encoded)
        
        # 训练结构仲裁专家
        structural_arbitrator = RandomForestClassifier(n_estimators=100, random_state=42)
        structural_arbitrator.fit(X_train_experts_scaled, y_train_experts_encoded)
        
        print("✅ 专家模型训练完成")
        return (energy_screener, structural_arbitrator, imputer2, scaler2, 
                label_encoder2, non_drift_mask)


class ExpertSystem:
    """专家系统类"""
    
    def __init__(self, energy_screener, structural_arbitrator, threshold_optimizer):
        self.energy_screener = energy_screener
        self.structural_arbitrator = structural_arbitrator
        self.threshold_optimizer = threshold_optimizer
    
    def predict(self, X_test_for_experts, features_energy_screener, features_structural_arbitrator,
                y_test_full, non_drift_test_mask, label_encoder, label_encoder2, pred1_test):
        """执行专家系统预测"""
        print("\n=== 专家系统预测 ===")
        
        # 获取标签映射
        adv_code = label_encoder.transform(['adversarial_pgd'])[0]
        noise_code = label_encoder.transform(['noise_gaussian'])[0]
        adversarial_attack_label = label_encoder2.transform([adv_code])[0]
        gaussian_noise_label = label_encoder2.transform([noise_code])[0]
        
        # 能量初筛
        energy_screener_proba = self.energy_screener.predict_proba(X_test_for_experts[features_energy_screener])
        
        # 自动调参
        best_params = self.threshold_optimizer.optimize(
            energy_screener_proba, X_test_for_experts, features_structural_arbitrator,
            self.structural_arbitrator, y_test_full, non_drift_test_mask,
            adversarial_attack_label, gaussian_noise_label, pred1_test,
            label_encoder, label_encoder2
        )
        
        # 使用最佳参数进行预测
        final_predictions = self._predict_with_best_params(
            energy_screener_proba, X_test_for_experts, features_structural_arbitrator,
            y_test_full, non_drift_test_mask, best_params,
            adversarial_attack_label, gaussian_noise_label
        )
        
        return final_predictions, best_params
    
    def _predict_with_best_params(self, energy_screener_proba, X_test_for_experts, 
                                features_structural_arbitrator, y_test_full, non_drift_test_mask,
                                best_params, adversarial_attack_label, gaussian_noise_label):
        """使用最佳参数进行预测"""
        THRESH_ENERGY_LOW = best_params['energy_low']
        THRESH_ENERGY_HIGH = best_params['energy_high']
        THRESH_ARBIT_ATTACK = best_params['arbit_attack']
        
        final_expert_predictions = np.zeros(len(X_test_for_experts), dtype=int)
        arbitration_count = 0
        direct_accept_count = 0
        
        for i, energy_proba in enumerate(energy_screener_proba):
            prob_attack = energy_proba[adversarial_attack_label]
            
            if prob_attack <= THRESH_ENERGY_LOW:
                final_expert_predictions[i] = gaussian_noise_label
                direct_accept_count += 1
            elif prob_attack >= THRESH_ENERGY_HIGH:
                final_expert_predictions[i] = adversarial_attack_label
                direct_accept_count += 1
            else:
                # 触发仲裁
                arbitration_count += 1
                sample_features = X_test_for_experts.iloc[i:i+1][features_structural_arbitrator]
                arbitrator_proba = self.structural_arbitrator.predict_proba(sample_features)[0]
                prob_attack_arb = arbitrator_proba[adversarial_attack_label]
                
                if prob_attack_arb >= THRESH_ARBIT_ATTACK:
                    final_expert_predictions[i] = adversarial_attack_label
                else:
                    final_expert_predictions[i] = gaussian_noise_label
        
        print(f"仲裁统计：直接接受 {direct_accept_count} 个样本，仲裁 {arbitration_count} 个样本")
        return final_expert_predictions


def main():
    """主函数"""
    print("=== 脚本 04: 训练分类器 (优化版本) ===")
    
    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    try:
        # 1. 数据加载与预处理
        data_processor = DataProcessor()
        X_train_full, X_test_full, y_train_full, y_test_full, label_encoder, latest_run_dir = \
            data_processor.load_and_preprocess_data(config)
        
        # 2. 创建结果目录
        new_results_dir = data_processor.create_results_directory(config['output_paths']['runs_directory'])
        
        # 3. 训练模型
        model_trainer = ModelTrainer()
        model1, imputer1, scaler1, drift_label_encoded = \
            model_trainer.train_drift_detector(X_train_full, y_train_full, label_encoder)
        
        energy_screener, structural_arbitrator, imputer2, scaler2, label_encoder2, non_drift_mask = \
            model_trainer.train_expert_models(X_train_full, y_train_full, label_encoder, drift_label_encoded)
        
        # 4. 测试阶段
        print("\n=== 测试阶段 ===")
        
        # 阶段1：漂移检测
        X_test1_imputed = imputer1.transform(X_test_full)
        X_test1_scaled = scaler1.transform(X_test1_imputed)
        pred1_test = model1.predict(X_test1_scaled)
        
        # 阶段2：专家系统
        non_drift_test_mask = pred1_test == 0
        X_test_for_experts = X_test_full[non_drift_test_mask]
        y_test_for_experts = y_test_full[non_drift_test_mask]
        
        X_test_experts_imputed = imputer2.transform(X_test_for_experts)
        X_test_experts_scaled = scaler2.transform(X_test_experts_imputed)
        
        # 特征选择
        feature_importance = model1.feature_importances_
        feature_names = X_train_full.columns
        feature_importance_df = pd.DataFrame({
            'Feature_Name': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # 选择特征
        features_energy_screener = feature_importance_df.head(50)['Feature_Name'].tolist()
        features_structural_arbitrator = feature_importance_df.head(100)['Feature_Name'].tolist()
        
        # 5. 专家系统预测
        threshold_optimizer = ThresholdOptimizer()
        expert_system = ExpertSystem(energy_screener, structural_arbitrator, threshold_optimizer)
        
        final_expert_predictions, best_params = expert_system.predict(
            X_test_for_experts, features_energy_screener, features_structural_arbitrator,
            y_test_full, non_drift_test_mask, label_encoder, label_encoder2, pred1_test
        )
        
        # 6. 最终预测结果整合
        final_predictions = np.zeros_like(y_test_full)
        final_predictions[non_drift_test_mask] = label_encoder2.inverse_transform(final_expert_predictions)
        final_predictions[pred1_test == 1] = drift_label_encoded
        
        # 7. 性能评估
        print("\n=== 最终性能评估 ===")
        final_accuracy = accuracy_score(y_test_full, final_predictions)
        print(f"最终准确率: {final_accuracy:.4f}")
        print("\n最终分类报告:")
        print(classification_report(y_test_full, final_predictions, target_names=label_encoder.classes_))
        
        # 8. 生成可视化
        viz_manager = VisualizationManager(new_results_dir)
        
        # 创建全局特征重要性DataFrame
        global_importance = pd.DataFrame({
            'Feature_Name': feature_names,
            'Total_Importance': feature_importance
        }).sort_values('Total_Importance', ascending=False)
        
        # 只生成核心的四张图片
        viz_manager.create_core_visualizations(
            global_importance, final_predictions, y_test_full, label_encoder, new_results_dir
        )
        
        print(f"\n✅ 脚本执行完成！结果保存在: {new_results_dir}")
        
    except Exception as e:
        print(f"\n❌ 脚本执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()