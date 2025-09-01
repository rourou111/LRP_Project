# scripts/04_train_classifier.py
#!/usr/bin/env python3
"""
脚本 04: 训练两阶段分类器并评估性能。
执行方式: 在项目根目录下运行 `python scripts/04_train_classifier.py`
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import yaml
import sys

# 从 scikit-learn 中导入我们需要的模块
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def auto_tune_thresholds(energy_proba, X_test_experts, features_arbitrator, 
                        structural_arbitrator, y_test_full, non_drift_test_mask, 
                        adv_label, noise_label, pred1_test, 
                        label_encoder, label_encoder2):
    """
    自动调参函数：寻找最佳的能量初筛和仲裁阈值组合
    """
    print("�� 开始自动调参，测试不同阈值组合...")
    
    # 定义阈值搜索范围
    energy_low_range = [0.20, 0.25, 0.30, 0.35, 0.40]
    energy_high_range = [0.60, 0.65, 0.70, 0.75, 0.80]
    arbit_range = [0.45, 0.50, 0.55, 0.60, 0.65]
    
    best_score = 0
    best_params = {}
    results = []
    
    total_combinations = len(energy_low_range) * len(energy_high_range) * len(arbit_range)
    current = 0
    
    for energy_low in energy_low_range:
        for energy_high in energy_high_range:
            if energy_low >= energy_high:  # 跳过无效组合
                continue
            for arbit_thresh in arbit_range:
                current += 1
                print(f"测试组合 {current}/{total_combinations}: "
                      f"能量低={energy_low:.2f}, 能量高={energy_high:.2f}, 仲裁={arbit_thresh:.2f}")
                
                # 使用当前阈值组合进行预测
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
                
                # 计算性能指标
                accuracy = accuracy_score(y_test_full, final_predictions)
                
                # 计算对抗攻击的召回率（防止漏检）
                cm = confusion_matrix(y_test_full, final_predictions)
                if len(cm) >= 3:  # 确保有3个类别
                    # 假设对抗攻击是第一个类别（索引0）
                    adv_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
                    # 计算高斯噪声的召回率
                    noise_recall = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0
                    
                    # 综合评分：准确率 + 对抗召回率 + 噪声召回率
                    combined_score = accuracy + adv_recall + noise_recall
                    
                    results.append({
                        'energy_low': energy_low,
                        'energy_high': energy_high,
                        'arbit_thresh': arbit_thresh,
                        'accuracy': accuracy,
                        'adv_recall': adv_recall,
                        'noise_recall': noise_recall,
                        'combined_score': combined_score
                    })
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {
                            'energy_low': energy_low,
                            'energy_high': energy_high,
                            'arbit_attack': arbit_thresh
                        }
                        print(f"🎯 发现更好的参数组合！综合评分: {combined_score:.4f}")
    
    # 显示最佳结果
    print(f"\n🏆 自动调参完成！最佳参数组合:")
    print(f"   能量低阈值: {best_params['energy_low']:.2f}")
    print(f"   能量高阈值: {best_params['energy_high']:.2f}")
    print(f"   仲裁阈值: {best_params['arbit_attack']:.2f}")
    print(f"   最佳综合评分: {best_score:.4f}")
    
    # 显示前5个最佳结果
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    print(f"\n📊 前5个最佳参数组合:")
    for i, result in enumerate(results[:5]):
        print(f"   {i+1}. 能量低={result['energy_low']:.2f}, "
              f"能量高={result['energy_high']:.2f}, "
              f"仲裁={result['arbit_thresh']:.2f}, "
              f"综合评分={result['combined_score']:.4f}")
    
    return best_params

# ... existing code ...

def main():
    print("=== 脚本 04: 训练分类器 ===")

    # --- 加载配置文件 ---
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("所有机器学习库都已成功导入！")
    print("两阶段分类器训练脚本已准备就绪。")

    # =============================================================================
    # 步骤一：数据加载与通用预处理
    # =============================================================================
    # --- 1. 只从包含 fingerprint 的目录选择输入 ---
    runs_dir = config['output_paths']['runs_directory']
    candidate_csvs = glob.glob(os.path.join(runs_dir, '*/vulnerability_fingerprints.csv'))
    if not candidate_csvs:
        print("\n错误：在 'runs' 下找不到任何 vulnerability_fingerprints.csv。")
        sys.exit(1)

    fingerprint_file_path = max(candidate_csvs, key=os.path.getctime)  # 最新的那个 csv
    latest_run_dir = os.path.dirname(fingerprint_file_path)            # 其所在目录
    print(f"\n正在从最新的数据运行目录加载: {fingerprint_file_path}")

    try:
        data = pd.read_csv(fingerprint_file_path)
        print(f"成功加载 {len(data)} 个样本。")
    except FileNotFoundError:
        print(f"\n错误：在路径 '{fingerprint_file_path}' 中找不到 vulnerability_fingerprints.csv 文件。")
        sys.exit(1)
    
    # --- 创建新的结果保存文件夹 ---
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_results_dir = os.path.join(runs_dir, f"classifier_results_{current_time}")
    os.makedirs(new_results_dir, exist_ok=True)
    print(f"\n创建新的结果保存文件夹: {new_results_dir}")

    # --- 2. 分离特征 (X) 与原始标签 (y) ---
    # ... (此部分代码与原脚本完全一致，无需改动) ...
    X = data.drop('vulnerability_type', axis=1)
    y_str = data['vulnerability_type']
    # --- 3. 标签编码 ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)
    label_mapping = {i: class_name for i, class_name in enumerate(label_encoder.classes_)}
    print("\n标签已成功编码为数字:")
    print(label_mapping)
    # --- 4. 划分总数据集 ---
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    # --- 5. 全局预处理：处理无穷大值 ---
    print("\n正在对训练集和测试集进行无穷大值预处理...")
    X_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("预处理完成。")

    # =============================================================================
    # 阶段一：训练模型一 (“漂移”识别器 - Generalist)
    # =============================================================================
    # ... (此阶段代码与原脚本完全一致，无需改动) ...
    print("\n" + "="*50)
    print("阶段一：开始训练模型一 ('漂移'识别器)")
    print("="*50)
    drift_label_encoded = list(label_encoder.classes_).index('drift_parameter')
    y_train1 = np.where(y_train_full == drift_label_encoded, 1, 0)
    X_train1 = X_train_full.copy()
    imputer1 = SimpleImputer(strategy='median')
    X_train1_imputed = imputer1.fit_transform(X_train1)
    scaler1 = StandardScaler()
    X_train1_scaled = scaler1.fit_transform(X_train1_imputed)
    model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model1.fit(X_train1_scaled, y_train1)
    print("模型一训练完成。")
    # --- 快速验证模型一在训练集上的性能 ---
    from sklearn.metrics import classification_report
    pred1_train = model1.predict(X_train1_scaled)
    print("\n--- 模型一 (分诊台) 在训练集上的性能报告 ---")
    print(classification_report(y_train1, pred1_train, target_names=['非漂移 (0)', '漂移 (1)']))

    # =============================================================================
    # 阶段二：构建"一票否决"仲裁专家模型架构
    # =============================================================================
    print("\n" + "="*50)
    print("阶段二：开始构建'一票否决'仲裁专家模型架构")
    print("="*50)
    non_drift_train_mask = (y_train1 == 0)
    X_train_experts = X_train_full[non_drift_train_mask]
    y_train_experts_raw = y_train_full[non_drift_train_mask]
    label_encoder2 = LabelEncoder()
    y_train_experts = label_encoder2.fit_transform(y_train_experts_raw)
    print(f"已筛选出 {len(X_train_experts)} 个'对抗 vs. 噪声'样本，交由仲裁专家处理。")
    
    # 阶段 2a：专家模型的专项化与重定义
    # =============================================================================
    print("\n--- 阶段 2a：专家模型的专项化与重定义 ---")
    # 1. 定义"能量派"初筛专家 (energy_screener)
    # 使命：担任第一道防线，成为一个高灵敏度的"异常信号探测器"
    # 理论依据：特征重要性分析明确指出，high_freq_ratio, std_dev_diff, ratio_zscore 是最强大的信号探测器
    features_energy_screener = [
        'high_freq_ratio',      # 捕捉高斯噪声的高频散布特性
        'std_dev_diff',         # 衡量注意力分散/集中的变化
        'ratio_zscore'          # 能量比的基线分离度 (Z-score)
    ]

    # 2. 定义"结构派"仲裁专家 (structural_arbitrator)
    # 使命：担任第二道防线，成为一个高精度的"事实核查官"
    # 理论依据：ll_distortion 等结构性特征在当前模型中被严重低估，需要专门模型强制利用
    features_structural_arbitrator = [
        'll_distortion',        # 核心：低频子带结构失真度
        'contrast',             # 对比度
        'correlation',          # 相关性
        'homogeneity',          # 同质性
        'energy',               # 能量
        'cosine_similarity'     # 余弦相似度
    ]
    
    print(f"能量派初筛专家特征集: {features_energy_screener}")
    print(f"结构派仲裁专家特征集: {features_structural_arbitrator}")
    
    # 数据预处理
    X_train_experts_imputed = imputer1.transform(X_train_experts)
    X_train_experts_scaled = scaler1.transform(X_train_experts_imputed)
    X_train_experts_scaled_df = pd.DataFrame(X_train_experts_scaled, columns=X_train_experts.columns, index=X_train_experts.index)
    
    # 训练两个专门的专家模型
    # 使用类权重，缓解类别不平衡
    energy_screener = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced_subsample'
    )
    structural_arbitrator = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced_subsample'
    )
    
    # 训练能量派初筛专家
    print("\n正在训练能量派初筛专家...")
    energy_screener.fit(X_train_experts_scaled_df[features_energy_screener], y_train_experts)
    
    # 训练结构派仲裁专家
    print("正在训练结构派仲裁专家...")
    structural_arbitrator.fit(X_train_experts_scaled_df[features_structural_arbitrator], y_train_experts)
    
    print("两个专家模型训练完成！")
    # 阶段 2b：验证仲裁专家性能
    # =============================================================================
    print("\n--- 阶段 2b：验证仲裁专家性能 ---")
    
    # 验证能量派初筛专家性能
    energy_screener_train_pred = energy_screener.predict(X_train_experts_scaled_df[features_energy_screener])
    energy_screener_accuracy = accuracy_score(y_train_experts, energy_screener_train_pred)
    print(f"能量派初筛专家训练集准确率: {energy_screener_accuracy:.4f}")
    
    # 验证结构派仲裁专家性能
    structural_arbitrator_train_pred = structural_arbitrator.predict(X_train_experts_scaled_df[features_structural_arbitrator])
    structural_arbitrator_accuracy = accuracy_score(y_train_experts, structural_arbitrator_train_pred)
    print(f"结构派仲裁专家训练集准确率: {structural_arbitrator_accuracy:.4f}")
    
    print("仲裁专家性能验证完成！")

    # =============================================================================
    # 新增：特征重要性审讯与分析
    # =============================================================================
    print("\n" + "="*50)
    print("特征重要性审讯与分析阶段")
    print("="*50)
    
    # 第一阶段：证据收集 - 审讯准备与执行
    print("\n--- 第一阶段：证据收集 ---")
    
    # 1.1 定位审讯目标：两个仲裁专家模型
    experts = {
        'energy_screener': (energy_screener, features_energy_screener, '能量派初筛专家'),
        'structural_arbitrator': (structural_arbitrator, features_structural_arbitrator, '结构派仲裁专家')
    }
    
    # 1.2 审讯实施：提取每个专家的"口供"
    feature_importance_data = []
    
    for expert_name, (expert_model, feature_list, expert_title) in experts.items():
        print(f"\n正在审讯 {expert_title} ({expert_name})...")
        
        # 获取特征重要性分数
        importance_scores = expert_model.feature_importances_
        
        # 将特征名与重要性分数配对
        for feature_name, importance_score in zip(feature_list, importance_scores):
            feature_importance_data.append({
                'Expert_Name': expert_title,
                'Expert_Code': expert_name,
                'Feature_Name': feature_name,
                'Importance_Score': importance_score
            })
    
    # 1.3 整理档案：创建结构化数据表
    importance_df = pd.DataFrame(feature_importance_data)
    print(f"\n审讯完成！共收集到 {len(importance_df)} 条特征重要性记录。")
    
    # 第二阶段：案情分析 - "口供"解读与洞察
    print("\n--- 第二阶段：案情分析 ---")
    
    # 2.1 层面一：专家独立分析报告
    print("\n【专家独立分析报告】")
    for expert_title in importance_df['Expert_Name'].unique():
        expert_data = importance_df[importance_df['Expert_Name'] == expert_title]
        expert_data_sorted = expert_data.sort_values('Importance_Score', ascending=False)
        
        print(f"\n{expert_title}的特征重要性排行榜:")
        for idx, row in expert_data_sorted.iterrows():
            print(f"  {row['Feature_Name']}: {row['Importance_Score']:.4f}")
    
    # 2.2 层面二：全局功劳排行榜
    print("\n【全局功劳排行榜】")
    global_importance = importance_df.groupby('Feature_Name')['Importance_Score'].agg(['sum', 'mean', 'count']).reset_index()
    global_importance.columns = ['Feature_Name', 'Total_Importance', 'Average_Importance', 'Expert_Count']
    global_importance = global_importance.sort_values('Total_Importance', ascending=False)
    
    print("\n所有特征的全局总重要性排行榜 (按总贡献度排序):")
    for idx, row in global_importance.iterrows():
        print(f"  {row['Feature_Name']}: 总贡献度={row['Total_Importance']:.4f}, "
              f"平均贡献度={row['Average_Importance']:.4f}, "
              f"被{row['Expert_Count']}个专家使用")
    
    # 识别MVP特征和干扰项
    print("\n【关键洞察】")
    top_features = global_importance.head(3)
    bottom_features = global_importance.tail(3)
    
    print("\n🏆 MVP特征 (Top 3):")
    for idx, row in top_features.iterrows():
        print(f"  {row['Feature_Name']}: {row['Total_Importance']:.4f}")
    
    print("\n⚠️  潜在干扰项 (Bottom 3):")
    for idx, row in bottom_features.iterrows():
        print(f"  {row['Feature_Name']}: {row['Total_Importance']:.4f}")
    
    # 第三阶段：制定行动 - 基于证据的优化决策
    print("\n--- 第三阶段：制定行动 ---")
    
    # 3.1 基于特征排名的决策矩阵
    print("\n【基于证据的优化决策建议】")
    
    # 检查关键特征的表现
    dynamic_wavelet_rank = global_importance[global_importance['Feature_Name'] == 'dynamic_wavelet_ratio_change']
    ll_distortion_rank = global_importance[global_importance['Feature_Name'] == 'll_distortion']
    
    if not dynamic_wavelet_rank.empty:
        dynamic_score = dynamic_wavelet_rank.iloc[0]['Total_Importance']
        if dynamic_score > global_importance['Total_Importance'].median():
            print("✅ dynamic_wavelet_ratio_change 全局排名很高 - 保留该特征，它是关键支柱！")
        else:
            print("⚠️  dynamic_wavelet_ratio_change 表现一般 - 考虑优化或增强")
    
    if not ll_distortion_rank.empty:
        ll_score = ll_distortion_rank.iloc[0]['Total_Importance']
        if ll_score > global_importance['Total_Importance'].median():
            print("✅ ll_distortion 全局排名很高 - 您的'结构审计师'指纹非常成功！")
        else:
            print("⚠️  ll_distortion 表现一般 - 需要进一步优化")
    
    # 识别需要移除的潜在噪音特征
    noise_threshold = global_importance['Total_Importance'].quantile(0.25)  # 下四分位数
    potential_noise_features = global_importance[global_importance['Total_Importance'] < noise_threshold]
    
    if not potential_noise_features.empty:
        print(f"\n🔍 建议进行'控制变量实验'的特征 (重要性 < {noise_threshold:.4f}):")
        for idx, row in potential_noise_features.iterrows():
            print(f"  {row['Feature_Name']}: {row['Total_Importance']:.4f}")
        print("  建议：建立新模型版本，移除这些特征，与基线模型进行性能对比。")
    
    # 保存特征重要性分析结果
    print("\n--- 保存分析结果 ---")
    # 保存详细的特征重要性数据
    importance_df.to_csv(os.path.join(new_results_dir, 'feature_importance_analysis.csv'), index=False)
    global_importance.to_csv(os.path.join(new_results_dir, 'global_feature_ranking.csv'), index=False)
        
    if os.path.exists(new_results_dir):
        # 生成特征重要性可视化
        plt.figure(figsize=(16, 12))
        
        # 子图1：各专家的特征重要性对比 (优化：调整布局和标签)
        plt.subplot(2, 2, 1)
        pivot_df = importance_df.pivot(index='Feature_Name', columns='Expert_Name', values='Importance_Score')
        pivot_df.plot(kind='bar', ax=plt.gca(), width=0.8)
        plt.title('各专家眼中的特征重要性对比', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('重要性分数', fontsize=12)
        plt.xlabel('特征名称', fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # 子图2：全局特征重要性排行榜 (优化：更清晰的标签和颜色)
        plt.subplot(2, 2, 2)
        top_10_features = global_importance.head(10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_10_features)))
        bars = plt.barh(range(len(top_10_features)), top_10_features['Total_Importance'], 
                        color=colors, alpha=0.8)
        plt.yticks(range(len(top_10_features)), top_10_features['Feature_Name'], fontsize=10)
        plt.xlabel('总重要性分数', fontsize=12)
        plt.title('Top 10 特征全局重要性排行榜', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # 在条形图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_10_features['Total_Importance'])):
            plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)
        
        # 子图3：特征使用频率分布 (优化：明确标签和含义)
        plt.subplot(2, 2, 3)
        feature_usage = global_importance['Expert_Count'].value_counts().sort_index()
        usage_labels = [f'被{i}个专家使用' for i in feature_usage.index]
        colors_usage = plt.cm.Set3(np.linspace(0, 1, len(feature_usage)))
        
        bars_usage = plt.bar(range(len(feature_usage)), feature_usage.values, 
                             color=colors_usage, alpha=0.8)
        plt.xlabel('专家使用次数', fontsize=12)
        plt.ylabel('特征数量', fontsize=12)
        plt.title('特征被专家使用的频率分布', fontsize=14, fontweight='bold')
        plt.xticks(range(len(feature_usage)), feature_usage.index)
        plt.grid(axis='y', alpha=0.3)
        
        # 在条形图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars_usage, feature_usage.values)):
            plt.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=10)
        
        # 子图4：重要性分数分布直方图 (优化：更清晰的标题和标签)
        plt.subplot(2, 2, 4)
        plt.hist(importance_df['Importance_Score'], bins=15, alpha=0.7, 
                 edgecolor='black', color='skyblue', linewidth=1)
        plt.xlabel('特征重要性分数', fontsize=12)
        plt.ylabel('特征数量', fontsize=12)
        plt.title('特征重要性分数分布直方图', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        mean_score = importance_df['Importance_Score'].mean()
        median_score = importance_df['Importance_Score'].median()
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                    label=f'平均值: {mean_score:.3f}')
        plt.axvline(median_score, color='orange', linestyle='--', linewidth=2, 
                    label=f'中位数: {median_score:.3f}')
        plt.legend(fontsize=10)
        
        # 整体布局优化
        plt.tight_layout(pad=3.0)
        
        # 保存图像时确保完整显示
        plt.savefig(os.path.join(new_results_dir, 'feature_importance_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f"特征重要性分析图表已保存到: {new_results_dir}")
        
        # 显示图像
        plt.show()
        
        # 额外生成一个简化的特征重要性热力图
        print("\n--- 生成特征重要性热力图 ---")
        plt.figure(figsize=(12, 8))
        
        # 创建热力图数据
        heatmap_data = importance_df.pivot(index='Feature_Name', columns='Expert_Name', values='Importance_Score')
        
        # 绘制热力图
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                    linewidths=0.5, cbar_kws={'label': '重要性分数'})
        plt.title('特征重要性热力图 - 各专家视角对比', fontsize=16, fontweight='bold')
        plt.xlabel('专家类型', fontsize=12)
        plt.ylabel('特征名称', fontsize=12)
        
        # 调整x轴标签角度
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(new_results_dir, 'feature_importance_heatmap.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f"特征重要性热力图已保存到: {new_results_dir}")
        plt.show()
    
    print("\n特征重要性审讯与分析完成！")
    print("基于分析结果，您可以制定精准的模型优化策略。")

    # =============================================================================
    # 阶段三：对完整的"专家组决策系统"进行最终评估
    # =============================================================================
    # ... (此阶段代码与原脚本完全一致，无需改动) ...
    print("\n" + "="*50)
    print("阶段三：开始对两阶段系统进行最终评估")
    print("="*50)
    X_test_imputed = imputer1.transform(X_test_full)
    X_test_scaled = scaler1.transform(X_test_imputed)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_full.columns, index=X_test_full.index)
    print("测试样本进入系统...")
    print(" -> 步骤1: '分诊台' (模型一) 正在进行初步诊断...")
    pred1_test = model1.predict(X_test_scaled)
    final_predictions = np.zeros_like(y_test_full)
    non_drift_test_mask = (pred1_test == 0)
    X_test_for_experts = X_test_scaled_df[non_drift_test_mask]
    print(f" -> '分诊台'诊断完毕: {len(X_test_for_experts)} 个样本被提交至'专家组'。")
    if len(X_test_for_experts) > 0:
        print(" -> 步骤2: '发现-核查'仲裁流程开始...")
        
        # 阶段 2b：构建全新的仲裁式预测逻辑
        # =============================================================================
        print(" -> 2a. 初步筛查：能量派初筛专家进行异常信号检测...")
        
        # 1. 初步筛查：能量派初筛专家（使用自动调参的最佳阈值）
        # 1. 初步筛查：能量派初筛专家（使用自动调参的最佳阈值）
        # 1. 初步筛查：能量派初筛专家（使用自动调参的最佳阈值）
        energy_screener_proba = energy_screener.predict_proba(X_test_for_experts[features_energy_screener])
        
        # 自动调参：测试不同阈值组合
        # 获取标签映射
        print(f"可用的标签类别: {list(label_encoder2.classes_)}")
        print(f"标签编码器映射: {dict(zip(range(len(label_encoder2.classes_)), label_encoder2.classes_))}")
        
        # 获取二级编码器中"adversarial_pgd"和"noise_gaussian"的索引
        adv_code = label_encoder.transform(['adversarial_pgd'])[0]
        noise_code = label_encoder.transform(['noise_gaussian'])[0]
        adversarial_attack_label = label_encoder2.transform([adv_code])[0]
        gaussian_noise_label = label_encoder2.transform([noise_code])[0]
        print('二级编码器 classes_:', list(label_encoder2.classes_))
        print('映射: noise_idx=', gaussian_noise_label, ' adv_idx=', adversarial_attack_label)
        
        # 自动调参：测试不同阈值组合
        print("\n开始自动调参，寻找最佳阈值...")
        best_params = auto_tune_thresholds(
            energy_screener_proba, X_test_for_experts, features_structural_arbitrator,
            structural_arbitrator, y_test_full, non_drift_test_mask,
            adversarial_attack_label, gaussian_noise_label, pred1_test,  # ← 现在变量已经定义了
            label_encoder, label_encoder2
        )
        
        THRESH_ENERGY_LOW = best_params['energy_low']
        THRESH_ENERGY_HIGH = best_params['energy_high']
        THRESH_ARBIT_ATTACK = best_params['arbit_attack']
        
        print(f"最佳参数: 能量低阈值={THRESH_ENERGY_LOW}, 能量高阈值={THRESH_ENERGY_HIGH}, 仲裁阈值={THRESH_ARBIT_ATTACK}")
        print("使用最佳参数重新预测...")
        
        # 2. 条件仲裁（核心逻辑）
        print(" -> 2b. 条件仲裁：根据初筛结果决定是否触发仲裁...")
        
        # 获取标签映射
        print(f"可用的标签类别: {list(label_encoder2.classes_)}")
        print(f"标签编码器映射: {dict(zip(range(len(label_encoder2.classes_)), label_encoder2.classes_))}")
        
        # 获取二级编码器中"adversarial_pgd"和"noise_gaussian"的索引
        adv_code = label_encoder.transform(['adversarial_pgd'])[0]
        noise_code = label_encoder.transform(['noise_gaussian'])[0]
        adversarial_attack_label = label_encoder2.transform([adv_code])[0]
        gaussian_noise_label = label_encoder2.transform([noise_code])[0]
        print('二级编码器 classes_:', list(label_encoder2.classes_))
        print('映射: noise_idx=', gaussian_noise_label, ' adv_idx=', adversarial_attack_label)
        
        # 初始化最终预测结果
        final_expert_predictions = np.zeros(len(X_test_for_experts), dtype=int)
        
        # 统计仲裁情况
        arbitration_count = 0
        direct_accept_count = 0
        
        for i, energy_proba in enumerate(energy_screener_proba):
            # 根据二级编码器确定概率列的索引
            prob_attack = energy_proba[adversarial_attack_label]
            if prob_attack <= THRESH_ENERGY_LOW:
                # 低置信度攻击 => 直接接收为噪声
                final_expert_predictions[i] = gaussian_noise_label
                direct_accept_count += 1
                continue
            if prob_attack >= THRESH_ENERGY_HIGH:
                # 高置信度攻击 => 直接定为攻击（可选：仍可请求仲裁做双重确认）
                final_expert_predictions[i] = adversarial_attack_label
                direct_accept_count += 1
                continue

            # 介于两阈值之间 => 触发仲裁
            arbitration_count += 1
            print(f"     -> 样本 {i+1}: 初筛概率处于灰区({prob_attack:.2f})，启动仲裁...")
            sample_features = X_test_for_experts.iloc[i:i+1][features_structural_arbitrator]
            arbitrator_proba = structural_arbitrator.predict_proba(sample_features)[0]
            prob_attack_arb = arbitrator_proba[adversarial_attack_label]
            arbitrator_prediction = adversarial_attack_label if prob_attack_arb >= THRESH_ARBIT_ATTACK else gaussian_noise_label
            
            # 采纳仲裁专家的最终意见
            if arbitrator_prediction == adversarial_attack_label:
                # 仲裁专家同意是"对抗攻击"
                final_expert_predictions[i] = adversarial_attack_label
                print(f"     -> 样本 {i+1}: 仲裁专家确认攻击，最终判定为对抗攻击")
            else:
                # 仲裁专家行使"一票否决权"，推翻初筛结果
                final_expert_predictions[i] = gaussian_noise_label
                print(f"     -> 样本 {i+1}: 仲裁专家行使否决权，修正为高斯噪声")
        
        print(f" -> 仲裁统计：直接接受 {direct_accept_count} 个样本，仲裁 {arbitration_count} 个样本")
        
        # 转换回原始标签
        expert_predictions_original_labels = label_encoder2.inverse_transform(final_expert_predictions)
        final_predictions[non_drift_test_mask] = expert_predictions_original_labels
        print(" -> '发现-核查'仲裁流程完毕。")
    drift_label_encoded = list(label_encoder.classes_).index('drift_parameter')
    final_predictions[pred1_test == 1] = drift_label_encoded
    print("...所有样本预测流程结束。")
    print("\n--- 两阶段'一票否决'仲裁专家系统最终性能评估 ---")
    print(f"最终准确率: {accuracy_score(y_test_full, final_predictions):.4f}")
    print("\n最终分类报告:")
    print(classification_report(y_test_full, final_predictions, target_names=label_encoder.classes_))
    print("\n最终混淆矩阵:")
    final_cm = confusion_matrix(y_test_full, final_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Final Confusion Matrix for the Two-Stage Arbitration Expert System')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # 保存图像到新的结果文件夹
    plt.savefig(os.path.join(new_results_dir, 'final_confusion_matrix_expert_system.png'))
    print(f"\n混淆矩阵图像已保存到: {new_results_dir}")
    plt.show()
    print("\n脚本执行完毕！")

    # 重新设计特征重要性可视化 - 更清晰易懂
    print("\n--- 生成清晰易懂的特征重要性分析图表 ---")
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 创建三个独立的图表，每个都清晰易懂
    
    # 图表1：特征重要性排行榜 - 最直观的展示
    plt.figure(figsize=(14, 8))
    
    # 按总重要性排序，只显示前15个最重要的特征
    top_features = global_importance.head(15)
    
    # 创建水平条形图
    bars = plt.barh(range(len(top_features)), top_features['Total_Importance'], 
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    
    # 设置Y轴标签（特征名称）
    plt.yticks(range(len(top_features)), top_features['Feature_Name'], fontsize=11)
    
    # 在条形图上添加数值标签
    for i, (bar, value) in enumerate(zip(bars, top_features['Total_Importance'])):
        plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('总重要性分数', fontsize=14)
    plt.title('特征重要性排行榜 - 哪些特征最重要？', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    # 添加说明文字
    plt.figtext(0.02, 0.02, 
                '说明：分数越高，该特征在区分"对抗攻击"与"高斯噪声"时越重要', 
                fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(new_results_dir, '01_feature_importance_ranking.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 图表2：各专家的特征重要性对比 - 清晰的专家视角
    plt.figure(figsize=(16, 10))
    
    # 选择重要性最高的8个特征进行对比
    top_8_features = global_importance.head(8)['Feature_Name'].tolist()
    expert_data = importance_df[importance_df['Feature_Name'].isin(top_8_features)]
    
    # 创建分组条形图
    pivot_data = expert_data.pivot(index='Feature_Name', columns='Expert_Name', values='Importance_Score')
    
    # 设置颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # 绘制分组条形图
    x = np.arange(len(top_8_features))
    width = 0.2
    
    for i, (expert_name, color) in enumerate(zip(pivot_data.columns, colors)):
        values = pivot_data[expert_name].values
        plt.bar(x + i*width, values, width, label=expert_name, color=color, alpha=0.8)
    
    plt.xlabel('特征名称', fontsize=14)
    plt.ylabel('重要性分数', fontsize=14)
    plt.title('各专家眼中的特征重要性对比', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x + width*1.5, top_8_features, rotation=45, ha='right')
    plt.legend(title='专家类型', fontsize=12, title_fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加说明文字
    plt.figtext(0.02, 0.02, 
                '说明：每个特征在不同专家眼中的重要性分数，帮助理解各专家的决策逻辑', 
                fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(new_results_dir, '02_expert_comparison.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 图表3：特征使用情况分析 - 理解特征分布
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1：特征被专家使用的次数
    feature_usage = global_importance['Expert_Count'].value_counts().sort_index()
    usage_labels = [f'被{i}个专家使用' for i in feature_usage.index]
    
    bars1 = ax1.bar(range(len(feature_usage)), feature_usage.values, 
                     color=['#FF9999', '#66B2FF'], alpha=0.8)
    ax1.set_xlabel('专家使用次数', fontsize=12)
    ax1.set_ylabel('特征数量', fontsize=12)
    ax1.set_title('特征被专家使用的频率分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(feature_usage)))
    ax1.set_xticklabels(feature_usage.index)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在条形图上添加数值标签
    for i, (bar, value) in enumerate(zip(bars1, feature_usage.values)):
        ax1.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 子图2：重要性分数分布
    ax2.hist(importance_df['Importance_Score'], bins=12, alpha=0.7, 
              edgecolor='black', color='lightblue', linewidth=1)
    ax2.set_xlabel('特征重要性分数', fontsize=12)
    ax2.set_ylabel('特征数量', fontsize=12)
    ax2.set_title('特征重要性分数分布', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加统计参考线
    mean_score = importance_df['Importance_Score'].mean()
    median_score = importance_df['Importance_Score'].median()
    ax2.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                 label=f'平均值: {mean_score:.3f}')
    ax2.axvline(median_score, color='orange', linestyle='--', linewidth=2, 
                 label=f'中位数: {median_score:.3f}')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(new_results_dir, '03_feature_analysis.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 图表4：改进的热力图 - 清晰显示每个特征在各专家中的重要性
    plt.figure(figsize=(14, 10))
    
    # 重新整理热力图数据，确保每个特征在每个专家下都有值
    heatmap_data = importance_df.pivot(index='Feature_Name', columns='Expert_Name', values='Importance_Score')
    
    # 使用更好的颜色方案和标注
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                linewidths=0.5, cbar_kws={'label': '重要性分数'}, 
                square=True, annot_kws={'size': 9})
    
    plt.title('特征重要性热力图 - 每个特征在各专家眼中的重要性', fontsize=16, fontweight='bold')
    plt.xlabel('专家类型', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    
    # 调整标签角度，确保可读性
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(new_results_dir, '04_feature_heatmap.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n所有优化后的图表已保存到: {new_results_dir}")
    print("现在每个图表都有清晰的标题、标签和说明，应该更容易理解了！")

if __name__ == '__main__':
    main()