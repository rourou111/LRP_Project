#!/usr/bin/env python3
"""
脚本 04: 训练优化的两阶段漏洞检测分类器。
执行方式: 在项目根目录下运行 `python scripts/04_train_classifier.py`

核心优化: 从"全科医生"升级为"专科医生"，专门解决"对抗攻击 vs 高斯噪声"的混淆问题
"""

import os
import pickle
import pandas as pd
import numpy as np
import yaml
import sys
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def clean_data(df):
    """
    清理数据中的无穷大值、NaN值和异常值
    """
    print("--- 数据清理 ---")
    
    # 检查原始数据状态
    print(f"原始数据形状: {df.shape}")
    print(f"无穷大值数量: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"NaN值数量: {df.isnull().sum().sum()}")
    
    # 复制数据框
    df_clean = df.copy()
    
    # 处理无穷大值：用该列的最大有限值替换
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col == 'vulnerability_type':  # 跳过标签列
            continue
            
        # 获取该列的最大有限值
        finite_values = df_clean[col][np.isfinite(df_clean[col])]
        if len(finite_values) > 0:
            max_finite = finite_values.max()
            min_finite = finite_values.min()
            
            # 替换正无穷大
            df_clean[col] = df_clean[col].replace([np.inf], max_finite * 1.1)
            # 替换负无穷大
            df_clean[col] = df_clean[col].replace([-np.inf], min_finite * 1.1)
    
    # 处理NaN值：用该列的中位数替换
    for col in numeric_columns:
        if col == 'vulnerability_type':  # 跳过标签列
            continue
            
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
    
    # 检查清理后的数据状态
    print(f"清理后数据形状: {df_clean.shape}")
    print(f"清理后无穷大值数量: {np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"清理后NaN值数量: {df_clean.isnull().sum().sum()}")
    
    return df_clean

def main():
    print("=== 脚本 04: 训练优化的两阶段漏洞检测分类器 ===")
    print("核心优化: 专科医生模式 - 精准区分对抗攻击与高斯噪声")
    
    # --- 加载配置文件 ---
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # --- 自动寻找并加载最新的指纹数据 ---
    runs_dir = config['output_paths']['runs_directory']
    list_of_run_dirs = glob.glob(os.path.join(runs_dir, '*/'))
    if not list_of_run_dirs:
        print("\n错误：在 'runs' 文件夹下找不到任何运行记录。")
        print("请确保您已经成功运行了 '03_extract_fingerprints.py' 脚本。")
        sys.exit(1)
    
    latest_run_dir = max(list_of_run_dirs, key=os.path.getctime)
    fingerprints_file_path = os.path.join(latest_run_dir, 'vulnerability_fingerprints.csv')
    print(f"\n正在从最新的运行记录中加载数据: {fingerprints_file_path}")
    
    try:
        fingerprints_df = pd.read_csv(fingerprints_file_path)
        print(f"成功加载 {len(fingerprints_df)} 条指纹数据。")
    except FileNotFoundError:
        print(f"\n错误：在路径 '{fingerprints_file_path}' 中找不到 vulnerability_fingerprints.csv 文件。")
        sys.exit(1)
    
    # --- 数据清理 ---
    fingerprints_df = clean_data(fingerprints_df)
    
    # --- 数据预处理 ---
    print("\n--- 数据预处理 ---")
    
    # 检查数据完整性
    print(f"数据类型分布:")
    print(fingerprints_df['vulnerability_type'].value_counts())
    
    # 检查特征列
    feature_columns = [col for col in fingerprints_df.columns if col != 'vulnerability_type']
    print(f"\n可用特征数量: {len(feature_columns)}")
    print(f"特征列表: {feature_columns}")
    
    # --- 第一阶段：分诊台模型 (保持不变) ---
    print("\n--- 第一阶段：分诊台模型训练 ---")
    print("目标：快速分离参数漂移样本")
    
    # 准备第一阶段数据
    stage1_data = fingerprints_df.copy()
    stage1_data['is_parameter_drift'] = (stage1_data['vulnerability_type'] == 'drift_parameter').astype(int)
    
    # 第一阶段特征：使用所有特征
    stage1_features = feature_columns
    X_stage1 = stage1_data[stage1_features]
    y_stage1 = stage1_data['is_parameter_drift']
    
    # 训练第一阶段模型
    model1 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # 交叉验证评估
    cv_scores_stage1 = cross_val_score(model1, X_stage1, y_stage1, cv=5)
    print(f"第一阶段交叉验证准确率: {cv_scores_stage1.mean():.4f} (+/- {cv_scores_stage1.std() * 2:.4f})")
    
    # 训练最终模型
    model1.fit(X_stage1, y_stage1)
    
    # --- 第二阶段：专科专家系统 (核心优化) ---
    print("\n--- 第二阶段：专科专家系统训练 ---")
    print("目标：精准区分对抗攻击与高斯噪声")
    
    # 准备第二阶段数据：只包含非参数漂移的样本
    stage2_data = fingerprints_df[fingerprints_df['vulnerability_type'] != 'drift_parameter'].copy()
    stage2_data = stage2_data[stage2_data['vulnerability_type'].isin(['adversarial_pgd', 'noise_gaussian'])]
    
    print(f"第二阶段样本数量: {len(stage2_data)}")
    print(f"第二阶段数据类型分布:")
    print(stage2_data['vulnerability_type'].value_counts())
    
    # 核心优化：专科专家特征集
    # 只使用对区分"对抗攻击 vs 高斯噪声"最关键的4个特征
    stage2_core_features = [
        'll_distortion',           # 指纹二：低频子带结构失真度
        'ratio_zscore',            # 指纹三：能量比Z-score
        'super_fingerprint',       # 超级指纹：结构加权能量
        'high_freq_ratio'          # 辅助特征：静态高频能量比
    ]
    
    print(f"\n专科专家核心特征集:")
    for i, feature in enumerate(stage2_core_features, 1):
        print(f"  {i}. {feature}")
    
    # 准备第二阶段数据
    X_stage2_core = stage2_data[stage2_core_features]
    y_stage2 = (stage2_data['vulnerability_type'] == 'adversarial_pgd').astype(int)  # 1=对抗攻击, 0=高斯噪声
    
    # 数据标准化
    scaler = StandardScaler()
    X_stage2_core_scaled = scaler.fit_transform(X_stage2_core)
    
    # 训练专科专家模型
    specialist_model = RandomForestClassifier(
        n_estimators=200,          # 增加树的数量
        max_depth=15,              # 限制深度，防止过拟合
        min_samples_split=10,      # 增加分裂阈值
        min_samples_leaf=5,        # 增加叶节点最小样本数
        random_state=42,
        n_jobs=-1
    )
    
    # 交叉验证评估专科专家
    cv_scores_specialist = cross_val_score(specialist_model, X_stage2_core_scaled, y_stage2, cv=5)
    print(f"\n专科专家交叉验证准确率: {cv_scores_specialist.mean():.4f} (+/- {cv_scores_specialist.std() * 2:.4f})")
    
    # 训练最终专科专家模型
    specialist_model.fit(X_stage2_core_scaled, y_stage2)
    
    # --- 对比评估：原有专家组系统 ---
    print("\n--- 对比评估：原有专家组系统 ---")
    print("目标：量化专科专家的性能提升")
    
    # 原有专家组使用所有特征
    X_stage2_all = stage2_data[feature_columns]
    X_stage2_all_scaled = scaler.fit_transform(X_stage2_all)
    
    # 训练原有专家组模型
    original_expert_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # 交叉验证评估原有系统
    cv_scores_original = cross_val_score(original_expert_model, X_stage2_all_scaled, y_stage2, cv=5)
    print(f"原有专家组交叉验证准确率: {cv_scores_original.mean():.4f} (+/- {cv_scores_original.std() * 2:.4f})")
    
    # 训练最终原有专家组模型
    original_expert_model.fit(X_stage2_all_scaled, y_stage2)
    
    # --- 性能对比分析 ---
    print("\n--- 性能对比分析 ---")
    
    # 计算性能提升
    accuracy_improvement = cv_scores_specialist.mean() - cv_scores_original.mean()
    print(f"专科专家 vs 原有专家组:")
    print(f"  准确率提升: {accuracy_improvement:.4f}")
    print(f"  相对提升: {(accuracy_improvement / cv_scores_original.mean() * 100):.2f}%")
    
    # --- 特征重要性分析 ---
    print("\n--- 专科专家特征重要性分析 ---")
    
    feature_importance = pd.DataFrame({
        'feature': stage2_core_features,
        'importance': specialist_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("特征重要性排序:")
    for i, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # --- 模型保存 ---
    print("\n--- 保存训练好的模型 ---")
    
    models_output_dir = os.path.join(latest_run_dir, 'trained_models')
    os.makedirs(models_output_dir, exist_ok=True)
    
    # 保存第一阶段模型
    with open(os.path.join(models_output_dir, 'stage1_model.pkl'), 'wb') as f:
        pickle.dump(model1, f)
    
    # 保存专科专家模型
    with open(os.path.join(models_output_dir, 'specialist_expert.pkl'), 'wb') as f:
        pickle.dump(specialist_model, f)
    
    # 保存原有专家组模型
    with open(os.path.join(models_output_dir, 'original_expert.pkl'), 'wb') as f:
        pickle.dump(original_expert_model, f)
    
    # 保存数据标准化器
    with open(os.path.join(models_output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 保存特征配置
    model_config = {
        'stage1_features': stage1_features,
        'stage2_core_features': stage2_core_features,
        'stage2_all_features': feature_columns,
        'training_date': datetime.now().isoformat(),
        'data_samples': len(fingerprints_df),
        'stage2_samples': len(stage2_data)
    }
    
    with open(os.path.join(models_output_dir, 'model_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"所有模型已保存到: {models_output_dir}")
    
    # --- 生成性能报告 ---
    print("\n--- 生成性能报告 ---")
    
    # 创建性能对比图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('专科专家 vs 原有专家组性能对比', fontsize=16)
    
    # 1. 交叉验证准确率对比
    models = ['原有专家组', '专科专家']
    cv_means = [cv_scores_original.mean(), cv_scores_specialist.mean()]
    cv_stds = [cv_scores_original.std(), cv_scores_specialist.std()]
    
    axes[0, 0].bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_title('交叉验证准确率对比')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 特征重要性
    axes[0, 1].barh(feature_importance['feature'], feature_importance['importance'])
    axes[0, 1].set_title('专科专家特征重要性')
    axes[0, 1].set_xlabel('重要性')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 专科专家特征分布
    for feature in stage2_core_features:
        axes[1, 0].hist(stage2_data[stage2_data['vulnerability_type'] == 'adversarial_pgd'][feature], 
                        alpha=0.5, label='对抗攻击', bins=30)
        axes[1, 0].hist(stage2_data[stage2_data['vulnerability_type'] == 'noise_gaussian'][feature], 
                        alpha=0.5, label='高斯噪声', bins=30)
    axes[1, 0].set_title('核心特征分布对比')
    axes[1, 0].set_xlabel('特征值')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 性能提升可视化
    improvement_data = {
        '原有专家组': cv_scores_original.mean(),
        '专科专家': cv_scores_specialist.mean()
    }
    axes[1, 1].pie(improvement_data.values(), labels=improvement_data.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('性能对比')
    
    plt.tight_layout()
    
    # 保存性能报告
    performance_report_path = os.path.join(latest_run_dir, 'performance_comparison.png')
    plt.savefig(performance_report_path, dpi=300, bbox_inches='tight')
    print(f"性能对比报告已保存到: {performance_report_path}")
    
    # --- 最终总结 ---
    print("\n" + "="*60)
    print("🎯 专科专家系统训练完成！")
    print("="*60)
    print(f"📊 性能提升:")
    print(f"   - 原有专家组准确率: {cv_scores_original.mean():.4f}")
    print(f"   - 专科专家准确率: {cv_scores_specialist.mean():.4f}")
    print(f"   - 绝对提升: {accuracy_improvement:.4f}")
    print(f"   - 相对提升: {(accuracy_improvement / cv_scores_original.mean() * 100):.2f}%")
    
    print(f"\n 核心特征:")
    for i, feature in enumerate(stage2_core_features, 1):
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
        print(f"   {i}. {feature}: {importance:.4f}")
    
    print(f"\n💾 模型保存位置: {models_output_dir}")
    print(f" 性能报告: {performance_report_path}")
    
    print("\n 系统已成功从'全科医生'升级为'专科医生'！")
    print("   现在可以精准区分对抗攻击与高斯噪声了！")

if __name__ == '__main__':
    main()