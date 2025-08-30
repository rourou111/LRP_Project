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

# 从 scikit-learn 中导入我们需要的模块
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

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
    # --- 1. 自动寻找最新的指纹数据文件 ---
    runs_dir = config['output_paths']['runs_directory']
    list_of_run_dirs = glob.glob(os.path.join(runs_dir, '*/'))
    if not list_of_run_dirs:
        print("\n错误：在 'runs' 文件夹下找不到任何运行记录。")
        sys.exit(1)

    latest_run_dir = max(list_of_run_dirs, key=os.path.getctime)
    fingerprint_file_path = os.path.join(latest_run_dir, 'vulnerability_fingerprints.csv')
    print(f"\n正在从最新的运行记录中加载数据: {fingerprint_file_path}")

    try:
        data = pd.read_csv(fingerprint_file_path)
        print(f"成功加载 {len(data)} 个样本。")
    except FileNotFoundError:
        print(f"\n错误：在路径 '{fingerprint_file_path}' 中找不到 vulnerability_fingerprints.csv 文件。")
        sys.exit(1)

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
    # 阶段二：构建“专家组决策系统” (Specialist System)
    # =============================================================================
    # ... (此阶段代码与原脚本完全一致，无需改动) ...
    print("\n" + "="*50)
    print("阶段二：开始构建'专家组决策系统'")
    print("="*50)
    non_drift_train_mask = (y_train1 == 0)
    X_train_experts = X_train_full[non_drift_train_mask]
    y_train_experts_raw = y_train_full[non_drift_train_mask]
    label_encoder2 = LabelEncoder()
    y_train_experts = label_encoder2.fit_transform(y_train_experts_raw)
    print(f"已筛选出 {len(X_train_experts)} 个'对抗 vs. 噪声'样本，交由专家组处理。")
    features_dynamic = [
        'wasserstein_dist',
        'cosine_similarity',
        'kl_divergence_pos',
        'kl_divergence_neg',
        'std_dev_diff',         # 衡量注意力分散/集中的变化 [cite: 26]
        'kurtosis_diff',        # 衡量注意力尖锐/平坦的变化 [cite: 27]
        'dynamic_wavelet_ratio_change', # 您新设计的特征一：动态小波能量比变化率
        'll_distortion'         # 您新设计的特征二：低频子带结构失真度
    ]

    # 专家2：“频域分析专家” (Frequency Domain Expert)
    # 关注的是 H_vuln 在频域上的内在特性
    features_frequency = [
        'high_freq_ratio',     # 旨在捕捉高斯噪声的高频散布特性 [cite: 23]
        'ratio_zscore'         # 您新设计的特征三：能量比的基线分离度 (Z-score)
    ]

    # 专家3：“纹理学专家” (Texture Expert)
    # 关注的是 H_vuln 热力图的“质感”
    features_texture = [
        'contrast',             # 对比度 [cite: 24]
        'homogeneity',          # 同质性 [cite: 24]
        'energy',               # 能量 [cite: 24]
        'correlation'           # 相关性 [cite: 24]
    ]

    # 专家4：“敏感性/内在性专家” (Sensitivity/Intrinsic Expert)
    # (这是一个建议的组合，您可以根据需要调整)
    # 关注的是 H_vuln 本身的统计特性，作为对其他专家的补充
    features_sensitivity = [
        'std_dev_diff',         # 注意：这里可以复用一些特征，让不同专家有交叉视角
        'kurtosis_diff',
        'high_freq_ratio'
    ]  
    X_train_experts_imputed = imputer1.transform(X_train_experts)
    X_train_experts_scaled = scaler1.transform(X_train_experts_imputed)
    X_train_experts_scaled_df = pd.DataFrame(X_train_experts_scaled, columns=X_train_experts.columns, index=X_train_experts.index)
    dynamic_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    frequency_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    texture_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    sensitivity_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    dynamic_opinions = cross_val_predict(dynamic_expert, X_train_experts_scaled_df[features_dynamic], y_train_experts, cv=5, method='predict_proba')
    frequency_opinions = cross_val_predict(frequency_expert, X_train_experts_scaled_df[features_frequency], y_train_experts, cv=5, method='predict_proba')
    texture_opinions = cross_val_predict(texture_expert, X_train_experts_scaled_df[features_texture], y_train_experts, cv=5, method='predict_proba')
    sensitivity_opinions = cross_val_predict(sensitivity_expert, X_train_experts_scaled_df[features_sensitivity], y_train_experts, cv=5, method='predict_proba')
    X_train_meta = np.hstack([dynamic_opinions, frequency_opinions, texture_opinions, sensitivity_opinions])
    print("专家会诊完成，已形成元特征集。")
    dynamic_expert.fit(X_train_experts_scaled_df[features_dynamic], y_train_experts)
    frequency_expert.fit(X_train_experts_scaled_df[features_frequency], y_train_experts)
    texture_expert.fit(X_train_experts_scaled_df[features_texture], y_train_experts)
    sensitivity_expert.fit(X_train_experts_scaled_df[features_sensitivity], y_train_experts)
    print("专家学习完成。")
    meta_classifier = LogisticRegression(random_state=42)
    meta_classifier.fit(X_train_meta, y_train_experts)
    print("最终决策者训练完成！")

    # =============================================================================
    # 阶段三：对完整的“专家组决策系统”进行最终评估
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
        print(" -> 步骤2: '专家组'开始对疑难样本进行会诊...")
        dynamic_opinions_test = dynamic_expert.predict_proba(X_test_for_experts[features_dynamic])
        frequency_opinions_test = frequency_expert.predict_proba(X_test_for_experts[features_frequency])
        texture_opinions_test = texture_expert.predict_proba(X_test_for_experts[features_texture])
        sensitivity_opinions_test = sensitivity_expert.predict_proba(X_test_for_experts[features_sensitivity])
        X_test_meta = np.hstack([dynamic_opinions_test, frequency_opinions_test, texture_opinions_test, sensitivity_opinions_test])
        expert_predictions = meta_classifier.predict(X_test_meta)
        expert_predictions_original_labels = label_encoder2.inverse_transform(expert_predictions)
        final_predictions[non_drift_test_mask] = expert_predictions_original_labels
        print(" -> '专家组'会诊完毕。")
    drift_label_encoded = list(label_encoder.classes_).index('drift_parameter')
    final_predictions[pred1_test == 1] = drift_label_encoded
    print("...所有样本预测流程结束。")
    print("\n--- 两阶段专家组决策系统最终性能评估 ---")
    print(f"最终准确率: {accuracy_score(y_test_full, final_predictions):.4f}")
    print("\n最终分类报告:")
    print(classification_report(y_test_full, final_predictions, target_names=label_encoder.classes_))
    print("\n最终混淆矩阵:")
    final_cm = confusion_matrix(y_test_full, final_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Final Confusion Matrix for the Two-Stage Expert System')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # 保存图像到最新的运行文件夹
    if os.path.exists(latest_run_dir):
        plt.savefig(os.path.join(latest_run_dir, 'final_confusion_matrix_expert_system.png'))
        print(f"\n混淆矩阵图像已保存到: {latest_run_dir}")
    plt.show()
    print("\n脚本执行完毕！")

if __name__ == '__main__':
    main()