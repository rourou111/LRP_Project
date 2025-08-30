# scripts/03_extract_fingerprints.py
#!/usr/bin/env python3
"""
脚本 03: 从成对的热力图中提取指纹特征。
执行方式: 在项目根目录下运行 `python scripts/03_extract_fingerprints.py`
"""
import os
import pickle
import pandas as pd
import numpy as np
import glob
import yaml
import sys
# 1. 从我们新创建的特征提取器模块中导入所有计算函数
from lrp_analysis.feature_extractor import (
    calculate_wasserstein,
    calculate_cosine_similarity,
    calculate_kl_divergences,
    calculate_std_dev_diff,
    calculate_kurtosis_diff,
    calculate_high_freq_energy_ratio,
    calculate_texture_features,
    calculate_dynamic_wavelet_ratio,
    calculate_ll_distortion
)

def main():
    print("=== 脚本 03: 提取指纹特征 ===")

    # --- 加载配置文件 ---
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- 自动寻找并加载最新的热力图数据 ---
    runs_dir = config['output_paths']['runs_directory']
    list_of_run_dirs = glob.glob(os.path.join(runs_dir, '*/'))
    if not list_of_run_dirs:
        print("\n错误：在 'runs' 文件夹下找不到任何运行记录。")
        print("请确保您已经成功运行了 '02_generate_heatmaps.py' 脚本。")
        sys.exit(1)

    latest_run_dir = max(list_of_run_dirs, key=os.path.getctime)
    heatmap_file_path = os.path.join(latest_run_dir, 'paired_heatmaps.pkl')
    print(f"\n正在从最新的运行记录中加载数据: {heatmap_file_path}")

    try:
        with open(heatmap_file_path, 'rb') as f:
            paired_heatmaps = pickle.load(f)
        print(f"成功加载 {len(paired_heatmaps)} 组配对的热力图数据。")
    except FileNotFoundError:
        print(f"\n错误：在路径 '{heatmap_file_path}' 中找不到 paired_heatmaps.pkl 文件。")
        sys.exit(1)

    # --- 特征三的前置步骤：为Z-score校准噪声基准 ---
    print("\n为特征三(Z-score)校准噪声基准...")
    # 注意：get_ratio 函数现在在 lrp_analysis.feature_extractor 中
    # 我们需要导入它，或者在这里重新定义一个简单的版本。
    # 这里选择重新定义一个简单的版本，因为它只在这个校准步骤中使用。
    def get_ratio_simple(h_tensor):
        """一个简化的版本，仅用于噪声基准校准"""
        import pywt
        h_np = h_tensor.detach().numpy().mean(axis=0)
        coeffs = pywt.dwt2(h_np, 'haar')
        LL, (LH, HL, HH) = coeffs
        energy_ll = np.sum(LL**2)
        energy_high_freq = np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)
        return energy_high_freq / (energy_ll + 1e-10)

    noise_ratios = []
    for data_pair in paired_heatmaps:
        if data_pair['vulnerability_type'] == 'noise_gaussian':
            h_vuln = data_pair['h_vuln']
            noise_ratios.append(get_ratio_simple(h_vuln))

    mu_noise = np.mean(noise_ratios) if noise_ratios else 0
    sigma_noise = np.std(noise_ratios) if noise_ratios else 1
    print(f"噪声基准校准完成: 平均值={mu_noise:.4f}, 标准差={sigma_noise:.4f}")

    # --- 批量处理所有样本，提取指纹 ---
    fingerprints_list = []
    print(f"\n--- 开始为 {len(paired_heatmaps)} 组热力图提取指纹 ---")

    for i, data_pair in enumerate(paired_heatmaps):
        h_clean = data_pair['h_clean']
        h_vuln = data_pair['h_vuln']
        vuln_type = data_pair['vulnerability_type']

        print(f"\r  正在处理样本 {i+1}/{len(paired_heatmaps)}", end="")

        # --- 调用从 lrp_analysis.feature_extractor 导入的函数 ---
        # 1. 对比性特征
        wasserstein = calculate_wasserstein(h_clean, h_vuln)
        cosine_sim = calculate_cosine_similarity(h_clean, h_vuln)
        kl_pos, kl_neg = calculate_kl_divergences(h_clean, h_vuln)

        # 2. 内在性特征
        std_diff = calculate_std_dev_diff(h_clean, h_vuln)
        kurt_diff = calculate_kurtosis_diff(h_clean, h_vuln)
        high_freq_ratio = calculate_high_freq_energy_ratio(h_vuln)

        # 3. 纹理特征
        texture_feats = calculate_texture_features(h_vuln)

        # 4. 新增特征
        dynamic_ratio_change = calculate_dynamic_wavelet_ratio(h_clean, h_vuln)
        ll_distortion = calculate_ll_distortion(h_clean, h_vuln)

        # 5. 计算特征三 (Z-score)
        static_ratio_sample = get_ratio_simple(h_vuln)
        ratio_zscore = np.abs(static_ratio_sample - mu_noise) / (sigma_noise + 1e-10)

        # --- 将所有结果存入一个字典 ---
        fingerprint_data = {
            'wasserstein_dist': wasserstein,
            'cosine_similarity': cosine_sim,
            'kl_divergence_pos': kl_pos,
            'kl_divergence_neg': kl_neg,
            'std_dev_diff': std_diff,
            'kurtosis_diff': kurt_diff,
            'high_freq_ratio': high_freq_ratio,
            **texture_feats, # 解包纹理特征字典
            'dynamic_wavelet_ratio_change': dynamic_ratio_change,
            'll_distortion': ll_distortion,
            'ratio_zscore': ratio_zscore,
            'vulnerability_type': vuln_type
        }
        fingerprints_list.append(fingerprint_data)

    print("\n--- 所有指纹已成功提取 ---")

    # --- 使用Pandas将结果保存为CSV文件 ---
    fingerprints_df = pd.DataFrame(fingerprints_list)
    output_filename = os.path.join(latest_run_dir, 'vulnerability_fingerprints.csv')
    if os.path.exists(output_filename):
        os.remove(output_filename)
        print(f"\n已删除旧的指纹文件: '{output_filename}'")
    fingerprints_df.to_csv(output_filename, index=False)
    print(f"\n指纹数据已成功保存到: {output_filename}")
    print("项目核心阶段已完成！您现在拥有了可用于训练机器学习模型的数据集。")

if __name__ == '__main__':
    main()