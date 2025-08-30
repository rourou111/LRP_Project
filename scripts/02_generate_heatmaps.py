# scripts/02_generate_heatmaps.py
#!/usr/bin/env python3
"""
脚本 02: 为漏洞样本生成成对的热力图 (H_clean 和 H_vuln)。
执行方式: 在项目根目录下运行 `python scripts/02_generate_heatmaps.py`
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import os
from datetime import datetime
import yaml

from captum.attr import LRP

def main():
    print("=== 脚本 02: 生成成对热力图 ===")

    # --- 加载配置文件 ---
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")

    # --- 加载模型 --- 
    # (此部分逻辑从原脚本复制，几乎不变)
    model = models.resnet18(weights=None)
    model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    print(f"正在从本地路径加载预训练模型: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except FileNotFoundError:
        print(f"错误: 在路径 {model_path} 未找到权重文件。")
        exit()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    model.eval()
    print("模型加载成功！")

    # --- 加载漏洞数据 ---
    vulnerabilities_file = config['data_paths']['vulnerabilities_pkl']
    print(f"正在从路径加载漏洞数据: {vulnerabilities_file}")
    try:
        with open(vulnerabilities_file, 'rb') as f:
            all_vulnerabilities = pickle.load(f)
        print(f"成功加载 {len(all_vulnerabilities)} 个漏洞样本！")
    except FileNotFoundError:
        print("错误：找不到漏洞数据文件。请先运行 01_generate_vulnerabilities.py")
        exit()

    # --- 创建输出目录 ---
    base_output_dir = config['output_paths']['runs_directory']
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"本次运行的所有输出将被保存在: {run_output_dir}")

    # --- 核心计算部分 --- 
    # (此部分逻辑从原脚本复制，因为它专属于这个脚本的流程)
    paired_heatmaps_data = []
    lrp = LRP(model)
    print(f"开始为 {len(all_vulnerabilities)} 个漏洞样本生成成对的热力图...")

    for i, sample in enumerate(all_vulnerabilities):
        print(f"\r  正在处理样本 {i+1}/{len(all_vulnerabilities)}", end="")

        # 确定原始图像和漏洞图像
        original_image = sample['original_image'].to(device)
        vuln_type = sample['vulnerability_type']

        if vuln_type == 'adversarial_pgd':
            vuln_image = sample['adversarial_image'].to(device)
        elif vuln_type == 'noise_gaussian':
            vuln_image = sample['noisy_image'].to(device)
        elif vuln_type == 'parameter_drift': # 根据您的其他脚本修正类型名称
            vuln_image = original_image # 对于漂移，图像不变
        else:
            continue

        # 确保图像有批次维度 (batch dimension)
        if original_image.dim() == 3:
            original_image = original_image.unsqueeze(0)
        if vuln_image.dim() == 3:
            vuln_image = vuln_image.unsqueeze(0)

        # 获取模型预测结果
        with torch.no_grad():
            original_pred = torch.argmax(model(original_image)).item()
            vuln_pred = torch.argmax(model(vuln_image)).item()

        # 计算LRP热力图
        h_clean = lrp.attribute(original_image, target=original_pred)
        h_vuln = lrp.attribute(vuln_image, target=vuln_pred)

        # 【关键部分】定义 paired_data 字典
        paired_data = {
            'h_clean': h_clean.cpu(),
            'h_vuln': h_vuln.cpu(),
            'vulnerability_type': vuln_type
        }

        paired_heatmaps_data.append(paired_data)

    print("\n--- 所有热力图已成功生成 ---")
    # --- 保存最终结果 ---
    output_filename = os.path.join(run_output_dir, 'paired_heatmaps.pkl')

    print(f"\n正在将 {len(paired_heatmaps_data)} 组成对的热力图保存到: {output_filename}")

    # 检查旧文件并删除（这是一个好习惯）
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # 使用 pickle 将数据写入文件
    with open(output_filename, 'wb') as f:
        pickle.dump(paired_heatmaps_data, f)

    print("保存成功！现在可以运行下一步了。")

if __name__ == '__main__':
    main()