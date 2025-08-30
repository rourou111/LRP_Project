# scripts/01_generate_vulnerabilities.py
#!/usr/bin/env python3
"""
脚本 01: 生成三种类型的漏洞样本。
执行方式: 在项目根目录下运行 `python scripts/01_generate_vulnerabilities.py`
"""
import os
import sys
# 将项目根目录添加到Python路径，这样才能找到 lrp_analysis 包
sys.path.insert(0, os.path.abspath('.')) 

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pickle
import yaml
from art.estimators.classification import PyTorchClassifier

# 从我们新创建的模块中导入核心函数
from lrp_analysis.data_generator import (
    generate_adversarial_samples,
    generate_noisy_samples,
    generate_drift_samples
)

def main():
    print("=== 脚本 01: 生成漏洞样本 ===")

    # --- 加载配置文件 ---
    # 路径现在是相对于项目根目录的 ./config/config.yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 准备工作：加载数据和预训练模型 --- 
    # (这部分逻辑直接从原脚本复制过来，几乎不需要改动)
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    model = torchvision.models.resnet18(pretrained=False)
    model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    print(f"从本地路径加载预训练模型: {model_path}")
    model.load_state_dict(torch.load(model_path), strict=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    classifier = PyTorchClassifier(
        model=model, clip_values=(0, 1), loss=criterion,
        optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10,
    )
    # --- 2. 执行并收集所有漏洞样本 ---
    # (直接调用我们从模块里导入的函数，代码变得非常简洁)
    print("--- 开始生成所有类型的漏洞样本 ---")
    adversarial_vulnerabilities = generate_adversarial_samples(classifier, testloader, classes)
    noisy_vulnerabilities = generate_noisy_samples(model, testloader, device, classes)
    drift_vulnerabilities, drifted_model = generate_drift_samples(model, testloader, device, classes)

    all_vulnerabilities = adversarial_vulnerabilities + noisy_vulnerabilities + drift_vulnerabilities
    print(f"\n总共收集到 {len(all_vulnerabilities)} 个漏洞样本。")

    # --- 3. 保存结果 ---
    output_filename = config['data_paths']['vulnerabilities_pkl']
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_filename):
        os.remove(output_filename)
        print(f"已删除旧的漏洞文件: '{output_filename}'")

    with open(output_filename, 'wb') as f:
        pickle.dump(all_vulnerabilities, f)
    print(f"漏洞样本已成功保存到: {output_filename}")

if __name__ == '__main__':
    main()