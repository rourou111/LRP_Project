# lrp_analysis/data_generator.py
import torch
import torch.nn as nn
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
import copy

def generate_adversarial_samples(classifier, data_loader, classes):
    """
    使用PGD攻击生成对抗样本
    """
    print("--- Step 2: Generating adversarial samples (Category 1) ---")
    adversarial_samples = []
    count = 0
    # 初始化PGD攻击
    attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=8/255, eps_step=2/255, max_iter=20, targeted=False)
    
    # 遍历数据加载器中的所有批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理对抗样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images, labels = images.numpy(), labels.numpy()
    
        # 生成对抗样本
        adversarial_images = attack.generate(x=images)
        
        # 验证攻击效果
        original_preds = np.argmax(classifier.predict(images), axis=1)
        adversarial_preds = np.argmax(classifier.predict(adversarial_images), axis=1)
        
        for i in range(len(images)):
            if original_preds[i] == labels[i] and adversarial_preds[i] != labels[i]:
                # 筛选出“原始预测正确”且“攻击后预测错误”的样本
                adversarial_samples.append({
                    "original_image": torch.tensor(images[i]),
                    "adversarial_image": torch.tensor(adversarial_images[i]),
                    "label": labels[i],
                    "original_pred": original_preds[i],
                    "adversarial_pred": adversarial_preds[i],
                    "vulnerability_type": "adversarial_pgd"
                })
                count += 1
                print(f"  Found an adversarial sample: Original={classes[labels[i]]}, Adversarial pred={classes[adversarial_preds[i]]}")
    print(f"Generated {len(adversarial_samples)} adversarial samples.\n")
    return adversarial_samples


# =============================================================================
# 3. 类别2：噪声漏洞 (Noise Vulnerability)
# =============================================================================
def add_gaussian_noise(image, std_dev=0.15):
    """向图像张量添加高斯噪声"""
    noise = torch.randn_like(image) * std_dev
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1) # 将像素值裁剪回[0, 1]范围

def generate_noisy_samples(model, data_loader, device, classes):
    """
    通过添加高斯噪声生成失效样本
    """
    noisy_samples = []
    count = 0
    
    # 遍历数据加载器中的所有批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理噪声样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images = images.to(device)
        labels = labels.to(device)
    
        # 原始预测
        original_outputs = model(images)
        original_preds = torch.argmax(original_outputs, dim=1)
        
        # 添加噪声并获取新预测
        noisy_images = add_gaussian_noise(images.clone())
        noisy_outputs = model(noisy_images)
        noisy_preds = torch.argmax(noisy_outputs, dim=1)
        
        for i in range(len(images)):
            if original_preds[i] == labels[i] and noisy_preds[i] != labels[i]:
                # 筛选出“原始预测正确”且“加噪后预测错误”的样本
                noisy_samples.append({
                    "original_image": images[i],
                    "noisy_image": noisy_images[i],
                    "label": labels[i],
                    "original_pred": original_preds[i].item(),
                    "noisy_pred": noisy_preds[i].item(),
                    "vulnerability_type": "noise_gaussian"
                })
                count += 1
                print(f"  Found a noisy sample: Original={classes[labels[i]]}, Noisy pred={classes[noisy_preds[i]]}")

    print(f"Generated {len(noisy_samples)} noisy samples.\n")
    return noisy_samples

# =============================================================================
# 4. 类别3：参数漂移漏洞 (Parameter Drift Vulnerability)
# =============================================================================
def perturb_model_weights(model, layer_name='layer4.1.conv2', std_dev=1e-2):
    """对特定层的权重添加微小扰动"""
    # 使用 copy.deepcopy() 来进行一次完美的、独立的“克隆”
    drifted_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for name, param in drifted_model.named_parameters():
            if name == layer_name + '.weight':
                print(f"  Perturbing weights of layer: {name}")
                noise = torch.randn_like(param) * std_dev
                param.add_(noise)
    
    drifted_model.eval()
    return drifted_model

def generate_drift_samples(original_model, data_loader, device, classes):
    print("--- Step 4: Generating parameter drift samples (Category 3) ---")
    drift_samples = []
    count = 0  # <-- 补充这一行，用于计数

    # 1. 只在开头创建一次漂移模型
    drifted_model = perturb_model_weights(original_model) 
    drifted_model.to(device)

    # 遍历数据加载器中的所有批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理漂移样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images, labels = images.to(device), labels.to(device)

        # 使用正确的模型获取预测结果
        with torch.no_grad():
            original_outputs = original_model(images)
            original_preds = torch.argmax(original_outputs, dim=1)

            drifted_outputs = drifted_model(images)
            drifted_preds = torch.argmax(drifted_outputs, dim=1)

        for i in range(len(images)):
            if original_preds[i] == labels[i] and drifted_preds[i] != labels[i]:
                # 筛选出“原始预测正确”且“漂移后预测错误”的样本
                drift_samples.append({
                    "original_image": images[i],
                    "drifted_image": images[i],  # 对于漂移，图像本身不变
                    "label": labels[i].item(),
                    "original_pred": original_preds[i].item(),
                    "drifted_pred": drifted_preds[i].item(),
                    "vulnerability_type": "parameter_drift" # 修正漏洞类型名称
                })
                count += 1
    # 这个打印语句应该在内层循环之外
    print(f"\n  找到了 {count} 个漂移样本。")

    print(f"Generated {len(drift_samples)} parameter drift samples.\n")
    # 2. 返回两个值
    return drift_samples, drifted_model