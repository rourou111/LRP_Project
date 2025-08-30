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
    print("--- Step 1: Generating adversarial samples (Category 1) ---")
    adversarial_samples = []
    
    attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=8/255, eps_step=2/255, max_iter=20, targeted=False)
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理对抗样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images_np, labels_np = images.numpy(), labels.numpy()
    
        adversarial_images = attack.generate(x=images_np)
        
        original_preds = np.argmax(classifier.predict(images_np), axis=1)
        adversarial_preds = np.argmax(classifier.predict(adversarial_images), axis=1)
        
        for i in range(len(images_np)):
            if original_preds[i] == labels_np[i] and adversarial_preds[i] != labels_np[i]:
                adversarial_samples.append({
                    "original_image": images[i], # 保存原始的 torch.Tensor
                    "adversarial_image": torch.tensor(adversarial_images[i]),
                    "label": labels_np[i],
                    "original_pred": original_preds[i],
                    "adversarial_pred": adversarial_preds[i],
                    "vulnerability_type": "adversarial_pgd"
                })

    print(f"\nGenerated {len(adversarial_samples)} adversarial samples.")
    return adversarial_samples


def add_gaussian_noise(image, std_dev=0.15):
    """向图像张量添加高斯噪声"""
    noise = torch.randn_like(image) * std_dev
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def generate_noisy_samples(model, data_loader, device, classes):
    """
    通过添加高斯噪声生成失效样本
    """
    print("--- Step 2: Generating noisy samples (Category 2) ---")
    noisy_samples = []
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理噪声样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images = images.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            original_outputs = model(images)
            original_preds = torch.argmax(original_outputs, dim=1)
            
            noisy_images = add_gaussian_noise(images.clone())
            noisy_outputs = model(noisy_images)
            noisy_preds = torch.argmax(noisy_outputs, dim=1)
        
        for i in range(len(images)):
            if original_preds[i] == labels[i] and noisy_preds[i] != labels[i]:
                noisy_samples.append({
                    "original_image": images[i].cpu(),
                    "noisy_image": noisy_images[i].cpu(),
                    "label": labels[i].item(),
                    "original_pred": original_preds[i].item(),
                    "noisy_pred": noisy_preds[i].item(),
                    "vulnerability_type": "noise_gaussian"
                })

    print(f"\nGenerated {len(noisy_samples)} noisy samples.")
    return noisy_samples


def perturb_model_weights(model, layer_name='layer4.1.conv2', std_dev=1e-2):
    """对特定层的权重添加微小扰动"""
    drifted_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in drifted_model.named_parameters():
            if name == layer_name + '.weight':
                noise = torch.randn_like(param) * std_dev
                param.add_(noise)
    drifted_model.eval()
    return drifted_model

def generate_drift_samples(original_model, data_loader, device, classes):
    """
    通过扰动模型权重生成参数漂移样本
    """
    print("--- Step 3: Generating parameter drift samples (Category 3) ---")
    drift_samples = []
    
    drifted_model = perturb_model_weights(original_model)
    drifted_model.to(device)

    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理漂移样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            original_outputs = original_model(images)
            original_preds = torch.argmax(original_outputs, dim=1)

            drifted_outputs = drifted_model(images)
            drifted_preds = torch.argmax(drifted_outputs, dim=1)

        for i in range(len(images)):
            if original_preds[i] == labels[i] and drifted_preds[i] != labels[i]:
                drift_samples.append({
                    "original_image": images[i].cpu(),
                    # "drifted_image": images[i].cpu(), # <-- 直接删除这一行
                    "label": labels[i].item(),
                    "original_pred": original_preds[i].item(),
                    "drifted_pred": drifted_preds[i].item(),
                    "vulnerability_type": "drift_parameter"
                })

    print(f"\nGenerated {len(drift_samples)} parameter drift samples.")
    return drift_samples, drifted_model