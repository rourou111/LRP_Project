# lrp_analysis/feature_extractor.py
import numpy as np
from numpy.fft import fft2, fftshift
import torch
from scipy.stats import wasserstein_distance, entropy as kl_divergence, kurtosis
from scipy.spatial.distance import cosine as cosine_distance
from skimage.feature import graycomatrix, graycoprops
import pywt

def calculate_wasserstein(h1_tensor, h2_tensor):
    """
    计算两张热力图之间的推土机距离 (Wasserstein Distance)。

    Args:
        h1_tensor (torch.Tensor): 第一张热力图。
        h2_tensor (torch.Tensor): 第二张热力图。

    Returns:
        float: 两张热力图之间的推土机距离。
    """
    # 步骤 1: 将输入的PyTorch Tensor转换为NumPy数组
    h1_np = h1_tensor.detach().numpy()
    h2_np = h2_tensor.detach().numpy()

    # 步骤 2: 将二维的热力图矩阵展平（flatten）为一维向量
    h1_flat = h1_np.flatten()
    h2_flat = h2_np.flatten()

    # 步骤 3: 调用scipy函数计算并返回推土机距离
    distance = wasserstein_distance(h1_flat, h2_flat)
    
    return distance    

def calculate_cosine_similarity(h1_tensor, h2_tensor):
    """
    计算两张热力图之间的余弦相似度 (Cosine Similarity)。
    (更健壮的版本，增加了对零向量的检查)
    """
    h1_np = h1_tensor.detach().numpy()
    h2_np = h2_tensor.detach().numpy()

    h1_flat = h1_np.flatten()
    h2_flat = h2_np.flatten()

    # --- 安全检查 ---
    # 如果任一向量的模长（L2范数）接近于零，则它们是零向量
    if np.linalg.norm(h1_flat) < 1e-10 or np.linalg.norm(h2_flat) < 1e-10:
        # 两个零向量之间的相似度可以定义为1，一个零和一个非零向量相似度为0
        return 1.0 if np.linalg.norm(h1_flat) < 1e-10 and np.linalg.norm(h2_flat) < 1e-10 else 0.0
    # -----------------

    distance = cosine_distance(h1_flat, h2_flat)
    similarity = 1 - distance

    return similarity
def calculate_kl_divergences(h_clean_tensor, h_vuln_tensor):
    """
    计算干净热力图与失效热力图之间，正、负贡献分布的KL散度。
    (新版：采用更稳健的平滑方法，从根源上防止无穷大值)
    
    Args:
        h_clean_tensor (torch.Tensor): 干净样本的热力图 (基准分布 P)。
        h_vuln_tensor (torch.Tensor): 失效样本的热力图 (近似分布 Q)。

    Returns:
        tuple[float, float]: 返回一个元组，包含 (正贡献KL散度, 负贡献KL散度)。
    """
    epsilon = 1e-10  # 定义一个极小值，用于平滑

    def _get_smoothed_distributions(h_tensor):
        """内部辅助函数，用于生成平滑后的正、负子分布"""
        h_flat = h_tensor.detach().flatten()
        
        # 分离正、负贡献
        h_pos = torch.clamp(h_flat, min=0)
        h_neg = torch.abs(torch.clamp(h_flat, max=0))
        
        # --- 核心修改：先平滑，再归一化 ---
        # 1. 给所有像素点都加上一个极小的基础概率值 (平滑)
        p_pos_smooth = h_pos + epsilon
        p_neg_smooth = h_neg + epsilon
        
        # 2. 在平滑后的新分布上进行归一化
        p_pos_normalized = p_pos_smooth / torch.sum(p_pos_smooth)
        p_neg_normalized = p_neg_smooth / torch.sum(p_neg_smooth)
        
        return p_pos_normalized.numpy(), p_neg_normalized.numpy()

    # 步骤 1: 为两张热力图分别准备平滑、归一化后的正、负子概率分布
    p_clean_pos, p_clean_neg = _get_smoothed_distributions(h_clean_tensor)
    p_vuln_pos, p_vuln_neg = _get_smoothed_distributions(h_vuln_tensor)

    # 步骤 2: 分别计算正、负贡献的KL散度
    kl_pos = kl_divergence(p_clean_pos, p_vuln_pos)
    kl_neg = kl_divergence(p_clean_neg, p_vuln_neg)
    
    return kl_pos, kl_neg
def calculate_std_dev_diff(h_clean_tensor, h_vuln_tensor):
    """
    计算 H_vuln 和 H_clean 之间标准差的差值 (动态特征)。
    """
    # 分别计算两张热力图的标准差
    std_clean = h_clean_tensor.detach().numpy().std()
    std_vuln = h_vuln_tensor.detach().numpy().std()

    # 返回差值
    return std_vuln - std_clean
def calculate_kurtosis_diff(h_clean_tensor, h_vuln_tensor):
    """
    计算 H_vuln 和 H_clean 之间峰度的差值 (动态特征)。
    """
    # 分别计算两张热力图的峰度
    kurt_clean = kurtosis(h_clean_tensor.detach().numpy().flatten())
    kurt_vuln = kurtosis(h_vuln_tensor.detach().numpy().flatten())

    # 返回差值
    return kurt_vuln - kurt_clean
def calculate_high_freq_energy_ratio(h_vuln_tensor, high_freq_band=0.25):
    if h_vuln_tensor.dim() == 4:
        h_vuln_tensor = h_vuln_tensor.squeeze(0)
    """
    计算单张热力图的高频能量占比。
    (修正版：增加了对三通道热力图的处理)

    Args:
        h_vuln_tensor (torch.Tensor): 失效样本的热力图。
        high_freq_band (float): 定义高频区域的阈值，例如0.25代表离中心25%以外的区域。

    Returns:
        float: 高频能量占总能量的比例。
    """
    # 步骤 1: 将Tensor转换为NumPy数组
    h_np_3channel = h_vuln_tensor.detach().numpy()
    
    # --- 这是新添加的步骤：将三通道热力图转换为单通道灰度图 ---
    # 我们通过在通道维度(axis=0)上取平均值来实现
    if h_np_3channel.ndim == 3 and h_np_3channel.shape[0] == 3:
        h_np = h_np_3channel.mean(axis=0)
    else:
        h_np = h_np_3channel # 如果已经是单通道，则直接使用
    # ----------------------------------------------------------

    # 步骤 2: 执行二维快速傅里叶变换
    f_transform = fft2(h_np)
    
    # 步骤 3: 将零频率分量移动到频谱中心
    f_transform_shifted = fftshift(f_transform)
    
    # 步骤 4: 计算能量谱（幅度的平方）
    magnitude_spectrum = np.abs(f_transform_shifted)**2
    
    # 步骤 5: 定义并计算高频区域的能量
    rows, cols = h_np.shape # <-- 现在这里的 h_np 已经是二维的了，不会再报错
    center_row, center_col = rows // 2, cols // 2
    
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    mask = x**2 + y**2 > (min(center_row, center_col) * high_freq_band)**2
    
    high_freq_energy = np.sum(magnitude_spectrum[mask])
    total_energy = np.sum(magnitude_spectrum)
    
    # 步骤 6: 计算并返回高频能量占比
    if total_energy < 1e-10:
        return 0.0
        
    ratio = high_freq_energy / total_energy
    
    return ratio

def calculate_texture_features(h_vuln_tensor):
    if h_vuln_tensor.dim() == 4:
        h_vuln_tensor = h_vuln_tensor.squeeze(0)
    """
    计算单张热力图的多种纹理特征。

    Args:
        h_vuln_tensor (torch.Tensor): 失效样本的热力图。

    Returns:
        dict: 一个包含多种纹理特征的字典。
    """
    # --- 步骤 1: 预处理热力图 ---
    
    # a. 将Tensor转换为NumPy数组
    h_np_3channel = h_vuln_tensor.detach().numpy()
    
    # b. 将三通道热力图转换为单通道灰度图 (通过取平均值)
    if h_np_3channel.ndim == 3 and h_np_3channel.shape[0] == 3:
        h_np_gray = h_np_3channel.mean(axis=0)
    else:
        h_np_gray = h_np_3channel

    # c. 将浮点数值归一化到 0-255 的整数范围
    # GLCM函数需要整数输入来代表不同的“灰度等级”
    # 我们先将数值范围缩放到 0-255
    h_min, h_max = h_np_gray.min(), h_np_gray.max()
    if h_max - h_min < 1e-10:
        # 如果图像是纯色的，则所有纹理特征都为0或1
        return {'contrast': 0.0, 'homogeneity': 1.0, 'energy': 1.0, 'correlation': 1.0}
        
    h_normalized = (h_np_gray - h_min) / (h_max - h_min) * 255.0
    # d. 转换为无符号8位整数类型 (uint8)
    h_int = h_normalized.astype(np.uint8)

# --- 步骤 2: 计算灰度共生矩阵 (GLCM) ---

    # a. 定义参数
    # distances: 我们考虑像素邻居的距离，这里只考虑紧邻的1个像素。
    # angles: 我们考虑4个方向（0度-水平，45度-斜对角，90度-垂直, 135度-反斜对角）。
    # levels: 灰度等级，我们之前归一化到了0-255，所以是256。
    # symmetric & normed: 标准参数，保持默认即可。
    glcm = graycomatrix(h_int, distances=[1], angles=[0, np.pi/4, np.pi/2, np.pi*3/4], levels=256, symmetric=True, normed=True)

    # --- 步骤 3: 从GLCM中提取四个纹理特征 ---
    
    # a. 对四个方向的结果取平均值，得到一个更稳健的、与方向无关的特征值
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # b. 将所有结果打包成一个字典返回
    texture_features = {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }
    
    return texture_features
def calculate_dynamic_wavelet_ratio(h_clean_tensor, h_vuln_tensor):
    """特征一：计算动态小波能量比变化率"""
    
    def get_ratio(h_tensor):
        h_np = h_tensor.detach().numpy().mean(axis=0) # 转为灰度图
        coeffs = pywt.dwt2(h_np, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # 计算能量 (平方和)
        energy_ll = np.sum(LL**2)
        energy_high_freq = np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)
        
        # 加上一个极小值防止除以零
        return energy_high_freq / (energy_ll + 1e-10)

    ratio_clean = get_ratio(h_clean_tensor)
    ratio_vuln = get_ratio(h_vuln_tensor)
    
    change_ratio = (ratio_vuln - ratio_clean) / (ratio_clean + 1e-10)
    return change_ratio

def calculate_ll_distortion(h_clean_tensor, h_vuln_tensor):
    if h_clean_tensor.dim() == 4:
        h_clean_tensor = h_clean_tensor.squeeze(0)
    if h_vuln_tensor.dim() == 4:
        h_vuln_tensor = h_vuln_tensor.squeeze(0)
    """特征二：计算低频子带结构失真度"""
    
    def get_ll_texture_vec(h_tensor):
        h_np = h_tensor.detach().numpy().mean(axis=0)
        coeffs = pywt.dwt2(h_np, 'haar')
        LL, _ = coeffs
        
        # 归一化到 0-255 的整数范围以计算GLCM
        ll_min, ll_max = LL.min(), LL.max()
        if ll_max - ll_min < 1e-10: return np.array([0.0, 1.0, 1.0, 1.0])
        ll_normalized = (LL - ll_min) / (ll_max - ll_min) * 255.0
        ll_int = ll_normalized.astype(np.uint8)
        
        glcm = graycomatrix(ll_int, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        return np.array([contrast, homogeneity, energy, correlation])

    vec_c = get_ll_texture_vec(h_clean_tensor)
    vec_v = get_ll_texture_vec(h_vuln_tensor)
    
    # 计算余弦距离: 1 - (a dot b) / (||a|| * ||b||)
    dist = 1 - np.dot(vec_c, vec_v) / (np.linalg.norm(vec_c) * np.linalg.norm(vec_v) + 1e-10)
    return dist