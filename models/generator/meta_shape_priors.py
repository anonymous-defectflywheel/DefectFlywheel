"""
随机形状先验 mask（线、点、自由形），用于无困难 mask 时的 fallback。
与创新思想一致：有困难 mask 时优先用困难区域，否则用随机形状扩充多样性。
"""
import random
import cv2
import numpy as np
from scipy.ndimage import label as ndi_label


def msp_line(W=256, H=256):
    M = np.zeros((H, W), dtype=np.uint8)
    c = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
    angle = np.random.uniform(0, 180)
    theta = np.deg2rad(angle)
    # Scale up line length
    l = np.random.uniform(100, 300)
    s = np.random.randint(20, 40)
    x = np.linspace(-l / 2, l / 2, s)
    y = np.zeros_like(x)
    if np.random.rand() < 0.5:
        eps = np.random.normal(0, 14, size=s).astype(np.float32).reshape(-1, 1)
        y += cv2.GaussianBlur(eps, (1, 5), 2).flatten()
    # Scale up line thickness
    min_t, max_t = 3, 15
    thickness_curve = (
        np.linspace(min_t, max_t, s // 2).tolist()
        + np.linspace(max_t, min_t, s - s // 2).tolist()
    )
    thickness_noise = np.random.uniform(-0.1 * (max_t - min_t), 0.1 * (max_t - min_t), size=s)
    thickness_curve = np.clip(np.array(thickness_curve) + thickness_noise, min_t, max_t)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for i in range(s - 1):
        pt1 = R @ np.array([x[i], y[i]]) + c
        pt2 = R @ np.array([x[i + 1], y[i + 1]]) + c
        cv2.line(
            M, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
            255, int(thickness_curve[i]), lineType=cv2.LINE_AA,
        )
    _, M = cv2.threshold(M, 127, 255, cv2.THRESH_BINARY)
    return M.astype(np.uint8)


def msp_dot(W=256, H=256):
    M = np.zeros((H, W), dtype=np.uint8)
    c = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
    # Scale up dot radius
    r = np.random.uniform(20, 80)
    s = np.random.randint(12, 30)
    theta = np.sort(np.random.uniform(0, 2 * np.pi, s))
    alpha = np.random.uniform(0.6, 1.4)
    beta = np.random.uniform(0.05, 0.35)
    u = np.random.uniform(0, 1)
    r_i = np.random.uniform(-beta * r, beta * r, s) if u >= 0.66 else np.random.normal(0, beta * r, s)
    x = c[0] + (r + r_i) * np.cos(theta) * alpha
    y = c[1] + (r + r_i) * np.sin(theta)
    contour = np.stack((x, y), axis=1).astype(np.int32)
    cv2.fillPoly(M, [contour], 255)
    if np.random.rand() < 0.5:
        k = np.random.choice([3, 5, 7])
        M = cv2.GaussianBlur(M, (k, k), 0)
    _, M = cv2.threshold(M, 127, 255, cv2.THRESH_BINARY)
    return M.astype(np.uint8)


def msp_freeform(W=256, H=256):
    M = np.zeros((H, W), dtype=np.uint8)
    n_step = np.random.randint(300, 18001)
    sigma = np.random.uniform(2, 12)
    x, y = np.random.randint(0, W), np.random.randint(0, H)
    for _ in range(n_step):
        M[y, x] = 1
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
        x = np.clip(x + dx, 0, W - 1)
        y = np.clip(y + dy, 0, H - 1)
    ksize = int(2 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    M = cv2.GaussianBlur(M.astype(np.float32), (ksize, ksize), sigmaX=sigma)
    if np.random.rand() < 0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        M = cv2.dilate(M, kernel, 1)
        M = cv2.erode(M, kernel, 1)
    _, M = cv2.threshold(M, 0.5, 1, cv2.THRESH_BINARY)
    labeled, num = ndi_label(M, structure=np.ones((3, 3)))
    if num > 1:
        areas = [(labeled == i).sum() for i in range(1, num + 1)]
        largest = 1 + np.argmax(areas)
        M = (labeled == largest).astype(np.uint8)
    return M.astype(np.uint8) * 255


def msp_perlin(W=256, H=256):
    """
    使用多尺度分形柏林噪声（Fractal Perlin Noise with Octaves）生成大面积、边缘分形的不规则斑点
    """
    noise = np.zeros((H, W), dtype=np.float32)
    octaves = 3
    persistence = 0.5
    scale = np.random.uniform(10, 30)
    
    amplitude = 1.0
    for i in range(octaves):
        # 随倍频增加，分辨率增高，控制边缘细节
        n_H, n_W = max(H // (2 ** (i + 2)), 1), max(W // (2 ** (i + 2)), 1)
        base_noise = np.random.randn(n_H, n_W).astype(np.float32)
        base_noise = cv2.resize(base_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # 不同倍频对应不同平滑度
        current_sigma = max(scale / (2 ** i), 1.0)
        smoothed_noise = cv2.GaussianBlur(base_noise, (0, 0), current_sigma)
        
        noise += amplitude * smoothed_noise
        amplitude *= persistence
    
    # 动态阈值以控制瑕疵生成面积比例
    threshold = np.percentile(noise, np.random.uniform(70, 95))
    mask = (noise > threshold).astype(np.uint8)
    
    # 形态学处理，使核心连通但也保留分形边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 仅保留最大连通域，避免太零碎
    labeled, num = ndi_label(mask, structure=np.ones((3, 3)))
    if num > 1:
        areas = [(labeled == i).sum() for i in range(1, num + 1)]
        largest = 1 + np.argmax(areas)
        mask = (labeled == largest).astype(np.uint8)
        
    return mask * 255


def mask_inside_region(region_mask, W, H, min_pixels=50):
    """
    在困难区域 mask 内叠加随机形状先验（线/点/自由形），返回 region ∩ shape，
    使「在困难区域上生成」同时具备 AnoStyler 式的点、线、自由形多样性。
    """
    region = np.asarray(region_mask).squeeze()
    if region.max() > 1:
        region = (region > 127).astype(np.uint8)
    else:
        region = (region > 0.5).astype(np.uint8)
    region = cv2.resize(region.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    for _ in range(10): # 增加试探次数
        shape_fn = random.choice([msp_line, msp_dot, msp_freeform, msp_perlin])
        meta = shape_fn(W=W, H=H)
        meta_bin = (meta > 127).astype(np.uint8)
        inter = region * meta_bin
        if inter.sum() >= min_pixels:
            return (inter * 255).astype(np.uint8)
    return (region * 255).astype(np.uint8)


def generate_meta_mask(W=256, H=256, m_max=5, alpha=0.7):
    """生成随机形状组合的 mask，用于无困难 mask 时的 fallback。"""
    base_W, base_H = 256, 256
    indices = np.arange(1, m_max + 1)
    logits = np.exp(-alpha * indices)
    probs = logits / logits.sum()
    m = np.random.choice(indices, p=probs)
    mask_final = np.zeros((base_H, base_W), dtype=np.uint8)
    shape_fns = [msp_line, msp_dot, msp_freeform, msp_perlin]
    for _ in range(m):
        fn = random.choice(shape_fns)
        mask_i = fn(W=base_W, H=base_H)
        mask_final = np.clip(mask_final + (mask_i > 0).astype(np.uint8), 0, 1)
    mask_final = (mask_final * 255).astype(np.uint8)
    resized = cv2.resize(mask_final, (W, H), interpolation=cv2.INTER_NEAREST)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return resized
