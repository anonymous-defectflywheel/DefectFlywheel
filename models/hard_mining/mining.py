"""
困难样本挖掘核心逻辑：从 patch 级异常分数图或正/负相似度图得到困难区域二值 mask。

- 困难区域定义：异常分数较高（模型易误判为异常）或正/负相似度接近（熵高、不确定）
- 输出与图像同尺寸的二值 mask，供生成器只在困难区域绘制异常
"""

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    ndimage = None


def anomaly_map_to_hard_mask(
    anomaly_map,
    out_hw=None,
    method="top_k",
    top_k_ratio=0.15,
    threshold=None,
    sigma_smooth=0.0,
    min_area_ratio=0.001,
):
    """
    从单通道异常分数图得到困难区域二值 mask。

    Args:
        anomaly_map: (H, W) 或 (1, H, W)，数值范围建议 [0, 1]，越高表示越像异常
        out_hw: (height, width) 目标尺寸，None 则与 anomaly_map 一致
        method: "top_k" 或 "threshold"
            - "top_k": 取分数最高的 top_k_ratio 比例像素作为困难区域
            - "threshold": 分数 >= threshold 的像素为困难区域
        top_k_ratio: method=="top_k" 时使用，取值 (0, 1]，例如 0.15 表示前 15%
        threshold: method=="threshold" 时使用
        sigma_smooth: 对异常图做高斯平滑的 sigma，0 表示不平滑
        min_area_ratio: 最小连通区域面积占整图比例，小于该比例的连通块会被剔除

    Returns:
        mask: (H, W) uint8，0 或 255，与 out_hw 对应
    """
    if hasattr(anomaly_map, "detach"):
        am = anomaly_map.detach().cpu().numpy().squeeze().astype(np.float64)
    else:
        am = np.asarray(anomaly_map, dtype=np.float64).squeeze()

    if am.ndim != 2:
        raise ValueError(f"anomaly_map 应为 2D 或可 squeeze 为 2D，当前 shape={am.shape}")

    # 可选：高斯平滑，减少噪声导致的碎片
    if sigma_smooth > 0:
        if not HAS_SCIPY:
            raise ImportError("sigma_smooth > 0 需要 scipy，请 pip install scipy")
        am = ndimage.gaussian_filter(am, sigma=sigma_smooth)

    # 二值化：困难 = 分数高
    if method == "top_k":
        k = max(1, int(am.size * top_k_ratio))
        flat = am.ravel()
        th = np.partition(flat, -k)[-k]
        # 稀疏图下 th 易为 0，导致 am>=0 全为真、mask 全白；改为仅保留恰好前 k 大像素
        if th <= 1e-9:
            idx = np.argpartition(flat, -k)[-k:]
            hard = np.zeros(flat.shape, dtype=bool)
            hard[idx] = True
            hard = hard.reshape(am.shape)
        else:
            hard = am >= th
    elif method == "threshold":
        th = threshold if threshold is not None else 0.5
        hard = (am >= th).astype(bool)
    else:
        raise ValueError(f"method 应为 'top_k' 或 'threshold'，当前为 {method}")

    # 连通区域过滤：去掉过小的碎片（需要 scipy）
    # 若过滤后全空（如 dummy 随机 .npy 导致像素分散），则回退为未过滤结果，保证至少有一块可绘制区域
    if min_area_ratio > 0 and HAS_SCIPY:
        hard_before_filter = hard.copy()
        labeled, num_features = ndimage.label(hard.astype(np.uint8))
        min_pixels = max(1, int(am.size * min_area_ratio))
        for i in range(1, num_features + 1):
            if (labeled == i).sum() < min_pixels:
                hard[labeled == i] = False
        if hard.sum() == 0:
            hard = hard_before_filter

    mask = (hard.astype(np.uint8)) * 255

    if out_hw is not None:
        from PIL import Image
        mask = np.asarray(
            Image.fromarray(mask).resize((out_hw[1], out_hw[0]), Image.NEAREST),
            dtype=np.uint8,
        )

    return mask


def similarity_map_to_hard_mask(
    pos_sim,
    neg_sim,
    out_hw=None,
    method="entropy",
    entropy_top_k_ratio=0.15,
    high_neg_ratio=0.15,
):
    """
    从正/负 prompt 与 patch 的相似度图得到困难区域。

    - 熵高：pos_sim 与 neg_sim 接近，模型不确定
    - 高 neg：neg_sim 高，模型易将正常区域判为异常（假阳性区域）

    Args:
        pos_sim: (H, W) 或 (1, H, W)，正类相似度，范围建议 [0, 1]
        neg_sim: (H, W) 或 (1, H, W)，负类相似度
        out_hw: 输出 mask 尺寸 (height, width)
        method: "entropy" | "high_neg" | "combined"
        entropy_top_k_ratio: 熵法取前多少比例
        high_neg_ratio: 高 neg 法取前多少比例

    Returns:
        mask: (H, W) uint8，0 或 255
    """
    def _to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().squeeze()
        return np.asarray(x).squeeze()

    p = _to_np(pos_sim).astype(np.float64)
    n = _to_np(neg_sim).astype(np.float64)

    # 熵：在 (p, n) 归一化后的分布上计算，熵高表示不确定
    s = p + n + 1e-8
    pn = p / s
    nn = n / s
    entropy = -pn * np.log(pn + 1e-8) - nn * np.log(nn + 1e-8)

    if method == "entropy":
        return anomaly_map_to_hard_mask(
            entropy,
            out_hw=out_hw,
            method="top_k",
            top_k_ratio=entropy_top_k_ratio,
        )
    if method == "high_neg":
        return anomaly_map_to_hard_mask(
            n,
            out_hw=out_hw,
            method="top_k",
            top_k_ratio=high_neg_ratio,
        )
    if method == "combined":
        mask_e = anomaly_map_to_hard_mask(
            entropy,
            out_hw=None,
            method="top_k",
            top_k_ratio=entropy_top_k_ratio,
        )
        mask_n = anomaly_map_to_hard_mask(
            n,
            out_hw=None,
            method="top_k",
            top_k_ratio=high_neg_ratio,
        )
        mask = np.clip(
            mask_e.astype(np.int32) + mask_n.astype(np.int32),
            0,
            255,
        ).astype(np.uint8)
        if out_hw is not None:
            from PIL import Image
            mask = np.asarray(
                Image.fromarray(mask).resize((out_hw[1], out_hw[0]), Image.NEAREST),
                dtype=np.uint8,
            )
        return mask

    raise ValueError(
        f"method 应为 'entropy' | 'high_neg' | 'combined'，当前为 {method}"
    )
