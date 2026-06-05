"""测试困难样本挖掘：anomaly_map -> hard mask."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.hard_mining import anomaly_map_to_hard_mask, similarity_map_to_hard_mask


def test_anomaly_map_to_hard_mask():
    # 模拟 16x16 异常图，右上角分数高
    np.random.seed(42)
    am = np.random.rand(16, 16).astype(np.float32)
    am[0:4, 12:16] = 0.9
    am[8:12, 8:12] = 0.85

    mask = anomaly_map_to_hard_mask(
        am,
        out_hw=(256, 256),
        method="top_k",
        top_k_ratio=0.15,
    )
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask.sum() > 0


def test_similarity_map_to_hard_mask():
    # 熵高区域：pos 和 neg 接近
    H, W = 8, 8
    pos_sim = np.random.rand(H, W).astype(np.float32) * 0.5 + 0.25
    neg_sim = 1.0 - pos_sim + np.random.rand(H, W).astype(np.float32) * 0.2

    mask = similarity_map_to_hard_mask(
        pos_sim, neg_sim,
        out_hw=(64, 64),
        method="entropy",
        entropy_top_k_ratio=0.2,
    )
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8


if __name__ == "__main__":
    test_anomaly_map_to_hard_mask()
    test_similarity_map_to_hard_mask()
    print("All tests passed.")
