"""
困难样本挖掘模块：从异常分数图得到困难区域 mask，供异常生成器使用。
"""
from .mining import (
    anomaly_map_to_hard_mask,
    similarity_map_to_hard_mask,
)

__all__ = [
    "anomaly_map_to_hard_mask",
    "similarity_map_to_hard_mask",
]
