#!/usr/bin/env python3
"""
困难样本挖掘脚本

从异常分数图得到困难区域二值 mask，供异常生成器只在困难区域绘制伪异常。

使用方式：
  1) 离线模式（推荐）：先用 One-For-All 对正常图前向并保存异常图为 .npy，
     再本脚本读取并生成 mask：
       python scripts/mine_hard.py --anomaly_map_dir /path/to/npy_dir --save_dir ./outputs/hard_masks

  2) 若已实现 OFA 接口，可通过 --ofa_root 指定 One-For-All 根目录并在脚本内完成前向（见文档）。
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

# 项目根目录加入 path，便于 import 项目内模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.hard_mining import anomaly_map_to_hard_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="困难样本挖掘：异常图 -> 困难区域 mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 输入输出
    parser.add_argument(
        "--anomaly_map_dir",
        type=str,
        default=None,
        help="存放异常图 .npy 的目录（离线模式）；每文件 shape 可为 (H,W) 或 (1,H,W)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/hard_masks",
        help="困难 mask 保存目录，保存为同名 _mask.png",
    )
    parser.add_argument(
        "--out_size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="输出 mask 尺寸，需与生成器输入一致",
    )
    # 挖掘策略
    parser.add_argument(
        "--method",
        type=str,
        default="top_k",
        choices=["top_k", "threshold"],
        help="困难区域选取方式：top_k=取分数最高的比例；threshold=按阈值",
    )
    parser.add_argument(
        "--top_k_ratio",
        type=float,
        default=0.15,
        help="method=top_k 时，取前多少比例像素（0~1）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="method=threshold 时使用的阈值",
    )
    parser.add_argument(
        "--sigma_smooth",
        type=float,
        default=0.0,
        help="对异常图做高斯平滑的 sigma，0 表示不平滑；>0 需安装 scipy",
    )
    parser.add_argument(
        "--min_area_ratio",
        type=float,
        default=0.001,
        help="最小连通区域面积占整图比例，过小区域会被剔除",
    )
    return parser.parse_args()


def run_offline_mode(args):
    """离线模式：从 anomaly_map_dir 读取 .npy，生成 mask 并保存到 save_dir。"""
    if not args.anomaly_map_dir or not os.path.isdir(args.anomaly_map_dir):
        raise FileNotFoundError(
            f"请指定存在的异常图目录: --anomaly_map_dir {args.anomaly_map_dir}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    out_hw = tuple(args.out_size)

    npy_files = sorted(
        f for f in os.listdir(args.anomaly_map_dir)
        if f.endswith(".npy")
    )
    if not npy_files:
        raise FileNotFoundError(
            f"目录下没有 .npy 文件: {args.anomaly_map_dir}"
        )

    for fname in tqdm(npy_files, desc="Mining hard masks"):
        path = os.path.join(args.anomaly_map_dir, fname)
        anomaly_map = np.load(path)

        mask = anomaly_map_to_hard_mask(
            anomaly_map,
            out_hw=out_hw,
            method=args.method,
            top_k_ratio=args.top_k_ratio,
            threshold=args.threshold,
            sigma_smooth=args.sigma_smooth,
            min_area_ratio=args.min_area_ratio,
        )

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.save_dir, f"{base}_mask.png")
        Image.fromarray(mask).save(out_path)

    print(f"已保存 {len(npy_files)} 个 mask 到: {args.save_dir}")


def main():
    args = parse_args()

    if args.anomaly_map_dir is not None:
        run_offline_mode(args)
        return

    # 未指定异常图目录时提示
    print(
        "请使用离线模式：\n"
        "  1) 使用 One-For-All 对正常图做前向，将每张图的 anomaly_map 保存为 .npy\n"
        "  2) 运行: python scripts/mine_hard.py --anomaly_map_dir <npy目录> --save_dir <输出目录>\n"
        "详见 README 或 docs/hard_mining.md"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
