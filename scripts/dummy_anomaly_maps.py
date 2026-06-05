#!/usr/bin/env python3
"""
生成测试用 dummy 异常图 .npy，用于在不运行 OFA 的情况下跑通「mine_hard → generate_anomaly」全流程。

根据 normal_list 或图像目录，为每张图生成同名 .npy，写入 output_dir。
模式：random / constant / blob（高斯块）/ diverse（点、线、自由形，与 AnoStyler 形状多样性一致）。
"""
import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from scripts.path_ids import sample_id_from_path

# 仅 diverse 模式需要 generator 子模块
def _load_meta_shape_priors():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from models.generator.meta_shape_priors import (
        generate_meta_mask,
        msp_line,
        msp_dot,
        msp_freeform,
    )
    return generate_meta_mask, msp_line, msp_dot, msp_freeform

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_image_paths(normal_list=None, image_dir=None):
    if normal_list and os.path.isfile(normal_list):
        with open(normal_list, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        return [p for p in paths if os.path.isfile(p)]
    if image_dir and os.path.isdir(image_dir):
        paths = []
        for name in sorted(os.listdir(image_dir)):
            if os.path.splitext(name)[1].lower() in EXTENSIONS:
                paths.append(os.path.abspath(os.path.join(image_dir, name)))
        return paths
    return []


def main():
    parser = argparse.ArgumentParser(
        description="生成测试用 dummy .npy，供 mine_hard 使用",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--normal_list",
        type=str,
        default=None,
        help="正常图路径列表文件，每行一条路径",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="或直接指定图像目录（与 --normal_list 二选一）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="数据集根目录，用于生成类别感知 sample_id",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=".npy 输出目录",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help=".npy 的 (H, W) 尺寸，与 mine_hard 的 out_size 一致",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="blob",
        choices=["random", "constant", "blob", "diverse"],
        help="random: 随机 [0,1]；constant: 固定 0.5；blob: 1~2 个高斯 blob；diverse: 点/线/freeform 随机形状（与 AnoStyler 一致，困难 mask 多样）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（mode=random 时有效）",
    )
    args = parser.parse_args()

    if not args.normal_list and not args.image_dir:
        raise ValueError("请指定 --normal_list 或 --image_dir 之一")
    if args.normal_list and args.image_dir:
        raise ValueError("请只指定 --normal_list 或 --image_dir 之一")

    paths = collect_image_paths(args.normal_list, args.image_dir)
    if not paths:
        raise FileNotFoundError("未找到任何图像路径")

    H, W = args.size
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    def make_blob(h, w):
        arr = np.zeros((h, w), dtype=np.float32)
        n_centers = np.random.randint(1, 3)
        for _ in range(n_centers):
            cy, cx = np.random.randint(0, h), np.random.randint(0, w)
            sy, sx = np.random.uniform(h * 0.08, h * 0.25), np.random.uniform(w * 0.08, w * 0.25)
            y = np.arange(h, dtype=np.float32)[:, None]
            x = np.arange(w, dtype=np.float32)[None, :]
            arr += np.exp(-((y - cy) ** 2 / (2 * sy**2) + (x - cx) ** 2 / (2 * sx**2)))
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr

    if args.mode == "diverse":
        generate_meta_mask, msp_line, msp_dot, msp_freeform = _load_meta_shape_priors()
        shape_fns = [msp_line, msp_dot, msp_freeform]

    for path in paths:
        base = sample_id_from_path(path, args.data_root)
        out_path = os.path.join(args.output_dir, base + ".npy")
        if args.mode == "random":
            arr = np.random.rand(H, W).astype(np.float32)
        elif args.mode == "constant":
            arr = np.full((H, W), 0.5, dtype=np.float32)
        elif args.mode == "blob":
            arr = make_blob(H, W)
        else:
            # diverse: 每张图随机选线/点/freeform，转为 [0,1] 异常图供 mine_hard
            fn = np.random.choice(shape_fns)
            mask_uint8 = fn(W=W, H=H)
            arr = (mask_uint8.astype(np.float32) / 255.0)
        np.save(out_path, arr)

    print(f"已生成 {len(paths)} 个 .npy 到: {args.output_dir}")


if __name__ == "__main__":
    main()
