#!/usr/bin/env python3
"""
从 MVTec / ZJU-Leaper 风格数据集目录生成正常图路径列表。

- mode=single：扫描 {data_root}/{category}/train/good/，可选 max_count。
- mode=shot_all：与 OFA main_zju 的 shot 选取规则一致，每类取相同索引（0, step, 2*step...），
  保证 baseline 与飞轮使用同一批 shot 图片，实验公平。
"""
import argparse
import json
import os

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SHOT_INDEX_STEP = 3  # 与 ofa/main_zju.py 一致：取第 1、4、7… 张


# 与 ofa/dataset.py generate_class_info 一致，保证 shot 顺序与 OFA 完全相同
OBJ_LIST_ZJU = ["p_id16", "p_id17", "p_id18", "p_id19"]


def get_obj_list_from_meta(data_root, dataset=None):
    """从 meta.json 或 dataset 得到类别列表，与 OFA main_zju 的 obj_list 顺序一致。"""
    if dataset == "wfdd":
        return ['grey_cloth', 'grid_cloth', 'pink_flower', 'yellow_cloth']
    elif dataset in ["fabric-mvtec", "fabric_mvtec"]:
        return ['fabric']
    
    meta_path = os.path.join(data_root, "meta.json")
    if not os.path.isfile(meta_path):
        obj_list = []
        for name in sorted(os.listdir(data_root)):
            if name.startswith("."):
                continue
            good_dir = os.path.join(data_root, name, "train", "good")
            if os.path.isdir(good_dir):
                obj_list.append(name)
        return obj_list
    with open(meta_path) as f:
        meta = json.load(f)
    train = meta.get("train", meta)
    return sorted(train.keys())


def run_shot_all(data_root, output, shot, shot_index_step, dataset=None):
    """与 OFA shot 规则一致：每类 train/good 按文件名排序，取索引 0, step, 2*step, ..."""
    data_root = os.path.abspath(data_root)
    obj_list = get_obj_list_from_meta(data_root, dataset=dataset)
    if not obj_list:
        raise FileNotFoundError(f"未在 {data_root} 下找到类别或 meta.json")
    paths = []
    for obj in obj_list:
        good_dir = os.path.join(data_root, obj, "train", "good")
        if not os.path.isdir(good_dir):
            raise FileNotFoundError(f"目录不存在: {good_dir}")
        # 与 ofa/main_zju.py 一致：支持多种图像后缀
        files = sorted([f for f in os.listdir(good_dir) if f.lower().endswith(EXTENSIONS)])
        if not files:
            raise FileNotFoundError(f"未在 {good_dir} 下找到图像")
        for i in range(shot):
            idx = min(i * shot_index_step, len(files) - 1)
            paths.append(os.path.abspath(os.path.join(good_dir, files[idx])))
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"shot_all: 已写入 {len(paths)} 条路径（与 OFA shot 一致）到: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="从数据集目录生成正常图路径列表（train/good）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="数据集根目录（如 .../ZJU-Leaper-Group5-MVTec_dev）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "shot_all"],
        help="single: 单类别 + category/max_count；shot_all: 每类按 OFA 规则取 shot 张，保证与 baseline 同 shot",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="单类别模式下的类别子目录名（如 p_id16）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="normal_list.txt",
        help="输出列表文件路径，每行一条图像绝对路径",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=None,
        help="单类别模式：最多写入条数；不指定则全部写入",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=2,
        help="shot_all 模式：每类取几张（与 OFA --shot 一致）",
    )
    parser.add_argument(
        "--shot_index_step",
        type=int,
        default=SHOT_INDEX_STEP,
        help="shot_all 模式：索引步长，与 main_zju SHOT_INDEX_STEP 一致",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="shot_all 时若为 zju，使用与 ofa/dataset.py 相同的 obj_list 顺序，保证与 baseline 完全一致",
    )
    args = parser.parse_args()

    if args.mode == "shot_all":
        run_shot_all(
            args.data_root, args.output, args.shot, args.shot_index_step, dataset=args.dataset
        )
        return

    if not args.category:
        raise ValueError("single 模式必须指定 --category")
    good_dir = os.path.join(args.data_root, args.category, "train", "good")
    if not os.path.isdir(good_dir):
        raise FileNotFoundError(f"目录不存在: {good_dir}")

    paths = []
    for name in sorted(os.listdir(good_dir)):
        if name.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() in EXTENSIONS:
            paths.append(os.path.abspath(os.path.join(good_dir, name)))
            if args.max_count is not None and len(paths) >= args.max_count:
                break

    if not paths:
        raise FileNotFoundError(f"未在 {good_dir} 下找到图像文件")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"已写入 {len(paths)} 条路径到: {args.output}")


if __name__ == "__main__":
    main()
