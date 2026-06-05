#!/usr/bin/env python3
"""
DefectFlywheel：调用 OFA 前向，对正常图列表逐张推理，将每张图的 anomaly map 按 basename 保存为 .npy，
供 scripts/mine_hard.py 的 --anomaly_map_dir 使用。

支持两种输入方式：
  - --normal_list list.txt：直接使用已有正常图路径列表；
  - --data_root DIR --category CAT：从数据集目录扫描 {DIR}/{CAT}/train/good/ 生成列表。
"""
import argparse
import os
import subprocess
import sys

# 与 prepare_normal_list 一致的扩展名
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_normal_list_from_dir(data_root, category, max_count=None):
    good_dir = os.path.join(data_root, category, "train", "good")
    if not os.path.isdir(good_dir):
        raise FileNotFoundError(f"目录不存在: {good_dir}")
    paths = []
    for name in sorted(os.listdir(good_dir)):
        if name.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() in EXTENSIONS:
            paths.append(os.path.abspath(os.path.join(good_dir, name)))
            if max_count is not None and len(paths) >= max_count:
                break
    if not paths:
        raise FileNotFoundError(f"未在 {good_dir} 下找到图像文件")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="OFA 前向导出 anomaly map 为 .npy，供 mine_hard 使用",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--normal_list",
        type=str,
        default=None,
        help="正常图路径列表文件（一行一条路径）",
    )
    inp.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="与 --category 合用：数据集根目录（如 .../ZJU-Leaper-Group5-MVTec_dev）",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="与 --data_root 合用：类别子目录名（如 p_id16）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，每张图对应一个 {basename}.npy",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="OFA 数据集根路径（用于构建 memory bank），默认与 data_root 一致或由 normal_list 推断",
    )
    parser.add_argument(
        "--blip_model_path",
        type=str,
        default=None,
        help="BlipDiffusion 模型路径；也可通过环境变量 BLIP_MODEL_PATH 设置",
    )
    parser.add_argument(
        "--export_obj",
        type=str,
        default=None,
        help="OFA 对象/类别名（如 p_id16），不指定时由 OFA 使用 obj_list 第一个",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="zju",
        help="OFA 数据集类型（如 zju, mvtec）",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="可选：导出时加载的 checkpoint（如 baseline/epoch_0.pt），不传则用当前预训练权重",
    )
    parser.add_argument("--epochs", type=int, default=20, help="透传给 main_zju.py，用于日志与导出时 memory bank 协议一致")
    parser.add_argument("--shot", type=int, default=2, help="透传给 main_zju.py，保证导出 anomaly map 使用同一 few-shot support")
    parser.add_argument("--seed", type=int, default=42, help="透传给 main_zju.py，保证 support sampling 与主 run 一致")
    parser.add_argument("--lr", type=float, default=1e-5, help="透传给 main_zju.py，用于完整配置记录")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="透传给 main_zju.py，用于完整配置记录")
    parser.add_argument("--scheduler_t_max", type=int, default=20, help="透传给 main_zju.py，用于完整配置记录")
    parser.add_argument(
        "--max_count",
        type=int,
        default=None,
        help="仅与 --data_root/--category 合用：最多导出张数（便于快速测试）",
    )
    args = parser.parse_args()

    project_root = _project_root()
    ofa_dir = os.path.join(project_root, "ofa")
    if not os.path.isdir(ofa_dir):
        raise FileNotFoundError(f"OFA 目录不存在: {ofa_dir}，请先完成 OFA 迁移")

    created_temp_list = False
    if args.normal_list:
        list_path = os.path.abspath(args.normal_list)
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"列表文件不存在: {list_path}")
        if args.data_path:
            data_path = os.path.abspath(args.data_path)
        else:
            with open(list_path) as f:
                first_line = f.readline().strip()
            data_path = os.path.abspath(args.data_path or "./data/visa")
            if first_line and os.path.isfile(first_line):
                p = os.path.dirname(first_line)
                for _ in range(6):
                    if p and os.path.isdir(p):
                        if os.path.isdir(os.path.join(p, "train", "good")):
                            data_path = p
                            break
                        p = os.path.dirname(p)
    else:
        if not args.category:
            parser.error("使用 --data_root 时需同时指定 --category")
        paths = _build_normal_list_from_dir(args.data_root, args.category, args.max_count)
        import tempfile
        fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="ofa_export_")
        try:
            with os.fdopen(fd, "w") as f:
                for p in paths:
                    f.write(p + "\n")
        except Exception:
            os.unlink(list_path)
            raise
        list_path = os.path.abspath(list_path)
        created_temp_list = True
        data_path = os.path.abspath(args.data_path or args.data_root)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "main_zju.py",
        "--data_path", data_path,
        "--export_npy", list_path,
        "--export_output_dir", output_dir,
        "--export_only",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--shot", str(args.shot),
        "--seed", str(args.seed),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--scheduler_t_max", str(args.scheduler_t_max),
    ]
    if args.export_obj:
        cmd += ["--export_obj", args.export_obj]
    if args.blip_model_path:
        cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        cmd += ["--load_checkpoint_for_export", os.path.abspath(args.checkpoint_path)]

    env = os.environ.copy()
    if args.blip_model_path:
        env["BLIP_MODEL_PATH"] = os.path.abspath(args.blip_model_path)
    env["OFA_DATA_PATH"] = data_path

    print("在 ofa/ 下执行:", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=ofa_dir, env=env)
    if created_temp_list and os.path.isfile(list_path):
        try:
            os.unlink(list_path)
        except Exception:
            pass
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
