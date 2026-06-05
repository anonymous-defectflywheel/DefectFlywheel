#!/usr/bin/env python3
"""
协同训练对接 OFA：整理模块二生成结果，得到「正常图列表 + 伪异常图列表」，
供 One-For-All 训练时替代原「正常特征相加+加噪」的伪异常数据源。

不修改 OFA 内部逻辑；仅生成 OFA 可读的列表文件，并可选调用 OFA 训练脚本。
"""
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="整理 DefectFlywheel 生成结果，供 OFA 训练使用",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="模块二输出目录（含 image/、mask/），如 ./outputs/generated_anomaly",
    )
    parser.add_argument(
        "--normal_list",
        type=str,
        required=True,
        help="生成时使用的正常图列表文件（与生成顺序一致，一行一张图）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/co_train_manifest",
        help="写入 normal_list.txt 与 synthetic_anomaly_list.txt 的目录",
    )
    parser.add_argument(
        "--ofa_root",
        type=str,
        default=None,
        help="OFA 仓库根目录；若指定，将设置环境变量供 OFA 读取列表路径",
    )
    parser.add_argument(
        "--ofa_train_cmd",
        type=str,
        default=None,
        help="OFA 训练命令，如 'python main_zju.py --data_path ... --dataset zju'；若与 --ofa_root 同时指定则在本脚本中执行",
    )
    parser.add_argument(
        "--blip_model_path",
        type=str,
        default=None,
        help="BlipDiffusion 模型目录；传入后通过环境变量 BLIP_MODEL_PATH 传给 OFA。不传时 main_zju 在飞轮模式下会尝试使用 ofa/blipdiffusion_model",
    )
    args = parser.parse_args()

    image_dir = os.path.join(args.generated_dir, "image")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"生成目录下缺少 image/: {args.generated_dir}")

    with open(args.normal_list, "r") as f:
        normal_paths = [line.strip() for line in f if line.strip()]
    if not normal_paths:
        raise ValueError(f"normal_list 为空或无效: {args.normal_list}")

    # 生成图按 gen_ano_0, gen_ano_1, ... 与 normal_list 顺序一一对应
    synthetic_paths = []
    for i in range(len(normal_paths)):
        jpg = os.path.join(image_dir, f"gen_ano_{i}.jpg")
        synthetic_paths.append(os.path.abspath(jpg) if os.path.isfile(jpg) else "")

    pairs = [(n, s) for n, s in zip(normal_paths, synthetic_paths) if s and os.path.isfile(s)]
    if not pairs:
        raise FileNotFoundError(f"未在 {image_dir} 找到与 normal_list 数量匹配的 gen_ano_*.jpg")
    normal_paths = [p[0] for p in pairs]
    synthetic_paths = [p[1] for p in pairs]

    os.makedirs(args.output_dir, exist_ok=True)
    normal_list_out = os.path.join(args.output_dir, "normal_list.txt")
    synthetic_list_out = os.path.join(args.output_dir, "synthetic_anomaly_list.txt")

    with open(normal_list_out, "w") as f:
        for p in normal_paths:
            f.write(os.path.abspath(p) + "\n")
    with open(synthetic_list_out, "w") as f:
        for p in synthetic_paths:
            f.write(p + "\n")

    print(f"已写入 {len(normal_paths)} 条正常图路径: {normal_list_out}")
    print(f"已写入 {len(synthetic_paths)} 条伪异常图路径: {synthetic_list_out}")

    if args.ofa_root and args.ofa_train_cmd:
        env = os.environ.copy()
        env["DEFECT_FLYWHEEL_NORMAL_LIST"] = os.path.abspath(normal_list_out)
        env["DEFECT_FLYWHEEL_SYNTHETIC_LIST"] = os.path.abspath(synthetic_list_out)
        env["DEFECT_FLYWHEEL_GENERATED_DIR"] = os.path.abspath(args.generated_dir)
        if args.blip_model_path:
            env["BLIP_MODEL_PATH"] = os.path.abspath(args.blip_model_path)
        cwd = args.ofa_root
        print(f"在 OFA 根目录执行: {args.ofa_train_cmd}")
        subprocess.run(args.ofa_train_cmd, shell=True, cwd=cwd, env=env)
    else:
        print(
            "OFA 接入说明：在 OFA 训练脚本中读取上述两个列表，"
            "用「正常图 + 伪异常图」作为训练数据，替代原特征加噪构造。"
        )
        print(
            "可选：指定 --ofa_root 与 --ofa_train_cmd 由本脚本直接调用 OFA 训练。"
        )
        print("详见 docs/co_train_ofa_interface.md 了解迁移后 OFA 如何读取两份列表。")


if __name__ == "__main__":
    main()
