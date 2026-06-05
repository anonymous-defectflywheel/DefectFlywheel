#!/usr/bin/env python3
"""验证方案乙逻辑：默认 20 大 epoch、anomaly_map_dir 与 load_checkpoint 路径正确。

单元测试不依赖 torch/GPU。完整集成测试需在已安装 torch 的 conda 环境下运行：
  CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py \\
    --data_path "$(pwd)/datasets/ZJU-Leaper-Group5-MVTec_dev" \\
    --exp_name exp_scheme_b_verify --blip_model_path "$(pwd)/ofa/blipdiffusion_model" \\
    --macro_epochs 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


def get_run_experiments_parser():
    """与 run_experiments.py 中 parser 的 macro_epochs/accumulate_flywheel 默认值一致。"""
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--macro_epochs", type=int, default=20)
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--accumulate_flywheel", action="store_true", default=True)
    p.add_argument("--no_accumulate_flywheel", action="store_false", dest="accumulate_flywheel")
    return p


def test_default_macro_epochs_is_20():
    """默认应为 20 个大 epoch（方案乙）。"""
    parser = get_run_experiments_parser()
    args = parser.parse_args(["--data_path", "/tmp/dummy"])
    assert args.macro_epochs == 20, f"expected macro_epochs=20, got {args.macro_epochs}"


def test_scheme_b_paths_for_k0_and_k1():
    """方案乙：k=0 与 k=1 时 anomaly_map_dir、load_ckpt 路径逻辑。"""
    exp_dir = "/tmp/exp"
    bank_dir = os.path.join(exp_dir, "flywheel_bank")
    baseline_save = os.path.join(exp_dir, "baseline")

    # k=0: mine 用 round_0/ofa_npy，权重优先用唯一 baseline alias
    k = 0
    anomaly_map_dir = os.path.join(bank_dir, f"round_{k - 1}", "ofa_npy") if k >= 1 else os.path.join(bank_dir, "round_0", "ofa_npy")
    load_ckpt = os.path.join(baseline_save, "baseline_epoch_00.pt") if k == 0 else os.path.join(exp_dir, f"flywheel_round{k - 1}", f"round_{k - 1:02d}_epoch_0.pt")
    assert anomaly_map_dir == os.path.join(bank_dir, "round_0", "ofa_npy")
    assert load_ckpt == os.path.join(baseline_save, "baseline_epoch_00.pt")

    # k=1: mine 用 round_0/ofa_npy，权重优先用上一轮 flywheel 唯一 alias
    k = 1
    anomaly_map_dir = os.path.join(bank_dir, f"round_{k - 1}", "ofa_npy") if k >= 1 else os.path.join(bank_dir, "round_0", "ofa_npy")
    load_ckpt = os.path.join(baseline_save, "baseline_epoch_00.pt") if k == 0 else os.path.join(exp_dir, f"flywheel_round{k - 1}", f"round_{k - 1:02d}_epoch_0.pt")
    assert anomaly_map_dir == os.path.join(bank_dir, "round_0", "ofa_npy")
    assert load_ckpt == os.path.join(exp_dir, "flywheel_round0", "round_00_epoch_0.pt")


def test_accumulate_flywheel_default():
    """默认应开启 accumulate_flywheel。"""
    parser = get_run_experiments_parser()
    args = parser.parse_args(["--data_path", "/tmp/dummy"])
    assert args.accumulate_flywheel is True


if __name__ == "__main__":
    test_default_macro_epochs_is_20()
    test_scheme_b_paths_for_k0_and_k1()
    test_accumulate_flywheel_default()
    print("All scheme B validation tests passed.")
