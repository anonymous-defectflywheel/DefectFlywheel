#!/usr/bin/env python3
"""
飞轮迭代脚本：多轮「困难挖掘 → 伪异常生成 → 伪样本库更新 → OFA 训练」。

每轮：OFA 导出 .npy → mine_hard → generate_anomaly → 更新伪样本库（accumulate 或 replace）→ OFA 训练。
伪样本库：bank_dir 下维护 bank/normal_list.txt 与 bank/synthetic_anomaly_list.txt，供 OFA 飞轮训练使用。

用法示例：
  python scripts/flywheel_iterate.py \\
    --normal_list ./outputs/normal_list.txt \\
    --data_path "$(pwd)/datasets/ZJU-Leaper-Group5-MVTec_dev" \\
    --num_rounds 2 \\
    --bank_dir ./outputs/flywheel_bank \\
    --accumulate \\
    --ofa_root ./ofa
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, cwd=None, env=None, check=True):
    if env is None:
        env = os.environ.copy()
    else:
        env = {**os.environ, **env}
    cwd = cwd or _project_root()
    print("[RUN]", " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"命令退出码 {r.returncode}")
    return r



def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _count_list_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _validate_hard_masks(hard_masks_dir, normal_list):
    expected = _count_list_lines(normal_list)
    actual = len([f for f in os.listdir(hard_masks_dir) if f.endswith("_mask.png")]) if os.path.isdir(hard_masks_dir) else 0
    if actual != expected:
        raise RuntimeError(
            f"hard mask count mismatch: expected {expected} masks from {normal_list}, "
            f"found {actual} in {hard_masks_dir}; possible sample_id collision or missing export"
        )
    print(f"[Validate] hard masks count OK: {actual}/{expected} -> {hard_masks_dir}")


def _write_checkpoint_alias(src, dst):
    if os.path.isfile(src):
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        return dst
    return None


def _write_round_config(path, args, payload):
    body = {
        "round_index": getattr(args, "round_index", None),
        "seed": args.seed,
        "shot": args.shot,
        "dataset": args.dataset,
        "mine_method": args.mine_method,
        "mine_top_k_ratio": args.mine_top_k_ratio,
        "mine_threshold": args.mine_threshold,
        "context_backend": args.context_backend,
        "accumulate": args.accumulate,
        "load_checkpoint": getattr(args, "load_checkpoint", None),
        "args": vars(args),
    }
    body.update(payload)
    _write_json(path, body)


def _run_one_round(
    args,
    root,
    bank_dir,
    bank_normal,
    bank_synthetic,
    normal_list,
    data_path,
    ofa_root,
    blip,
    all_normals,
    all_synthetics,
    _run,
):
    """方案乙单轮：mine(anomaly_map_dir) → generate → 更新伪样本库 → OFA 1 epoch（从 load_checkpoint 起）。"""
    k = args.round_index
    rd = os.path.join(bank_dir, f"round_{int(k):02d}")
    os.makedirs(rd, exist_ok=True)
    anomaly_map_dir = os.path.abspath(args.anomaly_map_dir)
    hard_masks_dir = os.path.join(rd, "hard_masks")
    gen_dir = os.path.join(rd, "gen")
    manifest_dir = os.path.join(rd, "manifest")

    reuse_static_generation = bool(
        getattr(args, "static_generation", False)
        and k is not None
        and k > 0
        and os.path.isfile(bank_normal)
        and os.path.isfile(bank_synthetic)
    )
    if reuse_static_generation:
        print(f"[Ablation Static] round {k}: reuse existing pseudo-defect bank without re-mining/re-generating.")
        env = {
            "DEFECT_FLYWHEEL_NORMAL_LIST": bank_normal,
            "DEFECT_FLYWHEEL_SYNTHETIC_LIST": bank_synthetic,
        }
        if blip:
            env["BLIP_MODEL_PATH"] = blip
        ofa_save = getattr(args, "ofa_save_path", None)
        if ofa_save:
            base = ofa_save.rstrip("/") + f"_round_{int(k):02d}"
            ofa_save_round = os.path.join(root, base) if not os.path.isabs(ofa_save) else base
            ofa_save_round = os.path.abspath(ofa_save_round)
            os.makedirs(ofa_save_round, exist_ok=True)
        ofa_cmd = [
            sys.executable, "main_zju.py",
            "--data_path", data_path,
            "--dataset", args.dataset,
            "--epochs", "1",
            "--shot", str(args.shot),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--scheduler_t_max", "1",
            "--seed", str(args.seed),
        ]
        if args.display_epoch:
            ofa_cmd += ["--display_epoch", args.display_epoch]
        if args.display_epochs:
            ofa_cmd += ["--display_epochs", args.display_epochs]
        if args.skip_test:
            ofa_cmd += ["--skip_test"]
        if ofa_save:
            ofa_cmd += ["--save_path", ofa_save_round]
        if blip:
            ofa_cmd += ["--blip_model_path", blip]
        if getattr(args, "load_checkpoint", None) and os.path.isfile(args.load_checkpoint):
            ofa_cmd += ["--load_checkpoint", os.path.abspath(args.load_checkpoint)]
        _run(ofa_cmd, cwd=ofa_root, env=env)
        if ofa_save:
            _write_checkpoint_alias(
                os.path.join(ofa_save_round, "epoch_0.pt"),
                os.path.join(ofa_save_round, f"round_{k:02d}_epoch_0.pt"),
            )
        _write_round_config(
            os.path.join(rd, "round_config.json"),
            args,
            {"static_generation_reuse": True, "normal_bank": bank_normal, "synthetic_bank": bank_synthetic},
        )
        return

    # accumulate 时先读取已有 bank
    normals = list(all_normals)
    synthetics = list(all_synthetics)
    if args.accumulate and os.path.isfile(bank_normal) and os.path.isfile(bank_synthetic):
        with open(bank_normal) as f:
            normals = [line.strip() for line in f if line.strip()]
        with open(bank_synthetic) as f:
            synthetics = [line.strip() for line in f if line.strip()]

    # 1) Mine or random-region ablation
    if args.mine_method == "random":
        print("[Ablation Random] single-round: skip hard mining and use generator meta masks.")
        hard_masks_dir_for_gen = None
    else:
        mine_cmd = [
            sys.executable, "scripts/mine_hard.py",
            "--anomaly_map_dir", anomaly_map_dir,
            "--save_dir", hard_masks_dir,
            "--method", args.mine_method,
        ]
        if args.mine_method == "top_k":
            mine_cmd += ["--top_k_ratio", str(args.mine_top_k_ratio)]
        else:
            mine_cmd += ["--threshold", str(args.mine_threshold)]
        _run(mine_cmd, cwd=root)
        _validate_hard_masks(hard_masks_dir, normal_list)
        hard_masks_dir_for_gen = hard_masks_dir

    # 2) Generate
    cmd_gen = [
            sys.executable, "scripts/generate_anomaly.py",
            "--config", args.config,
            "--normal_list", normal_list,
            "--data_root", data_path,
            "--save_dir", gen_dir,
            "--num_gen", "1",
            "--context_backend", args.context_backend,
            "--seed", str(args.seed),
    ]
    if hard_masks_dir_for_gen:
        cmd_gen.extend(["--hard_mask_dir", hard_masks_dir_for_gen])
    if getattr(args, 'blip_model_path', None):
        cmd_gen.extend(["--blip_model_path", blip])
    _run(cmd_gen, cwd=root)

    _write_round_config(
        os.path.join(rd, "round_config.json"),
        args,
        {
            "anomaly_map_dir": anomaly_map_dir,
            "hard_masks_dir": hard_masks_dir,
            "generated_dir": gen_dir,
            "manifest_dir": manifest_dir,
            "normal_list": normal_list,
        },
    )

    # 3) Manifest
    _run(
        [
            sys.executable, "scripts/co_train.py",
            "--generated_dir", gen_dir,
            "--normal_list", normal_list,
            "--output_dir", manifest_dir,
        ],
        cwd=root,
    )

    # 4) 更新伪样本库（replace 或 accumulate）
    with open(os.path.join(manifest_dir, "normal_list.txt")) as f:
        r_normals = [line.strip() for line in f if line.strip()]
    with open(os.path.join(manifest_dir, "synthetic_anomaly_list.txt")) as f:
        r_synthetics = [line.strip() for line in f if line.strip()]
    if len(r_normals) != len(r_synthetics):
        raise ValueError(f"本轮 manifest 行数不一致: normal {len(r_normals)} vs synthetic {len(r_synthetics)}")
    if args.accumulate:
        normals.extend(r_normals)
        synthetics.extend(r_synthetics)
    else:
        normals = list(r_normals)
        synthetics = list(r_synthetics)

    with open(bank_normal, "w") as f:
        for p in normals:
            f.write(os.path.abspath(p) + "\n")
    with open(bank_synthetic, "w") as f:
        for p in synthetics:
            f.write(p + "\n")
    print(f"[Step 5/6] 伪样本库当前共 {len(normals)} 对 (正常, 伪异常)")

    # 5) OFA 飞轮训练 1 个 epoch（从 load_checkpoint 起，方案乙）
    env = {
        "DEFECT_FLYWHEEL_NORMAL_LIST": bank_normal,
        "DEFECT_FLYWHEEL_SYNTHETIC_LIST": bank_synthetic,
    }
    if blip:
        env["BLIP_MODEL_PATH"] = blip
    ofa_save = getattr(args, "ofa_save_path", None)
    if ofa_save:
        base = ofa_save.rstrip("/") + f"_round_{int(k):02d}"
        ofa_save_round = os.path.join(root, base) if not os.path.isabs(ofa_save) else base
        ofa_save_round = os.path.abspath(ofa_save_round)
        os.makedirs(ofa_save_round, exist_ok=True)
    ofa_cmd = [
        sys.executable, "main_zju.py",
        "--data_path", data_path,
        "--dataset", args.dataset,
        "--epochs", "1",
        "--shot", str(args.shot),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--scheduler_t_max", "1",
        "--seed", str(args.seed),
    ]
    if args.display_epoch:
        ofa_cmd += ["--display_epoch", args.display_epoch]
    if args.display_epochs:
        ofa_cmd += ["--display_epochs", args.display_epochs]
    if args.round_index is not None:
        ofa_cmd += ["--checkpoint_name", f"flywheel_epoch_{int(args.round_index):02d}.pt"]
    if args.skip_test:
        ofa_cmd += ["--skip_test"]

    if ofa_save:
        ofa_cmd += ["--save_path", ofa_save_round]
    if blip:
        ofa_cmd += ["--blip_model_path", blip]
    if getattr(args, "load_checkpoint", None) and os.path.isfile(args.load_checkpoint):
        ofa_cmd += ["--load_checkpoint", os.path.abspath(args.load_checkpoint)]
    _run(ofa_cmd, cwd=ofa_root, env=env)
    if ofa_save:
        _write_checkpoint_alias(
            os.path.join(ofa_save_round, "epoch_0.pt"),
            os.path.join(ofa_save_round, f"round_{k:02d}_epoch_0.pt"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="飞轮迭代：多轮 困难挖掘→生成→伪样本库→OFA 训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--normal_list", type=str, required=True, help="正常图路径列表（每轮共用）")
    parser.add_argument("--data_path", type=str, required=True, help="OFA 数据集根目录")
    parser.add_argument("--num_rounds", type=int, default=2, help="迭代轮数")
    parser.add_argument("--bank_dir", type=str, default="./outputs/flywheel_bank", help="伪样本库根目录")
    parser.add_argument(
        "--accumulate",
        action="store_true",
        default=True,
        help="每轮将新 (正常,伪异常) 对追加到伪样本库；否则每轮替换",
    )
    parser.add_argument("--no_accumulate", action="store_false", dest="accumulate")
    parser.add_argument("--ofa_root", type=str, default="./ofa", help="OFA 根目录")
    parser.add_argument("--dataset", type=str, default="zju", help="OFA 数据集名，与 run_experiments/main_zju 一致以保证公平")
    parser.add_argument("--blip_model_path", type=str, default=None, help="BlipDiffusion 路径，不传则用 ofa/blipdiffusion_model")
    parser.add_argument("--mine_method", type=str, default="top_k", choices=["top_k", "threshold", "random"])
    parser.add_argument("--mine_top_k_ratio", type=float, default=0.1)
    parser.add_argument("--mine_threshold", type=float, default=0.5)
    parser.add_argument("--config", type=str, default="configs/generator.yaml", help="生成器配置")
    parser.add_argument("--context_backend", type=str, default="blipdiffusion_qformer", choices=["blipdiffusion_qformer", "blip_caption", "static_prompt"], help="传给 generate_anomaly.py 的上下文来源")
    parser.add_argument("--disable_blip", action="store_true", help="兼容旧消融脚本：强制 context_backend=static_prompt")
    parser.add_argument("--static_generation", action="store_true", help="消融：只在第 0 轮生成一次伪异常，后续轮次复用同一伪样本库")
    parser.add_argument("--ofa_save_path", type=str, default=None, help="OFA 每轮结果与 log 写入该目录下 round_1, round_2…，便于与 baseline 对比")
    parser.add_argument("--baseline_save_dir", type=str, default=None, help="baseline 保存目录；第 2 轮及以后导出时使用 baseline/epoch_0.pt 作为权重")
    parser.add_argument("--anomaly_map_dir", type=str, default=None, help="单轮模式：直接用该目录的 .npy 做 mine（上一轮 baseline 的 round_{k-1}/ofa_npy）")
    parser.add_argument("--round_index", type=int, default=None, help="单轮模式：当前大 epoch 下标 k")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="单轮模式：OFA 飞轮训练起点 ckpt")
    # 与 run_experiments 一致的训练参数，保证实验公平
    parser.add_argument("--epochs", type=int, default=20, help="OFA 训练轮数")
    parser.add_argument("--shot", type=int, default=2, help="每类 few-shot 数量")
    parser.add_argument("--lr", type=float, default=1e-5, help="OFA 学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="OFA weight_decay")
    parser.add_argument("--scheduler_t_max", type=int, default=20, help="OFA CosineAnnealingLR T_max")
    parser.add_argument("--seed", type=int, default=42, help="OFA 随机种子；首轮初步实验固定为 42")
    parser.add_argument("--display_epoch", type=str, default=None)
    parser.add_argument("--display_epochs", type=str, default=None)
    parser.add_argument("--skip_test", action="store_true", help="如果设置了，则传递给 main_zju.py 跳过测试")
    args = parser.parse_args()
    if getattr(args, "disable_blip", False):
        args.context_backend = "static_prompt"

    root = _project_root()
    bank_dir = os.path.abspath(args.bank_dir)
    ofa_root = os.path.abspath(args.ofa_root)
    normal_list = os.path.abspath(args.normal_list)
    data_path = os.path.abspath(args.data_path)
    blip = args.blip_model_path or os.path.join(root, "ofa", "blipdiffusion_model")

    os.makedirs(bank_dir, exist_ok=True)
    bank_normal = os.path.join(bank_dir, "normal_list.txt")
    bank_synthetic = os.path.join(bank_dir, "synthetic_anomaly_list.txt")

    # 若为 accumulate 且非首轮，从上一轮继承；否则清空
    if args.accumulate:
        all_normals = []
        all_synthetics = []
    else:
        all_normals = []
        all_synthetics = []

    baseline_save_dir = getattr(args, "baseline_save_dir", None) and os.path.abspath(args.baseline_save_dir) or None

    # 方案乙单轮模式：由 run_experiments 调用，不在此做 export，mine 用传入的 anomaly_map_dir
    if args.anomaly_map_dir is not None and args.round_index is not None:
        _run_one_round(
            args=args,
            root=root,
            bank_dir=bank_dir,
            bank_normal=bank_normal,
            bank_synthetic=bank_synthetic,
            normal_list=normal_list,
            data_path=data_path,
            ofa_root=ofa_root,
            blip=blip,
            all_normals=all_normals,
            all_synthetics=all_synthetics,
            _run=_run,
        )
        print("\n飞轮单轮结束。伪样本库列表:", bank_normal, bank_synthetic)
        return

    for round_no in range(1, args.num_rounds + 1):
        print("\n" + "=" * 60 + f" 飞轮第 {round_no}/{args.num_rounds} 轮 " + "=" * 60)
        rd = os.path.join(bank_dir, f"round_{int(round_no):02d}")
        os.makedirs(rd, exist_ok=True)
        ofa_npy_dir = os.path.join(rd, "ofa_npy")
        hard_masks_dir = os.path.join(rd, "hard_masks")
        gen_dir = os.path.join(rd, "gen")
        manifest_dir = os.path.join(rd, "manifest")

        # 1) [Step 1/6] OFA 导出异常分数图：第 1 轮用初始权重，第 2 轮用 baseline 第 1 轮 ckpt
        export_ckpt = None
        if round_no >= 2 and baseline_save_dir:
            epoch0_pt = os.path.join(baseline_save_dir, "epoch_0.pt")
            if os.path.isfile(epoch0_pt):
                export_ckpt = epoch0_pt
        export_cmd = [
            sys.executable,
            "scripts/ofa_export_npy.py",
            "--normal_list", normal_list,
            "--output_dir", ofa_npy_dir,
            "--data_path", data_path,
            "--blip_model_path", blip,
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--shot", str(args.shot),
            "--seed", str(args.seed),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--scheduler_t_max", str(args.scheduler_t_max),
        ]
        if export_ckpt:
            export_cmd += ["--checkpoint_path", export_ckpt]
            print("[Step 1/6] OFA 导出异常分数图（本轮：baseline 第 1 轮 ckpt）")
        else:
            print("[Step 1/6] OFA 导出异常分数图（本轮：初始预训练权重）")
        _run(export_cmd, cwd=root)

        # 2) [Step 2/6] 困难挖掘
        mine_cmd = [
            sys.executable, "scripts/mine_hard.py",
            "--anomaly_map_dir", ofa_npy_dir,
            "--save_dir", hard_masks_dir,
            "--method", args.mine_method,
        ]
        if args.mine_method == "top_k":
            mine_cmd += ["--top_k_ratio", str(args.mine_top_k_ratio)]
        else:
            mine_cmd += ["--threshold", str(args.mine_threshold)]
        _run(mine_cmd, cwd=root)
        _validate_hard_masks(hard_masks_dir, normal_list)
        print("[Step 2/6] 困难挖掘完成")

        # 3) [Step 3/6] 在困难区域上生成伪异常
        cmd_gen = [
                sys.executable, "scripts/generate_anomaly.py",
                "--config", args.config,
                "--normal_list", normal_list,
                "--data_root", data_path,
                "--hard_mask_dir", hard_masks_dir,
                "--save_dir", gen_dir,
                "--num_gen", "1",
                "--context_backend", args.context_backend,
                "--seed", str(args.seed),
        ]
        if getattr(args, 'blip_model_path', None):
            cmd_gen.extend(["--blip_model_path", blip])
        
        _run(cmd_gen, cwd=root)
        _write_round_config(
            os.path.join(rd, "round_config.json"),
            args,
            {
                "ofa_npy_dir": ofa_npy_dir,
                "hard_masks_dir": hard_masks_dir,
                "generated_dir": gen_dir,
                "manifest_dir": manifest_dir,
                "normal_list": normal_list,
            },
        )
        print(f"[Step 3/6] 伪异常生成完成")

        # 4) [Step 4/6] 本轮 manifest（正常-伪异常对列表）
        _run(
            [
                sys.executable, "scripts/co_train.py",
                "--generated_dir", gen_dir,
                "--normal_list", normal_list,
                "--output_dir", manifest_dir,
            ],
            cwd=root,
        )

        # 5) [Step 5/6] 更新伪样本库
        with open(os.path.join(manifest_dir, "normal_list.txt")) as f:
            r_normals = [line.strip() for line in f if line.strip()]
        with open(os.path.join(manifest_dir, "synthetic_anomaly_list.txt")) as f:
            r_synthetics = [line.strip() for line in f if line.strip()]
        if len(r_normals) != len(r_synthetics):
            raise ValueError(f"本轮 manifest 行数不一致: normal {len(r_normals)} vs synthetic {len(r_synthetics)}")
        if args.accumulate:
            all_normals.extend(r_normals)
            all_synthetics.extend(r_synthetics)
        else:
            all_normals = list(r_normals)
            all_synthetics = list(r_synthetics)

        with open(bank_normal, "w") as f:
            for p in all_normals:
                f.write(os.path.abspath(p) + "\n")
        with open(bank_synthetic, "w") as f:
            for p in all_synthetics:
                f.write(p + "\n")
        print(f"[Step 5/6] 伪样本库当前共 {len(all_normals)} 对 (正常, 伪异常)")

        # 6) [Step 6/6] OFA 飞轮训练（用当前伪样本库，每轮保存 ckpt + 性能表）
        env = {
            "DEFECT_FLYWHEEL_NORMAL_LIST": bank_normal,
            "DEFECT_FLYWHEEL_SYNTHETIC_LIST": bank_synthetic,
        }
        if blip:
            env["BLIP_MODEL_PATH"] = blip
        ofa_save = getattr(args, "ofa_save_path", None)
        if ofa_save:
            base = ofa_save.rstrip("/") + f"_round_{int(round_no):02d}"
            ofa_save_round = os.path.join(root, base) if not os.path.isabs(ofa_save) else base
            ofa_save_round = os.path.abspath(ofa_save_round)
            os.makedirs(ofa_save_round, exist_ok=True)
        ofa_cmd = [
            sys.executable, "main_zju.py",
            "--data_path", data_path,
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--shot", str(args.shot),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--scheduler_t_max", str(args.scheduler_t_max),
            "--seed", str(args.seed),
        ]
        if ofa_save:
            ofa_cmd += ["--save_path", ofa_save_round]
        if blip:
            ofa_cmd += ["--blip_model_path", blip]
        _run(ofa_cmd, cwd=ofa_root, env=env)
        if ofa_save:
            _write_checkpoint_alias(
                os.path.join(ofa_save_round, "epoch_0.pt"),
                os.path.join(ofa_save_round, f"round_{round_no:02d}_epoch_0.pt"),
            )

    print("\n飞轮迭代结束。伪样本库列表:", bank_normal, bank_synthetic)


if __name__ == "__main__":
    main()
