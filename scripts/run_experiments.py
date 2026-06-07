#!/usr/bin/env python3
"""
DefectFlywheel 实验脚本。支持三种模式：
  - baseline_chain_legacy（final 默认）：每个 macro epoch 训练 baseline 1 epoch 并导出热图；flywheel round k 使用旧链路 baseline heatmap。
  - flywheel_closed_loop：先训练 baseline_aux_epoch_00；round00 用 baseline_aux，round01+ 用上一轮 flywheel checkpoint 导出热图。
  - 旧模式：--macro_epochs 0 时，先跑完 baseline（epochs 轮），再跑 flywheel num_rounds 轮。

正式实验默认 eval_policy=final_only：中间 round 只训练和保存 checkpoint，只有最后一轮评测测试集。
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, cwd=None, env=None, check=True):
    env = {**os.environ, **(env or {})}
    cwd = cwd or _project_root()
    print("[RUN]", " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"命令退出码 {r.returncode}")
    return r


def _parse_log_table(log_path):
    """Parse the last markdown pipe table from a log file."""
    if not os.path.isfile(log_path):
        return None
    with open(log_path) as f:
        lines = [line.strip() for line in f]

    tables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
            sep = lines[i + 1]
            if sep.startswith("|") and sep.endswith("|") and set(sep.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")) == set():
                table = [line, sep]
                j = i + 2
                while j < len(lines) and lines[j].startswith("|") and lines[j].endswith("|"):
                    table.append(lines[j])
                    j += 1
                if len(table) >= 3:
                    tables.append(table)
                i = j
                continue
        i += 1

    if not tables:
        return None
    last = tables[-1]
    headers = [c.strip() for c in last[0].split("|")[1:-1]]
    rows = []
    for line in last[2:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells:
            rows.append(cells)
    return headers, rows



def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _capture(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return r.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def _snapshot_environment(exp_dir, root):
    env_dir = os.path.join(exp_dir, "environment")
    os.makedirs(env_dir, exist_ok=True)
    _write_text(os.path.join(env_dir, "python_version.txt"), sys.version + "\n")
    _write_text(os.path.join(env_dir, "platform.txt"), platform.platform() + "\n")
    _write_text(os.path.join(env_dir, "pip_freeze.txt"), _capture([sys.executable, "-m", "pip", "freeze"], cwd=root) + "\n")
    _write_text(os.path.join(env_dir, "nvidia_smi.txt"), _capture(["nvidia-smi"], cwd=root) + "\n")
    _write_text(os.path.join(env_dir, "git_status.txt"), _capture(["git", "status", "--short"], cwd=root) + "\n")
    _write_text(os.path.join(env_dir, "git_commit.txt"), _capture(["git", "rev-parse", "HEAD"], cwd=root) + "\n")
    return env_dir


def _serialize_args(args):
    return {k: (os.path.abspath(v) if k.endswith("path") and isinstance(v, str) else v) for k, v in vars(args).items()}


def _write_run_config(exp_dir, root, args, data_path, ofa_root, normal_list):
    env_dir = _snapshot_environment(exp_dir, root)
    command = " ".join([sys.executable] + sys.argv)
    _write_text(os.path.join(exp_dir, "command.sh"), command + "\n")
    payload = {
        "run_name": os.path.basename(exp_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_eligibility": f"reproduction/seed{args.seed}" if args.seed == 4 else "custom-seed",
        "source_repo": root,
        "source_role": "defectflywheel-source",
        "baseline_source": "/mnt/data/ouxuewen/gra_project/OneforAll",
        "method": "DefectFlywheel",
        "dataset": args.dataset,
        "data_path": data_path,
        "shot": args.shot,
        "seed": args.seed,
        "macro_epochs": args.macro_epochs,
        "heatmap_source": args.heatmap_source,
        "eval_policy": args.eval_policy,
        "baseline_eval_policy": args.baseline_eval_policy,
        "context_backend": args.context_backend,
        "mine_method": args.mine_method,
        "mine_top_k_ratio": args.mine_top_k_ratio,
        "mine_threshold": args.mine_threshold,
        "static_generation": args.static_generation,
        "ofa_context_backend": "blipdiffusion_qformer_in_ofa",
        "checkpoint_selection": f"fixed_final_flywheel_round_{max(args.macro_epochs - 1, 0):02d}" if args.macro_epochs > 0 else "legacy_mode",
        "normal_list": normal_list,
        "ofa_root": ofa_root,
        "args": _serialize_args(args),
        "command": command,
        "environment_snapshot": env_dir,
    }
    _write_json(os.path.join(exp_dir, "run_config.json"), payload)


def _round_name(round_index):
    return f"flywheel_round_{int(round_index):02d}"


def _bank_round_name(round_index):
    return f"round_{int(round_index):02d}"


def _checkpoint_path_for_round(exp_dir, round_index):
    round_dir = os.path.join(exp_dir, _round_name(round_index))
    preferred = os.path.join(round_dir, f"flywheel_epoch_{int(round_index):02d}.pt")
    legacy_alias = os.path.join(round_dir, f"round_{int(round_index):02d}_epoch_0.pt")
    legacy = os.path.join(round_dir, "epoch_0.pt")
    for candidate in (preferred, legacy_alias, legacy):
        if os.path.isfile(candidate):
            return candidate
    return preferred


def _baseline_aux_checkpoint(exp_dir):
    preferred = os.path.join(exp_dir, "baseline", "baseline_aux_epoch_00.pt")
    legacy = os.path.join(exp_dir, "baseline", "epoch_0.pt")
    return preferred if os.path.isfile(preferred) else legacy

def _write_checkpoint_alias(src, dst):
    if os.path.isfile(src):
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        return dst
    return None


def _write_checkpoint_manifest(exp_dir):
    ckpts = []
    for dirpath, _, filenames in os.walk(exp_dir):
        for name in filenames:
            if name.endswith(".pt"):
                path = os.path.join(dirpath, name)
                ckpts.append({
                    "path": path,
                    "relative_path": os.path.relpath(path, exp_dir),
                    "bytes": os.path.getsize(path),
                    "mtime": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
                })
    _write_json(os.path.join(exp_dir, "checkpoint_manifest.json"), {"checkpoints": sorted(ckpts, key=lambda x: x["relative_path"])})


def _write_metric_summaries(exp_dir, results):
    records = []
    selected_run = next(reversed(results), None) if results else None
    for run_name, log_path in results.items():
        parsed = _parse_log_table(log_path)
        if not parsed:
            records.append({"run": run_name, "log_path": log_path, "parse_status": "missing_table"})
            continue
        headers, rows = parsed
        for row in rows:
            record = {"run": run_name, "log_path": log_path, "parse_status": "ok"}
            record.update({h: row[i] if i < len(row) else "" for i, h in enumerate(headers)})
            records.append(record)
    _write_json(os.path.join(exp_dir, "raw_metrics.json"), {"selected_run": selected_run, "records": records})
    if records:
        fieldnames = sorted({k for rec in records for k in rec.keys()})
        with open(os.path.join(exp_dir, "metrics_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        with open(os.path.join(exp_dir, "metrics_summary.jsonl"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="2-shot 实验：baseline vs 飞轮，输出 log 与对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, required=True, help="数据集根目录（ZJU 格式）")
    parser.add_argument("--dataset", type=str, default="zju", help="OFA 数据集名，与 main_zju --dataset 一致；fair_shot 时用于 prepare_normal_list 的类别顺序")
    parser.add_argument("--exp_name", type=str, default="exp_2shot", help="实验名，结果写入 experiments/<exp_name>/")
    parser.add_argument("--max_normal", type=int, default=10, help="参与飞轮的正常图数量（prepare_normal_list 与 export 用）")
    parser.add_argument("--category", type=str, default="p_id16", help="用于生成 normal_list 的类别")
    parser.add_argument("--ofa_root", type=str, default="./ofa", help="OFA 根目录")
    parser.add_argument("--blip_model_path", type=str, default=None, help="BlipDiffusion 路径，不传则用 ofa/blipdiffusion_model")
    parser.add_argument("--context_backend", type=str, default="blipdiffusion_qformer", choices=["blipdiffusion_qformer", "blip_caption", "static_prompt"], help="传给 DefectFlywheel 生成器的上下文来源")
    parser.add_argument("--mine_method", type=str, default="top_k", choices=["top_k", "threshold", "random"], help="hard-region mining 或 random-region 消融")
    parser.add_argument("--mine_top_k_ratio", type=float, default=0.1)
    parser.add_argument("--mine_threshold", type=float, default=0.5)
    parser.add_argument("--static_generation", action="store_true", help="消融：只生成一次伪异常，后续轮次复用")
    parser.add_argument("--disable_blip", action="store_true", help="兼容旧消融脚本：强制 context_backend=static_prompt")
    parser.add_argument("--skip_baseline_eval", action="store_true", help="兼容旧消融脚本：baseline 训练时跳过测试")
    parser.add_argument("--skip_baseline", action="store_true", help="跳过 baseline，只跑飞轮")
    parser.add_argument("--macro_epochs", type=int, default=20, help="方案乙：大 epoch 数。每大 epoch = baseline 1 epoch + 导出 .npy + flywheel 1 轮（mine 用上一轮 baseline 的 .npy）。设为 0 时走旧模式（baseline 整段 + flywheel num_rounds）")
    parser.add_argument("--heatmap_source", type=str, default="baseline_chain_legacy", choices=["baseline_chain_legacy", "flywheel_closed_loop"], help="hard-region heatmap 来源：baseline_chain_legacy 恢复旧 baseline 链；flywheel_closed_loop 为调试/消融闭环")
    parser.add_argument("--eval_policy", type=str, default="final_only", choices=["final_only", "all"], help="final_only 表示仅最后一个 flywheel round 测试集评估；all 用于调试")
    parser.add_argument("--baseline_eval_policy", type=str, default="skip", choices=["skip", "final_only", "all"], help="baseline 链评测策略：skip=不评测；final_only=仅最后一个 baseline epoch 评测；all=每轮 baseline 都评测")
    parser.add_argument("--num_rounds", type=int, default=2, help="旧模式下的飞轮轮数（macro_epochs=0 时生效）")
    parser.add_argument("--accumulate_flywheel", action="store_true", default=True, help="方案乙：每轮将新 (正常,伪异常) 对追加到伪样本库；否则每轮替换")
    parser.add_argument("--no_accumulate_flywheel", action="store_false", dest="accumulate_flywheel")
    parser.add_argument("--fair_shot", action="store_true", default=True, help="飞轮使用与 baseline 相同的 shot 图片（每类按 OFA 规则取 shot 张），保证公平")
    parser.add_argument("--no_fair_shot", action="store_false", dest="fair_shot", help="关闭 fair_shot，改用单类别 + max_normal 生成 normal_list")
    # 固定训练参数，保证 baseline 与飞轮实验公平
    parser.add_argument("--epochs", type=int, default=2, help="旧模式下每段 OFA 训练轮数；方案乙下每大 epoch 内 baseline 与 flywheel 各 1 个 epoch")
    parser.add_argument("--shot", type=int, default=2, help="每类 few-shot 数量")
    parser.add_argument("--lr", type=float, default=1e-5, help="OFA 学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="OFA weight_decay")
    parser.add_argument("--scheduler_t_max", type=int, default=20, help="CosineAnnealingLR T_max")
    parser.add_argument("--seed", type=int, default=4, help="random seed; release reproduction default is 4")
    args = parser.parse_args()
    if getattr(args, "disable_blip", False):
        args.context_backend = "static_prompt"

    root = _project_root()
    data_path = os.path.abspath(args.data_path)
    ofa_root = os.path.abspath(args.ofa_root)
    # 实验目录名附带日期与时分，便于区分多次运行
    exp_name_with_time = f"{args.exp_name}_{datetime.now().strftime('%Y-%m-%d_%H%M')}"
    exp_dir = os.path.join(root, "experiments", exp_name_with_time)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"实验目录: {exp_dir}")

    normal_list = os.path.join(exp_dir, "normal_list.txt")
    _write_run_config(exp_dir, root, args, data_path, ofa_root, normal_list)

    # 1) [实验步骤 1/4] 生成正常图列表：公平对比时用与 baseline 相同的 shot 图片
    print("\n" + "=" * 60 + " [实验步骤 1/4] 生成正常图列表（fair_shot） " + "=" * 60)
    if args.fair_shot:
        _run(
            [
                sys.executable, "scripts/prepare_normal_list.py",
                "--data_root", data_path,
                "--mode", "shot_all",
                "--output", normal_list,
                "--shot", str(args.shot),
                "--shot_index_step", "3",
                "--dataset", args.dataset,
            ],
            cwd=root,
        )
    else:
        _run(
            [
                sys.executable, "scripts/prepare_normal_list.py",
                "--data_root", data_path,
                "--category", args.category,
                "--output", normal_list,
                "--max_count", str(args.max_normal),
            ],
            cwd=root,
        )

    results = {}
    baseline_save = os.path.join(exp_dir, "baseline")
    bank_dir = os.path.join(exp_dir, "flywheel_bank")
    flywheel_save_prefix = os.path.join(exp_dir, "flywheel")
    train_params_1epoch = [
        "--epochs", "1",
        "--shot", str(args.shot),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--scheduler_t_max", "1",
        "--seed", str(args.seed),
    ]
    train_params_multi = [
        "--epochs", str(args.epochs),
        "--shot", str(args.shot),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--scheduler_t_max", str(args.scheduler_t_max),
        "--seed", str(args.seed),
    ]

    if args.macro_epochs > 0:
        os.makedirs(baseline_save, exist_ok=True)
        os.makedirs(bank_dir, exist_ok=True)

        if args.heatmap_source == "baseline_chain_legacy":
            if args.skip_baseline:
                raise ValueError("baseline_chain_legacy requires baseline training; do not use --skip_baseline for final DefectFlywheel slots")

            print("\n" + "=" * 60 + " [Baseline-Chain Legacy] baseline heatmap chain + final-only flywheel eval " + "=" * 60)
            for k in range(args.macro_epochs):
                round_label = _round_name(k)
                bank_round_current = _bank_round_name(k)
                print("\n" + "=" * 60 + f" [Baseline-Chain] Macro Epoch {k + 1}/{args.macro_epochs}: {round_label} " + "=" * 60)

                baseline_epoch_ckpt = os.path.join(baseline_save, f"baseline_epoch_{k:02d}.pt")
                baseline_cmd = [
                    sys.executable, "main_zju.py",
                    "--data_path", data_path,
                    "--dataset", args.dataset,
                    "--save_path", baseline_save,
                    "--display_epoch", str(k + 1),
                    "--display_epochs", str(args.macro_epochs),
                    "--checkpoint_name", f"baseline_epoch_{k:02d}.pt",
                ] + train_params_1epoch
                if args.baseline_eval_policy == "skip" or (args.baseline_eval_policy == "final_only" and k < args.macro_epochs - 1):
                    baseline_cmd += ["--skip_test"]
                if args.blip_model_path:
                    baseline_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
                if k >= 1:
                    prev_baseline = os.path.join(baseline_save, f"baseline_epoch_{k - 1:02d}.pt")
                    if not os.path.isfile(prev_baseline):
                        prev_baseline = os.path.join(baseline_save, f"epoch_{k - 1}.pt")
                    if os.path.isfile(prev_baseline):
                        baseline_cmd += ["--load_checkpoint", os.path.abspath(prev_baseline)]
                _run(baseline_cmd, cwd=ofa_root)

                if not os.path.isfile(baseline_epoch_ckpt):
                    fallback = os.path.join(baseline_save, "epoch_0.pt")
                    if os.path.isfile(fallback):
                        shutil.copy2(fallback, baseline_epoch_ckpt)
                if not os.path.isfile(baseline_epoch_ckpt):
                    raise FileNotFoundError(f"baseline checkpoint not found for macro epoch {k}: {baseline_epoch_ckpt}")
                _write_checkpoint_alias(baseline_epoch_ckpt, os.path.join(baseline_save, "epoch_0.pt"))
                _write_checkpoint_alias(baseline_epoch_ckpt, os.path.join(baseline_save, f"epoch_{k}.pt"))
                baseline_log = os.path.join(baseline_save, "log.txt")
                if os.path.isfile(baseline_log):
                    if args.baseline_eval_policy == "all":
                        results[f"baseline_epoch_{k:02d}"] = baseline_log
                    elif args.baseline_eval_policy == "final_only" and k == args.macro_epochs - 1:
                        results["baseline_final"] = baseline_log

                round_k_npy = os.path.join(bank_dir, bank_round_current, "ofa_npy")
                os.makedirs(round_k_npy, exist_ok=True)
                export_cmd = [
                    sys.executable, "scripts/ofa_export_npy.py",
                    "--normal_list", normal_list,
                    "--output_dir", round_k_npy,
                    "--data_path", data_path,
                    "--checkpoint_path", baseline_epoch_ckpt,
                    "--dataset", args.dataset,
                    "--epochs", "1",
                    "--shot", str(args.shot),
                    "--seed", str(args.seed),
                    "--lr", str(args.lr),
                    "--weight_decay", str(args.weight_decay),
                    "--scheduler_t_max", "1",
                ]
                if args.blip_model_path:
                    export_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
                print(f"[BaselineChain] baseline_epoch={k:02d} export checkpoint={baseline_epoch_ckpt} -> {round_k_npy}")
                _run(export_cmd, cwd=root)

                if k == 0:
                    anomaly_map_dir = os.path.join(bank_dir, _bank_round_name(0), "ofa_npy")
                    load_ckpt = baseline_epoch_ckpt
                else:
                    anomaly_map_dir = os.path.join(bank_dir, _bank_round_name(k - 1), "ofa_npy")
                    load_ckpt = _checkpoint_path_for_round(exp_dir, k - 1)
                if not os.path.isdir(anomaly_map_dir):
                    raise FileNotFoundError(f"baseline-chain anomaly map dir not found for round {k}: {anomaly_map_dir}")
                if not os.path.isfile(load_ckpt):
                    raise FileNotFoundError(f"flywheel training checkpoint not found for round {k}: {load_ckpt}")

                flywheel_cmd = [
                    sys.executable, "scripts/flywheel_iterate.py",
                    "--normal_list", normal_list,
                    "--data_path", data_path,
                    "--dataset", args.dataset,
                    "--bank_dir", bank_dir,
                    "--ofa_root", ofa_root,
                    "--ofa_save_path", flywheel_save_prefix,
                    "--anomaly_map_dir", os.path.abspath(anomaly_map_dir),
                    "--round_index", str(k),
                    "--display_epoch", str(k + 1),
                    "--display_epochs", str(args.macro_epochs),
                    "--epochs", "1",
                    "--shot", str(args.shot),
                    "--lr", str(args.lr),
                    "--weight_decay", str(args.weight_decay),
                    "--scheduler_t_max", "1",
                    "--seed", str(args.seed),
                    "--context_backend", args.context_backend,
                    "--mine_method", args.mine_method,
                    "--mine_top_k_ratio", str(args.mine_top_k_ratio),
                    "--mine_threshold", str(args.mine_threshold),
                    "--load_checkpoint", os.path.abspath(load_ckpt),
                ]
                if args.eval_policy == "final_only" and k < args.macro_epochs - 1:
                    flywheel_cmd += ["--skip_test"]
                if args.static_generation:
                    flywheel_cmd += ["--static_generation"]
                if args.accumulate_flywheel:
                    flywheel_cmd += ["--accumulate"]
                else:
                    flywheel_cmd += ["--no_accumulate"]
                if args.blip_model_path:
                    flywheel_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
                print(f"[BaselineChain] flywheel_round={k:02d} hard_mask_source={anomaly_map_dir} load_checkpoint={load_ckpt}")
                _run(flywheel_cmd, cwd=root)

                round_dir = os.path.join(exp_dir, round_label)
                expected_ckpt = os.path.join(round_dir, f"flywheel_epoch_{k:02d}.pt")
                if not os.path.isfile(expected_ckpt):
                    legacy_ckpt = os.path.join(round_dir, "epoch_0.pt")
                    if os.path.isfile(legacy_ckpt):
                        shutil.copy2(legacy_ckpt, expected_ckpt)
                if not os.path.isfile(expected_ckpt):
                    raise FileNotFoundError(f"flywheel checkpoint not found for round {k}: {expected_ckpt}")
                _write_checkpoint_manifest(exp_dir)

                lp = os.path.join(round_dir, "log.txt")
                if os.path.isfile(lp) and (args.eval_policy == "all" or k == args.macro_epochs - 1):
                    results[round_label] = lp

        elif args.heatmap_source == "flywheel_closed_loop":
            # 闭环模式：baseline_aux 只作为 round00 起点；round01+ 用上一轮 flywheel checkpoint 导出热图。
            os.makedirs(baseline_save, exist_ok=True)
            os.makedirs(bank_dir, exist_ok=True)

            print("\n" + "=" * 60 + " [闭环模式] Baseline auxiliary warm-up (not a paper metric) " + "=" * 60)
            baseline_aux_ckpt = os.path.join(baseline_save, "baseline_aux_epoch_00.pt")
            if not args.skip_baseline:
                baseline_cmd = [
                    sys.executable, "main_zju.py",
                    "--data_path", data_path,
                    "--dataset", args.dataset,
                    "--save_path", baseline_save,
                    "--display_epoch", "0",
                    "--display_epochs", str(args.macro_epochs),
                    "--checkpoint_name", "baseline_aux_epoch_00.pt",
                    "--skip_test",
                ] + train_params_1epoch
                if args.blip_model_path:
                    baseline_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
                _run(baseline_cmd, cwd=ofa_root)
            if not os.path.isfile(baseline_aux_ckpt):
                fallback = os.path.join(baseline_save, "epoch_0.pt")
                if os.path.isfile(fallback):
                    shutil.copy2(fallback, baseline_aux_ckpt)
            if not os.path.isfile(baseline_aux_ckpt):
                raise FileNotFoundError(f"baseline auxiliary checkpoint not found: {baseline_aux_ckpt}")

            for k in range(args.macro_epochs):
                round_label = _round_name(k)
                bank_round = _bank_round_name(k)
                print("\n" + "=" * 60 + f" [闭环模式] Flywheel Round {k + 1}/{args.macro_epochs}: {round_label} " + "=" * 60)

                if k == 0:
                    source_ckpt = baseline_aux_ckpt
                    source_name = "baseline_aux_epoch_00"
                else:
                    source_ckpt = _checkpoint_path_for_round(exp_dir, k - 1)
                    source_name = _round_name(k - 1)
                if not os.path.isfile(source_ckpt):
                    raise FileNotFoundError(f"heatmap/training source checkpoint not found for round {k}: {source_ckpt}")

                round_k_npy = os.path.join(bank_dir, bank_round, "ofa_npy")
                os.makedirs(round_k_npy, exist_ok=True)
                export_cmd = [
                    sys.executable, "scripts/ofa_export_npy.py",
                    "--normal_list", normal_list,
                    "--output_dir", round_k_npy,
                    "--data_path", data_path,
                    "--checkpoint_path", source_ckpt,
                    "--dataset", args.dataset,
                    "--epochs", "1",
                    "--shot", str(args.shot),
                    "--seed", str(args.seed),
                    "--lr", str(args.lr),
                    "--weight_decay", str(args.weight_decay),
                    "--scheduler_t_max", "1",
                ]
                if args.blip_model_path:
                    export_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
                print(f"[ClosedLoop] round={k:02d} heatmap_source={source_name} checkpoint={source_ckpt}")
                _run(export_cmd, cwd=root)

                flywheel_cmd = [
                    sys.executable, "scripts/flywheel_iterate.py",
                    "--normal_list", normal_list,
                    "--data_path", data_path,
                    "--dataset", args.dataset,
                    "--bank_dir", bank_dir,
                    "--ofa_root", ofa_root,
                    "--ofa_save_path", flywheel_save_prefix,
                    "--anomaly_map_dir", os.path.abspath(round_k_npy),
                    "--round_index", str(k),
                    "--display_epoch", str(k + 1),
                    "--display_epochs", str(args.macro_epochs),
                    "--epochs", "1",
                    "--shot", str(args.shot),
                    "--lr", str(args.lr),
                    "--weight_decay", str(args.weight_decay),
                    "--scheduler_t_max", "1",
                    "--seed", str(args.seed),
                    "--context_backend", args.context_backend,
                    "--mine_method", args.mine_method,
                    "--mine_top_k_ratio", str(args.mine_top_k_ratio),
                    "--mine_threshold", str(args.mine_threshold),
                    "--load_checkpoint", os.path.abspath(source_ckpt),
                ]
                if args.eval_policy == "final_only" and k < args.macro_epochs - 1:
                    flywheel_cmd += ["--skip_test"]
                if args.static_generation:
                    flywheel_cmd += ["--static_generation"]
                if args.accumulate_flywheel:
                    flywheel_cmd += ["--accumulate"]
                else:
                    flywheel_cmd += ["--no_accumulate"]
                if args.blip_model_path:
                    flywheel_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
                _run(flywheel_cmd, cwd=root)

                round_dir = os.path.join(exp_dir, round_label)
                expected_ckpt = os.path.join(round_dir, f"flywheel_epoch_{k:02d}.pt")
                if not os.path.isfile(expected_ckpt):
                    legacy_ckpt = os.path.join(round_dir, "epoch_0.pt")
                    if os.path.isfile(legacy_ckpt):
                        shutil.copy2(legacy_ckpt, expected_ckpt)
                if not os.path.isfile(expected_ckpt):
                    raise FileNotFoundError(f"flywheel checkpoint not found for round {k}: {expected_ckpt}")
                _write_checkpoint_manifest(exp_dir)

                lp = os.path.join(round_dir, "log.txt")
                if os.path.isfile(lp) and (args.eval_policy == "all" or k == args.macro_epochs - 1):
                    results[round_label] = lp
        else:
            raise ValueError(f"Unsupported heatmap_source: {args.heatmap_source}")
    else:
        # 旧模式：baseline 整段 epochs 轮，再 flywheel num_rounds 轮
        if not args.skip_baseline:
            os.makedirs(baseline_save, exist_ok=True)
            print("\n" + "=" * 60 + " [旧模式] Baseline（OFA 原版训练） " + "=" * 60)
            baseline_cmd = [
                sys.executable, "main_zju.py",
                "--data_path", data_path,
                "--dataset", args.dataset,
                "--save_path", baseline_save,
            ] + train_params_multi
            if args.skip_baseline_eval:
                baseline_cmd += ["--skip_test"]
            if args.blip_model_path:
                baseline_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
            _run(baseline_cmd, cwd=ofa_root)
            results["baseline"] = os.path.join(baseline_save, "log.txt")
        flywheel_cmd = [
            sys.executable, "scripts/flywheel_iterate.py",
            "--normal_list", normal_list,
            "--data_path", data_path,
            "--dataset", args.dataset,
            "--num_rounds", str(args.num_rounds),
            "--bank_dir", bank_dir,
            "--accumulate",
            "--ofa_root", ofa_root,
            "--ofa_save_path", flywheel_save_prefix,
            "--epochs", str(args.epochs),
            "--shot", str(args.shot),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--scheduler_t_max", str(args.scheduler_t_max),
            "--seed", str(args.seed),
            "--context_backend", args.context_backend,
            "--mine_method", args.mine_method,
            "--mine_top_k_ratio", str(args.mine_top_k_ratio),
            "--mine_threshold", str(args.mine_threshold),
        ]
        if args.static_generation:
            flywheel_cmd += ["--static_generation"]
        if args.blip_model_path:
            flywheel_cmd += ["--blip_model_path", os.path.abspath(args.blip_model_path)]
        if not args.skip_baseline and os.path.isdir(baseline_save):
            flywheel_cmd += ["--baseline_save_dir", os.path.abspath(baseline_save)]
        _run(flywheel_cmd, cwd=root)
        for r in range(1, args.num_rounds + 1):
            lp = os.path.join(exp_dir, f"flywheel_round{r}", "log.txt")
            if os.path.isfile(lp):
                results[f"flywheel_round{r}"] = lp

    # 4) 解析并打印对比（每轮性能表见各 log.txt）
    print("\n" + "=" * 60 + " [实验步骤 4/4] 指标对比（最后一表） " + "=" * 60)
    for name, log_path in results.items():
        parsed = _parse_log_table(log_path)
        print(f"\n--- {name} ---")
        if parsed:
            headers, rows = parsed
            print(" | ".join(headers))
            for row in rows:
                print(" | ".join(row))
        else:
            print("(未解析到表格，请直接查看)", log_path)
    _write_metric_summaries(exp_dir, results)
    _write_checkpoint_manifest(exp_dir)
    print("\n完整 log 路径:", list(results.values()))
    print("机器可读结果:", os.path.join(exp_dir, "raw_metrics.json"), os.path.join(exp_dir, "metrics_summary.csv"))


if __name__ == "__main__":
    main()
