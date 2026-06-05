#!/usr/bin/env python3
"""
从 Hugging Face 下载 BlipDiffusion 模型到项目内指定目录，便于复现与 4.2/4.3 测试。
下载后目录根下含 model_index.json，可直接作为 --blip_model_path 使用。

用法:
  python scripts/download_blipdiffusion.py
  python scripts/download_blipdiffusion.py --output_dir ./ofa/blipdiffusion_model
"""
import argparse
import os


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Download BlipDiffusion from Hugging Face for OFA (4.2/4.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="保存目录，默认: <项目根>/ofa/blipdiffusion_model",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Salesforce/blipdiffusion",
        help="Hugging Face 模型 ID",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(_project_root(), "ofa", "blipdiffusion_model")

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("请先安装: pip install huggingface_hub")

    print(f"Downloading {args.repo_id} to {args.output_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
    )
    index_path = os.path.join(args.output_dir, "model_index.json")
    if not os.path.isfile(index_path):
        raise SystemExit(f"下载完成但未找到 {index_path}，请检查 repo_id 或网络。")

    print("Done.")
    print(f"使用方式: --blip_model_path \"$(pwd)/ofa/blipdiffusion_model\"")
    print(f"或: --blip_model_path \"{args.output_dir}\"")


if __name__ == "__main__":
    main()
