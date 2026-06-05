#!/bin/bash
# 方案乙逻辑验证：先跑单元测试，再提示完整集成测试命令（需在含 torch 的 conda 环境下执行）
set -e
cd "$(dirname "$0")/.."
echo "=== 1. 单元测试（默认 20 大 epoch、路径逻辑） ==="
python tests/test_run_experiments_scheme_b.py
echo ""
echo "=== 2. 完整集成测试（2 大 epoch）需在已安装 torch 的 conda 环境下运行 ==="
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py \\"
echo "    --data_path \"\$(pwd)/datasets/ZJU-Leaper-Group5-MVTec_dev\" \\"
echo "    --exp_name exp_scheme_b_verify --blip_model_path \"\$(pwd)/ofa/blipdiffusion_model\" \\"
echo "    --macro_epochs 2"
echo ""
echo "默认 20 大 epoch（不传 --macro_epochs 即可）："
echo "  python scripts/run_experiments.py --data_path ... --blip_model_path ..."
