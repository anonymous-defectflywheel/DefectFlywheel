#!/bin/bash
# DefectFlywheel 独立环境搭建（任选其一执行）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "项目目录: $PROJECT_ROOT"
echo ""

# 方式一：conda（推荐）
if command -v conda &>/dev/null; then
  echo "=== 使用 conda 创建环境 defect_flywheel ==="
  conda env create -f environment.yaml
  echo ""
  echo "创建完成。激活环境: conda activate defect_flywheel"
  exit 0
fi

# 方式二：venv + pip
echo "=== 使用 venv 创建环境 ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "创建完成。激活环境: source .venv/bin/activate  (Linux/Mac)"
