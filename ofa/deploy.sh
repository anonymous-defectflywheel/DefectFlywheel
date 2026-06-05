#!/bin/bash

# One-For-All 少样本异常检测 - 一键部署脚本
# 使用方法: bash deploy.sh

echo "🚀 开始部署 One-For-All 少样本异常检测环境..."

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  警告: 未检测到NVIDIA GPU，将使用CPU版本"
    CUDA_AVAILABLE=false
else
    echo "✅ 检测到NVIDIA GPU"
    CUDA_AVAILABLE=true
fi

# 创建conda环境
echo "📦 创建conda环境 IIPAD..."
conda create -n IIPAD python=3.8 -y

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate IIPAD

# 安装PyTorch
echo "🔥 安装PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # 检测CUDA版本
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "检测到CUDA版本: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "11.7" ]]; then
        conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
    elif [[ "$CUDA_VERSION" == "11.8" ]]; then
        conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "⚠️  未识别的CUDA版本，安装CPU版本"
        conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch -y
    fi
else
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch -y
fi

# 安装项目依赖
echo "📚 安装项目依赖..."
if [ -f "requirements.txt2" ]; then
    pip install -r requirements.txt2
else
    echo "❌ 错误: 未找到requirements.txt2文件"
    exit 1
fi

# 设置Hugging Face镜像
echo "🌐 配置Hugging Face镜像..."
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 验证安装
echo "🔍 验证安装..."
python -c "
import torch
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU数量: {torch.cuda.device_count()}')
    print(f'✅ GPU名称: {torch.cuda.get_device_name(0)}')
"

python -c "
from transformers import CLIPProcessor
print('✅ Transformers安装成功')
"

python -c "
from diffusers import BlipDiffusionPipeline
print('✅ Diffusers安装成功')
"

echo ""
echo "🎉 部署完成！"
echo ""
echo "📋 下一步操作："
echo "1. 激活环境: conda activate IIPAD"
echo "2. 下载模型: python -c \"from diffusers import BlipDiffusionPipeline; BlipDiffusionPipeline.from_pretrained('Salesforce/blipdiffusion', cache_dir='./blipdiffusion_model')\""
echo "3. 准备数据集: 将ZJU-Leaper-Group5-MVTec数据集放在dataset/目录下"
echo "4. 运行训练: bash test_zju.sh"
echo ""
echo "📖 详细说明请查看 DEPLOYMENT_GUIDE.md"


