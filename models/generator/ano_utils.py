"""生成器用工具：图像加载、VGG 特征提取等。"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image2(img_path, img_height=None, img_width=None):
    """加载图像为 [0,1] 的 tensor，(1, 3, H, W)。"""
    image = Image.open(img_path).convert("RGB")
    if img_width is not None and img_height is not None:
        image = image.resize((img_width, img_height))
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def get_features(image, model, layers=None):
    """从 VGG 提取指定层特征，用于内容损失。"""
    if layers is None:
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1",
            "31": "conv5_2",
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features
