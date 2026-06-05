"""
风格迁移式异常生成：在 mask 区域内使图像特征沿「正常→异常」文本方向偏移，
与 AnoStyler 一致。支持从文件读 mask 或直接传入困难 mask 数组（与模块一对接）。
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms, utils as vutils
from tqdm import tqdm

from . import style_net as StyleNet
from . import ano_utils as utils


def _img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
    return (image - mean) / std


def _clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode="bicubic")
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, -1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, -1, 1, 1)
    return (image - mean) / std


def _get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    return torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)


def run_style_transfer(
    content_image,
    clip_model,
    VGG,
    device,
    img_height,
    img_width,
    lambda_tv,
    lambda_pdir,
    lambda_gdir,
    lambda_c,
    lambda_mclip,
    crop_size,
    num_crops,
    max_step,
    lr,
    thresh,
    save_img_path,
    source_features,
    text_features,
    text_source,
    mask_path=None,
    mask_array=None,
):
    """
    在 mask 区域内生成异常外观，区域外保持内容一致。

    Args:
        content_image: (1, 3, H, W) tensor，[0,1]，正常图
        clip_model, VGG, device: 模型与设备
        img_height, img_width: 内容图尺寸
        lambda_*: 各损失权重
        crop_size, num_crops, max_step, lr, thresh: 优化超参
        save_img_path: 生成图保存路径
        source_features, text_features, text_source: CLIP 特征（正常图、异常文本、正常文本）
        mask_path: 可选，mask 图像路径（与 AnoStyler 原版一致）
        mask_array: 可选，困难 mask 数组 (H,W) 或 (1,H,W)，0~255 或 0~1；与模块一输出对接时使用
    """
    # 确定 mask：优先使用困难 mask 数组（数据飞轮）
    if mask_array is not None:
        mask = np.asarray(mask_array).squeeze().astype(np.float64)
        if mask.max() > 1.0:
            mask = mask / 255.0
    elif mask_path is not None:
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0
    else:
        raise ValueError("run_style_transfer 需要 mask_path 或 mask_array 之一")

    if np.max(mask) == 0:
        out = F.interpolate(content_image, size=(256, 256), mode="bilinear", align_corners=False)
        vutils.save_image(out, save_img_path, normalize=False)
        return

    mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    mask = F.interpolate(mask, size=(img_height, img_width), mode="nearest")

    style_net = StyleNet.UNet().to(device)
    clip_model.eval()

    optimizer = optim.Adam(style_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    content_features = utils.get_features(_img_normalize(content_image, device), VGG)

    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224),
    ])

    for epoch in tqdm(range(max_step + 1), desc="Style transfer", leave=False):
        full_output = style_net(content_image, use_sigmoid=True)
        target = content_image * (1 - mask) + full_output * mask
        target.requires_grad_(True)

        target_features = utils.get_features(_img_normalize(target, device), VGG)
        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)
        content_loss = content_loss + torch.mean((target_features["conv5_2"] - content_features["conv5_2"]) ** 2)

        _, _, H, W = target.shape
        img_proc_anom = []
        crop_coords = []
        for _ in range(num_crops):
            top = torch.randint(0, H - crop_size + 1, (1,))
            left = torch.randint(0, W - crop_size + 1, (1,))
            crop_coords.append((top.item(), left.item()))
            patch_anom = target[:, :, top : top + crop_size, left : left + crop_size]
            patch_anom = F.interpolate(patch_anom, size=224, mode="bilinear", align_corners=False)
            patch_anom = augment(patch_anom)
            img_proc_anom.append(patch_anom)

        img_anom = torch.cat(img_proc_anom, dim=0)
        feat_anom = clip_model.encode_image(_clip_normalize(img_anom, device))
        feat_anom = feat_anom / feat_anom.norm(dim=-1, keepdim=True)
        delta_I = feat_anom - source_features
        delta_I = delta_I / delta_I.norm(dim=-1, keepdim=True)
        delta_T = (text_features - text_source).repeat(num_crops, 1)
        delta_T = delta_T / delta_T.norm(dim=-1, keepdim=True)
        loss_temp = 1 - torch.cosine_similarity(delta_I, delta_T, dim=1)
        loss_temp[loss_temp < thresh] = 0

        weights = []
        for top, left in crop_coords:
            mask_crop = mask[:, :, top : top + crop_size, left : left + crop_size]
            weights.append(mask_crop.float().mean().item())
        weights = torch.tensor(weights).to(device)
        if weights.sum() > 1e-8:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        loss_patch = (weights * loss_temp).sum()

        glob_features = clip_model.encode_image(_clip_normalize(target, device))
        glob_features = glob_features / glob_features.norm(dim=-1, keepdim=True)
        glob_direction = glob_features - source_features
        glob_direction = glob_direction / glob_direction.norm(dim=-1, keepdim=True)
        gtext_direction = (text_features - text_source) / (text_features - text_source).norm(dim=-1, keepdim=True)
        loss_glob = (1 - torch.cosine_similarity(glob_direction, gtext_direction)).mean()

        reg_tv = lambda_tv * _get_image_prior_losses(target)
        masked_target = target * mask
        masked_target_resized = F.interpolate(masked_target, size=224, mode="bicubic")
        masked_clip_feat = clip_model.encode_image(_clip_normalize(masked_target_resized, device))
        masked_clip_feat = masked_clip_feat / masked_clip_feat.norm(dim=-1, keepdim=True)
        loss_clip_sim = 1 - torch.cosine_similarity(masked_clip_feat, text_features).mean()

        total_loss = (
            lambda_pdir * loss_patch
            + lambda_c * content_loss
            + reg_tv
            + lambda_gdir * loss_glob
            + lambda_mclip * loss_clip_sim
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    output = torch.clamp(target.clone().detach(), 0, 1)
    output_resized = F.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
    vutils.save_image(output_resized, save_img_path, normalize=False)
