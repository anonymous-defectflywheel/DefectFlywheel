import VVCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import torch.nn.init as init
from torch.optim import lr_scheduler
import prompt_generator

import torch.nn as nn
from prompt_ensemble import encode_text_with_prompt_ensemble
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import open_clip
from torch import optim

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from path_ids import sample_id_from_path
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from utils import aug

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    loss = 0
    for item in range(len(a)):
        loss += torch.dot(a[item], b[item]) / (torch.sqrt(torch.sum(a[item]**2)) * torch.sqrt(torch.sum(b[item]**2)))
    return loss / len(a)

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from PIL import Image
import sys


class ExportListDataset(torch.utils.data.Dataset):
    """Dataset from a list of image paths for OFA export_npy mode (DefectFlywheel)."""
    def __init__(self, path_list, preprocess, target_transform, obj_list, export_obj, class_name_map_class_id):
        self.path_list = [p.strip() for p in path_list if p.strip()]
        self.preprocess = preprocess
        self.target_transform = target_transform
        self.obj_list = obj_list
        self.export_obj = export_obj or obj_list[0]
        self.class_name_map_class_id = class_name_map_class_id
        self.img_size = 224

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = self.path_list[index]
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img) if self.preprocess else img
        img_mask = torch.zeros(1, self.img_size, self.img_size)
        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': self.export_obj,
            'cls_id': self.class_name_map_class_id[self.export_obj],
            'anomaly': torch.tensor(0),
            'img_path': path,
        }


class FlywheelPairDataset(torch.utils.data.Dataset):
    """(normal_img, synthetic_anomaly_img) pairs from DefectFlywheel co_train lists."""
    def __init__(self, normal_paths, synthetic_paths, preprocess_fn):
        self.pairs = [(n, s) for n, s in zip(normal_paths, synthetic_paths) if os.path.isfile(n) and os.path.isfile(s)]
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        npath, spath = self.pairs[index]
        normal_t = self.preprocess_fn(npath)
        synthetic_t = self.preprocess_fn(spath)
        return normal_t, synthetic_t


def main(args):
    img_size = args.image_size
    features_list = args.features_list

    logger = get_logger(args.save_path)
    # 实验 log 开头附上超参数，便于复现与对比
    logger.info("========== 超参数 ==========")
    logger.info("data_path: %s", getattr(args, 'data_path', ''))
    logger.info("dataset: %s", getattr(args, 'dataset', ''))
    logger.info("epochs: %s", getattr(args, 'epochs', 20))
    logger.info("shot: %s", getattr(args, 'shot', 2))
    logger.info("lr: %s", getattr(args, 'lr', 1e-5))
    logger.info("weight_decay: %s", getattr(args, 'weight_decay', 1e-4))
    logger.info("scheduler_t_max: %s", getattr(args, 'scheduler_t_max', 20))
    logger.info("seed: %s", getattr(args, 'seed', 4))
    logger.info("flywheel: %s", bool(getattr(args, 'flywheel_normal_list', None) and getattr(args, 'flywheel_synthetic_list', None)))
    logger.info("============================")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #this parameter are not used in our model, they just use to make VVCLIP to be built successfully.
    VVCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    img_size = 224

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", img_size, pretrained="openai")

    #introduing VV-attention mechanism ONLY use its visual encoder
    model, _ = VVCLIP_lib.load("ViT-L-14", device=device, design_details = VVCLIP_parameters)

    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model.eval()
    model.eval()
    # ZJU 模式下强制 224，与原 OFA 实现中 16×16 patch grid 假设一致，避免 reshape 报错
    if args.dataset in ['zju', 'wfdd', 'fabric-mvtec', 'fabric_mvtec']:
        args.image_size = 224
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    #introduing Q-former from BLIP-diffusion
    blip_model_path = getattr(args, 'blip_model_path', "/home/lwx/model_card")
    # 兼容 Hugging Face 缓存目录：若根目录无 model_index.json，则在子目录中查找（如 snapshots/<rev>/）
    _blip_root = blip_model_path
    if not os.path.isfile(os.path.join(blip_model_path, "model_index.json")):
        for _root, _dirs, _files in os.walk(blip_model_path):
            if "model_index.json" in _files:
                _blip_root = _root
                break
        if not os.path.isfile(os.path.join(_blip_root, "model_index.json")):
            raise FileNotFoundError(
                f"BlipDiffusion: no model_index.json in {blip_model_path} or subdirs. "
                "Run: python scripts/download_blipdiffusion.py (see docs/功能测试指南.md)"
            )
    blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
        _blip_root, torch_dtype=torch.float32
    ).to(device)

    results = {}
    metrics = {}

    
    model.to(device)
    clip_model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    
    # few-shot / 训练参数（可由命令行指定，便于实验公平对比）
    shot = getattr(args, 'shot', 2)
    epochs = getattr(args, 'epochs', 20)

    padding = tokenizer("").to(device)
    repersent_vec = {}
    visual_feature_bank_1 = {}
    visual_feature_bank_2 = {}
    soft_prompt_list = {}
    optimizer_list = {}
    cos_loss = nn.CosineSimilarity(dim=2)
    criterion = nn.CrossEntropyLoss().to(device)
    best_pixel_auroc = 0
    best_result = None

    #obtain embedding of manual prompt
    with torch.no_grad():
        text_prompts, text_prompts_list = encode_text_with_prompt_ensemble(clip_model, ['object'], tokenizer, device, dataset = args.dataset)

    soft_prompt = prompt_generator.SoftPrompt()
    soft_prompt = soft_prompt.to(device)

    lr = getattr(args, 'lr', 1e-5)
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    scheduler_t_max = getattr(args, 'scheduler_t_max', 20)
    optimizer = optim.Adam(soft_prompt.parameters(), lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=0.05)

    # 方案乙：训练时从指定 ckpt 恢复 soft_prompt（如上一轮 flywheel 或 baseline）
    if not (getattr(args, 'export_npy', None) and getattr(args, 'export_output_dir', None) and getattr(args, 'export_only', False)):
        load_ckpt = getattr(args, 'load_checkpoint', None)
        if load_ckpt and os.path.isfile(load_ckpt):
            ckpt = torch.load(load_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'soft_prompt' in ckpt:
                soft_prompt.load_state_dict(ckpt['soft_prompt'], strict=False)
            else:
                soft_prompt.load_state_dict(ckpt, strict=False)
            print(f"[Train] 已加载 checkpoint: {load_ckpt}")

    # DefectFlywheel: (normal, synthetic) pair dataloader when using co_train lists as data source
    flywheel_loader = None
    if getattr(args, 'flywheel_normal_list', None) and getattr(args, 'flywheel_synthetic_list', None):
        with open(args.flywheel_normal_list) as f:
            normal_paths = [line.strip() for line in f if line.strip()]
        with open(args.flywheel_synthetic_list) as f:
            synthetic_paths = [line.strip() for line in f if line.strip()]
        if len(normal_paths) != len(synthetic_paths):
            raise ValueError("flywheel normal_list and synthetic_list must have the same number of lines")
        def _preprocess_flywheel(path):
            cond = load_image(path)
            t = blip_diffusion_pipe.image_processor.preprocess(
                cond, do_resize=True, image_mean=blip_diffusion_pipe.config.mean,
                image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
            )["pixel_values"]
            return t.squeeze(0)
        flywheel_dataset = FlywheelPairDataset(normal_paths, synthetic_paths, _preprocess_flywheel)
        flywheel_loader = torch.utils.data.DataLoader(flywheel_dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f"DefectFlywheel: training from {len(flywheel_dataset)} (normal, synthetic) pairs")

    # DefectFlywheel: export_only 模式——不训练，用当前权重（或加载的 ckpt）构建 memory bank 后导出 .npy
    if getattr(args, 'export_npy', None) and getattr(args, 'export_output_dir', None) and getattr(args, 'export_only', False):
        load_ckpt = getattr(args, 'load_checkpoint_for_export', None)
        if load_ckpt and os.path.isfile(load_ckpt):
            ckpt = torch.load(load_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'soft_prompt' in ckpt:
                soft_prompt.load_state_dict(ckpt['soft_prompt'], strict=False)
            else:
                soft_prompt.load_state_dict(ckpt, strict=False)
            print(f"[Export] 已加载 checkpoint: {load_ckpt}")
        else:
            print("[Export] 使用初始 OFA 权重（未加载 checkpoint）")
        # 构建 memory bank（与训练后一致）
        shot_represents_bank = []
        pos_shot_represents = []
        neg_shot_represents = []
        visual_feature_bank_1['object'] = []
        visual_feature_bank_2['object'] = []
        query_bank = []
        for obj in obj_list:
            data_path = args.data_path + '/' + obj + '/train/good'
            for i in range(shot):
                with torch.no_grad():
                    train_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    idx = min(i * 3, len(train_files) - 1) if train_files else 0
                    selected_file = train_files[idx] if train_files else None
                    if selected_file:
                        cond_image = load_image(data_path + '/' + selected_file)
                    else:
                        raise FileNotFoundError(f"未在 {data_path} 下找到图像")
                    reference_image = blip_diffusion_pipe.image_processor.preprocess(
                        cond_image, image_mean=blip_diffusion_pipe.config.mean, image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
                    )["pixel_values"].to(device)
                    query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)
                    image_embedding, patch_embedding, patch_token_memory = model.encode_image(reference_image, features_list, DPAM_layer=20, ffn=False)
                    image_embedding = image_embedding.mean(dim=0).unsqueeze(dim=0)
                    pos_prompt_query, neg_prompt_query = soft_prompt(query, image_embedding)
                    pos_query_embedding, pos_token = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                    neg_query_embedding, neg_token = clip_model.encode_text_prompt(neg_prompt_query, padding, device)
                    pos_token = pos_token / pos_token.norm(dim=-1, keepdim=True)
                    neg_token = neg_token / neg_token.norm(dim=-1, keepdim=True)
                    visual_feature_bank_1['object'].append(patch_token_memory[0][0][1:])
                    visual_feature_bank_2['object'].append(patch_token_memory[2][0][1:])
                    pos_shot_represents.append(pos_query_embedding)
                    neg_shot_represents.append(neg_query_embedding)
                    query = query / query.norm(dim=-1, keepdim=True)
                    query_bank.append(query)
                    shot_represents_bank.append(pos_query_embedding)
                    shot_represents_bank.append(neg_query_embedding)
        visual_feature_bank_1['object'] = torch.stack(visual_feature_bank_1['object'], dim=0)
        visual_feature_bank_2['object'] = torch.stack(visual_feature_bank_2['object'], dim=0)
        visual_feature_bank_1['object'] = F.normalize(visual_feature_bank_1['object'], dim=-1)
        visual_feature_bank_2['object'] = F.normalize(visual_feature_bank_2['object'], dim=-1)
        shot_represents_bank = torch.stack(shot_represents_bank, dim=0).view(-1, 2, 768)
        query_bank = torch.vstack(query_bank)
        shot_represents_bank /= shot_represents_bank.norm(dim=-1, keepdim=True)
        pos_shot_represents = torch.vstack(pos_shot_represents)
        neg_shot_represents = torch.vstack(neg_shot_represents)
        pos_shot_represents = pos_shot_represents.mean(dim=0)
        neg_shot_represents = neg_shot_represents.mean(dim=0)
        pos_shot_represents /= pos_shot_represents.norm(dim=-1, keepdim=True)
        neg_shot_represents /= neg_shot_represents.norm(dim=-1, keepdim=True)
        shot_represents = text_prompts['object'].clone().T
        shot_represents[0] = pos_shot_represents
        shot_represents[1] = neg_shot_represents
        shot_represents = shot_represents.T
        # 执行导出
        export_obj = getattr(args, 'export_obj', None) or obj_list[0]
        with open(args.export_npy) as f:
            path_list = f.readlines()
        export_dataset = ExportListDataset(
            path_list, preprocess, target_transform, obj_list, export_obj,
            test_data.class_name_map_class_id
        )
        export_loader = torch.utils.data.DataLoader(export_dataset, batch_size=1, shuffle=False)
        os.makedirs(args.export_output_dir, exist_ok=True)
        model.to(device)
        for batch_idx, items in enumerate(tqdm(export_loader, desc="Export .npy")):
            image = items['img'].to(device)
            cls_name = items['cls_name']
            img_path = items['img_path'][0] if isinstance(items['img_path'], (list, tuple)) else items['img_path']
            cond_image = load_image(img_path)
            reference_image = blip_diffusion_pipe.image_processor.preprocess(
                cond_image, do_resize=True,
                image_mean=blip_diffusion_pipe.config.mean,
                image_std=blip_diffusion_pipe.config.std,
                return_tensors="pt",
            )["pixel_values"].to(device)
            with torch.no_grad():
                image_features, patch_features, patch_token_memory = model.encode_image(image, features_list, DPAM_layer=20, ffn=False)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                query = query.mean(dim=0).unsqueeze(dim=0)
                pos_prompt_query, neg_prompt_query = soft_prompt(query, image_features)
                pos_query_embedding, _ = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                neg_query_embedding, _ = clip_model.encode_text_prompt(neg_prompt_query, padding, device)
                pos_token_n = pos_token / pos_token.norm(dim=-1, keepdim=True)
                neg_token_n = neg_token / neg_token.norm(dim=-1, keepdim=True)
                pos_query_embedding = pos_query_embedding / pos_query_embedding.norm()
                neg_query_embedding = neg_query_embedding / neg_query_embedding.norm()
                query_n = query / query.norm(dim=-1, keepdim=True)
                query_n = query_n.expand(len(obj_list) * shot, -1, -1)
                query_sim = torch.mean(torch.sum(torch.mul(query_bank, query_n), dim=-1), dim=-1)
                obj_idx = torch.topk(query_sim, k=shot)[1].cpu().numpy().tolist()
                cur_visual_feature_bank_1 = visual_feature_bank_1['object'][obj_idx].view(-1, 1024)
                cur_visual_feature_bank_2 = visual_feature_bank_2['object'][obj_idx].view(-1, 1024)
                text_features = shot_represents_bank.clone()
                cur_text_features = text_features[obj_idx].clone().mean(dim=0)
                cur_text_features[0] = text_features[obj_idx][:, 0, :].clone().mean(dim=0)
                cur_text_features[1] = text_features[obj_idx][:, 1, :].clone().mean(dim=0)
                cur_text_features[0] = (cur_text_features[0].unsqueeze(dim=0) + pos_query_embedding) / 2
                cur_text_features[1] = (cur_text_features[1].unsqueeze(dim=0) + neg_query_embedding) / 2
                cur_text_features[0] = cur_text_features[0] / cur_text_features[0].norm()
                cur_text_features[1] = cur_text_features[1] / cur_text_features[1].norm()
                cur_text_features = cur_text_features.T
                anomaly_map_list = []
                for idx, patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                        similarity, _ = VVCLIP_lib.compute_similarity(patch_feature, cur_text_features.T)
                        similarity_map = similarity[:, 1:, :]
                        similarity_map = similarity_map.reshape(similarity_map.shape[0], 16, 16, -1).permute(0, 3, 1, 2)
                        similarity_map = similarity_map.permute(0, 2, 3, 1)
                        anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                        anomaly_map_list.append(anomaly_map)
                vis_feature_1 = patch_token_memory[0][0][1:]
                vis_feature_1 = vis_feature_1 / vis_feature_1.norm(dim=-1, keepdim=True)
                vis_feature_2 = patch_token_memory[2][0][1:]
                vis_feature_2 = vis_feature_2 / vis_feature_2.norm(dim=-1, keepdim=True)
                score1, _ = (1.0 - vis_feature_1 @ cur_visual_feature_bank_1.t()).min(dim=-1)
                score1 /= 2.0
                score2, _ = (1.0 - vis_feature_2 @ cur_visual_feature_bank_2.t()).min(dim=-1)
                score2 /= 2.0
                score = score1 + score2
                vis_score = score.reshape(1, 1, 16, 16)
                anomaly_map = torch.stack(anomaly_map_list)
                textual_anomaly_map = anomaly_map.sum(dim=0) / 4.0
                textual_anomaly_map = textual_anomaly_map.reshape(1, 1, 16, 16)
                anomaly_map = textual_anomaly_map + vis_score
                anomaly_map = F.interpolate(anomaly_map, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
                out_name = sample_id_from_path(items['img_path'][0], args.data_path) + '.npy'
                np.save(os.path.join(args.export_output_dir, out_name), anomaly_map.numpy())
        print(f"Exported {len(export_dataset)} anomaly maps to {args.export_output_dir}")
        sys.exit(0)

    #training
    if getattr(args, "eval_only", False):
        epochs = 1
        print("== Forced epochs=1 for eval_only mode ==")

    for epoch in range(epochs):
        disp_epoch = args.display_epoch if args.display_epoch is not None else (epoch + 1)
        disp_total = args.display_epochs if args.display_epochs is not None else epochs
        print(f"[Epoch {disp_epoch}/{disp_total}]")
        results = {}
        metrics = {}
        if getattr(args, "eval_only", False):
            print("== Eval Only Mode: Skipping Feature & Prompt Finetuning for this epoch ==")
            pass
        elif flywheel_loader is not None:
            # DefectFlywheel: use (normal, synthetic) pairs; neg_patch_embedding from synthetic image (no noise)
            for normal_t, synthetic_t in flywheel_loader:
                normal_t = normal_t.to(device)
                synthetic_t = synthetic_t.to(device)
                with torch.no_grad():
                    query = blip_diffusion_pipe.get_query_embeddings(normal_t, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)
                    image_embedding, patch_embedding, _ = model.encode_image(normal_t, features_list, DPAM_layer=20, ffn=False)
                    image_embedding = image_embedding.mean(dim=0).unsqueeze(dim=0)
                    image_embedding = image_embedding / image_embedding.norm()
                pos_query, neg_query = soft_prompt(query, image_embedding)
                pos_query_embedding, pos_token = clip_model.encode_text_prompt(pos_query, padding, device)
                neg_query_embedding, neg_token = clip_model.encode_text_prompt(neg_query, padding, device)
                pos_token = pos_token / pos_token.norm(dim=-1, keepdim=True)
                neg_token = neg_token / neg_token.norm(dim=-1, keepdim=True)
                pos_query_embedding = pos_query_embedding / pos_query_embedding.norm(dim=-1, keepdim=True)
                neg_query_embedding = neg_query_embedding / neg_query_embedding.norm(dim=-1, keepdim=True)
                p_pt = torch.dot(pos_query_embedding[0], text_prompts['object'][:,0]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                n_nt = torch.dot(neg_query_embedding[0], text_prompts['object'][:,1]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                n_pt = torch.dot(neg_query_embedding[0], text_prompts['object'][:,0]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                p_nt = torch.dot(pos_query_embedding[0], text_prompts['object'][:,1]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                text_loss = ((1 - p_pt) + (1 - n_nt) + n_pt + p_nt) / 4
                with torch.no_grad():
                    _, neg_patch_embedding_full, _ = model.encode_image(synthetic_t, features_list, DPAM_layer=20, ffn=False)
                patch_loss = 0
                for i in range(4):
                    patch_embedding[i] = patch_embedding[i] / patch_embedding[i].norm(dim=-1, keepdim=True)
                    neg_patch_embedding = neg_patch_embedding_full[i] / neg_patch_embedding_full[i].norm(dim=-1, keepdim=True)
                    p_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], pos_query_embedding[0]).mean().mean()
                    n_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], neg_query_embedding[0]).mean().mean()
                    n_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], neg_query_embedding[0]).mean().mean()
                    p_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], pos_query_embedding[0]).mean().mean()
                    patch_loss += ((1 - p_ppatch_sim) + (1 - n_npatch_sim) + n_ppatch_sim + p_npatch_sim) / 4
                patch_loss /= 4.0
                loss = text_loss + 0.125 * patch_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            random.shuffle(obj_list)
            for obj in obj_list:
                print(obj)
                results[obj] = {}
                results[obj]['gt_sp'] = []
                results[obj]['pr_sp'] = []
                results[obj]['imgs_masks'] = []
                results[obj]['anomaly_maps'] = []
                metrics[obj] = {}
                metrics[obj]['pixel-auroc'] = 0
                metrics[obj]['pixel-aupro'] = 0
                metrics[obj]['image-auroc'] = 0
                metrics[obj]['image-ap'] = 0

                # 适配ZJU数据集结构
                data_path = args.data_path + '/' + obj + '/train/good'
                # shot 选取规则：按文件名排序后取第 1、4、7… 张（索引 0, 3, 6…），即 i 与 i+3, i+6
                SHOT_INDEX_STEP = 3
                for i in range(shot):
                    train_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    idx = min(i * SHOT_INDEX_STEP, len(train_files) - 1) if train_files else 0
                    selected_file = train_files[idx] if train_files else None
                    if selected_file:
                        cond_image = load_image(data_path + '/' + selected_file)
                    else:
                        raise FileNotFoundError(f"未在 {data_path} 下找到图像")                
                    reference_image = blip_diffusion_pipe.image_processor.preprocess(
                                cond_image, do_resize=True, image_mean=blip_diffusion_pipe.config.mean, image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
                            )["pixel_values"]
                    reference_image = reference_image.to(device)


                    with torch.no_grad():
                        query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                        query = query.mean(dim=0).unsqueeze(dim=0)
                        image_embedding, patch_embedding, _ = model.encode_image(reference_image, features_list, DPAM_layer = 20, ffn=False)
                        image_embedding = image_embedding.mean(dim=0).unsqueeze(dim=0)
                        image_embedding = image_embedding / image_embedding.norm()
                        
                    pos_query, neg_query = soft_prompt(query, image_embedding)
                    pos_query_embedding, pos_token = clip_model.encode_text_prompt(pos_query, padding, device)
                    neg_query_embedding, neg_token = clip_model.encode_text_prompt(neg_query, padding, device)

                    pos_token = pos_token / pos_token.norm(dim = -1, keepdim = True)
                    neg_token = neg_token / neg_token.norm(dim = -1, keepdim = True)
                    pos_query_embedding = pos_query_embedding / pos_query_embedding.norm(dim = -1, keepdim = True)
                    neg_query_embedding = neg_query_embedding / neg_query_embedding.norm(dim = -1, keepdim = True)

                    # text-prompt alignment
                    p_ptext_sim = torch.dot(pos_query_embedding[0], text_prompts['object'][:,0]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                    n_ptext_sim = torch.dot(neg_query_embedding[0], text_prompts['object'][:,0]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                    p_ntext_sim = torch.dot(pos_query_embedding[0], text_prompts['object'][:,1]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                    n_ntext_sim = torch.dot(neg_query_embedding[0], text_prompts['object'][:,1]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                    text_loss = ((1 - p_ptext_sim) + (1 - n_ntext_sim) + n_ptext_sim + p_ntext_sim) / 4
                    
                    patch_loss = 0
                    fg_pos_it_patch = 0
                    fg_pos_ti_patch = 0
                    fg_neg_it_patch = 0
                    fg_neg_ti_patch = 0
                    thhold = 1 / 256


                    for i in range(4):
                        noise = torch.randn_like(patch_embedding[i]) * 2
                        patch_embedding[i] = patch_embedding[i] / patch_embedding[i].norm(dim = -1, keepdim = True)

                        neg_patch_embedding = patch_embedding[i - 1] + patch_embedding[i] + noise
                        neg_patch_embedding = neg_patch_embedding / neg_patch_embedding.norm(dim = -1, keepdim = True)
                        
                        # patch-prompt alignment
                        p_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], pos_query_embedding[0]).mean().mean()
                        n_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], neg_query_embedding[0]).mean().mean()
                        n_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], neg_query_embedding[0]).mean().mean()
                        p_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], pos_query_embedding[0]).mean().mean()
                        patch_loss += ((1 - p_ppatch_sim) + (1 - n_npatch_sim) + n_ppatch_sim + p_npatch_sim) / 4

                        # patch-token alignment
                        pos_similarity = torch.einsum('btd,bpd->btp', pos_token, patch_embedding[i][:, 1:, :])
                        pos_similarity = (pos_similarity - torch.min(pos_similarity, dim = -1, keepdim = True)[0]) / (torch.max(pos_similarity, dim = -1, keepdim = True)[0] - torch.min(pos_similarity, dim = -1, keepdim = True)[0])
                        pos_similarity = torch.where(pos_similarity < thhold, 0.0, pos_similarity)
                        pos_weights = pos_similarity / torch.sum(pos_similarity, dim=-1).T
                        pos_group_embed = torch.einsum('btp,bpd->btd', pos_weights, patch_embedding[i][:, 1:, :])
                        pos_group_embed = pos_group_embed / pos_group_embed.norm(dim = -1, keepdim = True)

                        pos_it_logits = torch.einsum('btd,bpd->btp', pos_group_embed, pos_token).squeeze(dim=0)
                        pos_it_labels = torch.eye(pos_it_logits.shape[1]).to(device)
                        pos_ti_logits = torch.einsum('btd,bpd->btp', pos_token, pos_group_embed).squeeze(dim=0)
                        pos_ti_labels = torch.eye(pos_ti_logits.shape[1]).to(device)
                        fg_pos_it_patch += criterion(pos_it_logits, pos_it_labels)
                        fg_pos_ti_patch += criterion(pos_ti_logits, pos_ti_labels)

                        neg_similarity = torch.einsum('btd,bpd->btp', neg_token, neg_patch_embedding[:, 1:, :])
                        neg_similarity = (neg_similarity - torch.min(neg_similarity, dim = -1, keepdim = True)[0]) / (torch.max(neg_similarity, dim = -1, keepdim = True)[0] - torch.min(neg_similarity, dim = -1, keepdim = True)[0])
                        neg_similarity = torch.where(neg_similarity < thhold, 0.0, neg_similarity)
                        neg_weights = neg_similarity / torch.sum(neg_similarity, dim=-1).T

                        neg_group_embed = torch.einsum('btp,bpd->btd', neg_weights, neg_patch_embedding[:, 1:, :])
                        neg_group_embed = neg_group_embed / neg_group_embed.norm(dim = -1, keepdim = True)

                        neg_it_logits = torch.einsum('btd,bpd->btp', neg_group_embed, neg_token).squeeze(dim=0)
                        neg_it_labels = torch.eye(neg_it_logits.shape[1]).to(device)
                        neg_ti_logits = torch.einsum('btd,bpd->btp', neg_token, neg_group_embed).squeeze(dim=0)

                        neg_ti_labels = torch.eye(neg_ti_logits.shape[1]).to(device)
                        fg_neg_it_patch += criterion(neg_it_logits, neg_it_labels)
                        fg_neg_ti_patch += criterion(neg_ti_logits, neg_ti_labels)
                    
                    patch_loss /= 4.0
                    fg_pos_it_patch /= 4.0
                    fg_pos_ti_patch /= 4.0
                    fg_neg_it_patch /= 4.0
                    fg_neg_ti_patch /= 4.0
                    fg_loss = (fg_pos_it_patch + fg_pos_ti_patch + fg_neg_it_patch + fg_neg_ti_patch) / 4.0
                
                    loss = text_loss + 0.125 * patch_loss + fg_loss 
                    print(text_loss.cpu(), patch_loss.cpu(), fg_loss.cpu())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        scheduler.step()
        
        #building memory bank
        with torch.no_grad():
            shot_represents_bank = []
            pos_shot_represents = []
            neg_shot_represents = []
            visual_feature_bank_1['object'] = []
            visual_feature_bank_2['object'] = []
            query_bank = []
            for obj in obj_list:
                print(obj)
                # 适配ZJU数据集结构
                data_path = args.data_path + '/' + obj + '/train/good'
                shot_represents = []
                # shot 选取规则：按文件名排序后取第 1、4、7… 张（索引 0, 3, 6…），即 i 与 i+3, i+6
                SHOT_INDEX_STEP = 3
                for i in range(shot):
                    train_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    idx = min(i * SHOT_INDEX_STEP, len(train_files) - 1) if train_files else 0
                    selected_file = train_files[idx] if train_files else None
                    if selected_file:
                        cond_image = load_image(data_path + '/' + selected_file)
                    else:
                        raise FileNotFoundError(f"未在 {data_path} 下找到图像")
                    reference_image = blip_diffusion_pipe.image_processor.preprocess(
                                cond_image, image_mean=blip_diffusion_pipe.config.mean, image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
                            )["pixel_values"]
                    reference_image = reference_image.to(device)
                    query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)
                    image_embedding, patch_embedding, patch_token_memory = model.encode_image(reference_image, features_list, DPAM_layer = 20, ffn=False)
                    image_embedding = image_embedding.mean(dim=0).unsqueeze(dim=0)
                    pos_prompt_query, neg_prompt_query = soft_prompt(query, image_embedding)
                    pos_query_embedding, _ = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                    neg_query_embedding, _ = clip_model.encode_text_prompt(neg_prompt_query, padding, device)

                    visual_feature_bank_1['object'].append(patch_token_memory[0][0][1:])
                    visual_feature_bank_2['object'].append(patch_token_memory[2][0][1:])
  
                    pos_shot_represents.append(pos_query_embedding)
                    neg_shot_represents.append(neg_query_embedding)
                    query /= query.norm(dim=-1, keepdim=True)
                    query_bank.append(query)

                    shot_represents_bank.append(pos_query_embedding)
                    shot_represents_bank.append(neg_query_embedding)

            visual_feature_bank_1['object'] = torch.stack(visual_feature_bank_1['object'], dim=0)
            visual_feature_bank_2['object'] = torch.stack(visual_feature_bank_2['object'], dim=0)
            visual_feature_bank_1['object'] = F.normalize(visual_feature_bank_1['object'], dim=-1)
            visual_feature_bank_2['object'] = F.normalize(visual_feature_bank_2['object'], dim=-1)

            shot_represents_bank = torch.stack(shot_represents_bank,dim=0).view(-1, 2, 768)
            query_bank = torch.vstack(query_bank)
            shot_represents_bank /= shot_represents_bank.norm(dim = -1, keepdim = True)

            pos_shot_represents = torch.vstack(pos_shot_represents)
            neg_shot_represents = torch.vstack(neg_shot_represents)

            pos_shot_represents = pos_shot_represents.mean(dim = 0)
            neg_shot_represents = neg_shot_represents.mean(dim = 0)
            pos_shot_represents /= pos_shot_represents.norm(dim = -1, keepdim = True)
            neg_shot_represents /= neg_shot_represents.norm(dim = -1, keepdim = True)
            shot_represents = text_prompts['object'].clone().T
            shot_represents[0] = pos_shot_represents
            shot_represents[1] = neg_shot_represents
            shot_represents = shot_represents.T

        # DefectFlywheel: export anomaly maps to .npy for mine_hard
        if getattr(args, 'export_npy', None) and getattr(args, 'export_output_dir', None):
            export_obj = getattr(args, 'export_obj', None) or obj_list[0]
            with open(args.export_npy) as f:
                path_list = f.readlines()
            export_dataset = ExportListDataset(
                path_list, preprocess, target_transform, obj_list, export_obj,
                test_data.class_name_map_class_id
            )
            export_loader = torch.utils.data.DataLoader(export_dataset, batch_size=1, shuffle=False)
            os.makedirs(args.export_output_dir, exist_ok=True)
            model.to(device)
            for batch_idx, items in enumerate(tqdm(export_loader, desc="Export .npy")):
                image = items['img'].to(device)
                cls_name = items['cls_name']
                # Blip Q-former 需要 224 输入；用 Blip 的 image_processor 预处理原图，与原 OFA shot 阶段一致
                img_path = items['img_path'][0] if isinstance(items['img_path'], (list, tuple)) else items['img_path']
                cond_image = load_image(img_path)
                reference_image = blip_diffusion_pipe.image_processor.preprocess(
                    cond_image, do_resize=True,
                    image_mean=blip_diffusion_pipe.config.mean,
                    image_std=blip_diffusion_pipe.config.std,
                    return_tensors="pt",
                )["pixel_values"].to(device)
                with torch.no_grad():
                    image_features, patch_features, patch_token_memory = model.encode_image(image, features_list, DPAM_layer=20, ffn=False)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)
                    pos_prompt_query, neg_prompt_query = soft_prompt(query, image_features)
                    pos_query_embedding, _ = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                    neg_query_embedding, _ = clip_model.encode_text_prompt(neg_prompt_query, padding, device)
                    pos_token_n = pos_token / pos_token.norm(dim=-1, keepdim=True)
                    neg_token_n = neg_token / neg_token.norm(dim=-1, keepdim=True)
                    pos_query_embedding = pos_query_embedding / pos_query_embedding.norm()
                    neg_query_embedding = neg_query_embedding / neg_query_embedding.norm()
                    query_n = query / query.norm(dim=-1, keepdim=True)
                    query_n = query_n.expand(len(obj_list) * shot, -1, -1)
                    query_sim = torch.mean(torch.sum(torch.mul(query_bank, query_n), dim=-1), dim=-1)
                    obj_idx = torch.topk(query_sim, k=shot)[1].cpu().numpy().tolist()
                    cur_visual_feature_bank_1 = visual_feature_bank_1['object'][obj_idx].view(-1, 1024)
                    cur_visual_feature_bank_2 = visual_feature_bank_2['object'][obj_idx].view(-1, 1024)
                    text_features = shot_represents_bank.clone()
                    cur_text_features = text_features[obj_idx].clone().mean(dim=0)
                    cur_text_features[0] = text_features[obj_idx][:, 0, :].clone().mean(dim=0)
                    cur_text_features[1] = text_features[obj_idx][:, 1, :].clone().mean(dim=0)
                    cur_text_features[0] = (cur_text_features[0].unsqueeze(dim=0) + pos_query_embedding) / 2
                    cur_text_features[1] = (cur_text_features[1].unsqueeze(dim=0) + neg_query_embedding) / 2
                    cur_text_features[0] = cur_text_features[0] / cur_text_features[0].norm()
                    cur_text_features[1] = cur_text_features[1] / cur_text_features[1].norm()
                    cur_text_features = cur_text_features.T
                    anomaly_map_list = []
                    for idx, patch_feature in enumerate(patch_features):
                        if idx >= args.feature_map_layer[0]:
                            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                            similarity, _ = VVCLIP_lib.compute_similarity(patch_feature, cur_text_features.T)
                            similarity_map = similarity[:, 1:, :]
                            similarity_map = similarity_map.reshape(similarity_map.shape[0], 16, 16, -1).permute(0, 3, 1, 2)
                            similarity_map = similarity_map.permute(0, 2, 3, 1)
                            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                            anomaly_map_list.append(anomaly_map)
                    vis_feature_1 = patch_token_memory[0][0][1:]
                    vis_feature_1 = vis_feature_1 / vis_feature_1.norm(dim=-1, keepdim=True)
                    vis_feature_2 = patch_token_memory[2][0][1:]
                    vis_feature_2 = vis_feature_2 / vis_feature_2.norm(dim=-1, keepdim=True)
                    score1, _ = (1.0 - vis_feature_1 @ cur_visual_feature_bank_1.t()).min(dim=-1)
                    score1 /= 2.0
                    score2, _ = (1.0 - vis_feature_2 @ cur_visual_feature_bank_2.t()).min(dim=-1)
                    score2 /= 2.0
                    score = score1 + score2
                    vis_score = score.reshape(1, 1, 16, 16)
                    anomaly_map = torch.stack(anomaly_map_list)
                    textual_anomaly_map = anomaly_map.sum(dim=0) / 4.0
                    textual_anomaly_map = textual_anomaly_map.reshape(1, 1, 16, 16)
                    anomaly_map = textual_anomaly_map + vis_score
                    anomaly_map = F.interpolate(anomaly_map, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
                out_name = sample_id_from_path(items['img_path'][0], args.data_path) + '.npy'
                np.save(os.path.join(args.export_output_dir, out_name), anomaly_map.numpy())
            print(f"Exported {len(export_dataset)} anomaly maps to {args.export_output_dir}")
            sys.exit(0)

        # 仅在最后一个 epoch 进行测试以节省时间（保留每轮的 ckpt 保存）
        if (epoch < epochs - 1) or getattr(args, 'skip_test', False):
            disp_epoch = args.display_epoch if args.display_epoch is not None else (epoch + 1)
            disp_total = args.display_epochs if args.display_epochs is not None else epochs
            logger.info("======== Epoch %d/%d ======== (Skip testing)", disp_epoch, disp_total)
            if args.save_path and not getattr(args, "export_npy", None) and not getattr(args, "eval_only", False):
                ckpt_template = getattr(args, "checkpoint_name", None)
                if ckpt_template:
                    ckpt_file = ckpt_template.format(epoch=epoch, display_epoch=disp_epoch, display_epoch0=disp_epoch - 1)
                else:
                    ckpt_file = f"epoch_{epoch}.pt"
                ckpt_path = os.path.join(args.save_path, ckpt_file)
                torch.save({"soft_prompt": soft_prompt.state_dict()}, ckpt_path)
                print(f"[Checkpoint] 已保存 {ckpt_path}")
            continue

        #testing 
        model.to(device)
        # 若前面未初始化 results（如直接走飞轮 co-train 分支），在测试前为每个 obj 建立结果容器
        if not results:
            for obj in obj_list:
                results[obj] = {
                    'gt_sp': [],
                    'pr_sp': [],
                    'imgs_masks': [],
                    'anomaly_maps': [],
                }
                metrics[obj] = {
                    'pixel-auroc': 0,
                    'pixel-aupro': 0,
                    'image-auroc': 0,
                    'image-ap': 0,
                }
        total_samples = len(test_dataloader)
        print(f"开始测试，总共 {total_samples} 个样本...")
        for idx, items in enumerate(test_dataloader):
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            image = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
            results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
            with torch.no_grad():
                image_features, patch_features, patch_token_memory = model.encode_image(image, features_list, DPAM_layer = 20, ffn=False)                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                query = blip_diffusion_pipe.get_query_embeddings(image, ['object']*10)
                query = query.mean(dim=0).unsqueeze(dim=0)

                pos_prompt_query, neg_prompt_query = soft_prompt(query, image_features)

                pos_query_embedding, pos_token = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                neg_query_embedding, neg_token = clip_model.encode_text_prompt(neg_prompt_query, padding, device)

                pos_token = pos_token / pos_token.norm(dim = -1, keepdim = True)
                neg_token = neg_token / neg_token.norm(dim = -1, keepdim = True)
                pos_query_embedding = pos_query_embedding / pos_query_embedding.norm()
                neg_query_embedding = neg_query_embedding / neg_query_embedding.norm()

                query /= query.norm(dim=-1, keepdim=True)
                query = query.expand(len(obj_list) * shot, -1, -1)
                query_sim = torch.mean(torch.sum(torch.mul(query_bank, query), dim=-1), dim = -1)
                
                obj_idx = torch.topk(query_sim, k=shot)[1].cpu().numpy().tolist()
                cur_visual_feature_bank_1 = visual_feature_bank_1['object'][obj_idx].view(-1, 1024)
                cur_visual_feature_bank_2 = visual_feature_bank_2['object'][obj_idx].view(-1, 1024)


                text_features = shot_represents_bank.clone()

                text_probs = torch.matmul(text_features, image_features.T).permute(0,2,1)
                text_probs = (text_probs).softmax(-1).view(len(obj_list) * shot, -1)
                text_probs = text_probs[obj_idx] + query_sim[obj_idx].view(len(obj_idx), -1)

                text_probs = text_probs.softmax(0)

                cur_text_features = text_features[obj_idx].clone().mean(dim=0)

                cur_text_features[0] = text_features[obj_idx][:, 0, :].clone().mean(dim=0)
                cur_text_features[1] = text_features[obj_idx][:, 1, :].clone().mean(dim=0)

                text_probs = torch.matmul(text_features, image_features.T).permute(0,2,1)
                text_probs = (text_probs/0.07).softmax(-1).view(len(obj_list) * shot, -1)
                text_probs = text_probs[obj_idx].mean(dim=0)

                cur_text_features[0] = (cur_text_features[0].unsqueeze(dim=0) + pos_query_embedding) / 2
                cur_text_features[1] = (cur_text_features[1].unsqueeze(dim=0) + neg_query_embedding) / 2

                cur_text_features[0] = cur_text_features[0] / cur_text_features[0].norm()
                cur_text_features[1] = cur_text_features[1] / cur_text_features[1].norm()
                cur_text_features = cur_text_features.T
                
                anomaly_map_list = []

                for idx, patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                        similarity, _ = VVCLIP_lib.compute_similarity(patch_feature, cur_text_features.T)
                        similarity_map = similarity[:, 1:, :]
                        similarity_map = similarity_map.reshape(similarity_map.shape[0], 16, 16, -1).permute(0, 3, 1, 2)
                        similarity_map = similarity_map.permute(0, 2, 3, 1)
                        anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                        anomaly_map_list.append(anomaly_map)

                        
                vis_feature_1 = patch_token_memory[0][0][1:]
                vis_feature_1 = vis_feature_1 / vis_feature_1.norm(dim=-1, keepdim=True)
                vis_feature_2 = patch_token_memory[2][0][1:]
                vis_feature_2 = vis_feature_2 / vis_feature_2.norm(dim=-1, keepdim=True)

                score1, _ = (1.0 - vis_feature_1 @ cur_visual_feature_bank_1.t()).min(dim=-1)
                score1 /= 2.0

                score2, _ = (1.0 - vis_feature_2 @ cur_visual_feature_bank_2.t()).min(dim=-1)
                score2 /= 2.0
                score = score1 + score2
                vis_score = score.reshape(1,1,16,16)
                
                anomaly_map = torch.stack(anomaly_map_list)
                textual_anomaly_map = anomaly_map.sum(dim = 0) / 4.0
                textual_anomaly_map = textual_anomaly_map.reshape(1,1,16,16)

                anomaly_map = textual_anomaly_map + vis_score 

                text_probs = -text_probs[0].unsqueeze(dim=0) - query_sim[obj_idx].mean(dim=0) + torch.max(textual_anomaly_map) + torch.max(vis_score)

                text_probs = text_probs.view(1)

                anomaly_map = F.interpolate(anomaly_map, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
                anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
                results[cls_name[0]]['anomaly_maps'].append(anomaly_map)

        table_ls = []
        image_auroc_list = []
        image_ap_list = []
        pixel_auroc_list = []
        pixel_aupro_list = []
        for obj in obj_list:
            table = []
            table.append(obj)
            results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
            results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
            if args.metrics == 'image-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
            elif args.metrics == 'pixel-level':
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            elif args.metrics == 'image-pixel-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            table_ls.append(table)

        if args.metrics == 'image-level':
            # logger
            table_ls.append(['mean', 
                            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
        elif args.metrics == 'pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                        ])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
        elif args.metrics == 'image-pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
        if np.mean(pixel_auroc_list) * 100 > best_pixel_auroc:
            best_pixel_auroc = np.mean(pixel_auroc_list) * 100
            best_result = results
        print(best_pixel_auroc, results)
        # 每轮都写入性能表，便于查看各 epoch 指标
        disp_epoch = args.display_epoch if args.display_epoch is not None else (epoch + 1)
        disp_total = args.display_epochs if args.display_epochs is not None else epochs
        logger.info("======== Epoch %d/%d ========", disp_epoch, disp_total)
        logger.info("\n%s", results)
        # 每轮保存 checkpoint，供飞轮第 2 轮导出时使用「第 1 轮 baseline」权重
        if args.save_path and not getattr(args, "export_npy", None) and not getattr(args, "eval_only", False):
            ckpt_template = getattr(args, "checkpoint_name", None)
            if ckpt_template:
                ckpt_file = ckpt_template.format(epoch=epoch, display_epoch=disp_epoch, display_epoch0=disp_epoch - 1)
            else:
                ckpt_file = f"epoch_{epoch}.pt"
            ckpt_path = os.path.join(args.save_path, ckpt_file)
            torch.save({"soft_prompt": soft_prompt.state_dict()}, ckpt_path)
            print(f"[Checkpoint] 已保存 {ckpt_path}")
    logger.info("\n======== 最佳 =========\n%s", best_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VVCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--blip_model_path", type=str, default="/home/lwx/model_card", help='path to BLIP-diffusion model')
    parser.add_argument("--export_npy", type=str, default=None, help='DefectFlywheel: path to normal list file; export anomaly maps to .npy')
    parser.add_argument("--export_output_dir", type=str, default=None, help='DefectFlywheel: output dir for .npy anomaly maps')
    parser.add_argument("--export_obj", type=str, default=None, help='DefectFlywheel: object/category name for export (default: first in obj_list)')
    parser.add_argument("--export_only", action="store_true", help='DefectFlywheel: 仅导出 .npy，不训练；与 export_npy 同用')
    parser.add_argument("--load_checkpoint_for_export", type=str, default=None, help='DefectFlywheel: export_only 时加载的 ckpt（如 baseline 第 1 轮）')
    parser.add_argument("--load_checkpoint", type=str, default=None, help='训练开始时加载的 ckpt（方案乙：k=0 用 baseline/epoch_0.pt，k≥1 用 flywheel_round_{k-1}/epoch_0.pt）')
    parser.add_argument("--flywheel_normal_list", type=str, default=None, help='DefectFlywheel: path to normal_list.txt for co-train data source')
    parser.add_argument("--flywheel_synthetic_list", type=str, default=None, help='DefectFlywheel: path to synthetic_anomaly_list.txt for co-train data source')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size (ZJU 模式下会强制为 224，与 16×16 grid 一致)")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    # 训练参数（实验公平性：baseline 与飞轮使用相同设置）
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数，实验对比时建议固定为 20")
    parser.add_argument("--display_epoch", type=int, default=None, help="仅用于日志显示的当前 Epoch（不改变真实计算轮数）")
    parser.add_argument("--display_epochs", type=int, default=None, help="仅用于日志显示的总 Epoch（不改变真实计算轮数）")
    parser.add_argument("--skip_test", action="store_true", help="不进行最后的 test")
    parser.add_argument("--eval_only", action="store_true", help="只评测，跳过训练阶段，不会存在任何参数的变化和微调。")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="可选 checkpoint 文件名模板，如 flywheel_epoch_{display_epoch0:02d}.pt；不传则使用 epoch_{epoch}.pt")
    parser.add_argument("--shot", type=int, default=2, help="每类 few-shot 数量")
    parser.add_argument("--lr", type=float, default=1e-5, help="Adam 学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight_decay")
    parser.add_argument("--scheduler_t_max", type=int, default=20, help="CosineAnnealingLR T_max，通常与 epochs 一致")

    args = parser.parse_args()
    # DefectFlywheel：环境变量覆盖路径，便于脚本传参
    if os.environ.get("BLIP_MODEL_PATH"):
        args.blip_model_path = os.environ.get("BLIP_MODEL_PATH")
    if os.environ.get("OFA_DATA_PATH"):
        args.data_path = os.environ.get("OFA_DATA_PATH")
    if os.environ.get("DEFECT_FLYWHEEL_NORMAL_LIST"):
        args.flywheel_normal_list = args.flywheel_normal_list or os.environ.get("DEFECT_FLYWHEEL_NORMAL_LIST")
    if os.environ.get("DEFECT_FLYWHEEL_SYNTHETIC_LIST"):
        args.flywheel_synthetic_list = args.flywheel_synthetic_list or os.environ.get("DEFECT_FLYWHEEL_SYNTHETIC_LIST")
    # DefectFlywheel：飞轮训练时若未通过环境变量传入 Blip 且默认路径不存在，使用 ofa 目录下的 blipdiffusion_model
    if args.flywheel_normal_list:
        _default_blip = "/home/lwx/model_card"
        if args.blip_model_path == _default_blip or not os.path.isdir(args.blip_model_path):
            _ofa_dir = os.path.dirname(os.path.abspath(__file__))
            _proj_blip = os.path.join(_ofa_dir, "blipdiffusion_model")
            if os.path.isdir(_proj_blip):
                args.blip_model_path = _proj_blip
    print(args)
    setup_seed(args.seed)
    main(args)
