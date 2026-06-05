#!/usr/bin/env python3
"""
异常样本生成脚本（模块二）

在 mask 区域内生成伪异常图，支持：
  - 困难 mask 模式：读取模块一输出的 *_mask.png，只在困难区域生成（数据飞轮核心）
  - 随机 mask 模式：使用随机形状先验，与 AnoStyler 原版一致

与创新思想一致：优先使用困难样本挖掘得到的 mask，使伪异常集中在模型薄弱区域。
"""
import os
import sys
import argparse
import json
import random
import time

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
import clip

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.generator.ano_utils import load_image2
from models.generator.def_train import run_style_transfer
from models.generator.meta_shape_priors import generate_meta_mask, mask_inside_region
from models.generator import prompts
from scripts.path_ids import mask_name_for_path


def _build_text_features(clip_model, device, category, defect=None):
    """构建正常/异常文本的 CLIP 特征（与 AnoStyler 一致）。"""
    normal_phrases = [
        tpl.format(prompt.format(category))
        for tpl in prompts.template_level_prompts
        for prompt in prompts.state_level_normal_prompts
    ]
    abnormal_phrases = []
    for tpl in prompts.template_level_prompts:
        for prompt in prompts.state_level_abnormal_prompts:
            abnormal_phrases.append(tpl.format(prompt.format(category)))
        for prompt in prompts.state_level_abnormality_specific_prompts:
            abnormal_phrases.append(tpl.format(prompt.format(category, defect or "defect")))

    with torch.no_grad():
        # 使用 truncate=True 防止长上下文溢出 CLIP 维度限制 (77 tokens)
        tokens_abn = clip.tokenize(abnormal_phrases, truncate=True).to(device)
        text_features = clip_model.encode_text(tokens_abn).mean(dim=0, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        tokens_src = clip.tokenize(normal_phrases, truncate=True).to(device)
        text_source = clip_model.encode_text(tokens_src).mean(dim=0, keepdim=True)
        text_source = text_source / text_source.norm(dim=-1, keepdim=True)

    return text_features, text_source


def _find_hard_mask_for_image(normal_path, hard_mask_dir, data_root=None):
    """Find the hard mask for a normal image using category-aware sample ids.

    Preferred: {class}__{stem}_mask.png. Legacy basename fallback is kept only
    for old debugging outputs.
    """
    candidates = [mask_name_for_path(normal_path, data_root)]
    base = os.path.splitext(os.path.basename(normal_path))[0]
    legacy_name = f"{base}_mask.png"
    if legacy_name not in candidates:
        candidates.append(legacy_name)
    for mask_name in candidates:
        mask_path = os.path.join(hard_mask_dir, mask_name)
        if os.path.isfile(mask_path):
            return mask_path
    return None

def _find_model_index_dir(model_path):
    """Resolve a local HuggingFace/BlipDiffusion cache directory containing model_index.json."""
    if not model_path:
        model_path = os.path.join(PROJECT_ROOT, "ofa", "blipdiffusion_model")
    candidates = []
    if model_path:
        candidates.append(os.path.abspath(model_path))
    candidates.append(os.path.join(PROJECT_ROOT, "ofa", "blipdiffusion_model"))
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        if os.path.isfile(os.path.join(candidate, "model_index.json")):
            return candidate
        for root, _, files in os.walk(candidate):
            if "model_index.json" in files:
                return root
    raise FileNotFoundError(
        "未找到 BlipDiffusion model_index.json"
    )


def _load_blipdiffusion_pipe(model_path, device):
    """Load BlipDiffusionPipeline for generator-side Q-Former context extraction."""
    from diffusers.pipelines import BlipDiffusionPipeline
    resolved = _find_model_index_dir(model_path)
    pipe = BlipDiffusionPipeline.from_pretrained(resolved, local_files_only=True)
    pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe, resolved


def _mean_pool_text_encoder(pipe, texts, device):
    tokens = pipe.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=getattr(pipe.tokenizer, "model_max_length", 77),
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = pipe.text_encoder(**tokens)
        hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
    mask = tokens.get("attention_mask")
    if mask is None:
        emb = hidden.mean(dim=1)
    else:
        emb = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return emb


def _qformer_select_context(pipe, raw_image, device, category):
    """Use BlipDiffusion Q-Former query embedding to select an image-specific fabric context phrase."""
    candidates = []
    for item in [category, "fabric", "textile fabric", "woven fabric", "patterned fabric", "cloth surface"]:
        if item and isinstance(item, str):
            candidates.append(item)
    candidates.extend(getattr(prompts, "FABRIC_CATEGORIES", []))
    candidates = list(dict.fromkeys([c for c in candidates if c]))
    preprocess_kwargs = {"do_resize": True, "return_tensors": "pt"}
    try:
        preprocess_kwargs.update({"image_mean": pipe.config.mean, "image_std": pipe.config.std})
    except Exception:
        pass
    reference_image = pipe.image_processor.preprocess(raw_image, **preprocess_kwargs)["pixel_values"].to(device)
    with torch.no_grad():
        query = pipe.get_query_embeddings(reference_image, ["fabric"] * 10)
    q = query
    while q.ndim > 1:
        q = q.mean(dim=0)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    text_emb = _mean_pool_text_encoder(pipe, candidates, device)
    if q.shape[-1] != text_emb.shape[-1]:
        raise RuntimeError(f"Q-Former dim {q.shape[-1]} != text encoder dim {text_emb.shape[-1]}")
    scores = torch.matmul(text_emb, q.reshape(-1, 1)).squeeze(-1)
    best_idx = int(torch.argmax(scores).item())
    return candidates[best_idx], {
        "context_backend": "blipdiffusion_qformer",
        "selected_context": candidates[best_idx],
        "selected_score": float(scores[best_idx].detach().cpu()),
        "candidate_count": len(candidates),
        "query_shape": list(query.shape),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="异常样本生成：支持困难 mask（模块一）或随机 mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="可选 yaml 配置，与下面命令行参数可叠加")
    # 输入输出
    parser.add_argument("--normal_image", type=str, default=None, help="单张正常图路径")
    parser.add_argument("--normal_list", type=str, default=None, help="正常图路径列表文件，每行一条路径")
    parser.add_argument("--data_root", type=str, default=None, help="数据集根目录，用于生成类别感知 sample_id 匹配 hard mask")
    parser.add_argument("--hard_mask_dir", type=str, default=None, help="模块一输出的 mask 目录，若存在则优先用困难 mask")
    parser.add_argument("--save_dir", type=str, default="./outputs/generated_anomaly", help="生成图与 mask 保存目录")
    parser.add_argument("--category", type=str, default=None, help="类别名；未指定则从布匹预设中随机选")
    parser.add_argument("--defect", type=str, default=None, help="缺陷类型；未指定则从布匹预设中随机选")
    # 图像尺寸
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--img_height", type=int, default=512)
    # 生成数量（仅随机 mask 时按 num_gen 重复；困难 mask 时每张正常图生成一张）
    parser.add_argument("--num_gen", type=int, default=1, help="每张正常图生成数量（随机 mask 时有效）")
    # 随机 mask 超参（无困难 mask 时使用）
    parser.add_argument("--m_max", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    # 风格迁移超参
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--num_crops", type=int, default=64)
    parser.add_argument("--max_step", type=int, default=75)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--thresh", type=float, default=0.7)
    parser.add_argument("--lambda_tv", type=float, default=0.002)
    parser.add_argument("--lambda_pdir", type=float, default=9000.0)
    parser.add_argument("--lambda_gdir", type=float, default=500.0)
    parser.add_argument("--lambda_c", type=float, default=150.0)
    parser.add_argument("--lambda_mclip", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP 模型名")
    parser.add_argument("--context_backend", type=str, default="blipdiffusion_qformer", choices=["blipdiffusion_qformer", "blip_caption", "static_prompt"], help="上下文来源；full method 使用 BlipDiffusion Q-Former；消融可用 blip_caption/static_prompt")
    parser.add_argument("--blip_model_path", type=str, default=None, help="BLIP 模型的本地路径，用于替代 HuggingFace 默认的自动下载")
    parser.add_argument("--disable_blip", action="store_true", help="兼容旧消融脚本：强制 context_backend=static_prompt")
    parser.add_argument(
        "--no_hard_mask_shape_diversity",
        action="store_false",
        dest="hard_mask_shape_diversity",
        default=True,
        help="关闭则仅用原始困难 mask；默认在困难区域内叠加随机形状（点/线/freeform）以增加多样性（与 AnoStyler 一致）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    if getattr(args, "disable_blip", False):
        args.context_backend = "static_prompt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 未指定 category/defect 时从布匹预设中随机选用
    if args.category is None or args.category.strip() == "":
        args.category = random.choice(prompts.FABRIC_CATEGORIES)
        print(f"未指定 category，从布匹预设随机选用: {args.category}")
    if args.defect is None or (isinstance(args.defect, str) and args.defect.strip() == ""):
        args.defect = random.choice(prompts.FABRIC_DEFECTS)
        print(f"未指定 defect，从布匹预设随机选用: {args.defect}")

    # 加载 CLIP 与 VGG
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()

    from torchvision import models
    VGG = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for p in VGG.parameters():
        p.requires_grad = False
        
    use_blip_caption = False
    use_qformer = False
    blip_processor = None
    blip_model = None
    qformer_pipe = None
    qformer_model_path = None
    if args.context_backend == "blipdiffusion_qformer":
        print("Loading BlipDiffusion pipeline for Q-Former context selection...")
        qformer_pipe, qformer_model_path = _load_blipdiffusion_pipe(args.blip_model_path, device)
        print(f"BlipDiffusion Q-Former model path: {qformer_model_path}")
        use_qformer = True
    elif args.context_backend == "blip_caption":
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("Loading BLIP model for caption-based context-aware prompt generation...")
            blip_path = args.blip_model_path if args.blip_model_path and os.path.exists(args.blip_model_path) else "Salesforce/blip-image-captioning-base"
            print(f"BLIP caption model path: {blip_path}")
            blip_processor = BlipProcessor.from_pretrained(blip_path)
            blip_model = BlipForConditionalGeneration.from_pretrained(blip_path).to(device)
            blip_model.eval()
            use_blip_caption = True
        except Exception as e:
            print(f"Failed to load BLIP caption model, fallback to static category. Error: {e}")
            use_blip_caption = False
    else:
        print("context_backend=static_prompt; skip multimodal context extraction.")

    # Base text features if we don't have blip context initially
    if not use_blip_caption and not use_qformer:
        text_features, text_source = _build_text_features(
            clip_model, device, args.category, args.defect
        )

    # 收集 (正常图路径, 可选困难 mask 路径) 列表
    normal_paths = []
    if args.normal_image and os.path.isfile(args.normal_image):
        normal_paths.append(args.normal_image)
    if args.normal_list and os.path.isfile(args.normal_list):
        with open(args.normal_list, "r") as f:
            for line in f:
                p = line.strip()
                if p and os.path.isfile(p):
                    normal_paths.append(p)
    if not normal_paths:
        print("未找到任何正常图，请指定 --normal_image 或 --normal_list")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)
    image_dir = os.path.join(args.save_dir, "image")
    mask_dir = os.path.join(args.save_dir, "mask")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    generation_records = []
    with open(os.path.join(args.save_dir, "generation_config.json"), "w") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    total_time = 0.0
    gen_idx = 0
    for normal_path in tqdm(normal_paths, desc="Images"):
        content_image = load_image2(
            normal_path,
            img_height=args.img_height,
            img_width=args.img_width,
        ).to(device)
        
        context_used = args.category
        context_source = "static_prompt"
        qformer_meta = {}
        if use_qformer:
            raw_image = Image.open(normal_path).convert('RGB')
            context_used, qformer_meta = _qformer_select_context(qformer_pipe, raw_image, device, args.category)
            context_source = "blipdiffusion_qformer"
            print(f"\n[Q-Former Context Selected] Image: {os.path.basename(normal_path)} -> Context: '{context_used}'")
            text_features, text_source = _build_text_features(
                clip_model, device, context_used, args.defect
            )
        elif use_blip_caption:
            try:
                raw_image = Image.open(normal_path).convert('RGB')
                inputs = blip_processor(raw_image, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = blip_model.generate(**inputs)
                    dynamic_context = blip_processor.decode(out[0], skip_special_tokens=True)
                context_used = dynamic_context
                context_source = "blip_caption"
                print(f"\n[BLIP Context Extracted] Image: {os.path.basename(normal_path)} -> Context: '{dynamic_context}'")
                text_features, text_source = _build_text_features(
                    clip_model, device, dynamic_context, args.defect
                )
            except Exception as e:
                print(f"Extraction failed for {normal_path}, fallback to static category: {e}")
                context_used = args.category
                context_source = "static_prompt_fallback"
                text_features, text_source = _build_text_features(
                    clip_model, device, args.category, args.defect
                )

        with torch.no_grad():
            source_features = clip_model.encode_image(
                _clip_normalize_static(content_image, device)
            ).detach()
            source_features = source_features / source_features.norm(dim=-1, keepdim=True)

        n_gen = args.num_gen if args.hard_mask_dir is None else 1
        for k in range(n_gen):
            t_start = time.perf_counter()
            mask_array = None
            mask_path = None

            mask_path = None
            if args.hard_mask_dir:
                hard_mask_path = _find_hard_mask_for_image(normal_path, args.hard_mask_dir, getattr(args, "data_root", None))
                if hard_mask_path:
                    mask_array = np.array(Image.open(hard_mask_path).convert("L"))
                    if getattr(args, "hard_mask_shape_diversity", True):
                        mask_array = mask_inside_region(
                            mask_array, args.img_width, args.img_height, min_pixels=50
                        )
                    m_save = cv2.resize(mask_array, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(mask_dir, f"gen_mask_{gen_idx}.png"), m_save)
                else:
                    raise FileNotFoundError(
                        f"hard_mask_dir was provided but no mask matched normal image: {normal_path}; "
                        f"searched category-aware and legacy names under {args.hard_mask_dir}"
                    )
            else:
                mask = generate_meta_mask(
                    W=args.img_width, H=args.img_height,
                    m_max=args.m_max, alpha=args.alpha,
                )
                while mask.sum() == 0:
                    mask = generate_meta_mask(
                        W=args.img_width, H=args.img_height,
                        m_max=args.m_max, alpha=args.alpha,
                    )
                mask_array = mask
                m_save = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(mask_dir, f"gen_mask_{gen_idx}.png"), m_save)

            save_img_path = os.path.join(image_dir, f"gen_ano_{gen_idx}.jpg")

            run_style_transfer(
                content_image=content_image,
                clip_model=clip_model,
                VGG=VGG,
                device=device,
                img_height=args.img_height,
                img_width=args.img_width,
                lambda_tv=args.lambda_tv,
                lambda_pdir=args.lambda_pdir,
                lambda_gdir=args.lambda_gdir,
                lambda_c=args.lambda_c,
                lambda_mclip=args.lambda_mclip,
                crop_size=args.crop_size,
                num_crops=args.num_crops,
                max_step=args.max_step,
                lr=args.lr,
                thresh=args.thresh,
                save_img_path=save_img_path,
                source_features=source_features,
                text_features=text_features,
                text_source=text_source,
                mask_path=None,
                mask_array=mask_array,
            )

            elapsed = time.perf_counter() - t_start
            total_time += elapsed
            generation_records.append({
                "index": gen_idx,
                "normal_path": os.path.abspath(normal_path),
                "generated_image": os.path.abspath(save_img_path),
                "mask_path": os.path.abspath(os.path.join(mask_dir, f"gen_mask_{gen_idx}.png")),
                "context_backend": args.context_backend,
                "context_source": context_source,
                "context_used": context_used,
                "qformer_context": qformer_meta,
                "defect": args.defect,
                "elapsed_seconds": elapsed,
            })
            gen_idx += 1

    with open(os.path.join(args.save_dir, "generation_manifest.json"), "w") as f:
        json.dump({"records": generation_records}, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"生成 {gen_idx} 张图像，总耗时 {total_time:.2f}s，平均 {total_time / max(1, gen_idx):.2f}s/张")
    print(f"保存目录: {args.save_dir}")


def _clip_normalize_static(image, device):
    import torch.nn.functional as F
    image = F.interpolate(image, size=224, mode="bicubic")
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, -1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, -1, 1, 1)
    return (image - mean) / std


if __name__ == "__main__":
    main()
