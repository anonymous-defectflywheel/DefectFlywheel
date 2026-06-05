"""轻量单测：模块二生成器（meta_shape_priors、可选 run_style_transfer 小图少步回归）."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_meta_shape_priors():
    """固定 seed 下 generate_meta_mask 输出形状与数值范围（仅用 numpy/cv2/scipy，不依赖 torch）。"""
    # 直接加载子模块，避免 generator/__init__.py 拉取 torch
    import importlib.util
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "models", "generator", "meta_shape_priors.py")
    spec = importlib.util.spec_from_file_location("meta_shape_priors", path)
    meta_shape_priors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meta_shape_priors)
    np.random.seed(42)
    h, w = 64, 64
    mask = meta_shape_priors.generate_meta_mask(w, h)
    assert mask.shape == (h, w), mask.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    mask2 = meta_shape_priors.generate_meta_mask(128, 128, m_max=3)
    assert mask2.shape == (128, 128)


def test_run_style_transfer_smoke():
    """固定 seed、小图、少步：run_style_transfer 能跑通并写出文件（需 CLIP/VGG）。"""
    try:
        import torch
        import clip
        from torchvision import models
        from models.generator import run_style_transfer
    except Exception as e:
        print("Skip run_style_transfer smoke test (missing deps):", e)
        return
    torch.manual_seed(1)
    np.random.seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = 64, 64
    content = torch.rand(1, 3, h, w, device=device)
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[16:48, 16:48] = 255
    clip_model, _ = clip.load("ViT-B/32", device=device)
    VGG = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for p in VGG.parameters():
        p.requires_grad = False
    # 构建 run_style_transfer 所需的 CLIP 特征
    from models.generator.def_train import _clip_normalize
    with torch.no_grad():
        content_224 = torch.nn.functional.interpolate(content, size=(224, 224), mode="bilinear")
        source_features = clip_model.encode_image(_clip_normalize(content_224, device))
        source_features = source_features / source_features.norm(dim=-1, keepdim=True)
        text_anom = clip.tokenize(["a defective fabric"]).to(device)
        text_norm = clip.tokenize(["a normal fabric"]).to(device)
        text_features = clip_model.encode_text(text_anom).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_source = clip_model.encode_text(text_norm).float()
        text_source = text_source / text_source.norm(dim=-1, keepdim=True)
    out_path = os.path.join(os.path.dirname(__file__), "_tmp_gen_out.jpg")
    try:
        run_style_transfer(
            content,
            clip_model,
            VGG,
            device,
            img_height=h,
            img_width=w,
            lambda_tv=1e-4,
            lambda_pdir=1.0,
            lambda_gdir=0.2,
            lambda_c=0.5,
            lambda_mclip=0.1,
            crop_size=64,
            num_crops=1,
            max_step=2,
            lr=0.02,
            thresh=0.0,
            save_img_path=out_path,
            source_features=source_features,
            text_features=text_features,
            text_source=text_source,
            mask_array=mask_arr,
        )
        assert os.path.isfile(out_path), out_path
    finally:
        if os.path.isfile(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    test_meta_shape_priors()
    test_run_style_transfer_smoke()
    print("Generator tests passed.")
