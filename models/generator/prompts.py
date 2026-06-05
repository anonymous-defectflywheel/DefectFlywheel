"""正常/异常文本模板，用于 CLIP 方向损失（ΔI ∝ ΔT）。"""

# 布匹预设：当用户未在配置/命令行指定 category 或 defect 时，从此列表随机选用
FABRIC_CATEGORIES = [
    "fabric",
    "textile",
    "cloth",
    "woven fabric",
]

FABRIC_DEFECTS = [
    "stain",
    "hole",
    "tear",
    "scratch",
    "fray",
    "contamination",
    "color defect",
    "weaving defect",
]

state_level_normal_prompts = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

state_level_abnormal_prompts = [
    "damaged {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

state_level_abnormality_specific_prompts = [
    "{} with {} defect",
    "{} with {} flaw",
    "{} with {} damage",
]

template_level_prompts = [
    "a photo of a {} for anomaly detection",
    "a photo of the {} for anomaly detection",
    "a photo of a {} for visual inspection",
    "a photo of the {} for visual inspection",
    "a close-up photo of a {}",
    "a close-up photo of the {}",
    "a cropped photo of a {}",
    "a cropped photo of the {}",
    "a blurry photo of a {}",
    "a blurry photo of the {}",
    "a dark photo of a {}",
    "a dark photo of the {}",
    "a bright photo of a {}",
    "a bright photo of the {}",
    "a low resolution photo of a {}",
    "a low resolution photo of the {}",
    "a photo of a small {}",
    "a photo of the small {}",
    "a photo of a large {}",
    "a photo of the large {}",
    "a dirty photo of a {}",
    "a dirty photo of the {}",
    "a bad photo of a {}",
    "a cropped close-up photo of a {}",
    "a cropped close-up photo of the {}",
]
