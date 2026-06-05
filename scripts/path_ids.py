#!/usr/bin/env python3
"""Path-safe sample identifiers for DefectFlywheel artifacts.

Use category-aware ids so multi-class datasets with repeated filenames
(e.g. WFDD/*/train/good/001.png) do not overwrite anomaly maps or masks.
"""
import os
import re
from pathlib import Path


def _safe_token(value):
    value = str(value).strip()
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def sample_id_from_path(image_path, data_root=None):
    """Return a stable filesystem-safe id for an image path.

    Preferred layout: {data_root}/{class}/train/good/{file}.png -> {class}__{stem}.
    Fallback: safe relative path without extension, preserving directory context.
    """
    image_abs = os.path.abspath(str(image_path))
    image = Path(image_abs)
    stem = _safe_token(image.stem)

    rel_parts = None
    if data_root:
        try:
            rel = os.path.relpath(image_abs, os.path.abspath(str(data_root)))
            if not rel.startswith("..") and not os.path.isabs(rel):
                rel_parts = Path(rel).parts
        except Exception:
            rel_parts = None

    if rel_parts:
        parts = list(rel_parts)
        class_name = None
        if "train" in parts:
            idx = parts.index("train")
            if idx > 0:
                class_name = parts[idx - 1]
        if class_name is None and len(parts) >= 2:
            class_name = parts[0]
        if class_name:
            return f"{_safe_token(class_name)}__{stem}"
        return "__".join(_safe_token(p) for p in parts[:-1] + [image.stem])

    parent = image.parent.name
    if parent and parent not in {"good", "train", "."}:
        return f"{_safe_token(parent)}__{stem}"
    return stem


def mask_name_for_path(image_path, data_root=None):
    return f"{sample_id_from_path(image_path, data_root)}_mask.png"
