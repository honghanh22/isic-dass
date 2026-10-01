"""Bước 4b2 — chuẩn hoá đặc tính kỹ thuật của ảnh sinh trước khi kết luận về fingerprint.

1. Kênh màu: nếu ảnh thật là ảnh xám thuần, chuyển ảnh sinh sang xám 3 kênh (với ISIC màu -> tự bỏ qua).
2. Chuỗi resize: cho ảnh sinh đi qua cùng chuỗi nội suy bilinear như ảnh thật (tuỳ chọn).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from ..data.preprocess import BILINEAR
from ..gan.generate import archive_pool, restore_pool
from .spectrum import channel_gap

log = logging.getLogger(__name__)


def resolve_harmonize_channels(setting: object, real_paths: list[str], n: int = 100) -> bool:
    """'auto' -> chỉ bật khi ảnh thật là ảnh xám thuần (chênh lệch kênh ≈ 0)."""
    if setting != "auto":
        return bool(setting)
    gap = float(np.mean([channel_gap(p) for p in real_paths[:n]]))
    decision = gap < 1e-6
    log.info("Chênh lệch kênh màu của ảnh thật = %.6f -> harmonize_channels = %s", gap, decision)
    return decision


def median_short_side(raw_dir: str | Path, n: int = 200) -> int:
    files = sorted(Path(raw_dir).glob("*"))[:n]
    return int(np.median([min(Image.open(p).size) for p in files]))


def harmonize_pool(paths: list[str], zip_path: Path, out_dir: Path, img_size: int, to_grayscale: bool,
                   resize_ref_side: int | None) -> list[str]:
    cached = restore_pool(zip_path, out_dir)
    if cached is not None:
        return cached
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    new_paths = []
    for p in tqdm(paths, desc=f"Chuẩn hoá {out_dir.name}"):
        img = Image.open(p).convert("RGB")
        if resize_ref_side:
            img = img.resize((resize_ref_side, resize_ref_side), BILINEAR).resize((img_size, img_size), BILINEAR)
        if to_grayscale:
            g = np.asarray(img.convert("L"))
            img = Image.fromarray(np.stack([g] * 3, axis=-1))
        q = out_dir / Path(p).name
        img.save(q, format="PNG")
        new_paths.append(str(q))
    archive_pool(out_dir, zip_path)
    return new_paths
