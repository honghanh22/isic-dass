"""Tiền xử lý: (tuỳ chọn) cắt viền tối -> (tuỳ chọn) đệm vuông -> resize -> PNG, giữ đúng số kênh."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .image_io import pil_mode, png_channels, read_image, to_pil

log = logging.getLogger(__name__)

BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _content_bbox(img: np.ndarray, dark_threshold: int) -> tuple[int, int, int, int] | None:
    mask = img.mean(axis=2) > dark_threshold
    if mask.sum() == 0:
        return None
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def crop_dark_border(img: np.ndarray, dark_threshold: int = 15, min_content_ratio: float = 0.5) -> np.ndarray:
    """Cắt viền đen, chỉ khi vùng nội dung chiếm < `min_content_ratio` một cạnh (ảnh dermoscopy). Ảnh (H, W, C)."""
    bbox = _content_bbox(img, dark_threshold)
    if bbox is None:
        return img
    rmin, rmax, cmin, cmax = bbox
    h, w = img.shape[:2]
    if (rmax - rmin + 1) / h < min_content_ratio or (cmax - cmin + 1) / w < min_content_ratio:
        return img[rmin:rmax + 1, cmin:cmax + 1]
    return img


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Đệm nền đen thành ảnh vuông, ảnh gốc ở giữa — giữ tỉ lệ giải phẫu, không bóp méo (ví dụ MRI)."""
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = img
    return canvas


def resize(img: np.ndarray, size: int) -> np.ndarray:
    out = np.asarray(to_pil(img).resize((size, size), BILINEAR))
    return out[..., None] if out.ndim == 2 else out


def preprocess_image(src: str | Path, dst: str | Path, img_size: int, channels: int, crop: bool = False,
                     resize_mode: str = "stretch") -> None:
    img = read_image(src, channels)
    if crop:
        img = crop_dark_border(img)
    if resize_mode == "pad_square":
        img = pad_to_square(img)
    to_pil(resize(img, img_size)).save(dst, format="PNG")


def preprocess_tree(src_root: str | Path, dst_root: str | Path, img_size: int, class_names: list[str],
                    channels: int, crop: bool = False, resize_mode: str = "stretch") -> None:
    """Tiền xử lý `src_root/<lớp>/*` sang `dst_root/<lớp>/*.png`.

    Bỏ qua ảnh đã xử lý ĐÚNG số kênh; ảnh cũ sai số kênh (ví dụ RGB từ bản trước) được xử lý lại.
    """
    for label in class_names:
        src_dir, dst_dir = Path(src_root) / label, Path(dst_root) / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        redone = 0
        for src in tqdm(sorted(src_dir.iterdir()), desc=f"Tiền xử lý {label} ({pil_mode(channels)})"):
            dst = dst_dir / (src.stem + ".png")
            if dst.exists():
                if png_channels(dst) == channels:
                    continue
                redone += 1
            preprocess_image(src, dst, img_size, channels, crop, resize_mode)
        if redone:
            log.warning("%s: xử lý lại %d ảnh cũ sai số kênh", label, redone)


def content_ratio_stats(paths: list[str], n: int = 200, dark_threshold: int = 15, seed: int = 0) -> dict[str, float]:
    """Tỉ lệ diện tích vùng có nội dung / ảnh — giúp quyết định có nên bật `crop_dark_border`.

    > 0.9 và ít dao động: không cần crop. < 0.7 và ít dao động: nên crop.
    std > 0.1: crop làm kích thước vật thể không đồng nhất giữa các ảnh, cân nhắc kỹ.
    """
    if not paths:
        return {}
    rng = np.random.default_rng(seed)
    ratios = []
    for p in rng.choice(paths, min(n, len(paths)), replace=False):
        img = read_image(p, 3)
        bbox = _content_bbox(img, dark_threshold)
        if bbox is not None:
            rmin, rmax, cmin, cmax = bbox
            ratios.append((rmax - rmin + 1) * (cmax - cmin + 1) / (img.shape[0] * img.shape[1]))
    if not ratios:
        return {}
    r = np.array(ratios)
    return {"median": float(np.median(r)), "min": float(r.min()), "max": float(r.max()), "std": float(r.std())}
