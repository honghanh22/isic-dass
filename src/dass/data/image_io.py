"""NƠI DUY NHẤT đọc / ghi ảnh trong pipeline — giữ nguyên số kênh gốc của bộ dữ liệu.

- Ảnh xám: mảng (H, W, 1), lưu PNG chế độ "L". Ảnh màu: (H, W, 3), lưu PNG "RGB".
- Không bước nào được tạo ra các kênh KHÁC NHAU từ ảnh xám (nguồn gốc lỗi "ảnh xám có màu" của bản cũ).
- Mạng pretrain ImageNet / Inception cần 3 kênh: nhân bản 1 -> 3 kênh ngay trước mạng, trên bộ nhớ
  (`to_three_channels`), không ghi ra đĩa. Ba kênh giống hệt nhau nên không mất / thêm thông tin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

GRAY_MODES = {"1", "L", "LA", "I", "I;16", "F"}
GRAY_TOLERANCE = 1.0   # chênh lệch kênh trung bình (mức xám) dưới ngưỡng này coi là ảnh xám (nhiễu nén JPEG)


class ChannelMismatchError(RuntimeError):
    """Ảnh được coi là xám nhưng các kênh khác nhau quá ngưỡng -> gộp về 1 kênh sẽ mất thông tin."""


def pil_mode(channels: int) -> str:
    if channels not in (1, 3):
        raise ValueError(f"Số kênh phải là 1 hoặc 3, nhận {channels}")
    return "L" if channels == 1 else "RGB"


def read_image(path: str | Path, channels: int) -> np.ndarray:
    """Đọc ảnh thành uint8 (H, W, channels). Ảnh xám đọc theo luminance của PIL (R = G = B -> giữ nguyên giá trị)."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert(pil_mode(channels)), dtype=np.uint8)
    return arr[..., None] if channels == 1 else arr


def to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode="L")
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def write_png(arr: np.ndarray, path: str | Path) -> None:
    to_pil(arr).save(path, format="PNG")


def png_channels(path: str | Path) -> int:
    with Image.open(path) as img:
        return 1 if img.mode in GRAY_MODES else 3


def channel_gap(arr: np.ndarray) -> float:
    """Chênh lệch trung bình giữa các kênh R, G, B (mức xám). Ảnh 1 kênh -> 0."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim < 3 or a.shape[-1] == 1:
        return 0.0
    return float(np.abs(np.diff(a, axis=-1)).mean())


def max_channel_diff(arr: np.ndarray) -> int:
    """Chênh lệch lớn nhất giữa các kênh (mức xám) trên toàn mảng (..., C)."""
    a = np.asarray(arr, dtype=np.int16)
    if a.shape[-1] == 1:
        return 0
    return int(np.abs(np.diff(a, axis=-1)).max())


def collapse_to_gray(arr: np.ndarray, tolerance: int = 0) -> np.ndarray:
    """(..., 3) có 3 kênh giống nhau -> (..., 1) không mất thông tin. Kênh khác nhau quá `tolerance` -> lỗi."""
    if arr.shape[-1] == 1:
        return arr
    diff = max_channel_diff(arr)
    if diff > tolerance:
        raise ChannelMismatchError(f"Các kênh khác nhau tới {diff} mức xám (ngưỡng {tolerance}): gộp về 1 kênh sẽ "
                                   "mất thông tin.")
    return arr[..., :1]


def to_three_channels(arr: np.ndarray) -> np.ndarray:
    """(..., 1) -> (..., 3) bằng nhân bản (chỉ dùng ngay trước mạng pretrain 3 kênh)."""
    return np.repeat(arr, 3, axis=-1) if arr.shape[-1] == 1 else arr


def detect_channels(paths: list[str], n: int = 200, seed: int = 0, tolerance: float = GRAY_TOLERANCE) -> int:
    """1 nếu mọi ảnh lấy mẫu là ảnh xám (chế độ xám, hoặc RGB có 3 kênh gần như bằng nhau), ngược lại 3."""
    if not paths:
        raise ValueError("Không có ảnh để nhận diện số kênh")
    rng = np.random.default_rng(seed)
    for p in rng.choice(paths, min(n, len(paths)), replace=False):
        with Image.open(p) as img:
            if img.mode in GRAY_MODES:
                continue
            if channel_gap(np.asarray(img.convert("RGB"))) > tolerance:
                return 3
    return 1


def load_images(paths: list[str], channels: int) -> np.ndarray:
    """(N, H, W, channels) uint8."""
    return np.stack([read_image(p, channels) for p in paths])
