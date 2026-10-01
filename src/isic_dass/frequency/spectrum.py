"""Các phép tính phổ numpy thuần (dùng chung cho fingerprint và mitigation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def channel_gap(path: str | Path) -> float:
    """Mức lệch giữa các kênh R, G, B. Ảnh xám thuần cho đúng 0.0."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return float(np.abs(np.diff(a, axis=2)).mean())


def radial_index(M: int) -> np.ndarray:
    yy, xx = np.mgrid[0:M, 0:M]
    return np.floor(np.sqrt((yy - 0.5 * M) ** 2 + (xx - 0.5 * M) ** 2)).astype(int)


def radial_power_profile(magnitude: np.ndarray) -> np.ndarray:
    """Algorithm 1 (Dong et al.): cộng biên độ theo bán kính, chuẩn hoá theo thành phần DC."""
    M = magnitude.shape[0]
    idx = radial_index(M)
    prof = np.bincount(idx.ravel(), weights=magnitude.ravel(), minlength=int(np.floor(np.sqrt(2) * M)))
    return prof / (prof[0] + 1e-12)


def fft_channels(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(biên độ, pha) của FFT 2D từng kênh, đã fftshift."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    F = np.fft.fftshift(np.fft.fft2(a, axes=(0, 1)), axes=(0, 1))
    return np.abs(F), np.angle(F)


def grayscale_power_profile(path: str | Path) -> np.ndarray:
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return radial_power_profile(np.abs(np.fft.fftshift(np.fft.fft2(g))))
