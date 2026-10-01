"""Inception-v3 của StyleGAN2-ADA (detector chuẩn cho KID / FID), nạp generator và sinh ảnh theo lớp.

Mọi hàm import torch / module của repo StyleGAN2-ADA khi được gọi (repo phải đã nằm trong sys.path).
Ảnh 1 kênh được nhân bản thành 3 kênh ngay trước Inception — giống `metric_utils` của chính StyleGAN2-ADA.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

INCEPTION_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"


def _device():
    import torch

    return torch.device("cuda")


@lru_cache(maxsize=1)
def get_inception():
    from metrics import metric_utils  # module của repo StyleGAN2-ADA

    return metric_utils.get_feature_detector(INCEPTION_URL, device=_device())


def inception_features(images_uint8_nhwc: np.ndarray, batch: int = 64) -> np.ndarray:
    """(N, H, W, C) uint8, C = 1 hoặc 3 -> (N, 2048) float64."""
    import torch

    det, device = get_inception(), _device()
    feats = []
    with torch.no_grad():
        for s in range(0, len(images_uint8_nhwc), batch):
            x = torch.from_numpy(np.ascontiguousarray(images_uint8_nhwc[s:s + batch])).permute(0, 3, 1, 2).to(device)
            if x.shape[1] == 1:
                x = x.repeat([1, 3, 1, 1])
            feats.append(det(x, return_features=True).cpu().numpy().astype(np.float64))
    return np.concatenate(feats, axis=0)


def load_generator(pkl_path: str | Path):
    import legacy  # module của repo StyleGAN2-ADA

    with open(pkl_path, "rb") as f:
        return legacy.load_network_pkl(f)["G_ema"].to(_device()).eval().requires_grad_(False)


def generate_class_images(G, n: int, class_idx: int, psi: float = 1.0, seed: int = 0, batch: int = 32) -> np.ndarray:
    """Sinh `n` ảnh uint8 NHWC (C = số kênh của generator) có điều kiện theo `class_idx`."""
    import torch

    device = _device()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    out = []
    with torch.no_grad():
        for s in range(0, n, batch):
            b = min(batch, n - s)
            z = torch.randn([b, G.z_dim], generator=gen, device=device)
            c = torch.zeros([b, G.c_dim], device=device)
            c[:, class_idx] = 1
            img = G(z, c, truncation_psi=psi, noise_mode="const")
            out.append((img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy())
    return np.concatenate(out, axis=0)
