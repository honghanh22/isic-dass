"""KID bằng Inception của chính StyleGAN2-ADA, nạp generator và sinh ảnh theo lớp.

`kid_from_features` là numpy thuần (test được không cần GPU); các hàm còn lại import torch khi gọi.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

INCEPTION_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"


def kid_from_features(real: np.ndarray, fake: np.ndarray, num_subsets: int = 50, max_subset_size: int = 1000,
                      seed: int = 0) -> float:
    """Giống metrics/kernel_inception_distance.py của StyleGAN2-ADA, thêm seed để lặp lại được."""
    rng = np.random.default_rng(seed)
    n = real.shape[1]
    m = min(real.shape[0], fake.shape[0], max_subset_size)
    t = 0.0
    for _ in range(num_subsets):
        x = fake[rng.choice(fake.shape[0], m, replace=False)]
        y = real[rng.choice(real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(t / num_subsets / m)


def load_images_uint8(paths: list[str]) -> np.ndarray:
    return np.stack([np.array(Image.open(p).convert("RGB")) for p in paths]).astype(np.uint8)


def _device():
    import torch

    return torch.device("cuda")


@lru_cache(maxsize=1)
def get_inception():
    from metrics import metric_utils  # module của repo StyleGAN2-ADA (đã thêm vào sys.path)

    return metric_utils.get_feature_detector(INCEPTION_URL, device=_device())


def inception_features(images_uint8_nhwc: np.ndarray, batch: int = 64) -> np.ndarray:
    import torch

    det, device = get_inception(), _device()
    feats = []
    with torch.no_grad():
        for s in range(0, len(images_uint8_nhwc), batch):
            x = torch.from_numpy(images_uint8_nhwc[s:s + batch]).permute(0, 3, 1, 2).to(device)
            feats.append(det(x, return_features=True).cpu().numpy().astype(np.float64))
    return np.concatenate(feats, axis=0)


def load_generator(pkl_path: str | Path):
    import legacy  # module của repo StyleGAN2-ADA

    with open(pkl_path, "rb") as f:
        return legacy.load_network_pkl(f)["G_ema"].to(_device()).eval().requires_grad_(False)


def generate_class_images(G, n: int, class_idx: int, psi: float = 1.0, seed: int = 0, batch: int = 32) -> np.ndarray:
    """Sinh `n` ảnh uint8 NHWC có điều kiện theo `class_idx`."""
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
