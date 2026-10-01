"""Chất lượng tập ảnh sinh được chọn: KID / FID trên đặc trưng E_v, độ đa dạng, SSIM nội bộ."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity

from ..utils import cosine_similarity_matrix


def compute_fid(real_emb: np.ndarray, fake_emb: np.ndarray) -> float:
    """FID trên ít mẫu -> ma trận hiệp phương sai gần suy biến; chỉ để tham khảo, KID là tiêu chí chính."""
    mu1, s1 = real_emb.mean(0), np.cov(real_emb, rowvar=False)
    mu2, s2 = fake_emb.mean(0), np.cov(fake_emb, rowvar=False)
    covmean = np.real(sqrtm(s1 @ s2))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def compute_diversity(emb: np.ndarray) -> float:
    """1 − cosine trung bình giữa các cặp khác nhau."""
    sim = cosine_similarity_matrix(emb, emb)
    n = sim.shape[0]
    return 1.0 - float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def compute_ssim(image_paths: list[str], n_pairs: int = 200, seed: int = 0) -> float:
    """SSIM trung bình giữa các cặp ảnh ngẫu nhiên (cửa sổ Gauss σ = 1.5, như định nghĩa của Wang et al.)."""
    rng = np.random.default_rng(seed)

    def load(p: str) -> np.ndarray:
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0

    scores = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(image_paths), size=2, replace=False)
        scores.append(structural_similarity(load(image_paths[i]), load(image_paths[j]), channel_axis=2,
                                            data_range=1.0, gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False))
    return float(np.mean(scores))
