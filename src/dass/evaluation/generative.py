"""Metric sinh ảnh trên đặc trưng Inception-v3 (detector của StyleGAN2-ADA) — chuẩn so sánh với các bài báo.

- KID (chỉ số CHÍNH): MMD² không chệch với kernel đa thức bậc 3, trung bình ± độ lệch chuẩn qua các tập con.
  Không chệch theo số mẫu -> phù hợp tập thật nhỏ (vài trăm ảnh).
- FID (THAM KHẢO): chệch mạnh khi số mẫu < 2048 chiều đặc trưng -> chỉ báo cáo kèm cảnh báo.
Các hàm ở đây là numpy thuần (test được không cần GPU); trích đặc trưng nằm ở `models.generator.inception`.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity

from ..data.image_io import read_image
from ..utils import cosine_similarity_matrix


def kid_subset_values(real: np.ndarray, fake: np.ndarray, num_subsets: int = 50, max_subset_size: int = 1000,
                      seed: int = 0) -> np.ndarray:
    """Giá trị KID trên từng tập con — giống metrics/kernel_inception_distance.py của StyleGAN2-ADA (có seed)."""
    rng = np.random.default_rng(seed)
    n = real.shape[1]
    m = min(real.shape[0], fake.shape[0], max_subset_size)
    values = np.empty(num_subsets)
    for i in range(num_subsets):
        x = fake[rng.choice(fake.shape[0], m, replace=False)]
        y = real[rng.choice(real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        values[i] = ((a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m) / m
    return values


def kid_from_features(real: np.ndarray, fake: np.ndarray, num_subsets: int = 50, max_subset_size: int = 1000,
                      seed: int = 0) -> float:
    return float(kid_subset_values(real, fake, num_subsets, max_subset_size, seed).mean())


def kid_with_std(real: np.ndarray, fake: np.ndarray, num_subsets: int = 50, max_subset_size: int = 1000,
                 seed: int = 0) -> tuple[float, float]:
    v = kid_subset_values(real, fake, num_subsets, max_subset_size, seed)
    return float(v.mean()), float(v.std())


def compute_fid(real: np.ndarray, fake: np.ndarray) -> float:
    mu1, s1 = real.mean(0), np.cov(real, rowvar=False)
    mu2, s2 = fake.mean(0), np.cov(fake, rowvar=False)
    covmean = np.real(sqrtm(s1 @ s2))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def compute_diversity(features: np.ndarray) -> float:
    """1 − cosine trung bình giữa các cặp ảnh khác nhau (cao = đa dạng)."""
    sim = cosine_similarity_matrix(features, features)
    n = sim.shape[0]
    return 1.0 - float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def compute_ssim(image_paths: list[str], channels: int, n_pairs: int = 200, seed: int = 0) -> float:
    """SSIM trung bình giữa các cặp ảnh ngẫu nhiên (thấp = đa dạng), tính trên đúng số kênh gốc.

    Cửa sổ Gauss σ = 1.5 như định nghĩa của Wang et al.
    """
    rng = np.random.default_rng(seed)

    def load(p: str) -> np.ndarray:
        img = read_image(p, channels).astype(np.float64) / 255.0
        return img[..., 0] if channels == 1 else img

    scores = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(image_paths), size=2, replace=False)
        scores.append(structural_similarity(load(image_paths[i]), load(image_paths[j]), data_range=1.0,
                                            channel_axis=None if channels == 1 else 2, gaussian_weights=True,
                                            sigma=1.5, use_sample_covariance=False))
    return float(np.mean(scores))
