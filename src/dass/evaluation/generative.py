"""Metric sinh ảnh trên đặc trưng Inception-v3 (detector của StyleGAN2-ADA) — chuẩn so sánh với các bài báo.

- KID (chỉ số CHÍNH): MMD² không chệch với kernel đa thức bậc 3, trung bình ± độ lệch chuẩn qua các tập con.
  Không chệch theo số mẫu -> phù hợp tập thật nhỏ (vài trăm ảnh).
- FID (THAM KHẢO): chệch mạnh khi số mẫu < 2048 chiều đặc trưng -> chỉ báo cáo kèm cảnh báo.
- Học thuộc (`memorization_stats`): khoảng cách tới ảnh train gần nhất của ảnh sinh, so với cùng khoảng cách của ảnh
  THẬT chưa thấy (val) -> KID thấp vì GAN chép lại ảnh train sẽ lộ ra ở đây.
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


def nn_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Khoảng cách cosine (1 − cos) từ mỗi dòng của `query` tới dòng gần nhất của `reference`."""
    return 1.0 - cosine_similarity_matrix(query, reference).max(axis=1)


def memorization_stats(fake: np.ndarray, train: np.ndarray, holdout: np.ndarray,
                       quantile: float = 0.05) -> dict[str, float]:
    """Ảnh sinh có gần ảnh train hơn mức ảnh thật chưa thấy (`holdout`, ví dụ val) gần ảnh train không.

    - `nn_dist_median`: trung vị khoảng cách tới ảnh train gần nhất của ảnh sinh.
    - `nn_dist_ratio`  = trung vị của ảnh sinh / trung vị của `holdout`: ≈ 1 như ảnh thật mới; < 1 rõ rệt -> ảnh sinh
      bám sát ảnh train (dấu hiệu học thuộc); > 1 -> ảnh sinh xa phân phối train.
    - `near_copy_rate`: tỉ lệ ảnh sinh gần ảnh train hơn phân vị `quantile` của `holdout`; ảnh thật mới cho ≈ `quantile`,
      cao hơn nhiều -> có nhiều ảnh gần như bản sao.
    """
    d_fake, d_ref = nn_distance(fake, train), nn_distance(holdout, train)
    ref_median = float(np.median(d_ref))
    return {"nn_dist_median": float(np.median(d_fake)),
            "nn_dist_ratio": float(np.median(d_fake) / ref_median) if ref_median > 0 else float("nan"),
            "near_copy_rate": float(np.mean(d_fake < np.quantile(d_ref, quantile)))}
