"""Chỉ số của TẬP ảnh sinh được chọn — chẩn đoán vì sao một cách chọn tốt / không tốt, KHÔNG phải kết quả cuối (kết
quả cuối là classifier trên test). numpy thuần; đọc từ `embeddings.npz` của bước `select` (lớp thiểu số).

- **Disease margin** (từng ảnh, tổng hợp theo tập): M_d = S_d⁺ − λ_d·S_d⁻ trong không gian E_d — ảnh GAN gần K ảnh
  BỆNH thật hơn K ảnh lớp đa số thật đến đâu. Báo trung bình, trung vị, độ lệch chuẩn và tỉ lệ ảnh có M_d > 0. Kèm M_v.
- **Feature diversity** (E_v — đúng không gian của S_div): 1 − cosine trung bình giữa mọi cặp ảnh trong tập, và khoảng
  cách trung bình tới ảnh gần nhất trong cùng tập (thứ mà thuật toán tham lam của DASS tối đa hoá).
- **Feature-space coverage / density** (Naeem et al., ICML 2020) so với ảnh BỆNH thật của tập train, trong E_d và E_v:
  coverage = tỉ lệ ảnh thật có ít nhất một ảnh GAN nằm trong bán kính k láng giềng gần nhất của nó (tập chọn phủ phân phối
  ảnh bệnh thật đến đâu); density = số ảnh GAN rơi vào các vùng đó, chia cho k × số ảnh GAN (≈ 1 khi ảnh GAN phân bố
  như ảnh thật, > 1 khi dồn vào vùng dày đặc).

Coverage và density phụ thuộc số ảnh: chỉ so giữa các phương pháp có CÙNG số ảnh (M1–M6); hàng "pool" có nhiều ảnh hơn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CHUNK = 1024


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bình phương khoảng cách Euclid giữa từng dòng của `a` và của `b`."""
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)


def knn_radii(real: np.ndarray, k: int) -> np.ndarray:
    """Bình phương khoảng cách từ mỗi ảnh thật tới láng giềng gần thứ k (không tính chính nó)."""
    real = np.asarray(real, dtype=np.float64)
    k = int(min(k, len(real) - 1))
    out = np.empty(len(real))
    for s in range(0, len(real), _CHUNK):
        d = _sq_dist(real[s:s + _CHUNK], real)
        d[np.arange(len(d)), np.arange(s, s + len(d))] = np.inf          # bỏ chính nó
        out[s:s + len(d)] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return out


def coverage_density(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict[str, float]:
    """Coverage và density của `fake` so với `real` (Naeem et al., ICML 2020), khoảng cách Euclid trên đặc trưng."""
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if len(real) < 2 or len(fake) == 0:
        return {"coverage": float("nan"), "density": float("nan")}
    radii = knn_radii(real, k)
    covered = np.zeros(len(real), dtype=bool)
    hits = 0
    for s in range(0, len(fake), _CHUNK):
        inside = _sq_dist(fake[s:s + _CHUNK], real) < radii[None, :]
        covered |= inside.any(axis=0)
        hits += int(inside.sum())
    k_eff = int(min(k, len(real) - 1))
    return {"coverage": float(covered.mean()), "density": float(hits / (k_eff * len(fake)))}


def pairwise_diversity(z: np.ndarray) -> float:
    """1 − cosine trung bình giữa mọi cặp ảnh khác nhau (đặc trưng đã hoặc chưa chuẩn hoá L2)."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) < 2:
        return float("nan")
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    total = 0.0
    for s in range(0, len(z), _CHUNK):
        total += float((z[s:s + _CHUNK] @ z.T).sum())
    n = len(z)
    return 1.0 - (total - n) / (n * (n - 1))


def nn_within(z: np.ndarray) -> float:
    """Khoảng cách cosine (1 − cos) trung bình từ mỗi ảnh tới ảnh gần nhất KHÁC nó trong cùng tập."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) < 2:
        return float("nan")
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    best = np.empty(len(z))
    for s in range(0, len(z), _CHUNK):
        sim = z[s:s + _CHUNK] @ z.T
        sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -np.inf
        best[s:s + len(sim)] = sim.max(axis=1)
    return float(np.mean(1.0 - best))


def _set_row(set_name: str, method: str, idx: np.ndarray, emb: dict, real_v: np.ndarray, real_d: np.ndarray,
             k: int) -> dict:
    md, mv = emb["score__M_d"][idx], emb["score__M_v"][idx]
    zv, zd = emb["z_v_pool"][idx], emb["z_d_pool"][idx]
    cov_d, cov_v = coverage_density(real_d, zd, k), coverage_density(real_v, zv, k)
    return {"set": set_name, "method": method, "n": int(len(idx)),
            "md_mean": float(md.mean()), "md_median": float(np.median(md)), "md_std": float(md.std(ddof=1)),
            "md_pos_share": float((md > 0).mean()), "mv_mean": float(mv.mean()), "mv_median": float(np.median(mv)),
            "diversity_ev": pairwise_diversity(zv), "nn_within_ev": nn_within(zv),
            "coverage_d": cov_d["coverage"], "density_d": cov_d["density"],
            "coverage_v": cov_v["coverage"], "density_v": cov_v["density"]}


def selection_quality(emb: dict, selections: dict[str, list[str]], minority: str, k: int = 5) -> pd.DataFrame:
    """Một dòng cho ảnh bệnh thật (mốc đa dạng), một dòng cho toàn pool đã dùng để chọn, một dòng cho mỗi phương pháp.

    `emb`: nội dung `embeddings.npz` (pool_names, z_v_pool, z_d_pool, z_v_real__<lớp>, z_d_real__<lớp>, score__M_v,
    score__M_d). `selections`: {phương pháp: [tên ảnh trong pool]} (selections.json, lớp thiểu số)."""
    names = [str(n) for n in emb["pool_names"]]
    index = {n: i for i, n in enumerate(names)}
    missing = [n for files in selections.values() for n in files if n not in index]
    if missing:
        raise ValueError(f"{len(missing)} ảnh trong selections.json không có trong embeddings.npz (pool đã đổi?)")
    real_v, real_d = emb[f"z_v_real__{minority}"], emb[f"z_d_real__{minority}"]
    rows = [{"set": "reference", "method": "real minority train", "n": int(len(real_v)),
             "diversity_ev": pairwise_diversity(real_v), "nn_within_ev": nn_within(real_v)},
            _set_row("pool", "all candidates", np.arange(len(names)), emb, real_v, real_d, k)]
    for method, files in selections.items():
        if files:
            idx = np.array([index[n] for n in files])
            rows.append(_set_row("selected", method, idx, emb, real_v, real_d, k))
    return pd.DataFrame(rows)
