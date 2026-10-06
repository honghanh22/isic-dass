"""Bước 5–6 — điểm lề trong một không gian đặc trưng.

    S⁺(x) = (1/K) Σ cos(z_x, z_i)   với K ảnh THẬT gần nhất thuộc lớp thiểu số
    S⁻(x) = (1/K) Σ cos(z_x, z_j)   với K ảnh THẬT gần nhất thuộc lớp đa số
    M(x)  = S⁺(x) − λ · S⁻(x)

Dùng trung bình TOP-K thay vì max vì max chỉ phụ thuộc một ảnh thật nên rất nhạy với ngoại lai.
"""

from __future__ import annotations

import logging

import numpy as np

from ..utils import cosine_similarity_matrix

log = logging.getLogger(__name__)

SCORE_KEYS = ("M_v", "S_v_pos", "S_v_neg", "M_d", "S_d_pos", "S_d_neg")


def topk_similarity(candidate: np.ndarray, reference: np.ndarray, topk: int) -> np.ndarray:
    sim = cosine_similarity_matrix(candidate, reference)
    k = int(min(topk, sim.shape[1]))
    return np.sort(sim, axis=1)[:, -k:].mean(axis=1)


def margin_score(candidate: np.ndarray, pos: np.ndarray, neg: np.ndarray, lam: float,
                 topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_pos = topk_similarity(candidate, pos, topk)
    s_neg = topk_similarity(candidate, neg, topk)
    return s_pos - lam * s_neg, s_pos, s_neg


def compute_pool_scores(z_v_pool: np.ndarray, z_v_real: dict[str, np.ndarray], z_d_pool: np.ndarray,
                        z_d_real: dict[str, np.ndarray], minority: str, majority: str, lambda_v: float,
                        lambda_d: float, topk: int) -> dict[str, np.ndarray]:
    M_v, s_v_pos, s_v_neg = margin_score(z_v_pool, z_v_real[minority], z_v_real[majority], lambda_v, topk)
    M_d, s_d_pos, s_d_neg = margin_score(z_d_pool, z_d_real[minority], z_d_real[majority], lambda_d, topk)
    log.info("M_v: TB %+.4f (std %.4f) | M_d: TB %+.4f (std %.4f)", M_v.mean(), M_v.std(), M_d.mean(), M_d.std())
    log.info("Tương quan M_v–M_d = %+.3f (gần 1 -> hai không gian đo cùng một thứ, M_d không thêm thông tin)",
             np.corrcoef(M_v, M_d)[0, 1])
    return {"M_v": M_v, "S_v_pos": s_v_pos, "S_v_neg": s_v_neg, "M_d": M_d, "S_d_pos": s_d_pos, "S_d_neg": s_d_neg}


def crossfit_folds(n: int, k: int, seed: int) -> np.ndarray:
    """Gán `n` ảnh vào `k` fold cân bằng (hoán vị ngẫu nhiên theo `seed`): fold[i] ∈ {0, …, k−1}."""
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return fold


def crossfit_margin(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], lam: float,
                    topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M_d cross-fitted: `parts` = [(z_pool_k, z_pos_heldout_k, z_neg_heldout_k)] — mỗi phần trong không gian của E_d
    thứ k, tham chiếu là ảnh thật mà E_d đó KHÔNG học. Trả về trung bình (M, S⁺, S⁻) qua k."""
    res = [margin_score(zp, pos, neg, lam, topk) for zp, pos, neg in parts]
    return tuple(np.mean([r[i] for r in res], axis=0) for i in range(3))
