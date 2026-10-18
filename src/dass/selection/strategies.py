"""Phương pháp so sánh và chiến lược chọn ảnh sinh (thiết kế 2.0.0).

Nhóm 1 — so sánh các phương pháp xử lý mất cân bằng (B4 = M1, B5 = M6 không train riêng):

| Mã | Phương pháp | Lớp thiểu số | Lớp đa số | Khi train |
|----|-------------|--------------|-----------|-----------|
| B0 | Imbalanced | ảnh thật | ảnh thật | — |
| B1 | Cost-sensitive | ảnh thật | ảnh thật | class-weighted loss |
| B2 | Random oversampling | ảnh thật + bản sao tới 1 : 1 | ảnh thật | — |
| B3 | Traditional augmentation | ảnh thật | ảnh thật | augmentation (lật / xoay / sáng / tương phản) |
| B4 | Balanced GAN: Random (= M1) | ảnh thật + g ảnh sinh | ảnh thật | — |
| B5 | Balanced GAN: DASS (= M6) | ảnh thật + g ảnh sinh | ảnh thật | — |

g = số ảnh thật lớp đa số − số ảnh thật lớp thiểu số (cân bằng 1 : 1). Ảnh sinh CHỈ ở lớp thiểu số.

Nhóm 2 — ablation DASS: mọi biến thể chọn đúng g ảnh từ CÙNG một pool ⌈k·g⌉ ảnh sinh lớp thiểu số:

| Mã | Điểm nền | Đa dạng |
|----|----------|---------|
| M1 | ngẫu nhiên | — |
| M2 | M̃_v | — |
| M3 | M̃_d | — |
| M4 | 0 | có (k-center greedy, γ = 1) |
| M5 | α·M̃_v + β·M̃_d | — |
| M6 | α·M̃_v + β·M̃_d | có (γ) — DASS |

    S_div(x, S) = min_{y ∈ S} [1 − cos(z_x^v, z_y^v)]
    S_DASS(x)   = α·M̃_v(x) + β·M̃_d(x) + γ·S̃_div(x)

Các thành phần chuẩn hoá min–max về [0, 1]; S_div chuẩn hoá lại ở MỖI vòng (giá trị đổi khi tập đã chọn lớn dần).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import minmax

IMBALANCED = "B0_imbalanced"
CLASS_WEIGHT = "B1_class_weight"
OVERSAMPLE = "B2_ros"
AUGMENT = "B3_augment"
BASELINES = (IMBALANCED, CLASS_WEIGHT, OVERSAMPLE, AUGMENT)     # không chọn gì từ pool
RANDOM, DASS = "M1_random", "M6_dass"                            # = B4, B5
SELECTION_METHODS = (RANDOM, "M2_visual", "M3_disease", "M4_diversity", "M5_visual_disease", DASS)
METHODS = (*BASELINES, *SELECTION_METHODS)                       # thứ tự train / bảng


def greedy_dass_select(z_v_pool: np.ndarray, base_score: np.ndarray, n_select: int, gamma: float) -> list[int]:
    """Chọn tham lam: score(x) = base(x) + γ·minmax(S_div(x, S)). `z_v_pool` phải đã chuẩn hoá L2."""
    base = np.asarray(base_score, dtype=np.float64)
    n = len(base)
    if n_select <= 0 or n == 0:
        return []
    first = int(np.argmax(base))
    selected = [first]
    remaining = np.ones(n, dtype=bool)
    remaining[first] = False
    max_sim = z_v_pool @ z_v_pool[first]          # cosine lớn nhất tới tập đã chọn
    while len(selected) < n_select and remaining.any():
        rl = np.flatnonzero(remaining)
        if gamma > 0 and len(rl) > 1:
            score = base[rl] + gamma * minmax(1.0 - max_sim[rl])
        else:
            score = base[rl]
        best = int(rl[int(np.argmax(score))])
        selected.append(best)
        remaining[best] = False
        max_sim = np.maximum(max_sim, z_v_pool @ z_v_pool[best])
    return selected


def diversity_only_select(z_v_pool: np.ndarray, n_select: int) -> list[int]:
    """M4: chỉ dùng S_div (k-center greedy, phủ đều không gian E_v), điểm nền bằng 0."""
    return greedy_dass_select(z_v_pool, np.zeros(len(z_v_pool)), n_select, gamma=1.0)


def top_n(score: np.ndarray, n: int) -> list[int]:
    return np.argsort(-np.asarray(score), kind="stable")[:n].tolist()


def select_all_methods(scores: dict[str, np.ndarray], z_v_pool: np.ndarray, n_select: int, alpha: float,
                       beta: float, gamma: float, seed: int) -> dict[str, list[int]]:
    """M1–M6: mỗi biến thể chọn đúng `n_select` ảnh từ cùng một pool -> khác biệt chỉ đến từ tiêu chí chọn."""
    base_vd = alpha * minmax(scores["M_v"]) + beta * minmax(scores["M_d"])
    selections = {
        RANDOM: np.random.default_rng(seed).choice(len(z_v_pool), size=n_select, replace=False).tolist(),
        "M2_visual": top_n(scores["M_v"], n_select),
        "M3_disease": top_n(scores["M_d"], n_select),
        "M4_diversity": diversity_only_select(z_v_pool, n_select),
        "M5_visual_disease": top_n(base_vd, n_select),
        DASS: greedy_dass_select(z_v_pool, base_vd, n_select, gamma),
    }
    return {m: [int(i) for i in idx] for m, idx in selections.items()}


def jaccard_matrix(selections: dict[str, list[int]]) -> pd.DataFrame:
    """Mức trùng lặp giữa các biến thể — kiểm tra các tiêu chí có thật sự khác nhau không."""
    methods = [m for m, idx in selections.items() if idx]
    J = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            sa, sb = set(selections[a]), set(selections[b])
            J.loc[a, b] = len(sa & sb) / len(sa | sb)
    return J
