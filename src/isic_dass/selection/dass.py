"""Bước 6–7 — thuật toán chọn tham lam DASS và bảy biến thể đối chứng M0–M6.

    S_div(x, S) = min_{y ∈ S} [1 − cos(z_x^v, z_y^v)]
    S_DASS(x)   = α·M̃_v(x) + β·M̃_d(x) + γ·S̃_div(x)

Các thành phần được chuẩn hoá min–max về [0, 1] trước khi cộng; S_div được chuẩn hoá lại ở MỖI vòng lặp
vì giá trị của nó thay đổi khi tập đã chọn lớn dần.

| ID | Điểm nền                 | Đa dạng          |
|----|--------------------------|------------------|
| M0 | — (chỉ ảnh thật + class weight) | —         |
| M1 | ngẫu nhiên               | —                |
| M2 | M̃_v                      | —                |
| M3 | M̃_d                      | —                |
| M4 | α·M̃_v + β·M̃_d            | —                |
| M5 | 0                        | có (γ = 1)       |
| M6 | α·M̃_v + β·M̃_d            | có (γ = gamma)   |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import minmax

BASELINE = "M0_real_only"
METHODS = ("M0_real_only", "M1_random", "M2_visual", "M3_disease", "M4_visual_disease", "M5_diversity", "M6_dass")


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
    """M5: chỉ dùng S_div (k-center greedy), điểm nền bằng 0."""
    return greedy_dass_select(z_v_pool, np.zeros(len(z_v_pool)), n_select, gamma=1.0)


def top_n(score: np.ndarray, n: int) -> list[int]:
    return np.argsort(-np.asarray(score), kind="stable")[:n].tolist()


def select_all_methods(scores: dict[str, np.ndarray], z_v_pool: np.ndarray, n_select: int, alpha: float,
                       beta: float, gamma: float, seed: int) -> dict[str, list[int]]:
    """Mọi biến thể chọn đúng `n_select` ảnh từ cùng một pool -> khác biệt chỉ đến từ tiêu chí chọn."""
    base_vd = alpha * minmax(scores["M_v"]) + beta * minmax(scores["M_d"])
    rng = np.random.default_rng(seed)
    selections = {
        "M0_real_only": [],
        "M1_random": rng.choice(len(z_v_pool), size=n_select, replace=False).tolist(),
        "M2_visual": top_n(scores["M_v"], n_select),
        "M3_disease": top_n(scores["M_d"], n_select),
        "M4_visual_disease": top_n(base_vd, n_select),
        "M5_diversity": diversity_only_select(z_v_pool, n_select),
        "M6_dass": greedy_dass_select(z_v_pool, base_vd, n_select, gamma),
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
