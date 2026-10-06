"""Chiến lược chọn ảnh sinh: thuật toán tham lam DASS và các biến thể đối chứng M0–M6 (+ M7 tuỳ chọn).

    S_div(x, S) = min_{y ∈ S} [1 − cos(z_x^v, z_y^v)]
    S_DASS(x)   = α·M̃_v(x) + β·M̃_d(x) + γ·S̃_div(x)

Các thành phần được chuẩn hoá min–max về [0, 1] trước khi cộng; S_div được chuẩn hoá lại ở MỖI vòng lặp
vì giá trị của nó thay đổi khi tập đã chọn lớn dần.

| ID | Điểm nền                 | Đa dạng          |
|----|--------------------------|------------------|
| M0 | — (chỉ ảnh thật; class weight nếu bật `classifier.baseline_class_weight`, mặc định tắt) | — |
| M0b | — (ảnh thật lớp thiểu số nhân bản lên 1 : 1, không class weight; ghép ở bước train) | — |
| M1 | ngẫu nhiên               | —                |
| M2 | M̃_v                      | —                |
| M3 | M̃_d                      | —                |
| M4 | α·M̃_v + β·M̃_d            | —                |
| M5 | 0                        | có (γ = 1)       |
| M6 | α·M̃_v + β·M̃_d            | có (γ = gamma)   |
| M7 | như M6 + `n_select` ảnh sinh vào lớp ĐA SỐ, có class weight (tuỳ chọn `both_classes_variant`) |

M0b (random oversampling, Buda et al. 2018): cùng số ảnh, cùng số bước train và cùng cách cân bằng (bằng dữ liệu)
với M1–M6, chỉ khác ở chỗ ảnh thêm vào là ảnh THẬT nhắc lại -> M6 vs M0b đo đúng đóng góp của NỘI DUNG ảnh sinh.
M0b không chọn gì từ pool nên không nằm trong `selections.json`; tập train được ghép ở bước train
(`data.variants.oversample_indices`).

M7: nếu ảnh sinh chỉ có ở lớp thiểu số thì "trông giống ảnh GAN" trở thành manh mối
của nhãn (shortcut). Thêm ảnh sinh vào cả lớp đa số làm dấu vết GAN không còn gắn với một nhãn; lớp đa số to ra
nên mất cân bằng quay lại một phần và được bù bằng class weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import minmax

BASELINE = "M0_real_only"
OVERSAMPLE = "M0b_real_oversample"   # không chọn từ pool -> không thuộc METHODS / selections.json
BOTH_CLASSES = "M7_dass_both_classes"
METHODS = ("M0_real_only", "M1_random", "M2_visual", "M3_disease", "M4_visual_disease", "M5_diversity", "M6_dass",
           BOTH_CLASSES)


DIV_NORMALIZATIONS = ("per_round", "fixed")
DIVERSITY_STARTS = ("first", "medoid")


def greedy_dass_select(z_v_pool: np.ndarray, base_score: np.ndarray, n_select: int, gamma: float,
                       div_normalization: str = "per_round", start: str = "first") -> list[int]:
    """Chọn tham lam: score(x) = base(x) + γ·S̃_div(x, S). `z_v_pool` phải đã chuẩn hoá L2.

    `div_normalization`:
      - "per_round" (mặc định, công thức gốc): S̃_div = min-max của S_div trên các ảnh còn lại, tính lại MỖI vòng ->
        ở các vòng cuối, khác biệt rất nhỏ của S_div bị kéo giãn ra [0, 1] và có thể lấn át phần nền.
      - "fixed": min / max lấy MỘT LẦN ở vòng đầu (sau ảnh thứ nhất) rồi giữ cố định (cắt về [0, 1]) -> trọng số thật
        của đa dạng không đổi theo vòng.
    `start`: ảnh đầu tiên khi điểm nền hằng (M5): "first" (mặc định: chỉ số 0 của pool) hoặc "medoid" (ảnh có cosine
    trung bình lớn nhất tới cả pool). Điểm nền không hằng -> luôn bắt đầu từ argmax(nền).
    """
    if div_normalization not in DIV_NORMALIZATIONS:
        raise ValueError(f"div_normalization phải thuộc {DIV_NORMALIZATIONS}, nhận {div_normalization!r}")
    if start not in DIVERSITY_STARTS:
        raise ValueError(f"start phải thuộc {DIVERSITY_STARTS}, nhận {start!r}")
    base = np.asarray(base_score, dtype=np.float64)
    n = len(base)
    if n_select <= 0 or n == 0:
        return []
    if start == "medoid" and np.ptp(base) < 1e-12:
        first = int(np.argmax(z_v_pool @ z_v_pool.mean(axis=0)))
    else:
        first = int(np.argmax(base))
    selected = [first]
    remaining = np.ones(n, dtype=bool)
    remaining[first] = False
    max_sim = z_v_pool @ z_v_pool[first]          # cosine lớn nhất tới tập đã chọn
    d0 = 1.0 - max_sim[remaining]
    lo, span = float(d0.min()), float(np.ptp(d0))  # thang cố định (div_normalization = "fixed")
    while len(selected) < n_select and remaining.any():
        rl = np.flatnonzero(remaining)
        if gamma > 0 and len(rl) > 1:
            d = 1.0 - max_sim[rl]
            norm = minmax(d) if div_normalization == "per_round" or span < 1e-12 else np.clip((d - lo) / span, 0, 1)
            score = base[rl] + gamma * norm
        else:
            score = base[rl]
        best = int(rl[int(np.argmax(score))])
        selected.append(best)
        remaining[best] = False
        max_sim = np.maximum(max_sim, z_v_pool @ z_v_pool[best])
    return selected


def diversity_only_select(z_v_pool: np.ndarray, n_select: int, div_normalization: str = "per_round",
                          start: str = "first") -> list[int]:
    """M5: chỉ dùng S_div (k-center greedy), điểm nền bằng 0."""
    return greedy_dass_select(z_v_pool, np.zeros(len(z_v_pool)), n_select, gamma=1.0,
                              div_normalization=div_normalization, start=start)


def top_n(score: np.ndarray, n: int) -> list[int]:
    return np.argsort(-np.asarray(score), kind="stable")[:n].tolist()


def select_all_methods(scores: dict[str, np.ndarray], z_v_pool: np.ndarray, n_select: int, alpha: float,
                       beta: float, gamma: float, seed: int, both_classes: bool = False,
                       div_normalization: str = "per_round", diversity_start: str = "first") -> dict[str, list[int]]:
    """Mọi biến thể chọn đúng `n_select` ảnh từ cùng một pool -> khác biệt chỉ đến từ tiêu chí chọn.

    `both_classes` -> thêm M7 (cùng ảnh thiểu số với M6; ảnh sinh lớp đa số được thêm khi ghép tập train).
    """
    base_vd = alpha * minmax(scores["M_v"]) + beta * minmax(scores["M_d"])
    rng = np.random.default_rng(seed)
    selections = {
        "M0_real_only": [],
        "M1_random": rng.choice(len(z_v_pool), size=n_select, replace=False).tolist(),
        "M2_visual": top_n(scores["M_v"], n_select),
        "M3_disease": top_n(scores["M_d"], n_select),
        "M4_visual_disease": top_n(base_vd, n_select),
        "M5_diversity": diversity_only_select(z_v_pool, n_select, div_normalization, diversity_start),
        "M6_dass": greedy_dass_select(z_v_pool, base_vd, n_select, gamma, div_normalization),
    }
    if both_classes:
        selections[BOTH_CLASSES] = list(selections["M6_dass"])
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
