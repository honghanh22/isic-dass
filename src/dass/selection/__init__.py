"""SELECTION STRATEGY — DASS (numpy thuần, không phụ thuộc framework).

- `scoring`:    lề M = S⁺ − λ·S⁻ (trung bình cosine TOP-K) trong không gian E_v và E_d
- `strategies`: các phương pháp so sánh B0–B3, ablation M1–M6, thuật toán tham lam S_DASS = α·M̃_v + β·M̃_d + γ·S̃_div
- `rc`:         DASS-RC — NC, gate, TMU, độ phủ ảnh thật (RDC), lazy greedy và các biến thể ablation A0–A6
"""

from .scoring import compute_pool_scores, margin_score, topk_similarity
from .strategies import (
    AUGMENT,
    BASELINES,
    CLASS_WEIGHT,
    DASS,
    IMBALANCED,
    METHODS,
    OVERSAMPLE,
    RANDOM,
    SELECTION_METHODS,
    greedy_dass_select,
    jaccard_matrix,
    select_all_methods,
)

__all__ = ["AUGMENT", "BASELINES", "CLASS_WEIGHT", "DASS", "IMBALANCED", "METHODS", "OVERSAMPLE", "RANDOM",
           "SELECTION_METHODS", "compute_pool_scores", "greedy_dass_select", "jaccard_matrix", "margin_score",
           "select_all_methods", "topk_similarity"]
