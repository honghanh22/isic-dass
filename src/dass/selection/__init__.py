"""SELECTION STRATEGY — DASS (numpy thuần, không phụ thuộc framework).

- `scoring`:    lề M = S⁺ − λ·S⁻ (trung bình cosine TOP-K) trong không gian E_v và E_d
- `strategies`: M0–M6 (+ M0b oversampling, M7 tuỳ chọn), thuật toán tham lam S_DASS = α·M̃_v + β·M̃_d + γ·S̃_div
"""

from .scoring import compute_pool_scores, margin_score, topk_similarity
from .strategies import (
    BASELINE,
    BOTH_CLASSES,
    METHODS,
    OVERSAMPLE,
    greedy_dass_select,
    jaccard_matrix,
    select_all_methods,
)

__all__ = ["BASELINE", "BOTH_CLASSES", "METHODS", "OVERSAMPLE", "compute_pool_scores", "greedy_dass_select",
           "jaccard_matrix", "margin_score", "select_all_methods", "topk_similarity"]
