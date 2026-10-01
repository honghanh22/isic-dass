"""Kiểm định thống kê: paired bootstrap ΔAUC (phân tầng theo lớp) so với baseline."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_bootstrap_auc(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int,
                         seed: int) -> tuple[float, float, float, float]:
    """ΔAUC = AUC(a) − AUC(b). Trả về (Δ quan sát, CI95 thấp, CI95 cao, p hai phía)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        diffs[i] = roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])
    obs = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(lo), float(hi), float(p_value)
