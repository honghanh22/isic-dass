"""Kiểm định thống kê: paired bootstrap ΔAUC (phân tầng theo lớp) so với baseline, hiệu chỉnh Holm cho nhiều phép so
sánh."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SCORE_FUNCS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}   # pr_auc = AP (sklearn), không phải hình thang


def paired_bootstrap_auc(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int,
                         seed: int, metric: str = "roc_auc") -> tuple[float, float, float, float]:
    """Δ = metric(a) − metric(b), metric ∈ {roc_auc, pr_auc (AP)}; bootstrap phân tầng theo lớp (mỗi mẫu đủ hai lớp).
    Trả về (Δ quan sát, CI95 thấp, CI95 cao, p hai phía)."""
    score = SCORE_FUNCS[metric]
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        diffs[i] = score(y[idx], p_a[idx]) - score(y[idx], p_b[idx])
    obs = score(y, p_a) - score(y, p_b)
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(lo), float(hi), float(p_value)


def holm_adjust(p_values) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni (kiểm soát xác suất có ít nhất một kết luận sai trong cả họ kiểm định); giữ thứ tự
    đầu vào. NaN giữ nguyên, không tính vào họ."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m, running = len(valid), 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out
