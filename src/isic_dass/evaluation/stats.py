"""Kiểm định: AUC của hồi quy logistic (5-fold) và paired bootstrap ΔAUC."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_logistic_auc(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> float:
    """AUC out-of-fold của hồi quy logistic chuẩn hoá (C = 0.1, class_weight='balanced').

    Dùng cho: AUC probe của không gian đặc trưng, kiểm tra shortcut thật-vs-sinh, detector profile công suất.
    """
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def separability_auc(emb_a: np.ndarray, emb_b: np.ndarray, seed: int) -> float:
    """AUC tách hai tập nhúng (0.5 = không tách được, 1.0 = tách hoàn toàn)."""
    X = np.concatenate([emb_a, emb_b])
    y = np.r_[np.zeros(len(emb_a)), np.ones(len(emb_b))]
    return cv_logistic_auc(X, y, seed)


def paired_bootstrap_auc(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int,
                         seed: int) -> tuple[float, float, float, float]:
    """ΔAUC = AUC(a) − AUC(b), bootstrap phân tầng theo lớp. Trả về (Δ quan sát, CI95 thấp, CI95 cao, p hai phía)."""
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
