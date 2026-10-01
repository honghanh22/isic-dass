"""Chẩn đoán trong không gian đặc trưng: AUC probe (hai lớp bệnh tách nhau đến đâu) và kiểm tra shortcut thật-vs-sinh."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_logistic_auc(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> float:
    """AUC out-of-fold của hồi quy logistic chuẩn hoá (C = 0.1, class_weight = 'balanced')."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def separability_auc(emb_a: np.ndarray, emb_b: np.ndarray, seed: int) -> float:
    """AUC tách hai tập nhúng (0.5 = không tách được, 1.0 = tách hoàn toàn)."""
    X = np.concatenate([emb_a, emb_b])
    y = np.r_[np.zeros(len(emb_a)), np.ones(len(emb_b))]
    return cv_logistic_auc(X, y, seed)


def probe_auc(emb_by_class: dict[str, np.ndarray], classes: dict[str, int], seed: int) -> float:
    X = np.concatenate([emb_by_class[c] for c in classes])
    y = np.concatenate([np.full(len(emb_by_class[c]), i) for c, i in classes.items()]).astype(int)
    return cv_logistic_auc(X, y, seed)


def shortcut_table(z_real: dict[str, np.ndarray], z_pool: np.ndarray, selections: dict[str, list[int]],
                   minority: str, majority: str, seed: int) -> pd.DataFrame:
    """AUC thật-vs-sinh của tập ảnh được chọn bởi từng phương pháp (E_v). Gần 1.0 -> classifier có thể học
    'thật hay sinh' thay vì 'bệnh hay không'. Dòng tham chiếu: hai lớp bệnh thật tách nhau đến đâu."""
    rows = [{"so_sanh": f"THAM CHIẾU: {majority} thật vs {minority} thật (E_v)", "method": "reference",
             "auc_5fold": separability_auc(z_real[majority], z_real[minority], seed)}]
    for m, idx in selections.items():
        if idx:
            rows.append({"so_sanh": f"{m}: thật vs sinh (E_v)", "method": m,
                         "auc_5fold": separability_auc(z_real[minority], z_pool[idx], seed)})
    return pd.DataFrame(rows)
