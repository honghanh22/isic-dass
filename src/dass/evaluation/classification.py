"""Metric phân loại nhị phân ở ngưỡng CỐ ĐỊNH (mặc định 0.5 — không dò trên val / test)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# thứ tự cột trong bảng báo cáo
KEY_METRICS = ["roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "balanced_accuracy", "g_mean", "mcc",
               "macro_f1", "precision", "accuracy"]
METRIC_LABELS = {"roc_auc": "AUC", "pr_auc": "PR-AUC", "f1": "F1", "sensitivity": "Sens.", "specificity": "Spec.",
                 "balanced_accuracy": "BA", "g_mean": "G-mean", "mcc": "MCC", "macro_f1": "Macro-F1",
                 "precision": "Prec.", "accuracy": "Acc."}


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    """Metric ngưỡng-độc-lập (ROC-AUC, PR-AUC) + metric tại ngưỡng `thr`. Lớp dương = chỉ số 1."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    both_classes = len(np.unique(y_true)) == 2
    return {"roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
            "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "g_mean": float(np.sqrt(sensitivity * specificity)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred))}
