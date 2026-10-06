"""Metric Keras dùng trong lúc train (chọn epoch)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class MacroRecall(tf.keras.metrics.Metric):
    """Recall trung bình macro nhị phân = (sensitivity + specificity) / 2 ở ngưỡng 0.5 — tiêu chí chọn epoch (CosSIF)."""

    def __init__(self, threshold: float = 0.5, name: str = "macro_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        # Keras Recall coi p > ngưỡng là dương; hạ ngưỡng một bước float32 -> p >= 0,5 là dương, khớp
        # evaluation.classification.binary_metrics (lớp âm: 1 − p > 0,5 <=> p < 0,5)
        self.pos = tf.keras.metrics.Recall(thresholds=float(np.nextafter(np.float32(threshold), np.float32(0))),
                                           name="rec_pos")
        self.neg = tf.keras.metrics.Recall(thresholds=threshold, name="rec_neg")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        self.pos.update_state(y_true, y_pred, sample_weight)
        self.neg.update_state(1.0 - y_true, 1.0 - y_pred, sample_weight)   # recall của lớp âm = specificity

    def result(self):
        return (self.pos.result() + self.neg.result()) / 2.0

    def reset_state(self):
        self.pos.reset_state()
        self.neg.reset_state()


def training_metrics(with_pr_auc: bool = True) -> list:
    metrics = [tf.keras.metrics.AUC(name="auc")]
    if with_pr_auc:
        metrics.append(tf.keras.metrics.AUC(name="pr_auc", curve="PR"))
    metrics.append(MacroRecall(name="macro_recall"))
    return metrics
