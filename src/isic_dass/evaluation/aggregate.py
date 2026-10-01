"""Bước 10b — gộp mọi file dự đoán trên Drive, bảng mean ± std và paired bootstrap so với baseline.

Đọc được cả file .npz cũ (không có trường `lam`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import binary_metrics
from .stats import paired_bootstrap_auc

KEY_METRICS = ["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "g_mean",
               "balanced_accuracy", "f1", "macro_f1", "mcc"]
GROUP_COLS = ["model", "method", "lam", "k"]

RunKey = tuple[str, str, str, float, int]   # (model, method, lam, k, seed)


def load_all_runs(pred_dir: str | Path, threshold: float = 0.5) -> tuple[pd.DataFrame, dict[RunKey, tuple]]:
    rows, probs = [], {}
    for path in sorted(Path(pred_dir).glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        model, method, k, seed = str(d["model"]), str(d["method"]), float(d["k"]), int(d["seed"])
        lam = str(d["lam"]) if "lam" in d.files else "-"
        y_val, p_val, y_test, p_test = d["y_val"], d["p_val"], d["y_test"], d["p_test"]
        row = {"model": model, "method": method, "lam": lam, "k": k, "seed": seed,
               "best_epoch": int(d["best_epoch"]),
               "val_roc_auc": roc_auc_score(y_val, p_val),
               "roc_auc": roc_auc_score(y_test, p_test),
               "pr_auc": average_precision_score(y_test, p_test)}
        row.update(binary_metrics(y_test, p_test, threshold))
        rows.append(row)
        probs[(model, method, lam, k, seed)] = (y_test, p_test)
    return pd.DataFrame(rows), probs


def mean_std_table(df: pd.DataFrame, cols: list[str] = KEY_METRICS) -> pd.DataFrame:
    g = df.groupby(GROUP_COLS)[cols]
    mean, std = g.mean(), g.std().fillna(0)
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])]
    out["n_seeds"] = g.size()
    return out.reset_index()


def _runs_of(probs: dict[RunKey, tuple], model: str, k: float, method: str,
             lam: str | None = None) -> dict[int, tuple]:
    """{seed: (y_test, p_test)} của một (model, k, phương pháp[, lambda])."""
    return {s: v for (m, me, lm, kk, s), v in probs.items()
            if m == model and me == method and kk == k and (lam is None or lm == lam)}


def compare_to_baseline(probs: dict[RunKey, tuple], baseline: str, n_boot: int, seed: int) -> pd.DataFrame:
    """So sánh AUC của từng (phương pháp, lambda) với baseline, dùng xác suất trung bình qua các seed chung."""
    rows = []
    for model, k in sorted({(m, kk) for m, _, _, kk, _ in probs}):
        base = _runs_of(probs, model, k, baseline)
        if not base:
            continue
        combos = sorted({(me, lm) for m, me, lm, kk, _ in probs if m == model and kk == k and me != baseline})
        for method, lam in combos:
            cur = _runs_of(probs, model, k, method, lam)
            common = sorted(set(base) & set(cur))
            if not common:
                continue
            y = base[common[0]][0]
            if not all(np.array_equal(y, cur[s][0]) for s in common):
                raise ValueError(f"Thứ tự nhãn test không khớp: {model}/{method}")
            p_b = np.mean([base[s][1] for s in common], axis=0)
            p_a = np.mean([cur[s][1] for s in common], axis=0)
            obs, lo, hi, pv = paired_bootstrap_auc(y, p_a, p_b, n_boot, seed)
            rows.append({"model": model, "k": k, "method": method, "lam": lam, "vs": baseline,
                         "n_seeds": len(common), "delta_auc": obs, "ci95_low": lo, "ci95_high": hi, "p_value": pv})
    return pd.DataFrame(rows)
