"""Gộp mọi file dự đoán .npz: metric từng lần chạy, thống kê mean / std qua seed, paired bootstrap so với baseline.

Đọc được cả .npz của notebook cũ (không có trường `lam`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .classification import KEY_METRICS, binary_metrics
from .statistics import paired_bootstrap_auc

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
               "best_epoch": int(d["best_epoch"]), "n_test": len(y_test),
               "val_roc_auc": float(roc_auc_score(y_val, p_val))}
        row.update(binary_metrics(y_test, p_test, threshold))
        rows.append(row)
        probs[(model, method, lam, k, seed)] = (y_test, p_test)
    return pd.DataFrame(rows), probs


def summary_stats(runs: pd.DataFrame, metrics: list[str] = KEY_METRICS) -> pd.DataFrame:
    """Một dòng mỗi (model, method, lam, k): <metric>_mean, <metric>_std, n_seeds (số liệu thô)."""
    g = runs.groupby(GROUP_COLS)
    mean, std = g[metrics].mean(), g[metrics].std(ddof=1).fillna(0.0)
    out = pd.concat([mean.add_suffix("_mean"), std.add_suffix("_std")], axis=1)
    out["n_seeds"] = g.size()
    ordered = [c for m in metrics for c in (f"{m}_mean", f"{m}_std")] + ["n_seeds"]
    return out[ordered].reset_index()


def _runs_of(probs: dict[RunKey, tuple], model: str, k: float, method: str,
             lam: str | None = None) -> dict[int, tuple]:
    return {s: v for (m, me, lm, kk, s), v in probs.items()
            if m == model and me == method and kk == k and (lam is None or lm == lam)}


def _compare(model: str, k: float, method: str, lam: str, cur: dict[int, tuple], ref_name: str,
             ref: dict[int, tuple], n_boot: int, seed: int) -> dict | None:
    """Paired bootstrap ΔAUC = AUC(method) − AUC(ref) trên xác suất trung bình qua các seed chung (ensemble)."""
    common = sorted(set(ref) & set(cur))
    if not common:
        return None
    y = ref[common[0]][0]
    if not all(np.array_equal(y, cur[s][0]) and np.array_equal(y, ref[s][0]) for s in common):
        raise ValueError(f"Thứ tự nhãn test không khớp: {model}/{method} vs {ref_name}")
    p_ref = np.mean([ref[s][1] for s in common], axis=0)
    p_cur = np.mean([cur[s][1] for s in common], axis=0)
    obs, lo, hi, pv = paired_bootstrap_auc(y, p_cur, p_ref, n_boot, seed)
    return {"model": model, "k": k, "method": method, "lam": lam, "vs": ref_name, "n_seeds": len(common),
            "delta_auc": obs, "ci95_low": lo, "ci95_high": hi, "p_value": pv}


def compare_to_baseline(probs: dict[RunKey, tuple], baseline: str, n_boot: int, seed: int) -> pd.DataFrame:
    """ΔAUC của từng (phương pháp, lambda) so với baseline, dùng xác suất trung bình qua các seed chung."""
    rows = []
    for model, k in sorted({(m, kk) for m, _, _, kk, _ in probs}):
        base = _runs_of(probs, model, k, baseline)
        if not base:
            continue
        combos = sorted({(me, lm) for m, me, lm, kk, _ in probs if m == model and kk == k and me != baseline})
        for method, lam in combos:
            row = _compare(model, k, method, lam, _runs_of(probs, model, k, method, lam), baseline, base, n_boot,
                           seed)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def compare_pairs(probs: dict[RunKey, tuple], pairs: list[list[str]], n_boot: int, seed: int) -> pd.DataFrame:
    """ΔAUC cho từng cặp [phương pháp, đối chứng] (ví dụ M6 vs M0b, M6 vs M1), trong mỗi (model, k).

    Cặp nào thiếu dự đoán của một trong hai phía (chưa train) thì bỏ qua.
    """
    rows = []
    for model, k in sorted({(m, kk) for m, _, _, kk, _ in probs}):
        for method, ref_name in pairs:
            ref = _runs_of(probs, model, k, ref_name)
            for lam in sorted({lm for m, me, lm, kk, _ in probs if m == model and kk == k and me == method}):
                row = _compare(model, k, method, lam, _runs_of(probs, model, k, method, lam), ref_name, ref,
                               n_boot, seed)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)
