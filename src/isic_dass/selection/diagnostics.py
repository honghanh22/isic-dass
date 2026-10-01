"""Bước 7b — chẩn đoán lựa chọn: AUC probe, kiểm tra shortcut thật-vs-sinh, hình ảnh được chọn / bị loại."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..evaluation.stats import cv_logistic_auc, separability_auc
from ..utils import minmax, save_figure


def probe_auc(emb_by_class: dict[str, np.ndarray], class_to_idx: dict[str, int], seed: int) -> float:
    """Không gian đặc trưng tách hai lớp tốt đến đâu (AUC hồi quy logistic 5-fold)."""
    X = np.concatenate([emb_by_class[c] for c in class_to_idx])
    y = np.concatenate([np.full(len(emb_by_class[c]), i) for c, i in class_to_idx.items()]).astype(int)
    return cv_logistic_auc(X, y, seed)


def shortcut_table(z_v_real: dict[str, np.ndarray], z_v_pool: np.ndarray, selections: dict[str, list[int]],
                   minority: str, majority: str, seed: int) -> pd.DataFrame:
    """AUC tách ảnh thật / ảnh sinh trong không gian E_v. Gần 1.0 -> classifier có thể học 'thật hay sinh'."""
    rows = [{"so_sanh": f"THAM CHIẾU: {majority} thật vs {minority} thật (E_v)",
             "method": "reference", "auc_5fold": separability_auc(z_v_real[majority], z_v_real[minority], seed)}]
    for m, idx in selections.items():
        if idx:
            rows.append({"so_sanh": f"{m}: thật vs sinh (E_v)", "method": m,
                         "auc_5fold": separability_auc(z_v_real[minority], z_v_pool[idx], seed)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ hình
def plot_score_scatter(scores: dict[str, np.ndarray], selections: dict[str, list[int]], path: Path) -> None:
    import matplotlib.pyplot as plt

    M_v, M_d = scores["M_v"], scores["M_d"]
    methods = [m for m, idx in selections.items() if idx]
    cols = 3
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.3 * rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, m in zip(axes, methods):
        sel = np.zeros(len(M_v), dtype=bool)
        sel[selections[m]] = True
        ax.scatter(M_v[~sel], M_d[~sel], s=7, c="lightgray", label=f"bị loại ({(~sel).sum()})")
        ax.scatter(M_v[sel], M_d[sel], s=7, c="tab:red", label=f"được chọn ({sel.sum()})")
        ax.set_title(m)
        ax.set_xlabel("M_v (lề thị giác)")
        ax.set_ylabel("M_d (lề bệnh học)")
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle("Ảnh được chọn / bị loại trong mặt phẳng (M_v, M_d)", fontsize=14)
    save_figure(fig, path)


def plot_selected_vs_removed(pool: list[str], scores: dict[str, np.ndarray], sel_idx: list[int], method: str,
                             alpha: float, beta: float, path: Path, n: int = 8, mode: str = "random",
                             seed: int = 0) -> None:
    """mode: 'random' | 'extreme' (điểm cao nhất vs thấp nhất) | 'boundary' (sát ranh giới chọn/loại)."""
    import matplotlib.pyplot as plt

    sel = np.zeros(len(pool), dtype=bool)
    sel[sel_idx] = True
    chosen, removed = np.flatnonzero(sel), np.flatnonzero(~sel)
    score = alpha * minmax(scores["M_v"]) + beta * minmax(scores["M_d"])
    rng = np.random.default_rng(seed)
    if mode == "random":
        a = rng.choice(chosen, min(n, len(chosen)), replace=False)
        b = rng.choice(removed, min(n, len(removed)), replace=False)
    elif mode == "extreme":
        a, b = chosen[np.argsort(-score[chosen])][:n], removed[np.argsort(score[removed])][:n]
    elif mode == "boundary":
        a, b = chosen[np.argsort(score[chosen])][:n], removed[np.argsort(-score[removed])][:n]
    else:
        raise ValueError(mode)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.75, 4.9))
    for row, (idxs, color, label) in enumerate([(a, "tab:green", "ĐƯỢC CHỌN"), (b, "tab:red", "BỊ LOẠI")]):
        for col in range(n):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(idxs):
                ax.axis("off")
                continue
            i = idxs[col]
            ax.imshow(Image.open(pool[i]))
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(4)
            ax.set_xlabel(f"M_v {scores['M_v'][i]:+.3f}\nM_d {scores['M_d'][i]:+.3f}", fontsize=7)
        axes[row, 0].set_ylabel(label, fontsize=11, color=color, fontweight="bold")
    fig.suptitle(f"{method} — chế độ '{mode}'", fontsize=12)
    save_figure(fig, path)


def plot_nearest_real(pool: list[str], real_paths: list[str], z_d_pool: np.ndarray, z_d_real: np.ndarray,
                      sel_idx: list[int], method: str, path: Path, n: int = 6) -> None:
    """Ảnh sinh cạnh ảnh thật gần nhất trong không gian E_d — kiểm tra GAN có học thuộc không."""
    import matplotlib.pyplot as plt

    idx = sel_idx[:n]
    sims = z_d_pool[idx] @ z_d_real.T
    nn = sims.argmax(axis=1)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.9, 4.4))
    for col, (i, j) in enumerate(zip(idx, nn)):
        axes[0, col].imshow(Image.open(pool[i]))
        axes[0, col].set_title(Path(pool[i]).name, fontsize=7)
        axes[1, col].imshow(Image.open(real_paths[j]))
        axes[1, col].set_title(f"cos = {sims[col, j]:.3f}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle(f"{method}: ảnh sinh và ảnh thật gần nhất (E_d) — kiểm tra học thuộc", fontsize=11)
    save_figure(fig, path)
