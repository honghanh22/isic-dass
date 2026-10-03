"""Hình chẩn đoán: phân bố lớp, đường KID của GAN, ảnh mẫu, mặt phẳng (M_v, M_d), ảnh được chọn / bị loại,
ảnh sinh cạnh ảnh thật gần nhất. Ảnh xám hiển thị bằng colormap xám (không tô màu giả)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..utils import minmax, save_figure


def _show(ax, path_or_array) -> None:
    arr = np.asarray(Image.open(path_or_array)) if isinstance(path_or_array, (str, Path)) else path_or_array
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)


def plot_class_distribution(counts: dict[str, list[int]], class_names: list[str], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(class_names))
    fig = plt.figure(figsize=(8, 5))
    width = 0.8 / max(1, len(counts))
    for i, (name, vals) in enumerate(counts.items()):
        bars = plt.bar(x + (i - (len(counts) - 1) / 2) * width, vals, width, label=name)
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), int(b.get_height()), ha="center", va="bottom")
    plt.xticks(x, class_names)
    plt.ylabel("Number of images")
    plt.title(title)
    plt.legend()
    save_figure(fig, path)


def plot_label_samples(rows: dict[str, list[tuple[np.ndarray, list, str]]], counts: dict[str, int], title: str,
                       path: Path) -> None:
    """Mỗi hàng một tên nhãn: ảnh mẫu, khung của chính nhãn đó (đỏ), dưới ảnh ghi "lớp cuối cùng · tư thế chụp"."""
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        raise ValueError("Không có ảnh mẫu nào để vẽ (không tìm thấy file DICOM của các ảnh có nhãn)")
    n = max(len(v) for v in rows.values())
    fig, axes = plt.subplots(len(rows), n, figsize=(2.2 * n + 1.6, 2.4 * len(rows)), squeeze=False)
    for r, (label, items) in enumerate(rows.items()):
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(items):
                ax.axis("off")
                continue
            img, boxes, caption = items[c]
            _show(ax, img)
            for x, y, w, h in boxes:
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1.2))
            ax.set_xlabel(caption, fontsize=8)
        head = textwrap.fill(label, 18) + (f"\n({counts[label]} ảnh)" if label in counts else "")
        axes[r, 0].set_ylabel(head, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    h = fig.get_figheight()                       # hình cao: lề trên mặc định (12 %) để lại khoảng trắng lớn
    fig.subplots_adjust(top=1 - 0.9 / h)
    fig.suptitle(title, fontsize=11, y=1 - 0.35 / h)
    save_figure(fig, path)


def plot_kid_history(history: pd.DataFrame, best_kimg: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history["cum_kimg"], history["kid"] * 1e3, marker="o")
    plt.axvline(best_kimg, color="red", ls="--", label=f"best @ {best_kimg} kimg")
    plt.xlabel("kimg")
    plt.ylabel("KID lớp thiểu số (×1e-3, thấp hơn = tốt hơn)")
    plt.title("StyleGAN2-ADA — KID theo snapshot")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(fig, path)


def plot_samples(samples: dict[str, np.ndarray], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    n = min(len(v) for v in samples.values())
    fig, axes = plt.subplots(len(samples), n, figsize=(2 * n, 2.2 * len(samples)), squeeze=False)
    for row, (label, imgs) in enumerate(samples.items()):
        for col in range(n):
            _show(axes[row, col], imgs[col])
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"sinh: {label}", loc="left", fontsize=11)
    fig.suptitle(title)
    save_figure(fig, path)


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
    """mode: 'random' | 'extreme' (điểm cao nhất vs thấp nhất) | 'boundary' (sát ranh giới chọn / loại)."""
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
            _show(ax, pool[i])
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
        _show(axes[0, col], pool[i])
        axes[0, col].set_title(Path(pool[i]).name, fontsize=7)
        _show(axes[1, col], real_paths[j])
        axes[1, col].set_title(f"cos = {sims[col, j]:.3f}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle(f"{method}: ảnh sinh và ảnh thật gần nhất (E_d) — kiểm tra học thuộc", fontsize=11)
    save_figure(fig, path)
