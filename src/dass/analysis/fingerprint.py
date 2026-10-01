"""Fingerprint miền tần số (Frank et al., ICML 2020) — chỉ ĐO, không biến đổi ảnh. Gọi code gốc của tác giả.

Chuỗi xử lý giữ đúng `prepare_dataset.py` của họ: load_image (grayscale) → dct2 → log_scale → chuẩn hoá welford.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from ..data.image_io import channel_gap, read_image
from ..utils import add_to_sys_path, run_command, save_figure

log = logging.getLogger(__name__)

FRANK_URL = "https://github.com/RUB-SysSec/GANDCTAnalysis.git"


def ensure_frank_repo(repo_dir: str | Path) -> SimpleNamespace:
    """Clone GANDCTAnalysis (MIT), vá cho Keras 3 và trả về các hàm gốc của tác giả."""
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        run_command(["git", "clone", "-q", FRANK_URL, str(repo_dir)])
    models_py = repo_dir / "src" / "models.py"
    src = models_py.read_text()
    if "layers.AveragePooling2D()" in src:   # Keras 3 bắt buộc truyền pool_size
        models_py.write_text(src.replace("layers.AveragePooling2D()", "layers.AveragePooling2D(pool_size=(2, 2))"))
        log.info("[patch] GANDCTAnalysis/src/models.py: AveragePooling2D(pool_size=(2, 2))")
    add_to_sys_path(repo_dir)
    image_np = importlib.import_module("src.image_np")
    fmath = importlib.import_module("src.math")
    models = importlib.import_module("src.models")
    return SimpleNamespace(dct2=image_np.dct2, load_image=image_np.load_image, log_scale=fmath.log_scale,
                           welford=fmath.welford, build_simple_cnn=models.build_simple_cnn,
                           build_regression=models.build_multinomial_regression)


def frank_dct_log(path: str, frank: SimpleNamespace) -> np.ndarray:
    return frank.log_scale(frank.dct2(frank.load_image(path, grayscale=True))).astype(np.float32)


def frank_frequency_detector(real_paths: list[str], fake_paths: list[str], tag: str, frank: SimpleNamespace,
                             fig_dir: Path | None, n_max: int = 400, epochs: int = 15, seed: int = 0) -> dict[str, float]:
    """log-DCT + chuẩn hoá welford + CNN nông / hồi quy của Frank et al. Trả về AUC của hai detector."""
    import matplotlib.pyplot as plt
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    rp = list(rng.choice(real_paths, min(n_max, len(real_paths)), replace=False))
    fp = list(rng.choice(fake_paths, min(n_max, len(fake_paths)), replace=False))
    X_real = np.stack([frank_dct_log(p, frank) for p in tqdm(rp, desc=f"DCT thật [{tag}]")])
    X_fake = np.stack([frank_dct_log(p, frank) for p in tqdm(fp, desc=f"DCT sinh [{tag}]")])
    X = np.concatenate([X_real, X_fake])
    y = np.r_[np.zeros(len(X_real)), np.ones(len(X_fake))].astype(np.float32)

    idx = rng.permutation(len(X))
    tr, te = idx[:int(0.8 * len(X))], idx[int(0.8 * len(X)):]
    mean, var = frank.welford(X[tr])                     # thống kê chỉ lấy từ phần train
    Xn = ((X - mean) / (np.sqrt(var) + 1e-12))[..., None].astype(np.float32)

    aucs = {}
    for name, builder in [("cnn", lambda: frank.build_simple_cnn(Xn.shape[1:], 1)),
                          ("regression", lambda: frank.build_regression(Xn.shape[1:], 1))]:
        tf.keras.utils.set_random_seed(seed)
        model = builder()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
        model.fit(Xn[tr], y[tr], validation_data=(Xn[te], y[te]), epochs=epochs, batch_size=32, verbose=0)
        aucs[name] = float(roc_auc_score(y[te], model.predict(Xn[te], verbose=0).ravel()))
        del model
        tf.keras.backend.clear_session()

    if fig_dir is not None:
        mean_real, mean_fake = X_real.mean(axis=0), X_fake.mean(axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (m, title) in zip(axes, [(mean_real, "log-DCT TB — ảnh THẬT"), (mean_fake, "log-DCT TB — ảnh SINH"),
                                         (mean_fake - mean_real, "Chênh lệch (fingerprint)")]):
            im = ax.imshow(m, cmap="viridis")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Frank et al. [{tag}] | AUC CNN = {aucs['cnn']:.3f} | AUC hồi quy = {aucs['regression']:.3f}")
        save_figure(fig, Path(fig_dir) / f"frank_dct_{tag}.png")
    return aucs


def fingerprint_report(stage: str, pools: dict[str, list[str]], real_paths: dict[str, list[str]],
                       frank: SimpleNamespace, fig_dir: Path | None, n_max: int, epochs: int, seed: int,
                       channels: int) -> list[dict]:
    """Một dòng mỗi lớp: AUC detector thật-vs-sinh của Frank et al. và chênh lệch kênh màu (phải = 0 với ảnh xám)."""
    rows = []
    for c, fake in pools.items():
        aucs = frank_frequency_detector(real_paths[c], fake, f"{stage}_{c}", frank, fig_dir, n_max, epochs, seed)
        rows.append({"giai_doan": stage, "lop": c, "auc_frank_cnn": aucs["cnn"],
                     "auc_frank_regression": aucs["regression"],
                     "chenh_lech_kenh_that": float(np.mean([channel_gap(read_image(p, channels))
                                                             for p in real_paths[c][:100]])),
                     "chenh_lech_kenh_sinh": float(np.mean([channel_gap(read_image(p, channels)) for p in fake[:100]]))})
        log.info("[%s/%s] AUC Frank CNN = %.4f | hồi quy = %.4f", stage, c, aucs["cnn"], aucs["regression"])
    return rows
