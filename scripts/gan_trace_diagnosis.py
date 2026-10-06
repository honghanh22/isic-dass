"""Dấu vết GAN nằm ở đâu: AUC thật-vs-sinh (logistic, 5-fold) trên từng loại đặc trưng chỉ chứa MỘT loại thông tin.

    python scripts/gan_trace_diagnosis.py -c configs/server/rsna_pneumonia.yaml --tag srv

So ảnh THẬT lớp thiểu số (train) với ảnh SINH lớp thiểu số (pool của lần chạy), tối đa `--n` ảnh mỗi bên.
Đặc trưng (AUC ≈ 0,5: không tách được; ≈ 1: tách hoàn toàn):
  - intensity      : mean, std, phân vị, histogram 32 bin của độ sáng   -> độ sáng / tương phản
  - thumb16        : ảnh thu nhỏ 16×16                                  -> bố cục, cấu trúc lớn
  - spectrum       : phổ năng lượng theo bán kính (log, 64 dải)          -> kết cấu, chi tiết nhỏ, nhiễu
  - spectrum_high  : chỉ nửa tần số cao của phổ                          -> chi tiết li ti
  - ev / ev_blur2 / ev_blur4 : E_v (EfficientNet-B0 ImageNet) trên ảnh gốc / làm mờ Gauss σ = 2 / 4 (CẢ HAI tập)
  - ev_histmatch   : E_v sau khi khớp histogram từng ảnh sinh theo một ảnh thật ngẫu nhiên
Chỉ là chẩn đoán (không đổi pipeline, không train mạng, không dùng test). Ghi
`results_<run>/metrics/gan_trace_diagnosis.csv`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from dass.config import load_config
from dass.utils import setup_logging

log = logging.getLogger("gan_trace")


def radial_spectrum(img: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """log năng lượng phổ Fourier trung bình theo bán kính (ảnh xám)."""
    g = img.astype(np.float64)
    g = g.mean(axis=2) if g.ndim == 3 else g
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = f.shape
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - h / 2, xx - w / 2)
    bins = np.minimum((r / r.max() * n_bins).astype(int), n_bins - 1)
    return np.log1p(np.bincount(bins.ravel(), f.ravel(), n_bins) / np.maximum(np.bincount(bins.ravel(), None, n_bins), 1))


def intensity_stats(img: np.ndarray) -> np.ndarray:
    v = img.astype(np.float64).ravel()
    hist = np.histogram(v, bins=32, range=(0, 255), density=True)[0]
    return np.r_[v.mean(), v.std(), np.percentile(v, [1, 5, 25, 50, 75, 95, 99]), hist]


def thumb(img: np.ndarray, size: int = 16) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.float64).ravel()


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(sigma)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    setup_logging()

    from skimage.exposure import match_histograms

    from dass.analysis.shortcut import separability_auc
    from dass.data.image_io import read_image
    from dass.pipeline.context import Context
    from dass.pipeline.pool import resolve_candidate_pool
    from dass.pipeline.stages import save_metrics

    cfg = load_config([Path(c) for c in args.configs], args.overrides, tag=args.tag)
    ctx = Context.create(cfg, "gan_trace")
    ch, rng = ctx.channels, np.random.default_rng(cfg.seed)
    real_paths = ctx.train_paths[ctx.budget.minority]
    synth_paths = resolve_candidate_pool(ctx, generate_if_missing=False).final
    n = min(args.n, len(real_paths), len(synth_paths))
    real_paths = [real_paths[i] for i in rng.choice(len(real_paths), n, replace=False)]
    synth_paths = [synth_paths[i] for i in rng.choice(len(synth_paths), n, replace=False)]

    def load(paths):
        imgs = [read_image(p, ch) for p in paths]
        return [im[..., 0] if ch == 1 else im for im in imgs]   # PIL: ảnh xám 2 chiều

    real, synth = load(real_paths), load(synth_paths)
    matched = [match_histograms(s, real[rng.integers(n)], channel_axis=None if ch == 1 else -1).clip(0, 255)
               .astype(np.uint8) for s in synth]
    log.info("So %d ảnh thật vs %d ảnh sinh lớp %s (%d kênh)", n, n, ctx.budget.minority, ch)

    def auc(fn, a=real, b=synth) -> float:
        return separability_auc(np.stack([fn(x) for x in a]), np.stack([fn(x) for x in b]), cfg.seed)

    rows = [{"features": "intensity", "info": "độ sáng / tương phản", "auc": auc(intensity_stats)},
            {"features": "thumb16", "info": "bố cục, cấu trúc lớn", "auc": auc(thumb)},
            {"features": "spectrum", "info": "kết cấu, chi tiết nhỏ (toàn phổ)", "auc": auc(radial_spectrum)},
            {"features": "spectrum_high", "info": "chỉ tần số cao",
             "auc": auc(lambda x: radial_spectrum(x)[32:])},
            {"features": "intensity_histmatch", "info": "độ sáng sau khớp histogram",
             "auc": auc(intensity_stats, real, matched)}]

    import tensorflow as tf

    from dass.data.loaders import configure_gpu, to_backbone_input
    configure_gpu()
    size = cfg.classifier.size
    ev = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg",
                                              input_shape=(size, size, 3))

    def embed(imgs):
        x = tf.image.resize(np.stack([im[..., None] if im.ndim == 2 else im for im in imgs]).astype("float32"),
                            [size, size])
        x = tf.keras.applications.efficientnet.preprocess_input(to_backbone_input(x))
        z = ev.predict(x, batch_size=32, verbose=0)
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    def ev_auc(a, b) -> float:
        return separability_auc(embed(a), embed(b), cfg.seed)

    rows += [{"features": "ev", "info": "E_v gốc (như shortcut_check)", "auc": ev_auc(real, synth)},
             {"features": "ev_blur2", "info": "E_v, cả hai làm mờ σ = 2",
              "auc": ev_auc([blur(x, 2) for x in real], [blur(x, 2) for x in synth])},
             {"features": "ev_blur4", "info": "E_v, cả hai làm mờ σ = 4",
              "auc": ev_auc([blur(x, 4) for x in real], [blur(x, 4) for x in synth])},
             {"features": "ev_histmatch", "info": "E_v, ảnh sinh khớp histogram", "auc": ev_auc(real, matched)}]
    df = pd.DataFrame(rows).assign(n_per_side=n)
    save_metrics(ctx.layout, "gan_trace_diagnosis", df)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
