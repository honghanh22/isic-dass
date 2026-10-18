"""Chẩn đoán (không đổi pipeline): làm mờ Gauss có xoá được dấu vết GAN mà vẫn giữ tín hiệu bệnh không?

    python scripts/blur_diagnosis.py -c configs/server/pcam.yaml --tag srv

Với mỗi σ (pixel, trên ảnh đã lưu ở `data.img_size`), làm mờ GIỐNG NHAU cho ảnh thật và ảnh sinh (PIL GaussianBlur,
cùng nhân cho mọi kênh -> không đổi màu), nhúng bằng E_v (EfficientNet-B0 ImageNet) và E_d của lần chạy, rồi đo
bằng probe logistic 5-fold:
- `auc_real_vs_synth`: tách ảnh THẬT lớp thiểu số (train) với ảnh SINH lớp thiểu số (pool) — mong muốn giảm về ~0,5;
- `auc_disease`: tách hai lớp ảnh THẬT (val, E_d chưa học) — mong muốn giữ cao.
Một σ có ích khi `auc_real_vs_synth` giảm rõ trong khi `auc_disease` gần như không đổi. Không train classifier.
Ghi `results_<run>/metrics/blur_diagnosis.csv`. Cần GPU (nhúng ảnh).
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

log = logging.getLogger("blur_diagnosis")


def blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """uint8 NHWC -> uint8 NHWC, cùng nhân Gauss cho mọi kênh."""
    if sigma <= 0:
        return images
    out = []
    for a in images:
        im = Image.fromarray(a.squeeze(-1) if a.shape[-1] == 1 else a)
        b = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=sigma)))
        out.append(b[..., None] if b.ndim == 2 else b)
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2, 3])
    ap.add_argument("--n", type=int, default=3000, help="số ảnh mỗi nhóm (lấy mẫu ngẫu nhiên, seed toàn cục)")
    args = ap.parse_args()
    setup_logging()

    from dass.analysis.shortcut import probe_auc, separability_auc
    from dass.data.image_io import load_images
    from dass.models.encoders import visual_encoder
    from dass.pipeline.context import Context
    from dass.pipeline.pool import resolve_candidate_pool
    from dass.pipeline.stages import init_tensorflow, save_metrics
    from dass.pipeline.stages.select import load_disease_encoder

    cfg = load_config([Path(c) for c in args.configs], args.overrides, tag=args.tag)
    init_tensorflow(cfg)
    ctx = Context.create(cfg, "blur_diagnosis")
    b, rng = ctx.budget, np.random.default_rng(cfg.seed)

    def sample(paths: list[str]) -> np.ndarray:
        idx = rng.choice(len(paths), size=min(args.n, len(paths)), replace=False)
        return load_images([paths[i] for i in sorted(idx)], ctx.channels)

    sets = {"real_min_train": sample(ctx.train_paths[b.minority]),
            "synth_min": sample(resolve_candidate_pool(ctx, generate_if_missing=False).final),
            "real_min_val": sample(ctx.val_paths[b.minority]),
            "real_maj_val": sample(ctx.val_paths[b.majority])}
    log.info("Số ảnh: %s", {k: len(v) for k, v in sets.items()})
    encoders = {"E_v": visual_encoder(cfg, ctx.channels), "E_d": load_disease_encoder(cfg, ctx)[0]}

    rows = []
    for sigma in args.sigmas:
        blurred = {k: blur(v, sigma) for k, v in sets.items()}
        for name, enc in encoders.items():
            z = {k: enc.embed_images(v) for k, v in blurred.items()}
            row = {"sigma": sigma, "space": name,
                   "auc_real_vs_synth": separability_auc(z["real_min_train"], z["synth_min"], cfg.seed),
                   "auc_disease": probe_auc({b.majority: z["real_maj_val"], b.minority: z["real_min_val"]},
                                            cfg.data.classes, cfg.seed)}
            rows.append(row)
            log.info("σ = %.1f | %s | thật vs sinh %.4f | bệnh (val) %.4f", sigma, name, row["auc_real_vs_synth"],
                     row["auc_disease"])
    df = pd.DataFrame(rows)
    save_metrics(ctx.layout, "blur_diagnosis", df)
    print(df.round(4).to_string(index=False))

    try:   # hình minh hoạ: cùng một ảnh thật và một ảnh sinh ở mọi σ
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, len(args.sigmas), figsize=(2.2 * len(args.sigmas), 4.6))
        for j, sigma in enumerate(args.sigmas):
            for i, key in enumerate(["real_min_train", "synth_min"]):
                img = blur(sets[key][:1], sigma)[0]
                axes[i, j].imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
                axes[i, j].set_title(f"{'thật' if i == 0 else 'sinh'} σ={sigma:g}", fontsize=9)
                axes[i, j].axis("off")
        out = ctx.layout.results_dir / "blur_diagnosis.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        log.info("Hình: %s", out)
    except Exception as e:   # hình không bắt buộc
        log.warning("Không vẽ được hình: %s", e)


if __name__ == "__main__":
    main()
