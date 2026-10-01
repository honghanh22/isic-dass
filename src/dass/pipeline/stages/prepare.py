"""Stage `prepare`: đọc nguồn, nhận diện số kênh, tiền xử lý, chia train/val/test, dataset card."""

from __future__ import annotations

import logging

from ...analysis.figures import plot_class_distribution
from ...config import Config
from ...data.transforms import content_ratio_stats
from ...utils import list_images
from ..context import Context

log = logging.getLogger(__name__)


def run(cfg: Config) -> Context:
    ctx = Context.create(cfg, "prepare")
    names, layout = cfg.data.class_names, ctx.layout
    counts = {"Train": [len(ctx.split[c]["train"]) for c in names],
              "Validation": [len(ctx.split[c]["val"]) for c in names],
              "Test": [len(list((layout.test_pp / c).iterdir())) for c in names]}
    log.info("Phân bố lớp: %s", {k: dict(zip(names, v)) for k, v in counts.items()})

    stats = content_ratio_stats([p for c in names for p in list_images(layout.train_raw / c, "*")], seed=cfg.seed)
    if stats:
        log.info("Tỉ lệ vùng có nội dung / ảnh gốc: trung vị %.2f, khoảng [%.2f, %.2f], std %.3f "
                 "(> 0.9 & ít dao động: không cần crop | < 0.7: nên crop | std > 0.1: cân nhắc) "
                 "— preprocess.crop_dark_border = %s", stats["median"], stats["min"], stats["max"], stats["std"],
                 cfg.data.preprocess.crop_dark_border)
    if cfg.save_figures:
        plot_class_distribution(counts, names, f"{cfg.data.name} — class distribution",
                                layout.results_dir / "class_distribution.png")
    return ctx
