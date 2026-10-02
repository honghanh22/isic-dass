"""Stage `report`: bảng cho bài báo -> `results_<run>/tables/` (mỗi bảng: .csv định dạng, .json số thô, .tex)."""

from __future__ import annotations

import logging

import pandas as pd

from ...config import Config, Layout
from ...evaluation import reporting
from ...utils import read_json
from ..context import record_stage
from . import load_metrics

log = logging.getLogger(__name__)


def run(cfg: Config) -> list:
    layout = Layout(cfg)
    layout.makedirs()
    out, latex, name = layout.tables_dir, cfg.evaluation.latex, cfg.data.name
    written = []

    card = read_json(layout.dataset_card)
    if card is not None:
        t = reporting.dataset_table(card)
        written += reporting.write_table(out, "dataset", t, t, f"{name}: số ảnh mỗi tập ({card['color']}, "
                                         f"{card['channels']} kênh, split {card['split']['type']})", latex)

    labels, groups = cfg.evaluation.method_labels, cfg.evaluation.method_groups
    summary = load_metrics(layout, "classification_summary")
    if summary is not None:
        table, bold = reporting.classification_table(summary, labels=labels, groups=groups)
        written += reporting.write_table(out, "classification", table, summary,
                                         f"{name}: kết quả phân loại trên test (mean ± std qua seed, ngưỡng "
                                         f"{cfg.evaluation.threshold})", latex, bold)

    cmp = load_metrics(layout, "significance_vs_baseline")
    if cmp is not None:
        written += reporting.write_table(out, "significance", reporting.significance_table(cmp, labels), cmp,
                                         f"{name}: paired bootstrap ΔAUC (cột vs = đối chứng)", latex)

    quality = load_metrics(layout, "generation_quality")
    if quality is not None:
        table, bold = reporting.generation_table(quality, labels)
        written += reporting.write_table(out, "generation_quality", table, quality,
                                         f"{name}: chất lượng ảnh sinh (Inception-v3; KID chính, FID tham khảo)",
                                         latex, bold)

    if not written:
        log.warning("Chưa có số liệu trong %s — chạy `dass evaluate` trước", layout.metrics_dir)
    for p in written:
        log.info("Bảng: %s", p)
    if summary is not None:
        pd.set_option("display.width", 250)
        printable = reporting.classification_table(summary, labels=labels, groups=groups)[0]
        print(printable.apply(lambda col: col.map(reporting.plain_label)).to_string(index=False))
    record_stage(layout, cfg, "report")
    return written
