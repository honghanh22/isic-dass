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

    # Chú thích bảng bằng tiếng Anh (bảng dán vào bài báo, cùng ngôn ngữ với tên phương pháp)
    card = read_json(layout.dataset_card)
    if card is not None:
        t = reporting.dataset_table(card)
        written += reporting.write_table(out, "dataset", t, t, f"{name}: images per subset ({card['color']}, "
                                         f"{card['channels']} channel(s), {card['split']['type']} split)", latex)

    # tên hiển thị + " (class-weighted)" theo đúng thiết lập đã train (trường class_weight trong .npz)
    labels = reporting.labels_for_runs(cfg.evaluation.method_labels, load_metrics(layout, "classification_runs"))
    groups = cfg.evaluation.method_groups
    summary = load_metrics(layout, "classification_summary")
    if summary is not None:
        table, bold = reporting.classification_table(summary, labels=labels, groups=groups)
        written += reporting.write_table(out, "classification", table, summary,
                                         f"{name}: test-set classification results (mean ± std over seeds, fixed "
                                         f"threshold {cfg.evaluation.threshold})", latex, bold)

    runs = load_metrics(layout, "classification_runs")
    if runs is not None:
        t = reporting.split_auc_table(runs, labels, groups)
        written += reporting.write_table(out, "auc_by_split", t, t,
                                         f"{name}: ROC-AUC on real train / val / test images (mean ± std over seeds); "
                                         "Train − Test: generalisation gap", latex)

    sig_caption = ("paired bootstrap ΔAUC on the seed-averaged (ensemble) predictions with 95% CI and p-value; "
                   "p (Holm): Holm-adjusted within each model; ΔAUC per seed: mean ± std of seed-paired differences")
    cmp = load_metrics(layout, "significance_vs_baseline")
    if cmp is not None:
        written += reporting.write_table(out, "significance", reporting.significance_table(cmp, labels), cmp,
                                         f"{name}: {sig_caption}", latex)

    sub = load_metrics(layout, "classification_by_subgroup")
    if sub is not None:
        table, bold = reporting.subgroup_table(sub, labels)
        written += reporting.write_table(out, "classification_by_subgroup", table, sub,
                                         f"{name}: AUC within each subgroup (mean ± std over seeds)", latex, bold)
    sub_cmp = load_metrics(layout, "significance_by_subgroup")
    if sub_cmp is not None:
        written += reporting.write_table(out, "significance_by_subgroup",
                                         reporting.significance_table(sub_cmp, labels), sub_cmp,
                                         f"{name}: within-subgroup {sig_caption} (Holm within model × subgroup)",
                                         latex)
    ref = load_metrics(layout, "subgroup_reference")
    if ref is not None:
        written += reporting.write_table(out, "subgroup_reference", reporting.subgroup_reference_table(ref), ref,
                                         f"{name}: attribute-only reference (test set)", latex)
    share = load_metrics(layout, "subgroup_share")
    if share is not None:
        written += reporting.write_table(out, "subgroup_share", reporting.subgroup_share_table(share, labels), share,
                                         f"{name}: attribute share in real and selected synthetic images "
                                         "(logistic probe on E_v features)", latex)

    quality = load_metrics(layout, "generation_quality")
    if quality is not None:
        table, bold = reporting.generation_table(quality, labels)
        written += reporting.write_table(out, "generation_quality", table, quality,
                                         f"{name}: synthetic image quality (Inception-v3; KID primary, FID for "
                                         "reference)", latex, bold)

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
