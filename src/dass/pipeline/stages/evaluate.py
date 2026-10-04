"""Stage `evaluate`: số liệu thô cho bài báo -> `results_<run>/metrics/`.

- Phân loại (từ dự đoán .npz trên test, ngưỡng cố định): từng lần chạy, mean / std qua seed, paired bootstrap ΔAUC
  (mọi phương pháp vs M0, cộng các cặp trong `evaluation.comparisons`), ΔAUC theo từng seed, p hiệu chỉnh Holm.
- Nhóm con (`evaluation.subgroup_columns`, ví dụ tư thế AP / PA của RSNA): AUC trong từng nhóm, mốc "chỉ dùng thuộc
  tính", tỉ lệ thuộc tính trong ảnh sinh (xem `evaluation.subgroups`).
- Sinh ảnh (Inception-v3): KID (chính, mean ± std), FID (tham khảo), đa dạng, SSIM nội bộ, AUC thật-vs-sinh
  cho từng tập ảnh được chọn; kèm hai hàng tham chiếu:
    * `reference / real val vs real train` — mức nền của metric (hai tập ảnh THẬT cùng lớp),
    * `pool / all candidates` — toàn bộ pool chưa lọc (phân phối của generator).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Config, Layout
from ...evaluation.aggregate import (
    add_holm,
    check_single_protocol,
    compare_pairs,
    compare_to_baseline,
    load_all_runs,
    summary_stats,
)
from ...evaluation.generative import compute_diversity, compute_fid, compute_ssim, kid_with_std
from ...utils import read_json
from ..context import Context, record_stage
from ..pool import load_selections, resolve_candidate_pool
from . import load_metrics, save_metrics

log = logging.getLogger(__name__)


def evaluate_classification(cfg: Config, layout: Layout) -> tuple[pd.DataFrame, dict]:
    ev = cfg.evaluation
    runs, probs = load_all_runs(layout.pred_dir, ev.threshold)
    if runs.empty:
        log.warning("Chưa có dự đoán nào trong %s -> bỏ qua phần phân loại", layout.pred_dir)
        return runs, probs
    check_single_protocol(runs)          # không gộp lần chạy có / không augmentation vào cùng một bảng
    save_metrics(layout, "classification_runs", runs)
    summary = summary_stats(runs)
    save_metrics(layout, "classification_summary", summary)
    # mọi phương pháp vs baseline (M0), cộng các cặp bổ sung (M6 vs M0b, M6 vs M1) — cùng một bảng, cột `vs`;
    # p_holm: hiệu chỉnh Holm trong mỗi model (cả họ so sánh của model đó)
    cmp = pd.concat([compare_to_baseline(probs, ev.baseline_method, ev.n_bootstrap, cfg.seed),
                     compare_pairs(probs, ev.comparisons, ev.n_bootstrap, cfg.seed)], ignore_index=True)
    if len(cmp):
        save_metrics(layout, "significance_vs_baseline", add_holm(cmp))
    log.info("Phân loại: %d lần chạy, %d nhóm (model × phương pháp)", len(runs), len(summary))
    return runs, probs


def evaluate_subgroups(cfg: Config, layout: Layout, probs: dict) -> None:
    """Theo từng cột `evaluation.subgroup_columns` (ví dụ ViewPosition): AUC trong từng nhóm, ΔAUC (Holm), mốc
    "chỉ dùng thuộc tính", tỉ lệ thuộc tính trong ảnh sinh. Không cần GPU."""
    from ...evaluation import subgroups as sg

    ev, names = cfg.evaluation, cfg.data.class_names
    if not ev.subgroup_columns:
        return
    if not layout.dicom_metadata_csv.exists():
        log.warning("evaluation.subgroup_columns = %s nhưng chưa có %s -> bỏ qua phân tích nhóm con",
                    ev.subgroup_columns, layout.dicom_metadata_csv)
        return
    split = read_json(layout.split_json) or {}
    files = sg.resolve_test_files(probs, sg.test_files_from_split(split, names), names)
    if probs and len(files) < len(probs):
        log.warning("%d / %d lần chạy không xác định được tên ảnh test -> bỏ khỏi phân tích nhóm con",
                    len(probs) - len(files), len(probs))
    runs_all, summ_all, cmp_all, ref_all, share_all = [], [], [], [], []
    for col in ev.subgroup_columns:
        attr = sg.load_attribute(layout.dicom_metadata_csv, col)
        if not attr:
            log.warning("Metadata không có cột %s -> bỏ qua", col)
            continue
        if files:
            runs = sg.subgroup_runs(probs, files, attr, col, ev.threshold)
            runs_all.append(runs)
            summ_all.append(sg.subgroup_summary(runs))
            cmp_all.append(sg.subgroup_comparisons(probs, files, attr, col, ev.baseline_method, ev.comparisons,
                                                   ev.n_bootstrap, cfg.seed))
            key = next(iter(files))
            ref_all.append(sg.attribute_only_auc(probs[key][0], files[key], attr, col))
        share = _attribute_share(cfg, layout, split, attr, col)
        if share is not None:
            share_all.append(share)
    for name, parts in [("classification_by_subgroup_runs", runs_all), ("classification_by_subgroup", summ_all),
                        ("significance_by_subgroup", cmp_all), ("subgroup_reference", ref_all),
                        ("subgroup_share", share_all)]:
        parts = [p for p in parts if p is not None and len(p)]
        if parts:
            save_metrics(layout, name, pd.concat(parts, ignore_index=True))
    for df in ref_all:
        log.info("Mốc chỉ dùng thuộc tính (test):\n%s", df.round(3).to_string(index=False))
    for df in share_all:
        log.info("Tỉ lệ thuộc tính trong ảnh thật / ảnh sinh (probe trên E_v):\n%s", df.round(3).to_string(index=False))


def _attribute_share(cfg: Config, layout: Layout, split: dict, attr: dict, col: str) -> pd.DataFrame | None:
    """Tỉ lệ thuộc tính trong ảnh sinh: dùng `embeddings.npz` (E_v) và `selections.json` của bước select."""
    from ...evaluation.subgroups import attribute_share_table

    selections = read_json(layout.selections_json)
    if not layout.embeddings_npz.exists() or selections is None or not split:
        log.warning("Chưa có embeddings.npz / selections.json -> bỏ qua tỉ lệ thuộc tính trong ảnh sinh")
        return None
    card = read_json(layout.dataset_card) or {}
    minority, majority = card.get("minority"), card.get("majority")
    if minority is None:
        return None
    with np.load(layout.embeddings_npz, allow_pickle=True) as e:
        z_real = {c: e[f"z_v_real__{c}"] for c in cfg.data.class_names}
        z_pool, pool_names = e["z_v_pool"], [str(n) for n in e["pool_names"]]
    # thứ tự z_v_real__<lớp> = thứ tự phần train của split đã lưu (Context.train_paths)
    names = {c: split[c]["train"] for c in cfg.data.class_names}
    if any(len(names[c]) != len(z_real[c]) for c in names):
        log.warning("embeddings.npz không khớp split hiện tại -> bỏ qua tỉ lệ thuộc tính trong ảnh sinh")
        return None
    df, probe_auc = attribute_share_table(z_real, names, z_pool, pool_names, selections, attr, col, minority,
                                          majority, cfg.seed)
    if df.empty:
        log.warning("Thuộc tính %s không phải nhị phân -> bỏ qua tỉ lệ trong ảnh sinh", col)
        return None
    log.info("Probe %s trên E_v (ảnh thật train): AUC ngoài fold = %.3f", col, probe_auc)
    return df.assign(probe_auc=probe_auc)


def evaluate_generation(cfg: Config) -> pd.DataFrame:
    from ...data.image_io import load_images
    from ...models.generator.inception import inception_features
    from ...models.generator.patches import ensure_stylegan_repo

    ctx = Context.create(cfg, "evaluate")
    layout, ev, ch, minority = ctx.layout, cfg.evaluation, ctx.channels, ctx.budget.minority
    ensure_stylegan_repo(layout.sg2_repo)          # Inception-v3 của StyleGAN2-ADA
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)

    def feats(paths: list[str]) -> np.ndarray:
        return inception_features(load_images(paths, ch))

    real_paths = ctx.train_paths[minority]
    real = feats(real_paths)
    pool_feats = feats(pool)
    index = {Path(p).name: i for i, p in enumerate(pool)}
    shortcut = load_metrics(layout, "shortcut_check")
    auc_by_method = dict(zip(shortcut["method"], shortcut["auc_5fold"])) if shortcut is not None else {}

    def row(set_name: str, method: str, paths: list[str], fake: np.ndarray, auc: float | None) -> dict:
        kid, kid_std = kid_with_std(real, fake, ev.kid_subsets, ev.kid_subset_size, cfg.seed)
        return {"set": set_name, "method": method, "n": len(paths), "n_real": len(real), "kid": kid, "kid_std": kid_std,
                "fid": compute_fid(real, fake), "diversity": compute_diversity(fake),
                "ssim": compute_ssim(paths, ch, ev.ssim_pairs, cfg.seed),
                "auc_real_vs_synth": np.nan if auc is None else auc}

    val_paths = ctx.val_paths[minority]
    rows = [row("reference", "real val vs real train", val_paths, feats(val_paths), None),
            row("pool", "all candidates", pool, pool_feats, None)]
    for method, paths in selections.items():
        if paths:
            rows.append(row("selected", method, paths, pool_feats[[index[Path(p).name] for p in paths]],
                            auc_by_method.get(method)))
    df = pd.DataFrame(rows)
    save_metrics(layout, "generation_quality", df)
    print(f"=== Chất lượng ảnh sinh (Inception-v3, {len(real)} ảnh thật {minority}) — KID chính, FID tham khảo ===\n"
          + df.round(5).to_string(index=False))
    return df


def run(cfg: Config, skip_generative: bool = False) -> None:
    layout = Layout(cfg)
    layout.makedirs()
    _, probs = evaluate_classification(cfg, layout)
    evaluate_subgroups(cfg, layout, probs)
    if not skip_generative:
        evaluate_generation(cfg)
    record_stage(layout, cfg, "evaluate")
