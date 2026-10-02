"""Stage `evaluate`: số liệu thô cho bài báo -> `results_<run>/metrics/`.

- Phân loại (từ dự đoán .npz trên test, ngưỡng cố định): từng lần chạy, mean / std qua seed, paired bootstrap ΔAUC
  (mọi phương pháp vs M0, cộng các cặp trong `evaluation.comparisons`).
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
from ...evaluation.aggregate import compare_pairs, compare_to_baseline, load_all_runs, summary_stats
from ...evaluation.generative import compute_diversity, compute_fid, compute_ssim, kid_with_std
from ..context import Context, record_stage
from ..pool import load_selections, resolve_candidate_pool
from . import load_metrics, save_metrics

log = logging.getLogger(__name__)


def evaluate_classification(cfg: Config, layout: Layout) -> None:
    ev = cfg.evaluation
    runs, probs = load_all_runs(layout.pred_dir, ev.threshold)
    if runs.empty:
        log.warning("Chưa có dự đoán nào trong %s -> bỏ qua phần phân loại", layout.pred_dir)
        return
    save_metrics(layout, "classification_runs", runs)
    summary = summary_stats(runs)
    save_metrics(layout, "classification_summary", summary)
    # mọi phương pháp vs baseline (M0), cộng các cặp bổ sung (M6 vs M0b, M6 vs M1) — cùng một bảng, cột `vs`
    cmp = pd.concat([compare_to_baseline(probs, ev.baseline_method, ev.n_bootstrap, cfg.seed),
                     compare_pairs(probs, ev.comparisons, ev.n_bootstrap, cfg.seed)], ignore_index=True)
    if len(cmp):
        save_metrics(layout, "significance_vs_baseline", cmp)
    log.info("Phân loại: %d lần chạy, %d nhóm (model × phương pháp)", len(runs), len(summary))


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
    evaluate_classification(cfg, layout)
    if not skip_generative:
        evaluate_generation(cfg)
    record_stage(layout, cfg, "evaluate")
