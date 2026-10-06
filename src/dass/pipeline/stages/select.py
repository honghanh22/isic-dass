"""Stage `select`: nhúng E_v / E_d, chấm điểm, chọn ảnh cho M0–M6 (+ M7 tuỳ chọn) và chẩn đoán."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...analysis import figures
from ...analysis.shortcut import probe_auc, separability_auc, shortcut_table
from ...config import Config
from ...data.variants import assemble_variant
from ...selection import (
    BOTH_CLASSES,
    compute_pool_scores,
    crossfit_folds,
    crossfit_margin,
    jaccard_matrix,
    select_all_methods,
)
from ...utils import write_json_atomic
from ..context import Context
from ..pool import resolve_candidate_pool, resolve_majority_pool
from . import init_tensorflow, save_metrics

log = logging.getLogger(__name__)


def _crossfit_md(cfg: Config, ctx: Context, pool: list[str], ckpt_dir: Path,
                 train_if_missing: bool) -> dict[str, np.ndarray]:
    """M_d cross-fitted (`encoder.e_d_crossfit_folds` = K): E_d thứ k train trên ảnh thật trừ fold k (val giữ nguyên),
    nhúng pool + ảnh của fold k; margin so với fold k (ảnh E_d đó KHÔNG học), rồi lấy trung bình qua K.
    Các chẩn đoán khác (probe, hình, embeddings.npz) vẫn dùng E_d đầy đủ."""
    from ...models.encoders import disease_encoder

    K, enc, sel, b, layout = cfg.encoder.e_d_crossfit_folds, cfg.encoder, cfg.selection, ctx.budget, ctx.layout
    names = {c: list(parts["train"]) for c, parts in ctx.split.items()}
    paths = ctx.train_paths
    folds = {c: crossfit_folds(len(names[c]), K, enc.e_d_seed + i) for i, c in enumerate(sorted(names))}
    parts = []
    for k in range(K):
        split_k = {c: {**p, "train": [f for f, fd in zip(names[c], folds[c]) if fd != k]} for c, p in ctx.split.items()}
        fit_dir = layout.variants / f"Ed_crossfit_{k}of{K}"
        assemble_variant(fit_dir, layout.train_pp, split_k, b.minority, [])
        ed = disease_encoder(cfg, ctx.channels, fit_dir,
                             ckpt_dir / f"Ed_{enc.e_d_model}_s{enc.e_d_seed}_cf{k}of{K}.weights.h5",
                             layout.clf_ckpt, train_if_missing=train_if_missing)
        held = {c: [p for p, fd in zip(paths[c], folds[c]) if fd == k] for c in names}
        parts.append((ed.embed(pool), ed.embed(held[b.minority]), ed.embed(held[b.majority])))
        log.info("Cross-fit E_d %d/%d: tham chiếu %d / %d ảnh không học", k + 1, K, len(held[b.minority]),
                 len(held[b.majority]))
    m, s_pos, s_neg = crossfit_margin(parts, sel.lambda_d, sel.sim_topk)
    log.info("M_d cross-fitted (K = %d): TB %+.4f (std %.4f)", K, m.mean(), m.std())
    return {"M_d": m, "S_d_pos": s_pos, "S_d_neg": s_neg}


def run(cfg: Config, force: bool = False) -> dict[str, list[int]] | None:
    """`force`: chọn lại dù đã có selections.json (dự đoán đã train với lựa chọn cũ sẽ không còn khớp)."""
    from ...models.encoders import disease_encoder, visual_encoder

    ctx = Context.create(cfg, "select")
    layout, sel, budget, ch = ctx.layout, cfg.selection, ctx.budget, ctx.channels
    if layout.selections_json.exists() and not force:
        log.warning("Đã có %s (các dự đoán đã train dùng lựa chọn này) -> giữ nguyên. Dùng `select --force` để "
                    "chọn lại, và đổi --tag / run_tag nếu đã có dự đoán.", layout.selections_json)
        return None
    init_tensorflow(cfg)
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    classes = cfg.data.class_names
    train_paths, val_paths = ctx.train_paths, ctx.val_paths

    ev = visual_encoder(cfg, ch)
    z_v_real = {c: ev.embed(train_paths[c]) for c in classes}
    z_v_pool = ev.embed(pool)

    shared = bool(cfg.encoder.e_d_from_run) and layout.e_d_run != cfg.paths.run_tag
    if shared:
        log.info("Dùng lại E_d của lần chạy '%s' (không train E_d mới)", layout.e_d_run)
    else:
        assemble_variant(layout.real_only, layout.train_pp, ctx.split, budget.minority, [])
    ed = disease_encoder(cfg, ch, layout.real_only, layout.e_d_ckpt, layout.clf_ckpt, train_if_missing=not shared)
    z_d_real = {c: ed.embed(train_paths[c]) for c in classes}
    z_d_pool = ed.embed(pool)

    # AUC probe: dòng "val" sát thực tế hơn (E_d đã học ảnh train; val chỉ dùng chọn epoch nên vẫn hơi lạc quan)
    c2i = cfg.data.classes
    probe = pd.DataFrame([
        {"space": "E_v (ImageNet)", "subset": "train (E_d đã thấy)", "auc_probe": probe_auc(z_v_real, c2i, cfg.seed)},
        {"space": "E_d (supervised)", "subset": "train (E_d đã thấy)", "auc_probe": probe_auc(z_d_real, c2i, cfg.seed)},
        {"space": "E_v (ImageNet)", "subset": "val (E_d chỉ dùng để chọn epoch)",
         "auc_probe": probe_auc({c: ev.embed(val_paths[c]) for c in classes}, c2i, cfg.seed)},
        {"space": "E_d (supervised)", "subset": "val (E_d chỉ dùng để chọn epoch)",
         "auc_probe": probe_auc({c: ed.embed(val_paths[c]) for c in classes}, c2i, cfg.seed)},
    ])
    save_metrics(layout, "probe_auc", probe)
    print(probe.round(4).to_string(index=False))

    scores = compute_pool_scores(z_v_pool, z_v_real, z_d_pool, z_d_real, budget.minority, budget.majority,
                                 sel.lambda_v, sel.lambda_d, sel.sim_topk)
    if cfg.encoder.e_d_crossfit_folds >= 2:
        scores.update(_crossfit_md(cfg, ctx, pool, layout.e_d_ckpt.parent, train_if_missing=not shared))
    selections = select_all_methods(scores, z_v_pool, budget.n_select, sel.alpha, sel.beta, sel.gamma, cfg.seed,
                                    both_classes=sel.both_classes_variant, div_normalization=sel.div_normalization,
                                    diversity_start=sel.diversity_start)
    for m, idx in selections.items():
        log.info("%s: chọn %d / %d ảnh sinh", m, len(idx), len(pool))

    write_json_atomic(layout.selections_json, {m: [Path(pool[i]).name for i in idx] for m, idx in selections.items()})
    np.savez(layout.embeddings_npz, pool_names=np.array([Path(p).name for p in pool]), z_v_pool=z_v_pool,
             z_d_pool=z_d_pool, **{f"z_v_real__{c}": z_v_real[c] for c in classes},
             **{f"z_d_real__{c}": z_d_real[c] for c in classes}, **{f"score__{k}": v for k, v in scores.items()})

    J = jaccard_matrix(selections)
    J.index.name = "method"
    save_metrics(layout, "selection_jaccard", J.reset_index())
    print("\n=== Jaccard giữa các biến thể ===\n" + J.round(2).to_string())

    shortcut = shortcut_table(z_v_real, z_v_pool, selections, budget.minority, budget.majority, cfg.seed)
    majority_pool = resolve_majority_pool(ctx, generate_if_missing=False)
    if majority_pool is not None:   # M7: ảnh sinh lớp đa số có tách được khỏi ảnh thật lớp đa số không
        shortcut.loc[len(shortcut)] = {
            "so_sanh": f"{BOTH_CLASSES}: {budget.majority} thật vs {budget.majority} sinh (E_v)",
            "method": f"{BOTH_CLASSES}__{budget.majority}",
            "auc_5fold": separability_auc(z_v_real[budget.majority], ev.embed(majority_pool.final), cfg.seed)}
    save_metrics(layout, "shortcut_check", shortcut)
    print(shortcut.round(4).to_string(index=False))

    if cfg.save_figures:
        fig_dir = layout.selection_fig_dir
        plotted = {m: idx for m, idx in selections.items() if m != BOTH_CLASSES}   # M7 trùng ảnh thiểu số với M6
        figures.plot_score_scatter(scores, plotted, fig_dir / "scatter_Mv_Md.png")
        for m, idx in plotted.items():
            if idx:
                figures.plot_selected_vs_removed(pool, scores, idx, m, sel.alpha, sel.beta,
                                                 fig_dir / f"grid_{m}_random.png")
        for mode in ["extreme", "boundary"]:
            figures.plot_selected_vs_removed(pool, scores, selections["M6_dass"], "M6_dass", sel.alpha, sel.beta,
                                             fig_dir / f"grid_M6_dass_{mode}.png", mode=mode)
        figures.plot_nearest_real(pool, train_paths[budget.minority], z_d_pool, z_d_real[budget.minority],
                                  selections["M6_dass"], "M6_dass", fig_dir / "nearest_real_M6_dass.png")
    return selections
