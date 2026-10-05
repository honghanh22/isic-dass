"""Stage `train`: ghép tập huấn luyện cho mọi biến thể rồi train một backbone với các seed."""

from __future__ import annotations

from ...config import Config
from ...data.variants import assert_clean_eval_sets, oversample_indices, prepare_variants
from ...selection import BASELINE, BOTH_CLASSES, OVERSAMPLE
from ..context import Context
from ..pool import load_selections, resolve_candidate_pool, resolve_majority_pool
from . import init_tensorflow


def run(cfg: Config, model_name: str, seeds: list[int] | None = None, archive_mismatched: bool = False) -> None:
    from ...engine.trainer import run_experiments

    ctx = Context.create(cfg, f"train:{model_name}")
    init_tensorflow(cfg)
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)
    majority_synth = {}
    if BOTH_CLASSES in selections:
        majority_pool = resolve_majority_pool(ctx, generate_if_missing=False)
        if majority_pool is None:
            raise RuntimeError(f"selections.json có {BOTH_CLASSES} nhưng selection.both_classes_variant = false. "
                               "Bật lại tuỳ chọn hoặc chạy lại `dass select`.")
        majority_synth[BOTH_CLASSES] = majority_pool.final
    extra_real = {}
    if cfg.selection.oversample_variant:
        # M0b không chọn gì từ pool (không nằm trong selections.json): nhân bản ảnh thật lớp thiểu số lên 1 : 1.
        # Đặt ngay sau M0 để thứ tự train / bảng giữ nguyên với các biến thể còn lại.
        real_minority = ctx.train_paths[ctx.budget.minority]
        extra_real[OVERSAMPLE] = [real_minority[i] for i in
                                  oversample_indices(len(real_minority), ctx.budget.n_select, cfg.seed)]
        selections = {**{m: v for m, v in selections.items() if m == BASELINE}, OVERSAMPLE: [],
                      **{m: v for m, v in selections.items() if m != BASELINE}}
    sel = cfg.selection
    # Class weight: M7 (lớp đa số to ra vì có ảnh sinh) luôn có; baseline M0 khi bật classifier.baseline_class_weight
    # (mặc định bật; tắt -> M0 train thẳng trên dữ liệu mất cân bằng, như ISIC v9 / RSNA).
    # (dùng hằng BASELINE, không dùng evaluation.baseline_method: đổi đối chứng thống kê không được đổi cách train)
    class_weight_methods = {BOTH_CLASSES} | ({BASELINE} if cfg.classifier.baseline_class_weight else set())
    variants = prepare_variants(ctx.layout.variants, ctx.layout.train_pp, ctx.split, ctx.budget.minority, selections,
                                sel.pool_mult, f"v{sel.lambda_v:g}_d{sel.lambda_d:g}",
                                class_weight_methods=class_weight_methods,
                                majority=ctx.budget.majority, majority_synth=majority_synth, extra_real=extra_real)
    assert_clean_eval_sets(variants, ctx.layout.test_pp, cfg.data.class_names)
    ros = next((v.dir for v in variants.values() if v.method == OVERSAMPLE), None)   # giai đoạn B (nếu bật)
    run_experiments(cfg, ctx.layout, model_name, variants, seeds or cfg.classifier.seeds, ctx.channels,
                    archive_mismatched, real_balanced_dir=ros)
