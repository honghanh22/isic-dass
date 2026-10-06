"""Stage `train`: ghép tập huấn luyện cho mọi biến thể rồi train một backbone với các seed."""

from __future__ import annotations

from ...config import Config
from ...data.variants import assert_clean_eval_sets, oversample_indices, prepare_variants, ros_fill
from ...selection import BALANCED, BASELINE, BOTH_CLASSES, OVERSAMPLE
from ..context import Context
from ..pool import load_selections, resolve_balanced_majority_pool, resolve_candidate_pool, resolve_majority_pool
from . import init_tensorflow


def variant_inputs(cfg: Config, ctx: Context) -> tuple[dict[str, list[str]], dict, dict, set[str]]:
    """Đầu vào ghép tập train của mọi biến thể: (ảnh sinh lớp thiểu số theo biến thể — thứ tự = thứ tự train / bảng,
    ảnh sinh lớp đa số (M7), ảnh thật nhân bản (M0b; phần bù khi `selection.synth_fraction` < 1), biến thể có class
    weight). Dùng chung cho `train` và các script đánh giá trên val."""
    sel, b = cfg.selection, ctx.budget
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)
    majority_synth = {}
    if BOTH_CLASSES in selections:
        majority_pool = resolve_majority_pool(ctx, generate_if_missing=False)
        if majority_pool is None:
            raise RuntimeError(f"selections.json có {BOTH_CLASSES} nhưng selection.both_classes_variant = false. "
                               "Bật lại tuỳ chọn hoặc chạy lại `dass select`.")
        majority_synth[BOTH_CLASSES] = majority_pool.final
    real_minority = ctx.train_paths[b.minority]
    ros = [real_minority[i] for i in oversample_indices(len(real_minority), b.n_select, cfg.seed)]   # = tập của M0b
    # synth_fraction < 1: M1–M6 có ít ảnh sinh hơn số cần bù -> bù phần thiếu bằng ảnh thật nhân bản (cùng seed)
    extra_real = ros_fill(selections, real_minority, b.n_select, cfg.seed, skip={BALANCED})
    if BALANCED in selections:   # M8: ảnh thật cân bằng bằng ROS (đúng tập của M0b) + cùng số ảnh sinh ở mỗi lớp
        majority_pool = resolve_balanced_majority_pool(ctx, generate_if_missing=False)
        if majority_pool is None:
            raise RuntimeError(f"selections.json có {BALANCED} nhưng selection.balanced_synth_ratio = 0. Bật lại tuỳ "
                               "chọn hoặc chạy lại `dass select`.")
        if len(majority_pool.final) != len(selections[BALANCED]):
            raise RuntimeError(f"{BALANCED}: {len(selections[BALANCED])} ảnh sinh lớp thiểu số nhưng "
                               f"{len(majority_pool.final)} ảnh sinh lớp đa số -> chạy lại `sample` / `select`")
        majority_synth[BALANCED] = majority_pool.final
        extra_real[BALANCED] = ros
    if sel.oversample_variant:
        # M0b không chọn gì từ pool (không nằm trong selections.json): nhân bản ảnh thật lớp thiểu số lên 1 : 1.
        # Đặt ngay sau M0 để thứ tự train / bảng giữ nguyên với các biến thể còn lại.
        extra_real[OVERSAMPLE] = ros
        selections = {**{m: v for m, v in selections.items() if m == BASELINE}, OVERSAMPLE: [],
                      **{m: v for m, v in selections.items() if m != BASELINE}}
    # Class weight: M7 (lớp đa số to ra vì có ảnh sinh) luôn có; baseline M0 khi bật classifier.baseline_class_weight
    # (dùng hằng BASELINE, không dùng evaluation.baseline_method: đổi đối chứng thống kê không được đổi cách train)
    class_weight_methods = {BOTH_CLASSES} | ({BASELINE} if cfg.classifier.baseline_class_weight else set())
    return selections, majority_synth, extra_real, class_weight_methods


def run(cfg: Config, model_name: str, seeds: list[int] | None = None, archive_mismatched: bool = False) -> None:
    from ...engine.trainer import run_experiments

    ctx = Context.create(cfg, f"train:{model_name}")
    init_tensorflow(cfg)
    selections, majority_synth, extra_real, class_weight_methods = variant_inputs(cfg, ctx)
    sel = cfg.selection
    variants = prepare_variants(ctx.layout.variants, ctx.layout.train_pp, ctx.split, ctx.budget.minority, selections,
                                sel.pool_mult, f"v{sel.lambda_v:g}_d{sel.lambda_d:g}",
                                class_weight_methods=class_weight_methods,
                                majority=ctx.budget.majority, majority_synth=majority_synth, extra_real=extra_real)
    assert_clean_eval_sets(variants, ctx.layout.test_pp, cfg.data.class_names)
    run_experiments(cfg, ctx.layout, model_name, variants, seeds or cfg.classifier.seeds, ctx.channels,
                    archive_mismatched)
