"""Stage `train`: ghép tập huấn luyện cho mọi phương pháp (B0–B3, M1–M6) rồi train một backbone với các seed."""

from __future__ import annotations

from ...config import Config
from ...data.variants import VariantSpec, assert_clean_eval_sets, oversample_indices, prepare_variants
from ...selection import AUGMENT, CLASS_WEIGHT, IMBALANCED, OVERSAMPLE, SELECTION_METHODS
from ..context import Context
from ..pool import load_selections, resolve_candidate_pool
from . import init_tensorflow


def variant_specs(cfg: Config, ctx: Context) -> dict[str, VariantSpec]:
    """Đầu vào ghép tập train của mọi phương pháp, theo thứ tự train / bảng.

    - B0: ảnh thật (mất cân bằng). B1: như B0 + class weight. B3: như B0 + augmentation.
    - B2: ảnh thật + g bản sao ảnh thật lớp thiểu số (cân bằng 1 : 1).
    - M1–M6: ảnh thật + g ảnh sinh lớp thiểu số chọn theo tiêu chí của từng biến thể (selections.json).
    g = n_đa số − n_thiểu số.
    """
    b = ctx.budget
    real_minority = ctx.train_paths[b.minority]
    ros = [real_minority[i] for i in oversample_indices(len(real_minority), b.n_select, cfg.seed)]
    specs = {IMBALANCED: VariantSpec(), CLASS_WEIGHT: VariantSpec(class_weight=True),
             OVERSAMPLE: VariantSpec(extra_real=ros), AUGMENT: VariantSpec(augment=True)}
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)
    missing = [m for m in SELECTION_METHODS if m not in selections]
    if missing:
        raise RuntimeError(f"selections.json thiếu {missing} -> chạy lại `dass select --force`")
    for m in SELECTION_METHODS:
        if len(selections[m]) != b.n_select:
            raise RuntimeError(f"{m}: {len(selections[m])} ảnh sinh, cần {b.n_select} -> chạy lại `dass select`")
        specs[m] = VariantSpec(synth=selections[m])
    return specs


def run(cfg: Config, model_name: str, seeds: list[int] | None = None, archive_mismatched: bool = False,
        methods: list[str] | None = None) -> None:
    """`methods`: chỉ train các phương pháp này (mặc định: tất cả)."""
    from ...engine.trainer import run_experiments

    ctx = Context.create(cfg, f"train:{model_name}")
    init_tensorflow(cfg)
    specs = variant_specs(cfg, ctx)
    if methods:
        unknown = sorted(set(methods) - set(specs))
        if unknown:
            raise ValueError(f"Phương pháp không có: {unknown}. Có: {list(specs)}")
        specs = {m: s for m, s in specs.items() if m in methods}
    variants = prepare_variants(ctx.layout.variants, ctx.layout.train_pp, ctx.split, ctx.budget.minority, specs,
                                cfg.selection.pool_mult)
    assert_clean_eval_sets(variants, ctx.layout.test_pp, cfg.data.class_names)
    run_experiments(cfg, ctx.layout, model_name, variants, seeds or cfg.classifier.seeds, ctx.channels,
                    archive_mismatched, train_real=ctx.train_paths)
