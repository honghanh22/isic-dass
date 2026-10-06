"""Candidate pool: ảnh sinh của lớp thiểu số (để DASS chọn) và — nếu bật M7 — của lớp đa số. Cache zip trên Drive."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..utils import read_json
from .context import Context

log = logging.getLogger(__name__)


@dataclass
class CandidatePool:
    raw: list[str]
    resized: list[str] | None = None    # sau `generator.match_resize_chain`

    @property
    def final(self) -> list[str]:
        return self.resized or self.raw


def best_kimg(ctx: Context) -> int:
    """kimg của snapshot tốt nhất; đồng thời kiểm tra GAN đúng bộ dữ liệu + split của lần chạy này."""
    from ..data.splits import split_hash
    from ..models.generator.trainer import check_gan_owner, load_state

    state = load_state(ctx.layout.gan_state_json)
    if state["best_cum_kimg"] is None:
        raise RuntimeError("Chưa có GAN đã train (gan_state.json trống). Chạy `dass gan` trước.")
    check_gan_owner(state, ctx.cfg.data.name, split_hash(ctx.split))
    return state["best_cum_kimg"]


def _resolve_pool(ctx: Context, cls: str, n_images: int, seed: int, generate_if_missing: bool,
                  label: str = "") -> CandidatePool:
    """`label` (vd "_probe"): thư mục / zip riêng -> hai pool cùng lớp không ghi đè nhau. Mặc định "" = tên gốc."""
    from ..models.generator.sampler import archive_pool, generate_images, match_resize_chain, pool_tag, restore_pool

    cfg, layout, g = ctx.cfg, ctx.layout, ctx.cfg.generator
    force_gray = cfg.data.force_grayscale
    tag = pool_tag(best_kimg(ctx), n_images, ctx.channels, force_gray)
    pool_name = f"pool_{cls}{label}"
    raw_zip = layout.data_dir / f"{pool_name}_{tag}.zip"
    raw_dir = layout.candidates / pool_name
    raw = restore_pool(raw_zip, raw_dir)
    if raw is None:
        if not generate_if_missing:
            raise RuntimeError(f"Chưa có candidate pool {raw_zip}. Chạy `dass sample` trước.")
        from ..models.generator.patches import ensure_stylegan_repo

        ensure_stylegan_repo(layout.sg2_repo)
        raw = generate_images(layout.sg2_repo, layout.gan_best_pkl, raw_dir, ctx.class_idx(cls), n_images,
                              g.trunc_psi, seed, ctx.channels, g.channel_tolerance, force_gray)
        archive_pool(raw_dir, raw_zip)
    pool = CandidatePool(raw=raw)

    if g.match_resize_chain:
        from PIL import Image

        from ..utils import list_images

        originals = list_images(layout.train_raw / cls, "*")[:200]
        sides = []
        for p in originals:
            with Image.open(p) as im:          # chỉ đọc kích thước; đóng file ngay
                sides.append(min(im.size))
        sides.sort()
        ref_side = sides[len(sides) // 2]
        pool.resized = match_resize_chain(pool.raw, layout.candidates / f"{pool_name}_resizechain",
                                          layout.data_dir / f"{pool_name}_resizechain{ref_side}_{tag}.zip",
                                          cfg.data.img_size, ref_side, ctx.channels)
    log.info("Pool %s: %d ảnh (%d kênh)", cls, len(pool.final), ctx.channels)
    return pool


def resolve_candidate_pool(ctx: Context, generate_if_missing: bool = True) -> CandidatePool:
    """`pool_size` ảnh sinh của lớp thiểu số để DASS chọn ra `n_select`."""
    return _resolve_pool(ctx, ctx.budget.minority, ctx.budget.pool_size, ctx.cfg.generator.gen_seed,
                         generate_if_missing)


def resolve_balanced_majority_pool(ctx: Context, generate_if_missing: bool = True) -> CandidatePool | None:
    """Ảnh sinh lớp đa số cho M8 (`selection.balanced_synth_ratio` > 0): đúng số ảnh sinh mỗi lớp, cùng seed với M7
    (gen_seed + 1), không lọc. None nếu tắt."""
    from ..data.variants import balanced_synth_count

    q = ctx.cfg.selection.balanced_synth_ratio
    if q <= 0:
        return None
    n = balanced_synth_count(q, ctx.budget.n_real[ctx.budget.majority])
    return _resolve_pool(ctx, ctx.budget.majority, n, ctx.cfg.generator.gen_seed + 1, generate_if_missing)


def resolve_majority_pool(ctx: Context, generate_if_missing: bool = True) -> CandidatePool | None:
    """`n_select` ảnh sinh của lớp đa số cho M7 (None nếu không bật `selection.both_classes_variant`)."""
    if not ctx.cfg.selection.both_classes_variant:
        return None
    return _resolve_pool(ctx, ctx.budget.majority, ctx.budget.n_select, ctx.cfg.generator.gen_seed + 1,
                         generate_if_missing)


def load_selections(ctx: Context, pool: list[str]) -> dict[str, list[str]]:
    """selections.json lưu tên file -> ánh xạ về đường dẫn trong pool hiện tại."""
    names = read_json(ctx.layout.selections_json)
    if names is None:
        raise RuntimeError(f"Chưa có {ctx.layout.selections_json}. Chạy `dass select` trước.")
    by_name = {Path(p).name: p for p in pool}
    missing = [n for files in names.values() for n in files if n not in by_name]
    if missing:
        raise RuntimeError(f"{len(missing)} ảnh trong selections.json không có trong pool hiện tại "
                           "(pool đã đổi?). Chạy lại `dass select`.")
    return {m: [by_name[n] for n in files] for m, files in names.items()}
