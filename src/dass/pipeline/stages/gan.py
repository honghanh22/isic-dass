"""Stage `gan-setup` (clone + vá StyleGAN2-ADA, biên dịch plugin) và `gan` (train có early stopping theo KID + báo cáo)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ...analysis.figures import plot_kid_history, plot_samples
from ...config import Config, Layout
from ...data.image_io import load_images
from ...data.splits import split_hash
from ...utils import run_command
from ..context import Context, record_stage

log = logging.getLogger(__name__)


def setup(cfg: Config, reset: bool = True, verify: bool = True, clear_cache: bool = False) -> None:
    from ...models.generator.patches import clear_torch_extension_cache, ensure_stylegan_repo, verify_cuda_plugins

    layout = Layout(cfg)
    if clear_cache:
        clear_torch_extension_cache()
    ensure_stylegan_repo(layout.sg2_repo, reset=reset)
    if verify:
        verify_cuda_plugins(layout.sg2_repo)


def run(cfg: Config, fresh_start: bool = False, dry_run: bool = False) -> dict:
    from ...models.generator.export import build_stylegan_dataset_zip
    from ...models.generator.patches import ensure_stylegan_repo
    from ...models.generator.trainer import StyleGanTrainer

    ctx = Context.create(cfg, "gan")
    layout, g = ctx.layout, cfg.generator
    ensure_stylegan_repo(layout.sg2_repo)
    build_stylegan_dataset_zip(layout.gan_dataset_zip, layout.train_pp, ctx.split, cfg.data.classes)
    if dry_run:   # kiểm tra dataset + cấu hình, không train
        run_command([sys.executable, "train.py", f"--outdir={layout.gan_runs / 'dryrun'}",
                     f"--data={layout.gan_dataset_zip}", "--gpus=1", "--cond=1", f"--mirror={int(g.mirror)}",
                     f"--cfg={g.cfg}", f"--batch={g.batch}", f"--gamma={g.gamma}", "--kimg=10",
                     "--metrics=none", "--dry-run"], cwd=layout.sg2_repo)
        return {}

    scorer, info = build_scorer(cfg, ctx)
    state = StyleGanTrainer(layout, g, cfg.seed, scorer, data_name=cfg.data.name, split_sha1=split_hash(ctx.split),
                            metric_info=info).train(fresh_start=fresh_start)
    report(cfg, ctx, state)
    return state


def build_scorer(cfg: Config, ctx: Context) -> tuple[Callable[[Path], tuple[float, dict[str, float]]], dict]:
    """Hàm chấm snapshot (thấp = tốt) theo `generator.select_metric` + thông tin ghi vào gan_state.json.

    Ảnh thật tham chiếu = TOÀN BỘ ảnh train của mỗi lớp được chấm (không bao giờ val). Mỗi lớp sinh `kid_n_gen` ảnh
    (seed `kid_seed`, ψ như lúc sinh pool); điểm snapshot = trung bình điểm các lớp trong `select_classes` (minority =
    chỉ lớp thiểu số; all = mọi lớp). fid_ed luôn chấm mọi lớp và trả về điểm từng lớp (lưu vào lịch sử).
    - kid_inception: KID trên Inception-v3 (giữ nguyên hành vi cũ khi select_classes = minority).
    - fid_ed: FID trên đặc trưng backbone của E_d (chưa chuẩn hoá L2) — CHÍNH file E_d mà `select` dùng cho M_d
      (Xue et al. 2021 chọn checkpoint bằng FID trên classifier của miền ảnh).
    """
    import torch

    from ...evaluation.generative import compute_fid, kid_from_features
    from ...models.generator.inception import generate_class_images, inception_features, load_generator

    g = cfg.generator
    classes = [ctx.budget.minority] if g.select_classes == "minority" else list(cfg.data.class_names)
    # fid_ed chấm MỌI lớp (ghi vào lịch sử để báo cáo), điểm chọn chỉ lấy các lớp trong `classes`
    scored = list(cfg.data.class_names) if g.select_metric == "fid_ed" else classes
    info = {"metric": g.select_metric, "select_classes": classes, "select_ema_alpha": g.select_ema_alpha}
    if g.select_metric == "kid_inception":
        def extract(images: np.ndarray) -> np.ndarray:
            return inception_features(images)

        real = {c: inception_features(load_images(ctx.train_paths[c], ctx.channels)) for c in scored}

        def distance(r: np.ndarray, f: np.ndarray) -> float:
            return kid_from_features(r, f, num_subsets=g.kid_subsets, seed=g.kid_seed)
    else:
        from . import init_tensorflow
        from .select import load_disease_encoder

        init_tensorflow(cfg)
        ed, _ = load_disease_encoder(cfg, ctx)
        info.update(e_d_ckpt=str(ctx.layout.e_d_ckpt), e_d_run=ctx.layout.e_d_run)

        def extract(images: np.ndarray) -> np.ndarray:
            return ed.embed_images(images, normalize=False).astype(np.float64)

        real = {c: ed.embed(ctx.train_paths[c], normalize=False).astype(np.float64) for c in scored}
        distance = compute_fid
    for c in scored:
        log.info("Đặc trưng ảnh thật %s (train) cho %s: %s", c, g.select_metric, real[c].shape)

    def scorer(pkl: Path) -> tuple[float, dict[str, float]]:
        """(điểm chọn = trung bình các lớp trong `select_classes`, điểm từng lớp đã chấm)."""
        G = load_generator(pkl)
        per_class = {}
        for c in scored:
            fake = generate_class_images(G, g.kid_n_gen, ctx.class_idx(c), psi=g.trunc_psi, seed=g.kid_seed)
            per_class[c] = float(distance(real[c], extract(fake)))
        del G
        torch.cuda.empty_cache()
        if len(per_class) > 1:
            log.info("            %s theo lớp: %s", g.select_metric, {c: round(v, 5) for c, v in per_class.items()})
        return float(np.mean([per_class[c] for c in classes])), per_class

    return scorer, info


def report(cfg: Config, ctx: Context, state: dict) -> None:
    """Đường KID theo snapshot + ảnh mẫu từng lớp của snapshot tốt nhất."""
    from ...models.generator.inception import generate_class_images, load_generator

    layout = ctx.layout
    hist = pd.DataFrame(state["history"])
    if len(hist):
        hist.to_csv(layout.metrics_dir / "gan_kid_history.csv", index=False)
        if cfg.save_figures:
            plot_kid_history(hist, state["best_cum_kimg"], layout.gan_fig_dir / "kid_history.png",
                             state.get("metric", "kid_inception"))
    if cfg.save_figures and layout.gan_best_pkl.exists():
        G = load_generator(layout.gan_best_pkl)
        samples = {c: generate_class_images(G, 8, i, psi=cfg.generator.trunc_psi, seed=1)
                   for c, i in cfg.data.classes.items()}
        plot_samples(samples, f"Ảnh sinh từ snapshot tốt nhất ({state['best_cum_kimg']} kimg)",
                     layout.gan_fig_dir / "samples_best.png")
    record_stage(layout, cfg, "gan", gan={k: state.get(k) for k in ("best_kid", "best_cum_kimg", "stop_reason")})
