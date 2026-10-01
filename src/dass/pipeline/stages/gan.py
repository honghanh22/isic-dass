"""Stage `gan-setup` (clone + vá StyleGAN2-ADA, biên dịch plugin) và `gan` (train có early stopping theo KID + báo cáo)."""

from __future__ import annotations

import logging
import sys

import pandas as pd

from ...analysis.figures import plot_kid_history, plot_samples
from ...config import Config, Layout
from ...data.image_io import load_images
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
    from ...models.generator.inception import inception_features
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

    minority = ctx.budget.minority
    real_feats = inception_features(load_images(ctx.train_paths[minority], ctx.channels))
    log.info("Inception features %s thật (train): %s", minority, real_feats.shape)
    state = StyleGanTrainer(layout, g, cfg.seed, real_feats, ctx.class_idx(minority)).train(fresh_start=fresh_start)
    report(cfg, ctx, state)
    return state


def report(cfg: Config, ctx: Context, state: dict) -> None:
    """Đường KID theo snapshot + ảnh mẫu từng lớp của snapshot tốt nhất."""
    from ...models.generator.inception import generate_class_images, load_generator

    layout = ctx.layout
    hist = pd.DataFrame(state["history"])
    if len(hist):
        hist.to_csv(layout.metrics_dir / "gan_kid_history.csv", index=False)
        if cfg.save_figures:
            plot_kid_history(hist, state["best_cum_kimg"], layout.gan_fig_dir / "kid_history.png")
    if cfg.save_figures and layout.gan_best_pkl.exists():
        G = load_generator(layout.gan_best_pkl)
        samples = {c: generate_class_images(G, 8, i, psi=cfg.generator.trunc_psi, seed=1)
                   for c, i in cfg.data.classes.items()}
        plot_samples(samples, f"Ảnh sinh từ snapshot tốt nhất ({state['best_cum_kimg']} kimg)",
                     layout.gan_fig_dir / "samples_best.png")
    record_stage(layout, cfg, "gan", gan={k: state.get(k) for k in ("best_kid", "best_cum_kimg", "stop_reason")})
