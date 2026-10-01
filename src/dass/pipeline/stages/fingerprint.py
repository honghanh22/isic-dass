"""Stage `fingerprint` (tuỳ chọn): đo fingerprint miền tần số của ảnh sinh bằng detector của Frank et al."""

from __future__ import annotations

import pandas as pd

from ...analysis.fingerprint import ensure_frank_repo, fingerprint_report
from ...config import Config
from ..context import Context
from ..pool import resolve_candidate_pool
from . import init_tensorflow, save_metrics


def run(cfg: Config) -> pd.DataFrame:
    ctx = Context.create(cfg, "fingerprint")
    init_tensorflow(cfg)
    pool = resolve_candidate_pool(ctx, generate_if_missing=False)
    frank = ensure_frank_repo(ctx.layout.frank_repo)
    a, minority = cfg.analysis, ctx.budget.minority
    fig_dir = ctx.layout.fingerprint_fig_dir if cfg.save_figures else None
    rows = []
    for stage, paths in [("goc", pool.raw), ("sau_resize_chain", pool.resized)]:
        if paths:
            rows += fingerprint_report(stage, {minority: paths}, ctx.train_paths, frank, fig_dir,
                                       a.fingerprint_n_max, a.fingerprint_epochs, cfg.seed, ctx.channels)
    df = pd.DataFrame(rows)
    save_metrics(ctx.layout, "frequency_fingerprint", df)
    print(df.round(4).to_string(index=False))
    return df
