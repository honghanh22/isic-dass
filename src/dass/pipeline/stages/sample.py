"""Stage `sample`: sinh (hoặc khôi phục) candidate pool ⌈k·g⌉ ảnh sinh lớp thiểu số."""

from __future__ import annotations

from ...config import Config
from ..context import Context
from ..pool import CandidatePool, resolve_candidate_pool


def run(cfg: Config) -> CandidatePool:
    ctx = Context.create(cfg, "sample")
    return resolve_candidate_pool(ctx, generate_if_missing=True)
