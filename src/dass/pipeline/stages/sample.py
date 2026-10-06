"""Stage `sample`: sinh (hoặc khôi phục) candidate pool của lớp thiểu số (+ lớp đa số nếu bật M7 / M8)."""

from __future__ import annotations

from ...config import Config
from ..context import Context
from ..pool import CandidatePool, resolve_balanced_majority_pool, resolve_candidate_pool, resolve_majority_pool


def run(cfg: Config) -> CandidatePool:
    ctx = Context.create(cfg, "sample")
    pool = resolve_candidate_pool(ctx, generate_if_missing=True)
    resolve_majority_pool(ctx, generate_if_missing=True)
    resolve_balanced_majority_pool(ctx, generate_if_missing=True)
    return pool
