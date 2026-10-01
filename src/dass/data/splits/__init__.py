"""Chiến lược chia — điểm khác biệt #3 giữa các bộ dữ liệu. Thêm chiến lược mới: một hàm + một nhánh trong `build_split`."""

from __future__ import annotations

from pathlib import Path

from ...config.schema import SplitConfig
from .base import (
    ClassBudget,
    Split,
    canonical,
    check_expected_split,
    compute_budget,
    load_or_create_split,
    materialize_subset,
    split_hash,
    split_paths,
    validate_split,
)
from .from_file import from_file
from .stratified import holdout_val, random_split, stratified


def build_split(sp: SplitConfig, pp_dir: str | Path, class_names: list[str], seed: int, has_test_set: bool,
                split_file: str | Path | None = None) -> Split:
    if sp.type == "holdout_val":
        return holdout_val(pp_dir, class_names, sp.val, seed)
    if sp.type == "stratified":
        return stratified(pp_dir, class_names, sp.val, sp.test, seed, sp.group_regex)
    if sp.type == "file":
        if split_file is None:
            raise ValueError("data.split.type = file cần data.split.file")
        return from_file(split_file, pp_dir, class_names, with_test=not has_test_set)
    raise ValueError(f"Chiến lược chia không hỗ trợ: {sp.type!r}")


__all__ = ["ClassBudget", "Split", "build_split", "canonical", "check_expected_split", "compute_budget", "from_file",
           "holdout_val", "load_or_create_split", "materialize_subset", "random_split", "split_hash", "split_paths",
           "stratified", "validate_split"]
