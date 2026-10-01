"""Chia ngẫu nhiên phân tầng theo lớp: [test | val | train] theo thứ tự sau khi xáo trộn.

- `holdout_val` (có tập test riêng, ví dụ ISIC): test = 0, chỉ tách val — tái lập đúng notebook ISIC v5.
- `stratified` (không có tập test riêng, ví dụ Brain Tumor): tách test rồi val — tái lập đúng notebook Brain v1.
  `group_regex` chia theo nhóm (mã bệnh nhân) để mọi ảnh của một bệnh nhân nằm cùng một phần.

Một RNG chung, duyệt lớp theo thứ tự cấu hình. KHÔNG đổi thứ tự gọi RNG: GAN cũ được train trên các split này.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np

from .base import Split

log = logging.getLogger(__name__)


def _take(groups: dict[str, list[str]], keys: list[str]) -> list[str]:
    return [f for k in keys for f in groups[k]]


def _sizes(n: int, val: float, test: float) -> tuple[int, int]:
    n_test = max(1, int(n * test)) if test > 0 else 0
    n_val = max(1, int((n - n_test) * val))
    return n_test, n_val


def random_split(pp_dir: str | Path, class_names: list[str], val: float, test: float, seed: int,
                 group_regex: str = "") -> Split:
    rng = np.random.default_rng(seed)
    split: Split = {}
    for label in class_names:
        files = sorted(os.listdir(Path(pp_dir) / label))
        if group_regex:
            groups: dict[str, list[str]] = {}
            for f in files:
                m = re.match(group_regex, f)
                groups.setdefault(m.group(1) if m else f, []).append(f)
            keys = sorted(groups)
            rng.shuffle(keys)
            n_test, n_val = _sizes(len(keys), val, test)
            parts = {"test": _take(groups, keys[:n_test]), "val": _take(groups, keys[n_test:n_test + n_val]),
                     "train": _take(groups, keys[n_test + n_val:])}
        else:
            rng.shuffle(files)
            n_test, n_val = _sizes(len(files), val, test)
            parts = {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}
        if test <= 0:
            del parts["test"]
        split[label] = parts
        log.info("%s: %d ảnh -> %s", label, len(files), " / ".join(f"{len(v)} {k}" for k, v in parts.items()))
    return split


def holdout_val(pp_dir: str | Path, class_names: list[str], val: float, seed: int) -> Split:
    return random_split(pp_dir, class_names, val=val, test=0.0, seed=seed)


def stratified(pp_dir: str | Path, class_names: list[str], val: float, test: float, seed: int,
               group_regex: str = "") -> Split:
    return random_split(pp_dir, class_names, val=val, test=test, seed=seed, group_regex=group_regex)
