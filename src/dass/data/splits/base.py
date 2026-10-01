"""Kiểu Split, kiểm tra hợp lệ, lưu / so khớp, tách phần test và ngân sách ảnh sinh — dùng chung mọi chiến lược."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...utils import read_json, write_json_atomic

log = logging.getLogger(__name__)

Split = dict[str, dict[str, list[str]]]   # {lớp: {"train": [file], "val": [file], ("test": [file])}}


@dataclass(frozen=True)
class ClassBudget:
    n_real: dict[str, int]
    minority: str
    majority: str
    n_select: int     # số ảnh sinh cần chọn = chênh lệch giữa hai lớp
    pool_size: int    # số ảnh ứng viên cần sinh


def canonical(split: Split) -> dict:
    return {c: {s: sorted(files) for s, files in parts.items()} for c, parts in split.items()}


def split_hash(split: Split) -> str:
    return hashlib.sha1(json.dumps(canonical(split), sort_keys=True).encode()).hexdigest()


def validate_split(split: Split, pp_dir: str | Path) -> None:
    """Các phần không giao nhau và phủ đúng toàn bộ ảnh trên đĩa."""
    for label, parts in split.items():
        names = list(parts)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = set(parts[a]) & set(parts[b])
                if common:
                    raise ValueError(f"Split lỗi: lớp {label} có {len(common)} ảnh vừa ở {a} vừa ở {b}")
        on_disk = set(os.listdir(Path(pp_dir) / label))
        listed = set().union(*parts.values())
        if listed != on_disk:
            raise ValueError(f"Split không khớp ảnh trên đĩa (lớp {label}: {len(listed)} trong split, "
                             f"{len(on_disk)} trên đĩa).")


def load_or_create_split(json_path: str | Path, pp_dir: str | Path, build: Callable[[], Split]) -> Split:
    """Lần đầu: tạo bằng `build()` và lưu. Các lần sau: phải trùng split đã lưu (kết quả của run_tag gắn với nó)."""
    expected = build()
    saved = read_json(json_path)
    if saved is None:
        write_json_atomic(json_path, expected)
        log.info("Đã lưu split: %s", json_path)
        split = expected
    else:
        if canonical(saved) != canonical(expected):
            raise ValueError(f"Cách chia hiện tại khác split đã lưu ở {json_path} (kết quả cũ dùng split đó). "
                             "Đổi paths.run_tag (hoặc dùng --tag) để bắt đầu lần chạy mới, hoặc đưa data.* về như cũ.")
        split = saved
    validate_split(split, pp_dir)
    return split


def check_expected_split(split: Split, expected_json: str | Path) -> None:
    """Split phải trùng split tham chiếu — ví dụ split mà GAN dùng lại đã được train trên phần train của nó.

    Khác nhau -> GAN có thể đã thấy ảnh val/test của lần chạy này (rò rỉ dữ liệu) -> dừng.
    """
    expected = read_json(expected_json)
    if expected is None:
        raise FileNotFoundError(f"Không tìm thấy split tham chiếu {expected_json} (data.split.expected)")
    if canonical(expected) != canonical(split):
        raise ValueError(f"Split khác split tham chiếu {expected_json}: GAN dùng lại có thể đã thấy ảnh val/test. "
                         "Giữ đúng cấu hình chia của lần train GAN, hoặc train GAN mới (đổi paths.gan_tag) "
                         "và bỏ data.split.expected.")
    log.info("Split trùng khớp split tham chiếu %s", Path(expected_json).name)


def materialize_subset(split: Split, subset: str, src_root: str | Path, dst_root: str | Path) -> dict[str, int]:
    """Đồng bộ `dst_root/<lớp>/` đúng bằng các ảnh của `subset` (copy ảnh thiếu, xoá ảnh thừa của split cũ)."""
    counts = {}
    for label, parts in split.items():
        dst = Path(dst_root) / label
        dst.mkdir(parents=True, exist_ok=True)
        wanted = set(parts[subset])
        for p in dst.iterdir():
            if p.name not in wanted:
                p.unlink()
        for f in wanted:
            if not (dst / f).exists():
                shutil.copy2(Path(src_root) / label / f, dst / f)
        counts[label] = len(wanted)
    return counts


def split_paths(root: str | Path, split: Split, subset: str) -> dict[str, list[str]]:
    return {c: [str(Path(root) / c / f) for f in parts[subset]] for c, parts in split.items()}


def compute_budget(split: Split, pool_mult: float) -> ClassBudget:
    n_real = {c: len(parts["train"]) for c, parts in split.items()}
    minority = min(n_real, key=n_real.get)
    majority = max(n_real, key=n_real.get)
    n_select = n_real[majority] - n_real[minority]
    return ClassBudget(n_real=n_real, minority=minority, majority=majority, n_select=n_select,
                       pool_size=int(np.ceil(pool_mult * n_select)))
