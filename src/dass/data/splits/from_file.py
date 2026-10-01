"""Chia theo CSV do người dùng định nghĩa: tiêu đề `image_id,split`, split = train | val (| test)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import Split

log = logging.getLogger(__name__)


def from_file(split_csv: str | Path, pp_dir: str | Path, class_names: list[str], with_test: bool = False) -> Split:
    """So khớp theo tên file không đuôi (image_id có hay không có đuôi đều được). Mọi ảnh phải có mặt.

    `with_test`: nguồn không có tập test riêng -> CSV phải gán cả phần test.
    """
    import pandas as pd

    allowed = ("train", "val", "test") if with_test else ("train", "val")
    df = pd.read_csv(split_csv)
    if not {"image_id", "split"} <= set(df.columns):
        raise ValueError(f"{split_csv}: cần các cột 'image_id' và 'split', có {list(df.columns)}")
    assignment = {Path(str(i).strip()).stem: str(s).strip().lower() for i, s in zip(df["image_id"], df["split"])}
    bad = sorted({s for s in assignment.values() if s not in allowed})
    if bad:
        raise ValueError(f"{split_csv}: giá trị split không hợp lệ {bad} (chỉ nhận {allowed})")

    split: Split = {}
    missing: list[str] = []
    for label in class_names:
        parts: dict[str, list[str]] = {s: [] for s in allowed}
        for f in sorted(os.listdir(Path(pp_dir) / label)):
            subset = assignment.get(Path(f).stem)
            if subset is None:
                missing.append(f"{label}/{f}")
            else:
                parts[subset].append(f)
        split[label] = parts
    if missing:
        raise ValueError(f"{split_csv}: thiếu {len(missing)} ảnh, ví dụ {missing[:3]}")
    for label, parts in split.items():
        empty = [s for s in allowed if not parts[s]]
        if empty:
            raise ValueError(f"{split_csv}: lớp {label} cần có ảnh ở cả {' / '.join(allowed)} (trống: {empty})")
        log.info("%s: %s (theo %s)", label, " / ".join(f"{len(v)} {k}" for k, v in parts.items()), Path(split_csv).name)
    return split
