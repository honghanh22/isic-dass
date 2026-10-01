"""Đóng gói ảnh train thành zip đúng định dạng StyleGAN2-ADA (PNG + dataset.json chứa nhãn lớp).

PNG giữ nguyên số kênh: ảnh xám ("L") -> StyleGAN2-ADA tự train generator 1 kênh.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

from ...data.splits import Split

log = logging.getLogger(__name__)


def build_stylegan_dataset_zip(zip_path: str | Path, train_pp_dir: str | Path, split: Split,
                               class_to_idx: dict[str, int]) -> dict[str, int]:
    """Chỉ dùng phần TRAIN của split (không gồm val/test). Trả về số ảnh mỗi lớp."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for label, idx in class_to_idx.items():
            for f in split[label]["train"]:
                arcname = f"{label}/{f}"
                zf.write(Path(train_pp_dir) / label / f, arcname=arcname)
                labels.append([arcname, idx])
        zf.writestr("dataset.json", json.dumps({"labels": labels}))
    counts = {c: sum(1 for _, i in labels if i == idx) for c, idx in class_to_idx.items()}
    log.info("Đã tạo %s: %d ảnh %s", zip_path, len(labels), counts)
    return counts
