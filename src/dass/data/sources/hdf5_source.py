"""Nguồn HDF5 đã tiền xử lý (ví dụ PCam: `scripts/preprocess_pcam.py`): một file chứa sẵn train / valid / test.

File cần các nhóm `train`, `valid`, `test`, mỗi nhóm có `x` (N, H, W, C) uint8, `y` (N,) và `src_split`, `src_index`
(định danh ảnh trong bộ gốc); `attrs["classes"]` = JSON {tên lớp: chỉ số}, mỗi nhóm có `attrs["md5_x"]`.
- `ingest("train")` ghi ảnh của `train` + `valid` (val tách bằng CSV chia sẵn đi kèm, split.type = file),
  `ingest("test")` ghi ảnh của `test`, thành PNG không mất dữ liệu `<dst>/<lớp>/<src_split>_<src_index>.png`.
- Thư mục đích ghi dấu `md5_x` của file: tạo lại file .h5 khác mà vẫn dùng ổ tạm cũ -> dừng (không trộn hai bản).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..image_io import write_png
from .base import DatasetSource

GROUPS = {"train": ("train", "valid"), "test": ("test",)}
FINGERPRINT = ".hdf5_source.json"


def image_id(src_split: str, src_index: int) -> str:
    return f"{src_split}_{int(src_index):06d}"


def file_fingerprint(path: str | Path) -> dict[str, str]:
    import h5py

    with h5py.File(path, "r") as f:
        return {g: str(f[g].attrs["md5_x"]) for g in ("train", "valid", "test")}


class Hdf5Source(DatasetSource):
    def __init__(self, classes: dict[str, int], h5_path: Path):
        super().__init__(classes)
        self.h5_path = Path(h5_path)

    @property
    def has_test_set(self) -> bool:
        return True

    def ingest(self, subset: str, dst_root: str | Path) -> dict[str, int]:
        import h5py

        dst_root = Path(dst_root)
        fp = {"file": str(self.h5_path), "md5_x": file_fingerprint(self.h5_path)}
        marker = dst_root / FINGERPRINT
        if marker.exists() and json.loads(marker.read_text()) != fp:
            raise ValueError(f"{dst_root} đã chứa ảnh từ một bản {self.h5_path.name} KHÁC (md5 không khớp) -> xoá thư "
                             "mục cục bộ của bộ dữ liệu này rồi chạy lại, hoặc đặt data.name mới.")
        idx_to_name = {i: n for n, i in self.classes.items()}
        counts = dict.fromkeys(self.classes, 0)
        with h5py.File(self.h5_path, "r") as f:
            file_classes = json.loads(f.attrs["classes"])
            if file_classes != self.classes:
                raise ValueError(f"{self.h5_path.name}: lớp trong file {file_classes} khác data.classes {self.classes}")
            for name in self.classes:
                (dst_root / name).mkdir(parents=True, exist_ok=True)
            for group in GROUPS[subset]:
                g = f[group]
                y, ss, si = g["y"][:], g["src_split"][:].astype(str), g["src_index"][:]
                todo = [i for i in range(len(y))
                        if not (dst_root / idx_to_name[int(y[i])] / f"{image_id(ss[i], si[i])}.png").exists()]
                for start in tqdm(range(0, len(todo), 1024), desc=f"{self.h5_path.name}:{group} -> PNG"):
                    chunk = todo[start:start + 1024]
                    x = g["x"][np.array(chunk)] if chunk else []
                    for i, img in zip(chunk, x):
                        write_png(img, dst_root / idx_to_name[int(y[i])] / f"{image_id(ss[i], si[i])}.png")
                for i in range(len(y)):
                    counts[idx_to_name[int(y[i])]] += 1
        marker.write_text(json.dumps(fp))
        return counts
