"""Nguồn dạng thư mục chia sẵn theo lớp `<root>/<thư mục lớp>/*` (ví dụ Brain Tumor)."""

from __future__ import annotations

import shutil
from pathlib import Path

from tqdm.auto import tqdm

from .base import IMAGE_EXTS, DatasetSource


def copy_class_folders(src_root: str | Path, dst_root: str | Path, class_names: list[str],
                       class_dirs: dict[str, str] | None = None) -> dict[str, int]:
    """Copy `<src_root>/<thư mục lớp>/*` sang `<dst_root>/<tên lớp>/`. `class_dirs`: tên lớp -> tên thư mục nguồn."""
    class_dirs = class_dirs or {}
    counts = {}
    for label in class_names:
        src_dir, dst_dir = Path(src_root) / class_dirs.get(label, label), Path(dst_root) / label
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Không có thư mục lớp {src_dir} (cần dạng <thư mục ảnh>/<thư mục lớp>/)")
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for src in tqdm(files, desc=f"Copy {src_dir.name} -> {label}"):
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        counts[label] = len(files)
    return counts


class FolderSource(DatasetSource):
    def __init__(self, classes: dict[str, int], train_root: Path, test_root: Path | None = None,
                 class_dirs: dict[str, str] | None = None):
        super().__init__(classes)
        self.roots = {"train": train_root, "test": test_root}
        self.class_dirs = class_dirs or {}

    @property
    def has_test_set(self) -> bool:
        return self.roots["test"] is not None

    def ingest(self, subset: str, dst_root: str | Path) -> dict[str, int]:
        root = self.roots[subset]
        if root is None:
            raise ValueError(f"Nguồn không có tập {subset}")
        return copy_class_folders(root, dst_root, list(self.classes), self.class_dirs)
