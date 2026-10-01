"""Nguồn dạng thư mục ảnh phẳng + CSV nhãn (ví dụ ISIC 2016)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .base import IMAGE_EXTS, DatasetSource

log = logging.getLogger(__name__)


def normalize_label(raw: object, classes: dict[str, int]) -> str:
    """Nhãn là tên lớp (không phân biệt hoa thường) hoặc chỉ số lớp (1, '1', '1.0')."""
    key = str(raw).strip()
    by_name = {name.lower(): name for name in classes}
    if key.lower() in by_name:
        return by_name[key.lower()]
    by_idx = {idx: name for name, idx in classes.items()}
    try:
        value = float(key)
    except ValueError:
        value = None
    if value is not None and value.is_integer() and int(value) in by_idx:
        return by_idx[int(value)]
    raise ValueError(f"Nhãn không nhận diện được: {raw!r} (lớp hợp lệ: {classes})")


def read_labels(labels_csv: str | Path, classes: dict[str, int], header: bool = False) -> pd.DataFrame:
    """Cột 1 = image_id, cột 2 = nhãn (các cột sau bị bỏ qua)."""
    df = pd.read_csv(labels_csv, header=0 if header else None)
    if df.shape[1] < 2:
        raise ValueError(f"{labels_csv}: cần ít nhất 2 cột (image_id, nhãn)")
    df = df.iloc[:, :2].copy()
    df.columns = ["image_id", "label"]
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["label"] = df["label"].map(lambda v: normalize_label(v, classes))
    return df


def image_file_name(image_id: str, image_ext: str) -> str:
    return image_id if Path(image_id).suffix.lower() in IMAGE_EXTS else f"{image_id}{image_ext}"


def copy_images_by_csv(labels_csv: str | Path, src_img_dir: str | Path, dst_root: str | Path,
                       classes: dict[str, int], header: bool = False, image_ext: str = ".jpg") -> dict[str, int]:
    labels = read_labels(labels_csv, classes, header)
    dst_root, src_img_dir = Path(dst_root), Path(src_img_dir)
    for label in classes:
        (dst_root / label).mkdir(parents=True, exist_ok=True)
    missing = 0
    counts = dict.fromkeys(classes, 0)
    for image_id, label in tqdm(zip(labels["image_id"], labels["label"]), total=len(labels),
                                desc=f"Copy -> {dst_root}"):
        src = src_img_dir / image_file_name(image_id, image_ext)
        if not src.exists():
            missing += 1
            continue
        dst = dst_root / label / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        counts[label] += 1
    if missing:
        log.warning("%d ảnh trong %s không có trong %s", missing, labels_csv, src_img_dir)
    return counts


class CsvSource(DatasetSource):
    def __init__(self, classes: dict[str, int], train_images: Path, train_labels: Path,
                 test_images: Path | None = None, test_labels: Path | None = None,
                 header: bool = False, image_ext: str = ".jpg"):
        super().__init__(classes)
        self.sets = {"train": (train_images, train_labels), "test": (test_images, test_labels)}
        self.header, self.image_ext = header, image_ext

    @property
    def has_test_set(self) -> bool:
        return self.sets["test"][0] is not None

    def ingest(self, subset: str, dst_root: str | Path) -> dict[str, int]:
        images, labels = self.sets[subset]
        if images is None:
            raise ValueError(f"Nguồn không có tập {subset}")
        return copy_images_by_csv(labels, images, dst_root, self.classes, self.header, self.image_ext)
