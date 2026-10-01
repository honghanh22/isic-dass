"""Nguồn dữ liệu — điểm khác biệt #1 giữa các bộ dữ liệu. Thêm nguồn mới: lớp con của `DatasetSource` + đăng ký."""

from __future__ import annotations

from ...config import Config, Layout
from .base import IMAGE_EXTS, DatasetSource
from .csv_source import CsvSource, normalize_label
from .folder_source import FolderSource


def build_source(cfg: Config, layout: Layout) -> DatasetSource:
    s, classes = cfg.data.source, cfg.data.classes
    if s.type == "csv":
        return CsvSource(classes, layout.drive_train_images, layout.drive_train_labels,
                         layout.drive_test_images, layout.drive_test_labels, s.header, s.image_ext)
    if s.type == "folders":
        return FolderSource(classes, layout.drive_train_images, layout.drive_test_images, s.class_dirs)
    raise ValueError(f"Nguồn không hỗ trợ: {s.type!r}")


__all__ = ["IMAGE_EXTS", "CsvSource", "DatasetSource", "FolderSource", "build_source", "normalize_label"]
