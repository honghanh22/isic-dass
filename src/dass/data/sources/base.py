"""Interface nguồn dữ liệu: copy ảnh gốc về ổ cục bộ, xếp theo thư mục lớp `<dst>/<tên lớp>/`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DatasetSource(ABC):
    def __init__(self, classes: dict[str, int]):
        self.classes = classes

    @property
    @abstractmethod
    def has_test_set(self) -> bool:
        """True nếu nguồn có tập test riêng (khi đó split chỉ tách val)."""

    @abstractmethod
    def ingest(self, subset: str, dst_root: str | Path) -> dict[str, int]:
        """Copy ảnh của `subset` ("train" | "test") vào `dst_root/<lớp>/`. Bỏ qua ảnh đã copy. Trả về số ảnh mỗi lớp."""
