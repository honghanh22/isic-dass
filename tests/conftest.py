from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_tree(root: Path, counts: dict[str, int], mode: str, size: int = 16, ext: str = ".png") -> Path:
    """root/<lớp>/<lớp>_XX<ext>: ảnh ngẫu nhiên chế độ `mode` ("L" hoặc "RGB")."""
    r = np.random.default_rng(1)
    for label, n in counts.items():
        (root / label).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            shape = (size, size) if mode == "L" else (size, size, 3)
            Image.fromarray(r.integers(0, 255, shape, dtype=np.uint8)).save(root / label / f"{label}_{i:02d}{ext}")
    return root


@pytest.fixture
def image_tree(tmp_path):
    """tmp/pp/<lớp>/*.png RGB: 10 benign, 4 malignant."""
    return make_tree(tmp_path / "pp", {"benign": 10, "malignant": 4}, "RGB")


@pytest.fixture
def gray_tree(tmp_path):
    """tmp/gray/<lớp>/*.png ảnh xám: 12 negative, 5 positive."""
    return make_tree(tmp_path / "gray", {"negative": 12, "positive": 5}, "L")
