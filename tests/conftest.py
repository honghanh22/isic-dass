import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def image_tree(tmp_path):
    """tmp/<lớp>/*.png: 10 benign, 4 malignant, ảnh 16×16 ngẫu nhiên."""
    root = tmp_path / "pp"
    r = np.random.default_rng(1)
    for label, n in [("benign", 10), ("malignant", 4)]:
        (root / label).mkdir(parents=True)
        for i in range(n):
            Image.fromarray(r.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(root / label / f"{label}_{i:02d}.png")
    return root
