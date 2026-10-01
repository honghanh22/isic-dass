import numpy as np
import pytest
from PIL import Image

from dass.data.image_io import (
    ChannelMismatchError,
    channel_gap,
    collapse_to_gray,
    detect_channels,
    png_channels,
    read_image,
    to_three_channels,
    write_png,
)


def test_gray_roundtrip_keeps_one_channel(tmp_path, rng):
    arr = rng.integers(0, 255, (8, 9, 1), dtype=np.uint8)
    write_png(arr, tmp_path / "g.png")
    assert Image.open(tmp_path / "g.png").mode == "L" and png_channels(tmp_path / "g.png") == 1
    np.testing.assert_array_equal(read_image(tmp_path / "g.png", 1), arr)


def test_rgb_roundtrip_keeps_three_channels(tmp_path, rng):
    arr = rng.integers(0, 255, (8, 9, 3), dtype=np.uint8)
    write_png(arr, tmp_path / "c.png")
    assert png_channels(tmp_path / "c.png") == 3
    np.testing.assert_array_equal(read_image(tmp_path / "c.png", 3), arr)


def test_equal_channel_rgb_to_gray_is_lossless(tmp_path, rng):
    """Ảnh xám lưu dạng RGB (R = G = B) đọc ở chế độ 1 kênh giữ nguyên giá trị (luminance của PIL)."""
    g = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    Image.fromarray(np.stack([g] * 3, -1)).save(tmp_path / "x.png")
    np.testing.assert_array_equal(read_image(tmp_path / "x.png", 1)[..., 0], g)


def test_detect_channels(tmp_path, rng, gray_tree, image_tree):
    gray = [str(p) for p in gray_tree.rglob("*.png")]
    color = [str(p) for p in image_tree.rglob("*.png")]
    assert detect_channels(gray) == 1 and detect_channels(color) == 3
    g = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    Image.fromarray(np.stack([g] * 3, -1)).save(tmp_path / "rgb_gray.png")
    assert detect_channels([str(tmp_path / "rgb_gray.png")]) == 1        # RGB nhưng 3 kênh bằng nhau
    with pytest.raises(ValueError):
        detect_channels([])


def test_collapse_to_gray_checks_channels(rng):
    g = rng.integers(0, 255, (4, 6, 6, 1), dtype=np.uint8)
    same = np.repeat(g, 3, axis=-1)
    np.testing.assert_array_equal(collapse_to_gray(same), g)
    diff = same.copy()
    diff[0, 0, 0, 1] = (int(diff[0, 0, 0, 0]) + 2) % 256
    with pytest.raises(ChannelMismatchError):
        collapse_to_gray(diff, tolerance=0)
    assert collapse_to_gray(diff, tolerance=255).shape[-1] == 1


def test_to_three_channels_and_gap(rng):
    g = rng.integers(0, 255, (5, 5, 1), dtype=np.uint8)
    t = to_three_channels(g)
    assert t.shape == (5, 5, 3) and channel_gap(t) == 0.0 and channel_gap(g) == 0.0
