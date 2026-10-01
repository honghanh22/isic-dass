"""Ảnh y tế xám lưu dạng RGB có nhiễu màu JPEG CHỈ ở một lớp (Brain Tumor: 129 / 2000 ảnh negative, 0 / 400 positive)
là manh mối màu giả gắn với nhãn. `force_grayscale` phải loại bỏ nó: mọi ảnh lưu 3 kênh BẰNG NHAU."""

import numpy as np
from PIL import Image

from dass.config import load_config
from dass.data.image_io import (
    detect_channels,
    is_stored_correctly,
    max_channel_diff,
    read_image,
)
from dass.data.transforms import preprocess_tree
from dass.models.generator.sampler import pool_tag
from dass.pipeline.context import Context


def _noisy_rgb(rng, size=(20, 24), noise=3):
    """Ảnh xám lưu RGB với nhiễu màu nhỏ kiểu JPEG (lệch kênh vài mức xám)."""
    g = rng.integers(20, 230, size, dtype=np.int16)
    rgb = np.stack([g, g + rng.integers(-noise, noise + 1, size), g + rng.integers(-noise, noise + 1, size)], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def test_read_image_force_gray_equals_pil_luminance(tmp_path, rng):
    arr = _noisy_rgb(rng)
    Image.fromarray(arr, "RGB").save(tmp_path / "x.png")
    out = read_image(tmp_path / "x.png", 3, force_gray=True)
    assert out.shape == (20, 24, 3) and max_channel_diff(out) == 0
    np.testing.assert_array_equal(out[..., 0], np.asarray(Image.fromarray(arr, "RGB").convert("L")))
    assert max_channel_diff(read_image(tmp_path / "x.png", 3)) > 0            # không ép xám -> giữ nguyên gốc


def test_detect_channels_tolerates_jpeg_chroma_noise(tmp_path, rng):
    for i in range(5):
        Image.fromarray(_noisy_rgb(rng), "RGB").save(tmp_path / f"{i}.png")
    assert detect_channels([str(p) for p in tmp_path.glob("*.png")]) == 1


def test_preprocess_tree_redoes_unequal_channel_files(tmp_path, rng):
    src = tmp_path / "raw" / "negative"
    src.mkdir(parents=True)
    Image.fromarray(_noisy_rgb(rng), "RGB").save(src / "a.png")
    dst = tmp_path / "pp"
    preprocess_tree(tmp_path / "raw", dst, 16, ["negative"], channels=3)                 # bản cũ: còn nhiễu màu
    assert not is_stored_correctly(dst / "negative" / "a.png", 3, force_gray=True)
    preprocess_tree(tmp_path / "raw", dst, 16, ["negative"], channels=3, force_gray=True)
    assert is_stored_correctly(dst / "negative" / "a.png", 3, force_gray=True)
    assert Image.open(dst / "negative" / "a.png").mode == "RGB"


def test_pool_tag_separates_gray_pools():
    assert pool_tag(1200, 1344) == "from1200kimg_n1344"
    assert pool_tag(1200, 1344, channels=3, force_gray=True) == "from1200kimg_n1344_gray"


def test_brain_like_dataset_has_no_label_correlated_color(tmp_path, rng):
    """Nhiễu màu chỉ ở lớp negative (như dữ liệu thật) -> sau tiền xử lý mọi ảnh 3 kênh bằng nhau ở cả hai lớp."""
    src = tmp_path / "Brain_Tumor_Dataset"
    for folder, n, noisy in [("Negative", 20, True), ("Positive", 8, False)]:
        (src / folder).mkdir(parents=True)
        for i in range(n):
            arr = _noisy_rgb(rng) if noisy and i % 3 == 0 else np.repeat(rng.integers(0, 255, (20, 24, 1),
                                                                                      dtype=np.uint8), 3, -1)
            Image.fromarray(arr, "RGB").save(src / folder / f"{folder}_{i:02d}.png")
    cfg_path = tmp_path / "bt.yaml"
    cfg_path.write_text(f"""
paths: {{drive_root: {tmp_path / 'drive'}, local_root: {tmp_path / 'local'}, run_tag: t}}
data:
  name: bt
  channels: 3
  force_grayscale: true
  img_size: 16
  classes: {{negative: 0, positive: 1}}
  source: {{type: folders, train_images: {src}, class_dirs: {{negative: Negative, positive: Positive}}}}
  preprocess: {{crop_dark_border: false, resize: pad_square}}
  split: {{type: stratified, test: 0.15, val: 0.176}}
""", encoding="utf-8")
    ctx = Context.create(load_config([cfg_path]), "test")
    for d in [ctx.layout.train_pp, ctx.layout.test_pp]:
        for p in d.rglob("*.png"):
            img = Image.open(p)
            assert img.mode == "RGB" and max_channel_diff(np.asarray(img)) == 0, p
