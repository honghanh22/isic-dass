"""Tích hợp (không cần GPU): chạy phần dữ liệu của pipeline trên bộ dữ liệu giả lập và kiểm tra ảnh xám giữ
nguyên 1 kênh, ảnh màu giữ 3 kênh, split / test / dataset card đúng chiến lược của từng bộ dữ liệu."""

import json

import numpy as np
from conftest import make_tree
from PIL import Image

from dass.config import load_config
from dass.pipeline.context import Context


def _write_yaml(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_grayscale_folder_dataset_stays_single_channel(tmp_path):
    # ảnh xám lưu dạng RGB có 3 kênh bằng nhau (như Brain Tumor) + đuôi .jpg lẫn .png
    src = tmp_path / "Brain_Tumor_Dataset"
    for folder, n in [("Negative", 20), ("Positive", 8)]:
        (src / folder).mkdir(parents=True)
        r = np.random.default_rng(len(folder))
        for i in range(n):
            g = r.integers(0, 255, (24, 20), dtype=np.uint8)
            Image.fromarray(np.stack([g] * 3, -1), "RGB").save(src / folder / f"{folder}_{i:02d}.png")
    cfg_path = _write_yaml(tmp_path / "bt.yaml", f"""
paths: {{drive_root: {tmp_path / 'drive'}, local_root: {tmp_path / 'local'}, run_tag: t}}
data:
  name: bt
  channels: auto
  img_size: 16
  classes: {{negative: 0, positive: 1}}
  source: {{type: folders, train_images: {src}, class_dirs: {{negative: Negative, positive: Positive}}}}
  preprocess: {{crop_dark_border: false, resize: pad_square}}
  split: {{type: stratified, test: 0.15, val: 0.176}}
""")
    ctx = Context.create(load_config([cfg_path]), "test")
    assert ctx.channels == 1
    for d in [ctx.layout.train_pp, ctx.layout.test_pp]:
        modes = {Image.open(p).mode for p in d.rglob("*.png")}
        assert modes == {"L"}, f"{d}: {modes}"
    sizes = {Image.open(p).size for p in ctx.layout.train_pp.rglob("*.png")}
    assert sizes == {(16, 16)}
    card = json.loads(ctx.layout.dataset_card.read_text(encoding="utf-8"))
    assert card["channels"] == 1 and card["color"] == "grayscale" and card["minority"] == "positive"
    assert card["counts"]["test"] == {c: len(ctx.split[c]["test"]) for c in ["negative", "positive"]}
    assert ctx.layout.run_manifest.exists()
    # chạy lại: idempotent, cùng split
    assert Context.create(load_config([cfg_path]), "test").split == ctx.split


def test_rgb_csv_dataset_stays_three_channels(tmp_path):
    imgs = make_tree(tmp_path / "tmp_rgb", {"x": 14}, "RGB", size=20, ext=".jpg")
    flat_train, flat_test = tmp_path / "drive" / "Train", tmp_path / "drive" / "Test"
    flat_train.mkdir(parents=True)
    flat_test.mkdir(parents=True)
    rows_train, rows_test = [], []
    for i, p in enumerate(sorted((imgs / "x").iterdir())):
        target, rows = (flat_test, rows_test) if i >= 10 else (flat_train, rows_train)
        p.rename(target / p.name)
        rows.append(f"{p.stem},{1 if i % 3 == 0 else 0}")
    (tmp_path / "drive" / "train.csv").write_text("\n".join(rows_train) + "\n")
    (tmp_path / "drive" / "test.csv").write_text("\n".join(rows_test) + "\n")
    cfg_path = _write_yaml(tmp_path / "isic.yaml", f"""
paths: {{drive_root: {tmp_path / 'drive'}, local_root: {tmp_path / 'local'}, run_tag: t}}
data:
  name: isic
  img_size: 16
  classes: {{benign: 0, malignant: 1}}
  source: {{type: csv, train_images: Train, train_labels: train.csv, test_images: Test, test_labels: test.csv}}
  preprocess: {{crop_dark_border: true, resize: stretch}}
  split: {{type: holdout_val, val: 0.15}}
""")
    ctx = Context.create(load_config([cfg_path]), "test")
    assert ctx.channels == 3
    assert {Image.open(p).mode for p in ctx.layout.train_pp.rglob("*.png")} == {"RGB"}
    assert {Image.open(p).mode for p in ctx.layout.test_pp.rglob("*.png")} == {"RGB"}
    assert all(set(parts) == {"train", "val"} for parts in ctx.split.values())     # test riêng, không tách
