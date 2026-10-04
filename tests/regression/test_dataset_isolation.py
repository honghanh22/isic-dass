"""Thêm bộ dữ liệu mới không được "dính" vào bộ dữ liệu cũ: ổ tạm cục bộ (data.name), GAN (gan_tag) và chỉ số lớp."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dass.config import Layout, load_config
from dass.data.splits import ClassBudget, split_hash
from dass.models.generator.trainer import StyleGanTrainer, check_gan_owner
from dass.pipeline.context import Context, warn_if_minority_not_positive
from dass.pipeline.pool import best_kimg


def _folder_dataset(root, seed):
    r = np.random.default_rng(seed)
    for folder, n in [("Negative", 12), ("Positive", 5)]:
        (root / folder).mkdir(parents=True)
        for i in range(n):
            Image.fromarray(r.integers(0, 255, (20, 20), dtype=np.uint8)).save(root / folder / f"{folder}_{i:02d}.png")
    return root


def _config(tmp_path, name, src, run_tag="t", gan_tag="g"):
    path = tmp_path / f"{name}_{src.name}.yaml"
    path.write_text(f"""
paths: {{drive_root: {tmp_path / 'drive'}, local_root: {tmp_path / 'local'}, run_tag: {run_tag}, gan_tag: {gan_tag}}}
data:
  name: {name}
  channels: auto
  img_size: 16
  classes: {{negative: 0, positive: 1}}
  source: {{type: folders, train_images: {src}, class_dirs: {{negative: Negative, positive: Positive}}}}
  preprocess: {{crop_dark_border: false, resize: stretch}}
  split: {{type: stratified, test: 0.2, val: 0.25}}
""", encoding="utf-8")
    return load_config([path])


def test_same_data_name_for_another_source_is_rejected(tmp_path):
    a, b = _folder_dataset(tmp_path / "A", 1), _folder_dataset(tmp_path / "B", 2)
    Context.create(_config(tmp_path, "same", a), "test")
    Context.create(_config(tmp_path, "same", a, run_tag="t2"), "test")       # cùng nguồn, run khác: được
    with pytest.raises(ValueError, match="data.name"):                         # nguồn khác, trùng tên: dừng
        Context.create(_config(tmp_path, "same", b, run_tag="t3"), "test")
    Context.create(_config(tmp_path, "other", b, run_tag="t3"), "test")      # tên riêng: được


def test_gan_is_bound_to_its_dataset_and_split(tmp_path):
    cfg = _config(tmp_path, "ds", _folder_dataset(tmp_path / "A", 1))
    lay = Layout(cfg)
    lay.gan_dir.mkdir(parents=True)
    trainer = StyleGanTrainer(lay, cfg.generator, 0, np.zeros((2, 4)), 1, data_name="ds", split_sha1="abc")

    def fake_segment(cmd, offset, state):                  # thay cho train.py: một snapshot rồi dừng sớm
        state.update(cum_kimg=100, best_kid=0.01, best_cum_kimg=100, finished=True, stop_reason="test")
        trainer._save(state)                               # như _consume_snapshots thật
        return True, 0, tmp_path / "log.txt"

    trainer._run_segment = fake_segment
    trainer.train()
    state = json.loads(lay.gan_state_json.read_text())
    assert (state["data_name"], state["split_sha1"]) == ("ds", "abc")     # GAN mới ghi lại dữ liệu + split
    check_gan_owner(state, "ds", "abc")
    check_gan_owner(state, None, None)
    with pytest.raises(ValueError, match="split_sha1"):
        StyleGanTrainer(lay, cfg.generator, 0, np.zeros((2, 4)), 1, data_name="ds", split_sha1="xyz").train()
    with pytest.raises(ValueError, match="data_name"):
        check_gan_owner(state, "other_dataset", "abc")
    check_gan_owner({"split_sha1": None, "data_name": None}, "ds", "xyz")  # GAN cũ chưa ghi: bỏ qua

    split = {"negative": {"train": ["a.png"], "val": [], "test": []}, "positive": {"train": ["b.png"], "val": [],
                                                                                   "test": []}}
    ctx = SimpleNamespace(layout=lay, cfg=cfg, split=split)
    with pytest.raises(ValueError, match="split_sha1"):                     # sample / select / train cũng kiểm tra
        best_kimg(ctx)
    state["split_sha1"] = split_hash(split)
    lay.gan_state_json.write_text(json.dumps(state))
    assert best_kimg(ctx) == 100


def test_warning_when_minority_is_not_class_one(tmp_path, caplog):
    cfg = _config(tmp_path, "ds", tmp_path)
    budget = ClassBudget(n_real={"negative": 3, "positive": 10}, minority="negative", majority="positive",
                         n_select=7, pool_size=11)
    with caplog.at_level(logging.WARNING):
        warn_if_minority_not_positive(cfg, budget)
    assert "chỉ số 0" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        warn_if_minority_not_positive(cfg, ClassBudget({"negative": 10, "positive": 3}, "positive", "negative", 7, 11))
    assert caplog.text == ""
