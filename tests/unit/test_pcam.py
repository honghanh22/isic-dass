"""PCam: nguồn HDF5 đã tiền xử lý, cấu hình và luật chọn snapshot GAN (FID-E_d làm mượt EMA, dừng sớm)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from conftest import CONFIGS

from dass.config import load_config
from dass.config.schema import GeneratorConfig
from dass.data.sources.hdf5_source import Hdf5Source, image_id
from dass.models.generator.trainer import default_state, update_selection


def _write_h5(path, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.attrs["classes"] = json.dumps({"negative": 0, "positive": 1})
        for k, (g, n) in enumerate((("train", 6), ("valid", 3), ("test", 4))):
            x = rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8)
            grp = f.create_group(g)
            grp.create_dataset("x", data=x)
            grp.create_dataset("y", data=np.array([i % 2 for i in range(n)], np.int8))
            grp.create_dataset("src_split", data=np.array([g] * n, dtype="S"))
            grp.create_dataset("src_index", data=np.arange(n) + 10 * k)
            grp.attrs["md5_x"] = str(rng.integers(1 << 30))


def test_hdf5_source_writes_lossless_pngs(tmp_path):
    from PIL import Image

    h5 = tmp_path / "d.h5"
    _write_h5(h5)
    src = Hdf5Source({"negative": 0, "positive": 1}, h5)
    assert src.ingest("train", tmp_path / "train") == {"negative": 5, "positive": 4}   # train + valid
    assert src.ingest("test", tmp_path / "test") == {"negative": 2, "positive": 2}
    with h5py.File(h5) as f:
        x0 = f["valid/x"][1]
        name = image_id("valid", int(f["valid/src_index"][1]))
    assert (np.asarray(Image.open(tmp_path / "train" / "positive" / f"{name}.png")) == x0).all()
    assert src.ingest("train", tmp_path / "train")["negative"] == 5                     # chạy lại: không ghi trùng


def test_hdf5_source_refuses_a_different_file_in_the_same_cache(tmp_path):
    h5 = tmp_path / "d.h5"
    _write_h5(h5, seed=0)
    Hdf5Source({"negative": 0, "positive": 1}, h5).ingest("test", tmp_path / "test")
    _write_h5(h5, seed=1)                                                             # tạo lại file khác
    with pytest.raises(ValueError, match="KHÁC"):
        Hdf5Source({"negative": 0, "positive": 1}, h5).ingest("test", tmp_path / "test")


def test_pcam_config():
    cfg = load_config([CONFIGS / "server" / "pcam.yaml"], tag="srv")
    g, d = cfg.generator, cfg.data
    assert (d.source.type, d.img_size, d.split.type, d.channels) == ("hdf5", 128, "file", 3)
    assert (g.select_metric, g.select_classes, g.select_aggregate, g.select_ema_alpha, g.keep_snapshots) == \
        ("fid_ed", "all", "max", 0.5, True)   # lớp kém nhất quyết định (khai báo trước GAN 6 : 1)
    # dừng sớm cùng luật với ISIC / RSNA
    isic = load_config([CONFIGS / "server" / "isic2016.yaml"], tag="srv").generator
    assert (g.min_kimg, g.patience, g.min_rel_delta, g.max_kimg) == \
        (isic.min_kimg, isic.patience, isic.min_rel_delta, isic.max_kimg)
    assert cfg.paths.run_tag == "pcam6_v1_srv" and cfg.encoder.e_d_from_run == ""   # E_d của chính lần chạy
    assert cfg.data.source.train_images.endswith("pcam_imb6.h5") and cfg.paths.gan_tag == "pcam6"
    with pytest.raises(ValueError, match="hdf5"):
        load_config([CONFIGS / "server" / "pcam.yaml"], ["data.split.type=stratified"])


def test_selection_ema_and_patience():
    g = GeneratorConfig(select_ema_alpha=0.5, min_kimg=200, patience=2, min_rel_delta=0.0)
    st = default_state()
    seq = [(100, 10.0), (200, 6.0), (300, 2.0), (400, 30.0), (500, 30.0)]
    out = []
    for cum, raw in seq:
        improved, sm = update_selection(st, raw, cum, g)
        st["history"].append({"cum_kimg": cum, "kid": raw, "smoothed": sm})
        out.append((improved, sm))
    assert [round(s, 3) for _, s in out] == [10.0, 8.0, 5.0, 17.5, 23.75]   # EMA α = 0,5
    assert [i for i, _ in out] == [True, True, True, False, False]
    assert st["best_cum_kimg"] == 300 and st["bad_count"] == 2              # 2 lần không cải thiện -> đủ patience


def test_selection_without_smoothing_is_unchanged():
    """α = 1 (mặc định ISIC / RSNA): điểm chọn = điểm thô, như trước 1.23.0."""
    g = GeneratorConfig()
    st = default_state()
    for cum, raw in [(100, 0.05), (200, 0.049), (300, 0.03)]:
        improved, sm = update_selection(st, raw, cum, g)
        assert sm == raw
        st["history"].append({"cum_kimg": cum, "kid": raw})
    assert st["best_kid"] == 0.03 and st["best_cum_kimg"] == 300


def test_selection_rule_cannot_change_mid_history():
    from dass.models.generator.trainer import check_selection_rule

    st = {**default_state(), "metric": "fid_ed", "select_classes": ["negative", "positive"], "select_ema_alpha": 0.5}
    check_selection_rule(st, {"metric": "fid_ed", "select_classes": ["positive"], "select_ema_alpha": 0.5})  # chưa có lịch sử
    st["history"].append({"cum_kimg": 100, "kid": 1.0})
    check_selection_rule(st, {"metric": "fid_ed", "select_classes": ["negative", "positive"], "select_ema_alpha": 0.5})
    with pytest.raises(ValueError, match="rescore"):
        check_selection_rule(st, {"metric": "fid_ed", "select_classes": ["positive"], "select_ema_alpha": 0.5})
    with pytest.raises(ValueError, match="rescore"):
        check_selection_rule({**st, "metric": "kid_inception"}, {"metric": "fid_ed"})


def test_select_aggregate_validation():
    assert load_config([CONFIGS / "server" / "isic2016.yaml"], tag="srv").generator.select_aggregate == "mean"
    with pytest.raises(ValueError, match="select_aggregate"):
        load_config([CONFIGS / "server" / "pcam.yaml"], ["generator.select_aggregate=min"])
