import json

import numpy as np
import pytest

from isic_dass.data.ingest import copy_images_by_class, normalize_label
from isic_dass.data.preprocess import crop_dark_border
from isic_dass.data.split import compute_budget, create_split, load_or_create_split, validate_split
from isic_dass.data.variants import SYNTH_PREFIX, assemble_variant, assert_clean_eval_sets, prepare_variants

CLASSES = ["benign", "malignant"]


@pytest.mark.parametrize("raw,expected", [("benign", "benign"), (0, "benign"), ("1.0", "malignant"),
                                          (" Malignant ", "malignant")])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_label_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_label("nevus")


def test_copy_images_by_class(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in ["a", "b"]:
        (src / f"{i}.jpg").write_bytes(b"x")
    csv = tmp_path / "gt.csv"
    csv.write_text("a,0\nb,1.0\nmissing,0\n")
    copy_images_by_class(csv, src, tmp_path / "dst", CLASSES)
    assert (tmp_path / "dst" / "benign" / "a.jpg").exists()
    assert (tmp_path / "dst" / "malignant" / "b.jpg").exists()


def test_crop_dark_border_only_crops_large_borders():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 30:70] = 200                      # nội dung chiếm 20 % chiều cao -> cắt
    assert crop_dark_border(img).shape == (20, 40, 3)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2[10:90, 10:90] = 200                     # nội dung chiếm 80 % -> giữ nguyên
    assert crop_dark_border(img2).shape == (100, 100, 3)
    assert crop_dark_border(np.zeros((8, 8, 3), dtype=np.uint8)).shape == (8, 8, 3)


def test_split_is_deterministic_and_disjoint(image_tree):
    s1 = create_split(image_tree, CLASSES, 0.25, seed=7)
    s2 = create_split(image_tree, CLASSES, 0.25, seed=7)
    assert s1 == s2
    validate_split(s1, image_tree)
    assert len(s1["benign"]["val"]) == 2 and len(s1["malignant"]["val"]) == 1


def test_load_or_create_split_persists(image_tree, tmp_path):
    path = tmp_path / "split.json"
    s = load_or_create_split(path, image_tree, CLASSES, 0.25, seed=7)
    assert json.loads(path.read_text()) == s
    assert load_or_create_split(path, image_tree, CLASSES, 0.25, seed=999) == s   # đọc lại, không tạo mới


def test_validate_split_detects_mismatch(image_tree):
    s = create_split(image_tree, CLASSES, 0.25, seed=7)
    s["benign"]["train"].pop()
    with pytest.raises(ValueError):
        validate_split(s, image_tree)


def test_compute_budget():
    split = {"benign": {"train": list("abcdefgh"), "val": []}, "malignant": {"train": list("xyz"), "val": []}}
    b = compute_budget(split, pool_mult=4.0)
    assert (b.minority, b.majority, b.n_select, b.pool_size) == ("malignant", "benign", 5, 20)


def test_assemble_variant_and_cache(image_tree, tmp_path):
    split = create_split(image_tree, CLASSES, 0.25, seed=7)
    synth = [str(p) for p in sorted((image_tree / "benign").iterdir())[:3]]
    out = tmp_path / "variant"
    counts = assemble_variant(out, image_tree, split, "malignant", synth)
    assert counts["train"]["malignant"] == len(split["malignant"]["train"]) + 3
    assert counts["val"] == {c: len(split[c]["val"]) for c in CLASSES}
    assert len(list((out / "train" / "malignant").glob(f"{SYNTH_PREFIX}*"))) == 3

    marker_mtime = (out / ".complete.json").stat().st_mtime
    assert assemble_variant(out, image_tree, split, "malignant", synth) == counts   # cache: không ghép lại
    assert (out / ".complete.json").stat().st_mtime == marker_mtime


def test_prepare_variants_and_clean_eval(image_tree, tmp_path):
    split = create_split(image_tree, CLASSES, 0.25, seed=7)
    synth = [str(p) for p in sorted((image_tree / "benign").iterdir())[:2]]
    variants = prepare_variants(tmp_path / "v", image_tree, split, "malignant",
                                {"M0_real_only": [], "M6_dass": synth}, 4.0, "v1_d1", "M0_real_only")
    assert set(variants) == {"M0_real_only_p4", "M6_dass_p4"}
    assert variants["M0_real_only_p4"].class_weight and not variants["M6_dass_p4"].class_weight
    assert_clean_eval_sets(variants, image_tree, CLASSES)
