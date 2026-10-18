import json

import pytest

from dass.data.splits import (
    check_expected_split,
    compute_budget,
    from_file,
    holdout_val,
    load_or_create_split,
    materialize_subset,
    stratified,
    validate_split,
)
from dass.data.variants import (
    DUP_PREFIX,
    SYNTH_PREFIX,
    VariantSpec,
    assemble_variant,
    assert_clean_eval_sets,
    oversample_indices,
    prepare_variants,
)

CLASSES = ["benign", "malignant"]


def test_holdout_and_stratified_are_disjoint_and_deterministic(image_tree):
    s1 = holdout_val(image_tree, CLASSES, 0.25, seed=7)
    assert s1 == holdout_val(image_tree, CLASSES, 0.25, seed=7)
    validate_split(s1, image_tree)
    assert set(s1["benign"]) == {"train", "val"}
    s2 = stratified(image_tree, CLASSES, 0.25, 0.2, seed=7)
    validate_split(s2, image_tree)
    assert set(s2["benign"]) == {"train", "val", "test"} and len(s2["benign"]["test"]) == 2


def test_load_or_create_split_persists(image_tree, tmp_path):
    path = tmp_path / "split.json"
    s = load_or_create_split(path, image_tree, lambda: holdout_val(image_tree, CLASSES, 0.25, 7))
    assert json.loads(path.read_text()) == s
    assert load_or_create_split(path, image_tree, lambda: holdout_val(image_tree, CLASSES, 0.25, 7)) == s
    with pytest.raises(ValueError, match="run_tag"):          # đổi cách chia mà giữ run_tag -> lỗi
        load_or_create_split(path, image_tree, lambda: holdout_val(image_tree, CLASSES, 0.25, 999))


def test_check_expected_split(image_tree, tmp_path):
    split = stratified(image_tree, CLASSES, 0.25, 0.2, seed=7)
    ref = tmp_path / "real_split.json"
    with pytest.raises(FileNotFoundError):
        check_expected_split(split, ref)
    ref.write_text(json.dumps({c: dict(reversed(list(p.items()))) for c, p in split.items()}))
    check_expected_split(split, ref)
    with pytest.raises(ValueError, match="val/test"):
        check_expected_split(stratified(image_tree, CLASSES, 0.25, 0.2, seed=8), ref)


def test_materialize_subset_syncs_exactly(image_tree, tmp_path):
    split = stratified(image_tree, CLASSES, 0.25, 0.2, seed=7)
    (tmp_path / "test" / "benign").mkdir(parents=True)
    (tmp_path / "test" / "benign" / "stale.png").write_bytes(b"x")      # ảnh test của split cũ
    counts = materialize_subset(split, "test", image_tree, tmp_path / "test")
    for c in CLASSES:
        assert sorted(p.name for p in (tmp_path / "test" / c).iterdir()) == sorted(split[c]["test"])
    assert counts == {c: len(split[c]["test"]) for c in CLASSES}


def _split_csv(path, tree, val=(), test=()):
    rows = ["image_id,split"]
    for label in CLASSES:
        for f in sorted((tree / label).iterdir()):
            rows.append(f"{f.stem},{'val' if f.stem in val else 'test' if f.stem in test else 'train'}")
    path.write_text("\n".join(rows) + "\n")


def test_from_file(image_tree, tmp_path):
    csv = tmp_path / "s.csv"
    _split_csv(csv, image_tree, val={"benign_00", "malignant_03"})
    s = from_file(csv, image_tree, CLASSES)
    assert s["malignant"]["val"] == ["malignant_03.png"]
    validate_split(s, image_tree)
    _split_csv(csv, image_tree, val={"benign_00", "malignant_00"}, test={"benign_01", "malignant_01"})
    assert from_file(csv, image_tree, CLASSES, with_test=True)["benign"]["test"] == ["benign_01.png"]
    with pytest.raises(ValueError, match="không hợp lệ"):
        from_file(csv, image_tree, CLASSES, with_test=False)
    csv.write_text("image_id,split\nbenign_00,val\n")
    with pytest.raises(ValueError, match="thiếu"):
        from_file(csv, image_tree, CLASSES)


def test_validate_split_detects_overlap_and_missing(image_tree):
    s = stratified(image_tree, CLASSES, 0.25, 0.2, seed=7)
    s["benign"]["test"].append(s["benign"]["train"][0])
    with pytest.raises(ValueError, match="vừa ở"):
        validate_split(s, image_tree)
    s = stratified(image_tree, CLASSES, 0.25, 0.2, seed=7)
    s["benign"]["train"].pop()
    with pytest.raises(ValueError, match="không khớp"):
        validate_split(s, image_tree)


def test_compute_budget():
    split = {"benign": {"train": list("abcdefgh"), "val": []}, "malignant": {"train": list("xyz"), "val": []}}
    b = compute_budget(split, pool_mult=4.0)
    assert (b.minority, b.majority, b.n_select, b.pool_size) == ("malignant", "benign", 5, 20)


def test_assemble_variant_excludes_test_and_caches(image_tree, tmp_path):
    split = stratified(image_tree, CLASSES, 0.25, 0.2, seed=7)
    synth = [str(p) for p in sorted((image_tree / "benign").iterdir())[:3]]
    out = tmp_path / "variant"
    counts = assemble_variant(out, image_tree, split, "malignant", synth)
    assert counts["train"]["malignant"] == len(split["malignant"]["train"]) + 3
    assert counts["val"] == {c: len(split[c]["val"]) for c in CLASSES}
    mtime = (out / ".complete.json").stat().st_mtime
    assert assemble_variant(out, image_tree, split, "malignant", synth) == counts
    assert (out / ".complete.json").stat().st_mtime == mtime


def test_prepare_variants_design_2_0(image_tree, tmp_path):
    """B0–B3 + M1–M6 (2.0.0): B1 chỉ khác B0 ở class weight, B3 chỉ khác ở augmentation; B2 = ảnh thật + g bản sao
    ảnh thật lớp thiểu số (1 : 1); M* = ảnh thật + g ảnh sinh lớp thiểu số; lớp đa số luôn chỉ có ảnh thật."""
    split = holdout_val(image_tree, CLASSES, 0.25, seed=7)
    b = compute_budget(split, pool_mult=2.0)
    real = [str(image_tree / b.minority / f) for f in split[b.minority]["train"]]
    dupes = [real[i] for i in oversample_indices(len(real), b.n_select, seed=7)]
    synth = [str(p) for p in sorted((image_tree / b.majority).iterdir())[:b.n_select]]   # ảnh "sinh" giả lập
    specs = {"B0_imbalanced": VariantSpec(), "B1_class_weight": VariantSpec(class_weight=True),
             "B2_ros": VariantSpec(extra_real=dupes), "B3_augment": VariantSpec(augment=True),
             "M6_dass": VariantSpec(synth=synth)}
    v = prepare_variants(tmp_path / "v", image_tree, split, b.minority, specs, 2.0)
    assert list(v) == ["B0_imbalanced_k2", "B1_class_weight_k2", "B2_ros_k2", "B3_augment_k2", "M6_dass_k2"]
    n_maj = len(split[b.majority]["train"])
    flags = {t: (x.class_weight, x.augment) for t, x in v.items()}
    assert flags == {"B0_imbalanced_k2": (False, False), "B1_class_weight_k2": (True, False),
                     "B2_ros_k2": (False, False), "B3_augment_k2": (False, True), "M6_dass_k2": (False, False)}
    for tag, n_min, prefix in [("B0_imbalanced_k2", len(real), None), ("B2_ros_k2", n_maj, DUP_PREFIX),
                               ("M6_dass_k2", n_maj, SYNTH_PREFIX)]:
        files = list((v[tag].dir / "train" / b.minority).iterdir())
        assert len(files) == n_min, tag                                          # B2, M*: cân bằng 1 : 1
        if prefix:
            assert len([f for f in files if f.name.startswith(prefix)]) == b.n_select
        assert len(list((v[tag].dir / "train" / b.majority).iterdir())) == n_maj   # lớp đa số: chỉ ảnh thật
        assert not [f for f in (v[tag].dir / "train" / b.majority).iterdir() if f.name.startswith(SYNTH_PREFIX)]
    assert_clean_eval_sets(v, image_tree, CLASSES)


@pytest.mark.parametrize("n_real,n_extra", [(281, 1120), (148, 470), (5, 3), (4, 8), (3, 0)])
def test_oversample_indices_balanced_and_deterministic(n_real, n_extra):
    idx = oversample_indices(n_real, n_extra, seed=2026)
    assert len(idx) == n_extra and idx == oversample_indices(n_real, n_extra, seed=2026)
    if n_extra:
        counts = [idx.count(i) for i in range(n_real)]
        assert max(counts) - min(counts) <= 1          # mọi ảnh thật được nhắc lại gần như đều nhau
    if n_extra % n_real:
        assert idx != oversample_indices(n_real, n_extra, seed=1)


def test_oversample_needs_real_images():
    with pytest.raises(ValueError):
        oversample_indices(0, 5, seed=0)
