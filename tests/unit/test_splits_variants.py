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


def test_prepare_variants_class_weight_and_both_classes(image_tree, tmp_path):
    split = holdout_val(image_tree, CLASSES, 0.25, seed=7)
    synth = [str(p) for p in sorted((image_tree / "benign").iterdir())[:2]]
    maj = [str(p) for p in sorted((image_tree / "malignant").iterdir())[:3]]
    v = prepare_variants(tmp_path / "v", image_tree, split, "malignant",
                         {"M0_real_only": [], "M6_dass": synth, "M7_dass_both_classes": synth}, 4.0, "v1_d1",
                         class_weight_methods={"M0_real_only", "M7_dass_both_classes"}, majority="benign",
                         majority_synth={"M7_dass_both_classes": maj})
    assert v["M0_real_only_p4"].class_weight and not v["M6_dass_p4"].class_weight
    assert len(list((v["M7_dass_both_classes_p4"].dir / "train" / "benign").glob(f"{SYNTH_PREFIX}*"))) == 3
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


def test_oversample_variant_reaches_one_to_one_with_real_copies(image_tree, tmp_path):
    """M0b: lớp thiểu số = ảnh thật + bản sao ảnh thật, đúng bằng lớp đa số; không ảnh sinh, không class weight."""
    split = holdout_val(image_tree, CLASSES, 0.25, seed=7)
    budget = compute_budget(split, pool_mult=3.0)
    real = [str(image_tree / budget.minority / f) for f in split[budget.minority]["train"]]
    dupes = [real[i] for i in oversample_indices(len(real), budget.n_select, seed=7)]
    v = prepare_variants(tmp_path / "v", image_tree, split, budget.minority,
                         {"M0_real_only": [], "M0b_real_oversample": []}, 3.0, "v1_d1",
                         class_weight_methods={"M0_real_only"}, extra_real={"M0b_real_oversample": dupes})
    m0b = v["M0b_real_oversample_p3"]
    train_min = list((m0b.dir / "train" / budget.minority).iterdir())
    assert len(train_min) == len(split[budget.majority]["train"])                 # 1 : 1
    assert len([p for p in train_min if p.name.startswith(DUP_PREFIX)]) == budget.n_select
    assert not [p for p in train_min if p.name.startswith(SYNTH_PREFIX)]
    assert not m0b.class_weight and v["M0_real_only_p3"].class_weight
    assert len(list((v["M0_real_only_p3"].dir / "train" / budget.minority).iterdir())) == len(real)
    assert_clean_eval_sets(v, image_tree, CLASSES)
