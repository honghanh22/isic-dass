import pytest

from dass.analysis.shortcut import separability_auc, shortcut_table
from dass.models.classifiers import MODEL_NAMES, resolve_model_name
from dass.models.generator.patches import PATCHES, Patch, apply_patch
from dass.models.generator.sampler import pool_tag
from dass.models.generator.trainer import default_state, load_state


def test_apply_patch_is_idempotent(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a = 1\n")
    patch = Patch("m.py", "a = 1\n", "a = 2  # [PATCH]\n", "test")
    assert apply_patch(tmp_path, patch) == "patched"
    assert apply_patch(tmp_path, patch) == "already"
    f.write_text("b = 1\n")
    with pytest.raises(RuntimeError):
        apply_patch(tmp_path, patch)


def test_patches_well_formed():
    assert len(PATCHES) == 9
    for p in PATCHES:
        assert p.old != p.new and p.rel_path and p.reason


def test_pool_tag_keeps_rgb_names_and_separates_gray():
    assert pool_tag(1200, 1880) == "from1200kimg_n1880"           # tương thích pool ISIC cũ
    assert pool_tag(1200, 4480, channels=1) == "from1200kimg_n4480_c1"


def test_gan_state_defaults(tmp_path):
    assert load_state(tmp_path / "missing.json") == default_state()


def test_resolve_model_name():
    assert resolve_model_name("resnet50") == "ResNet50" and resolve_model_name("vit_b16") == "ViT-B16"
    assert len(MODEL_NAMES) == 6
    with pytest.raises(KeyError):
        resolve_model_name("vgg16")


def test_separability_and_shortcut_table(rng):
    a = rng.normal(0, 1, (60, 4))
    assert separability_auc(a, a + 5, seed=0) > 0.99
    real = {"neg": rng.normal(0, 1, (30, 4)), "pos": rng.normal(3, 1, (30, 4))}
    pool = rng.normal(3, 1, (40, 4))
    t = shortcut_table(real, pool, {"M0_real_only": [], "M6_dass": list(range(20))}, "pos", "neg", seed=0)
    assert list(t["method"]) == ["reference", "M6_dass"]
    assert t["auc_5fold"].iloc[0] > 0.9 > t["auc_5fold"].iloc[1]     # thật/sinh cùng phân phối -> khó tách


def test_copy_atomic_keeps_old_file_when_interrupted(tmp_path, monkeypatch):
    """Bị ngắt giữa lúc chép (hết giờ GPU) -> file đích cũ còn nguyên, không bị ghi dở."""
    import shutil

    import pytest

    from dass.utils import copy_atomic

    src, dst = tmp_path / "new.pkl", tmp_path / "out" / "latest.pkl"
    src.write_bytes(b"new" * 1000)
    copy_atomic(src, dst)
    assert dst.read_bytes() == src.read_bytes()

    def interrupted(s, d):
        open(d, "wb").write(b"par")
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copy2", interrupted)
    src.write_bytes(b"newer")
    with pytest.raises(KeyboardInterrupt):
        copy_atomic(src, dst)
    assert dst.read_bytes() == b"new" * 1000
