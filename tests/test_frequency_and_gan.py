from pathlib import Path

import numpy as np
from PIL import Image

from isic_dass.frequency.harmonize import harmonize_pool, resolve_harmonize_channels
from isic_dass.frequency.mitigation import mitigate_image
from isic_dass.frequency.spectrum import channel_gap, radial_power_profile
from isic_dass.gan.generate import pool_tag
from isic_dass.gan.setup import PATCHES, Patch, apply_patch
from isic_dass.gan.train import default_state, load_state


def test_radial_power_profile_normalized_to_dc(rng):
    prof = radial_power_profile(np.abs(rng.normal(size=(32, 32))) + 1)
    assert prof[0] == 1.0 or abs(prof[0] - 1.0) < 1e-9
    assert (prof >= 0).all()


def test_channel_gap_and_auto_harmonize(tmp_path, rng):
    gray = tmp_path / "gray.png"
    g = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    Image.fromarray(np.stack([g] * 3, -1)).save(gray)
    color = tmp_path / "color.png"
    Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(color)
    assert channel_gap(gray) == 0.0 and channel_gap(color) > 0
    assert resolve_harmonize_channels("auto", [str(gray)]) is True
    assert resolve_harmonize_channels("auto", [str(color)]) is False
    assert resolve_harmonize_channels(True, [str(color)]) is True


def test_harmonize_pool_grayscale_and_cache(tmp_path, rng):
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(3):
        p = src / f"synth_{i:05d}.png"
        Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    zip_path, out = tmp_path / "cache" / "h.zip", tmp_path / "h"
    new = harmonize_pool(paths, zip_path, out, 8, to_grayscale=True, resize_ref_side=None)
    assert zip_path.exists() and all(channel_gap(p) == 0 for p in new)
    restored = harmonize_pool(paths, zip_path, out, 8, True, None)   # lần 2: khôi phục từ zip
    assert [Path(p).name for p in restored] == [f"synth_{i:05d}.png" for i in range(3)]


def test_mitigation_identity_when_nothing_to_remove(tmp_path, rng):
    p = tmp_path / "x.png"
    img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(img).save(p)
    out = mitigate_image(str(p), delta=np.zeros((16, 16, 3)), power_dict=np.zeros((1, 3, 23)), mode="sdn")
    np.testing.assert_array_equal(out, img)


def test_apply_patch_is_idempotent(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a = 1\n")
    patch = Patch("m.py", "a = 1\n", "a = 2  # [PATCH]\n", "test")
    assert apply_patch(tmp_path, patch) == "patched"
    assert apply_patch(tmp_path, patch) == "already"
    assert f.read_text() == "a = 2  # [PATCH]\n"
    f.write_text("b = 1\n")
    try:
        apply_patch(tmp_path, patch)
        raise AssertionError("phải báo lỗi khi không tìm thấy đoạn cần vá")
    except RuntimeError:
        pass


def test_patches_are_well_formed():
    for p in PATCHES:
        assert p.old != p.new and p.rel_path and p.reason


def test_gan_state_defaults(tmp_path):
    assert load_state(tmp_path / "missing.json") == default_state()
    assert pool_tag(1200, 1880) == "from1200kimg_n1880"
