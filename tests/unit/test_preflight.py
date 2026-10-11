"""`dass preflight`: báo thiếu GAN / split tham chiếu / E_d dùng lại trước khi chạy; nhận ra file M0b / M0c thừa."""

from conftest import CONFIGS

from dass.config import Layout, load_config
from dass.pipeline.preflight import preflight_checks, unexpected_predictions


def _cfg(tmp_path):
    return load_config([CONFIGS / "colab" / "rsna_pneumonia.yaml"],
                       [f"paths.drive_root={tmp_path / 'out'}", f"data.source.train_images={tmp_path / 'data'}"])


def test_preflight_lists_required_artifacts_of_the_colab_rsna_run(tmp_path):
    cfg = _cfg(tmp_path)
    checks = {c.what: c for c in preflight_checks(cfg)}
    assert set(checks) == {"dữ liệu gốc", "GAN (best.pkl)", "GAN (gan_state.json)", "split tham chiếu của GAN",
                           "E_d dùng lại (rsna_v3_srv)"}
    assert not any(c.ok for c in checks.values())                      # thư mục trống -> thiếu hết
    lay = Layout(cfg)
    for p in (lay.drive_train_images, lay.gan_best_pkl, lay.gan_state_json, lay.expected_split, lay.e_d_ckpt):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.mkdir() if p == lay.drive_train_images else p.write_bytes(b"x")
    assert all(c.ok for c in preflight_checks(cfg))
    assert lay.e_d_ckpt.parent.parent.name == "checkpoints_rsna_v3_srv"


def test_preflight_flags_stray_m0b_predictions(tmp_path):
    cfg = _cfg(tmp_path)
    pred = Layout(cfg).pred_dir
    pred.mkdir(parents=True)
    for name in ["EfficientNetV2B0__M0b_real_oversample_p1.5__s2026.npz", "EfficientNetV2B0__M6_dass_p1.5__s2026.npz"]:
        (pred / name).write_bytes(b"x")
    assert unexpected_predictions(cfg) == ["EfficientNetV2B0__M0b_real_oversample_p1.5__s2026.npz"]
