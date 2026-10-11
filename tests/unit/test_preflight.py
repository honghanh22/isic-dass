"""`dass preflight`: báo thiếu GAN / split tham chiếu / E_d dùng lại trước khi chạy; `--fix` khôi phục bản nằm trong
thư mục trùng tên do Drive tạo; nhận ra file M0b / M0c thừa; không dừng khi Drive mất kết nối."""

from pathlib import Path

from conftest import CONFIGS

from dass.config import Layout, load_config
from dass.pipeline import preflight
from dass.pipeline.preflight import preflight_checks, unexpected_predictions


def _cfg(tmp_path, *overrides):
    return load_config([CONFIGS / "colab" / "rsna_pneumonia.yaml"],
                       [f"paths.drive_root={tmp_path / 'out'}", f"data.source.train_images={tmp_path / 'data'}",
                        *overrides])


def test_colab_rsna_trains_its_own_e_d(tmp_path):
    """Config Colab mặc định tự train E_d -> preflight không đòi file E_d của server."""
    checks = {c.what for c in preflight_checks(_cfg(tmp_path))}
    assert checks == {"dữ liệu gốc", "GAN (best.pkl)", "GAN (gan_state.json)", "split tham chiếu của GAN"}


def test_preflight_lists_required_artifacts_when_reusing_e_d(tmp_path):
    cfg = _cfg(tmp_path, "encoder.e_d_from_run=rsna_v3_srv")
    checks = {c.what: c for c in preflight_checks(cfg)}
    assert "E_d dùng lại (rsna_v3_srv)" in checks and not any(c.ok for c in checks.values())
    lay = Layout(cfg)
    for p in (lay.drive_train_images, lay.gan_best_pkl, lay.gan_state_json, lay.expected_split, lay.e_d_ckpt):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.mkdir() if p == lay.drive_train_images else p.write_bytes(b"x")
    assert all(c.ok for c in preflight_checks(cfg))
    assert lay.e_d_ckpt.parent.parent.name == "checkpoints_rsna_v3_srv"


def test_fix_restores_a_single_copy_from_a_drive_duplicate_folder(tmp_path):
    cfg = _cfg(tmp_path)
    lay = Layout(cfg)
    dup = lay.expected_split.parents[1].with_name("checkpoints_rsna_v2 (1)") / "data" / "real_val_split.json"
    dup.parent.mkdir(parents=True)
    dup.write_text('{"negative": {}}')
    check = next(c for c in preflight_checks(cfg) if "split" in c.what)
    assert not check.ok and check.candidates == [dup]
    assert preflight.restore(check) and lay.expected_split.read_text() == '{"negative": {}}'


def test_fix_does_not_guess_between_several_copies(tmp_path):
    cfg = _cfg(tmp_path)
    lay = Layout(cfg)
    for name in ("checkpoints_rsna_v2 (1)", "checkpoints_rsna_v2 (2)"):
        f = lay.expected_split.parents[1].with_name(name) / "data" / "real_val_split.json"
        f.parent.mkdir(parents=True)
        f.write_text("{}")
    check = next(c for c in preflight_checks(cfg) if "split" in c.what)
    assert len(check.candidates) == 2 and not preflight.restore(check) and not lay.expected_split.exists()


def test_other_runs_splits_are_never_used_as_reference(tmp_path):
    """Split của lần chạy khác (vd rsna_v4 vừa tạo) KHÔNG phải bản của rsna_v2 -> không được đề xuất."""
    cfg = _cfg(tmp_path)
    lay = Layout(cfg)
    lay.split_json.parent.mkdir(parents=True)
    lay.split_json.write_text("{}")
    assert next(c for c in preflight_checks(cfg) if "split" in c.what).candidates == []


def test_preflight_reports_unreadable_drive_instead_of_crashing(tmp_path, monkeypatch):
    """Drive FUSE ngắt (Errno 103 khi stat) -> mục đó "LỖI ĐỌC", không làm hỏng cả lệnh."""
    cfg = _cfg(tmp_path)
    real_exists = Path.exists

    def flaky(self, *a, **k):
        if "data" in self.parts:
            raise ConnectionAbortedError(103, "Software caused connection abort")
        return real_exists(self, *a, **k)

    monkeypatch.setattr(Path, "exists", flaky)
    checks = {c.what: c for c in preflight.preflight_checks(cfg)}
    assert checks["dữ liệu gốc"].error and not checks["dữ liệu gốc"].ok
    assert not preflight.run(cfg)


def test_preflight_flags_stray_m0b_predictions(tmp_path):
    cfg = _cfg(tmp_path)
    pred = Layout(cfg).pred_dir
    pred.mkdir(parents=True)
    for name in ["EfficientNetV2B0__M0b_real_oversample_p1.5__s2026.npz", "EfficientNetV2B0__M6_dass_p1.5__s2026.npz"]:
        (pred / name).write_bytes(b"x")
    assert unexpected_predictions(cfg) == ["EfficientNetV2B0__M0b_real_oversample_p1.5__s2026.npz"]
