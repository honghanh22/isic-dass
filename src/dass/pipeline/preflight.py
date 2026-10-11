"""`dass preflight`: kiểm tra TRƯỚC khi chạy (không cần GPU / TF / torch) — artefact bắt buộc của lần chạy có sẵn chưa
(dữ liệu gốc, GAN dùng lại, split tham chiếu, E_d dùng lại) và thư mục dự đoán đang có gì (số lần chạy, file của
phương pháp không thuộc thiết kế hiện tại)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config, Layout


@dataclass
class Check:
    what: str
    path: Path
    ok: bool
    required: bool = True


def preflight_checks(cfg: Config) -> list[Check]:
    lay = Layout(cfg)
    checks = [Check("dữ liệu gốc", lay.drive_train_images, lay.drive_train_images.exists()),
              Check("GAN (best.pkl)", lay.gan_best_pkl, lay.gan_best_pkl.exists()),
              Check("GAN (gan_state.json)", lay.gan_state_json, lay.gan_state_json.exists())]
    if lay.expected_split is not None:
        checks.append(Check("split tham chiếu của GAN", lay.expected_split, lay.expected_split.exists()))
    if cfg.encoder.e_d_from_run and lay.e_d_run != cfg.paths.run_tag:
        checks.append(Check(f"E_d dùng lại ({lay.e_d_run})", lay.e_d_ckpt, lay.e_d_ckpt.exists()))
    return checks


def unexpected_predictions(cfg: Config) -> list[str]:
    """File dự đoán của phương pháp mà cấu hình hiện tại không train (vd M0b / M0c khi `oversample_variant` tắt) — để
    lại sẽ lẫn vào bảng; chuyển sang thư mục khác (không xoá) trước khi `evaluate`."""
    pred_dir = Layout(cfg).pred_dir
    if cfg.selection.oversample_variant or not pred_dir.exists():
        return []
    return sorted(p.name for p in pred_dir.glob("*.npz") if "__M0b_" in p.name or "__M0c_" in p.name)


def run(cfg: Config) -> bool:
    lay = Layout(cfg)
    checks = preflight_checks(cfg)
    print(f"Lần chạy: {cfg.paths.run_tag} | thiết kế {cfg.selection.design} | model {cfg.classifier.models} | "
          f"seed {cfg.classifier.seeds}")
    for c in checks:
        print(f"  [{'OK ' if c.ok else 'THIẾU'}] {c.what}: {c.path}")
    n_pred = len(list(lay.pred_dir.glob("*.npz"))) if lay.pred_dir.exists() else 0
    print(f"  Dự đoán đã có trong {lay.pred_dir}: {n_pred} file (sẽ được bỏ qua nếu đúng giao thức)")
    extra = unexpected_predictions(cfg)
    if extra:
        print(f"  [CHÚ Ý] {len(extra)} file M0b / M0c không thuộc thiết kế hiện tại -> chuyển sang "
              f"{lay.results_dir / 'predictions_archived_M0b_M0c'} (không xoá) trước `evaluate`: {extra[:6]}")
    missing = [c for c in checks if c.required and not c.ok]
    print("=> Sẵn sàng chạy." if not missing else f"=> THIẾU {len(missing)} mục bắt buộc — bổ sung trước khi chạy.")
    return not missing
