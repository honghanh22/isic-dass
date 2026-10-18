"""`dass preflight`: kiểm tra TRƯỚC khi chạy (không cần GPU / TF / torch) — artefact bắt buộc của lần chạy có sẵn chưa
(dữ liệu gốc, GAN dùng lại, split tham chiếu, E_d dùng lại) và thư mục dự đoán đang có gì (số lần chạy, file của
phương pháp không thuộc thiết kế hiện tại).

`--fix`: file bắt buộc bị thiếu nhưng có ĐÚNG MỘT bản trong thư mục cùng tên mà Google Drive tạo khi trùng tên
("checkpoints_rsna_v2 (1)") hoặc ở thư mục cha của drive_root (bố cục cũ) -> chép bản đó về đúng chỗ. Không tạo file
mới, không chọn khi có nhiều bản.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config, Layout


@dataclass
class Check:
    what: str
    path: Path
    ok: bool
    required: bool = True
    error: str = ""        # không đọc được đường dẫn (vd Google Drive mất kết nối: Errno 103 / 107)
    candidates: list[Path] = field(default_factory=list)   # bản cùng tên ở chỗ khác (khi thiếu)


def _check(what: str, path: Path) -> Check:
    try:
        return Check(what, path, path.exists())
    except OSError as e:   # Drive FUSE ngắt giữa chừng -> báo rõ thay vì dừng cả lệnh
        return Check(what, path, False, error=f"{type(e).__name__}: {e.strerror or e}")


def relocated_copies(path: Path, drive_root: Path) -> list[Path]:
    """Bản của `path` (= drive_root/<thư mục>/<phần còn lại>) nằm trong thư mục cùng tên hoặc "<tên> (N)" (Drive tự
    đổi tên khi trùng), ở drive_root hoặc thư mục cha của nó. Chỉ duyệt các thư mục con trực tiếp (Drive chậm)."""
    try:
        rel = path.relative_to(drive_root)
    except ValueError:
        return []
    top, rest = rel.parts[0], Path(*rel.parts[1:])
    found = []
    for root in (drive_root, drive_root.parent):
        try:
            children = [d for d in root.iterdir() if d.is_dir()]
        except OSError:
            continue
        for d in children:
            if d.name == top or d.name.startswith(f"{top} ("):
                cand = d / rest
                try:
                    if cand != path and cand.is_file():
                        found.append(cand)
                except OSError:
                    continue
    return sorted(found)


def preflight_checks(cfg: Config) -> list[Check]:
    lay = Layout(cfg)
    checks = [_check("dữ liệu gốc", lay.drive_train_images),
              _check("GAN (best.pkl)", lay.gan_best_pkl),
              _check("GAN (gan_state.json)", lay.gan_state_json)]
    if lay.expected_split is not None:
        checks.append(_check("split tham chiếu của GAN", lay.expected_split))
    if cfg.encoder.e_d_from_run and lay.e_d_run != cfg.paths.run_tag:
        checks.append(_check(f"E_d dùng lại ({lay.e_d_run})", lay.e_d_ckpt))
    root = Path(cfg.paths.drive_root)
    for c in checks:
        if not c.ok and not c.error and c.path != lay.drive_train_images:
            c.candidates = relocated_copies(c.path, root)
    return checks


def restore(check: Check) -> bool:
    """Chép bản duy nhất tìm được về đúng chỗ. Nhiều bản -> không đoán, để người dùng chọn."""
    if check.ok or len(check.candidates) != 1:
        return False
    check.path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(check.candidates[0], check.path)
    check.ok = True
    return True


def run(cfg: Config, fix: bool = False) -> bool:
    lay = Layout(cfg)
    checks = preflight_checks(cfg)
    print(f"Lần chạy: {cfg.paths.run_tag} | k = {cfg.selection.pool_mult:g} | model {cfg.classifier.models} | "
          f"seed {cfg.classifier.seeds} | E_d: {cfg.encoder.e_d_from_run or 'train trong lần chạy này'}")
    for c in checks:
        if fix and restore(c):
            print(f"  [ĐÃ KHÔI PHỤC] {c.what}: chép {c.candidates[0]} -> {c.path}")
            continue
        state = "OK " if c.ok else ("LỖI ĐỌC" if c.error else "THIẾU")
        print(f"  [{state}] {c.what}: {c.path}" + (f" — {c.error}" if c.error else ""))
        for cand in c.candidates:
            print(f"      có bản ở: {cand}" + ("" if fix else "  (chạy `preflight --fix` để chép về đúng chỗ)"))
    if any(c.error for c in checks):
        print("  => Không đọc được Drive (mất kết nối). Chạy lại ô Mount Drive với "
              "drive.mount('/content/drive', force_remount=True) rồi chạy lại ô này.")
        return False
    n_pred = len(list(lay.pred_dir.glob("*.npz"))) if lay.pred_dir.exists() else 0
    print(f"  Dự đoán đã có trong {lay.pred_dir}: {n_pred} file (sẽ được bỏ qua nếu đúng giao thức)")
    missing = [c for c in checks if c.required and not c.ok]
    if not missing:
        print("=> Sẵn sàng chạy.")
        return True
    print(f"=> THIẾU {len(missing)} mục bắt buộc — bổ sung trước khi chạy:")
    for c in missing:
        if "split tham chiếu" in c.what:
            print(f"   - {c.what}: KHÔNG tạo lại được (là bằng chứng GAN chưa thấy ảnh val / test). Chép đúng file này "
                  f"từ server (cùng đường dẫn tương đối dưới thư mục kết quả) vào {c.path}")
        else:
            print(f"   - {c.what}: đặt file vào {c.path}")
    return False
