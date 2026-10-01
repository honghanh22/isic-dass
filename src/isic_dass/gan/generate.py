"""Bước 4 — sinh Candidate Pool cho lớp thiểu số.

Pool được nén lên Drive; phiên sau giải nén lại đúng pool đó thay vì sinh lại,
để mọi lần chạy classifier dùng cùng một bộ ảnh.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from ..utils import list_images

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("_generate_worker.py")


def pool_tag(best_cum_kimg: int, pool_size: int) -> str:
    return f"from{best_cum_kimg}kimg_n{pool_size}"


def generate_images(repo_dir: str | Path, pkl: str | Path, out_dir: str | Path, class_idx: int, n_images: int,
                    psi: float, seed: int, batch: int = 16) -> list[str]:
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    cmd = [sys.executable, str(WORKER), str(repo_dir), str(pkl), str(out_dir),
           str(class_idx), str(n_images), str(psi), str(seed), str(batch)]
    log.info("Sinh ảnh (tiến trình con): %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-25:]
        if "/" in line or line.startswith("DONE"):
            log.info("    %s", line)
    proc.wait()
    if proc.returncode != 0:
        print("\n".join(tail))
        raise RuntimeError(f"Tiến trình sinh ảnh lỗi (mã {proc.returncode})")
    return list_images(out_dir)


def restore_pool(zip_path: str | Path, pool_dir: str | Path) -> list[str] | None:
    """Giải nén pool đã cache trên Drive. Trả về None nếu chưa có cache."""
    zip_path, pool_dir = Path(zip_path), Path(pool_dir)
    if not zip_path.exists():
        return None
    shutil.rmtree(pool_dir, ignore_errors=True)
    shutil.unpack_archive(zip_path, pool_dir)
    paths = list_images(pool_dir)
    log.info("Khôi phục %d ảnh từ %s", len(paths), zip_path)
    return paths


def archive_pool(pool_dir: str | Path, zip_path: str | Path) -> None:
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", pool_dir)
    log.info("Đã lưu %s", zip_path)
