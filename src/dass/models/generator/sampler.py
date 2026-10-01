"""Sinh candidate pool trong tiến trình con; cache zip trên Drive để mọi lần chạy dùng đúng cùng một bộ ảnh."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from ...data.image_io import ChannelMismatchError
from ...utils import list_images

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("_sample_worker.py")
EXIT_CHANNEL_MISMATCH = 3


def pool_tag(best_cum_kimg: int, n_images: int, channels: int = 3, force_gray: bool = False) -> str:
    """Tag cache của pool. Pool 1 kênh / ảnh đã ép xám có hậu tố riêng để không lẫn với pool cũ."""
    suffix = "_c1" if channels == 1 else ("_gray" if force_gray else "")
    return f"from{best_cum_kimg}kimg_n{n_images}{suffix}"


def generate_images(repo_dir: str | Path, pkl: str | Path, out_dir: str | Path, class_idx: int, n_images: int,
                    psi: float, seed: int, channels: int, tolerance: int = 0, force_gray: bool = False,
                    batch: int = 16) -> list[str]:
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    cmd = [sys.executable, str(WORKER), str(repo_dir), str(pkl), str(out_dir), str(class_idx), str(n_images),
           str(psi), str(seed), str(batch), str(channels), str(tolerance), str(int(force_gray))]
    log.info("Sinh ảnh (tiến trình con): %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-25:]
        if "/" in line or line.startswith(("DONE", "CHANNEL_MISMATCH")):
            log.info("    %s", line)
    proc.wait()
    if proc.returncode == EXIT_CHANNEL_MISMATCH:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise ChannelMismatchError(
            "Generator 3 kênh sinh ảnh có các kênh KHÁC nhau cho bộ dữ liệu ảnh xám -> không gộp về 1 kênh được mà "
            f"không mất thông tin ({tail[-1]}). Train lại GAN trên ảnh 1 kênh (đổi paths.gan_tag, chạy `dass gan`).")
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


def match_resize_chain(paths: list[str], out_dir: Path, zip_path: Path, img_size: int, ref_side: int,
                       channels: int) -> list[str]:
    """Cho ảnh sinh đi qua cùng chuỗi nội suy bilinear (ref_side -> img_size) với ảnh thật. Giữ đúng số kênh."""
    from ...data.image_io import read_image, write_png
    from ...data.transforms import resize

    cached = restore_pool(zip_path, out_dir)
    if cached is not None:
        return cached
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    new_paths = []
    for p in paths:
        q = out_dir / Path(p).name
        write_png(resize(resize(read_image(p, channels), ref_side), img_size), q)
        new_paths.append(str(q))
    archive_pool(out_dir, zip_path)
    return new_paths
