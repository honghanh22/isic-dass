"""Bước 4c — spectral artifact mitigation (Dong et al., CVPR 2022), cài đặt theo Algorithm 1–3 của bài báo.

Thành phần tham khảo từ nghiên cứu trước để kiểm soát shortcut tổng hợp, không phải đóng góp của nghiên cứu này.
SDN (Algorithm 2) trừ phổ biên độ trung bình chênh lệch; PDC (Algorithm 3) khớp profile công suất với từ điển
ảnh thật. Phổ pha luôn giữ nguyên.

Giấy phép mã nguồn của tác giả: bảo lưu mọi quyền, cấm dùng vì mục đích thương mại.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm.auto import tqdm

from ..gan.generate import archive_pool, restore_pool
from ..utils import save_figure
from .spectrum import fft_channels, radial_index, radial_power_profile

log = logging.getLogger(__name__)

DONG_URL = "http://www.comp.polyu.edu.hk/~csajaykr/CVPR2022.zip"


def fetch_reference_code(dong_dir: str | Path) -> list[Path]:
    """Tải gói mã nguồn gốc để đối chiếu (không bắt buộc). Trả về danh sách file; lỗi mạng -> danh sách rỗng."""
    dong_dir = Path(dong_dir)
    dong_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dong_dir / "CVPR2022.zip"
    try:
        if not zip_path.exists():
            urllib.request.urlretrieve(DONG_URL, zip_path)
        if not any(p.is_dir() for p in dong_dir.iterdir()):
            shutil.unpack_archive(zip_path, dong_dir)
    except Exception as e:  # noqa: BLE001 — chỉ để tham khảo, không được làm hỏng pipeline
        log.warning("Không tải được mã nguồn Dong et al.: %r", e)
        return []
    files = sorted(p for p in dong_dir.rglob("*") if p.is_file())
    log.info("Mã nguồn tham khảo Dong et al.: %d file (%d Python, %d MATLAB) trong %s", len(files),
             sum(p.suffix == ".py" for p in files), sum(p.suffix == ".m" for p in files), dong_dir)
    return files


def mean_magnitude(paths: list[str], desc: str) -> np.ndarray:
    acc = None
    for p in tqdm(paths, desc=desc):
        mag, _ = fft_channels(p)
        acc = mag if acc is None else acc + mag
    return acc / len(paths)


def build_power_dictionary(paths: list[str]) -> np.ndarray:
    """Algorithm 1: profile công suất bán kính của từng ảnh thật, từng kênh. Shape (N, C, K)."""
    dic = []
    for p in tqdm(paths, desc="Từ điển công suất (Algorithm 1)"):
        mag, _ = fft_channels(p)
        dic.append(np.stack([radial_power_profile(mag[..., c]) for c in range(mag.shape[-1])]))
    return np.stack(dic)


def mitigate_image(path: str, delta: np.ndarray, power_dict: np.ndarray, mode: str = "sdn+pdc") -> np.ndarray:
    mag, phase = fft_channels(path)
    if "sdn" in mode:
        mag = np.clip(mag - delta, 0.0, None)
    if "pdc" in mode:
        idx = radial_index(mag.shape[0])
        K = power_dict.shape[-1]
        q = max(4, K // 4)
        for c in range(mag.shape[-1]):
            Ps = radial_power_profile(mag[..., c])
            r = int(np.argmin(((Ps[:q] - power_dict[:, c, :q]) ** 2).sum(axis=1)))
            mag[..., c] = mag[..., c] * (power_dict[r, c] / (Ps + 1e-12))[idx]
    out = np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * phase), axes=(0, 1)), axes=(0, 1)).real
    return np.clip(out, 0, 255).round().astype(np.uint8)


def mitigate_pool(paths: list[str], real_paths: list[str], zip_path: Path, out_dir: Path,
                  mode: str) -> tuple[list[str], np.ndarray | None]:
    """Trả về (đường dẫn ảnh sau mitigation, điểm SSIM trước/sau — None nếu khôi phục từ cache)."""
    cached = restore_pool(zip_path, out_dir)
    if cached is not None:
        return cached, None
    delta = mean_magnitude(paths, "Phổ TB ảnh sinh") - mean_magnitude(real_paths, "Phổ TB ảnh thật")
    power_dict = build_power_dictionary(real_paths)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    ssim_scores, new_paths = [], []
    for p in tqdm(paths, desc=f"Mitigation ({mode})"):
        fixed = mitigate_image(p, delta, power_dict, mode)
        ssim_scores.append(structural_similarity(np.asarray(Image.open(p).convert("RGB")), fixed,
                                                 channel_axis=2, data_range=255))
        q = out_dir / Path(p).name
        Image.fromarray(fixed).save(q, format="PNG")
        new_paths.append(str(q))
    archive_pool(out_dir, zip_path)
    scores = np.array(ssim_scores)
    log.info("SSIM trước/sau: TB %.4f (min %.4f, max %.4f) — bài báo ~0.93–0.98",
             scores.mean(), scores.min(), scores.max())
    return new_paths, scores


def plot_before_after(before: list[str], after: list[str], title: str, path: Path, n: int = 6) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n, figsize=(n * 2.0, 4.4))
    for c in range(n):
        axes[0, c].imshow(Image.open(before[c]))
        axes[1, c].imshow(Image.open(after[c]))
        for r in range(2):
            axes[r, c].axis("off")
    axes[0, 0].set_title("trước mitigation", loc="left", fontsize=10)
    axes[1, 0].set_title("sau mitigation", loc="left", fontsize=10)
    fig.suptitle(title)
    save_figure(fig, path)
