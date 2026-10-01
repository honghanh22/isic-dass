"""Tiện ích dùng chung: logging, seed / tính tất định, chạy lệnh, JSON ghi nguyên tử, vector, hình, phiên bản."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S", force=True)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Đặt seed cho random, numpy và (nếu đã import) TensorFlow / PyTorch.

    `deterministic` bật op tất định của TensorFlow (chậm hơn; vài op GPU không hỗ trợ sẽ báo lỗi).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def run_command(cmd: str | list[str], cwd: str | Path | None = None, tail: int = 3000) -> subprocess.CompletedProcess:
    """Chạy lệnh, in phần cuối stdout; báo lỗi kèm stderr nếu mã thoát khác 0."""
    shown = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    log.info("$ %s", shown)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout[-tail:])
    if r.returncode != 0:
        print(r.stderr[-tail:])
        raise RuntimeError(f"Lệnh lỗi (mã {r.returncode}): {shown}")
    return r


def add_to_sys_path(path: str | Path) -> None:
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """Ghi ra file tạm rồi đổi tên -> không để lại JSON hỏng nếu Colab ngắt giữa chừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Không ghi JSON được kiểu {type(o).__name__}")


def list_images(directory: str | Path, pattern: str = "*.png") -> list[str]:
    return sorted(str(p) for p in Path(directory).glob(pattern) if p.is_file())


def minmax(x: np.ndarray) -> np.ndarray:
    """Chuẩn hoá về [0, 1]; vector hằng trả về 0.5 để không thiên vị."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return np.full_like(x, 0.5) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def save_figure(fig, path: str | Path, dpi: int = 120) -> Path:
    """Lưu hình rồi đóng (pipeline chạy qua CLI, không hiển thị trực tiếp)."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Đã lưu hình: %s", path)
    return path


def package_versions() -> dict[str, str]:
    """Phiên bản các thư viện chính (ghi vào run_manifest để tái lập)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for name in ["dass", "numpy", "pandas", "scikit-learn", "scikit-image", "tensorflow", "keras", "torch", "pillow"]:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out
