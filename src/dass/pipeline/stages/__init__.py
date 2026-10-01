"""Các stage của pipeline. Mỗi stage đọc / ghi artefact trên Drive nên chạy độc lập và chạy lại được.

    prepare → gan → sample → [fingerprint] → select → train (mỗi model) → evaluate → report

Chạy mỗi stage trong một tiến trình riêng (CLI `dass run` tự làm) để PyTorch (GAN, Inception) và
TensorFlow (classifier) không tranh VRAM.
"""

from __future__ import annotations

import json

import pandas as pd

from ...config import Config, Layout
from ...utils import set_seed

PIPELINE = ("prepare", "gan", "sample", "select", "train", "evaluate", "report")
OPTIONAL = ("fingerprint",)


def init_tensorflow(cfg: Config) -> None:
    """Gọi đầu mọi stage dùng TensorFlow: bật memory growth rồi đặt seed (kể cả seed của TF)."""
    from ...data.loaders import configure_gpu

    configure_gpu(cfg.classifier.mixed_precision)
    set_seed(cfg.seed, cfg.deterministic)


def save_metrics(layout: Layout, name: str, df: pd.DataFrame) -> None:
    """Số liệu thô: `metrics/<name>.csv` + `metrics/<name>.json` (records)."""
    layout.metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(layout.metrics_dir / f"{name}.csv", index=False)
    records = json.loads(df.to_json(orient="records"))
    (layout.metrics_dir / f"{name}.json").write_text(json.dumps(records, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")


def load_metrics(layout: Layout, name: str) -> pd.DataFrame | None:
    path = layout.metrics_dir / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None
