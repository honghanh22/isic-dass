"""Context dùng chung cho mọi stage: đọc nguồn -> nhận diện số kênh -> tiền xử lý -> chia -> ngân sách ảnh sinh.

Mọi bước idempotent (ảnh đã có thì bỏ qua), nên mỗi stage tự gọi `Context.create` mà không tốn thời gian lặp lại.
Mỗi lần chạy ghi `run_manifest.json` (config đã hợp nhất, phiên bản thư viện, thời điểm từng stage) để tái lập.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import __version__
from ..config import Config, Layout
from ..data.image_io import detect_channels
from ..data.manifest import build_dataset_card
from ..data.sources import build_source
from ..data.splits import (
    ClassBudget,
    Split,
    build_split,
    check_expected_split,
    compute_budget,
    load_or_create_split,
    materialize_subset,
    split_paths,
)
from ..data.transforms import preprocess_tree
from ..utils import code_version, list_images, package_versions, read_json, set_seed, write_json_atomic

log = logging.getLogger(__name__)


def record_stage(layout: Layout, cfg: Config, stage: str, **extra) -> None:
    """`run_manifest.json`: cấp trên cùng = lần ghi gần nhất (giữ tương thích); `stages[<stage>]` lưu RIÊNG commit
    code, phiên bản và cấu hình của từng stage (ví dụ `train:ResNet50`) -> biết mỗi kết quả được tạo bằng code nào."""
    manifest = read_json(layout.run_manifest, default={}) or {}
    code = code_version()
    manifest.update(dass_version=__version__, code_version=code, config=cfg.to_dict(), versions=package_versions(),
                    **extra)
    manifest.setdefault("stages", {})[stage] = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                "argv": sys.argv, "dass_version": __version__, "code_version": code,
                                                "config": cfg.to_dict()}
    write_json_atomic(layout.run_manifest, manifest)


def resolve_channels(cfg: Config, layout: Layout) -> int:
    """`data.channels: auto` -> nhận diện từ ảnh gốc (ảnh xám = 1 kênh, ảnh màu = 3 kênh)."""
    if cfg.data.channels != "auto":
        return int(cfg.data.channels)
    paths = [p for c in cfg.data.class_names for p in list_images(layout.train_raw / c, "*")]
    channels = detect_channels(paths, seed=cfg.seed)
    log.info("Nhận diện số kênh từ %d ảnh gốc: %d (%s)", len(paths), channels, "ảnh xám" if channels == 1 else "RGB")
    return channels


@dataclass
class Context:
    cfg: Config
    layout: Layout
    split: Split
    budget: ClassBudget
    channels: int

    @classmethod
    def create(cls, cfg: Config, stage: str) -> Context:
        set_seed(cfg.seed, cfg.deterministic)
        layout = Layout(cfg)
        layout.makedirs()
        d = cfg.data

        source = build_source(cfg, layout)
        source.ingest("train", layout.train_raw)
        if source.has_test_set:
            source.ingest("test", layout.test_raw)
        channels = resolve_channels(cfg, layout)

        pp = d.preprocess
        preprocess_tree(layout.train_raw, layout.train_pp, d.img_size, d.class_names, channels,
                        pp.crop_dark_border, pp.resize, d.force_grayscale)
        if source.has_test_set:
            preprocess_tree(layout.test_raw, layout.test_pp, d.img_size, d.class_names, channels,
                            pp.crop_dark_border, pp.resize, d.force_grayscale)

        split = load_or_create_split(
            layout.split_json, layout.train_pp,
            lambda: build_split(d.split, layout.train_pp, d.class_names, cfg.seed, source.has_test_set,
                                layout.split_file))
        if layout.expected_split is not None:
            check_expected_split(split, layout.expected_split)
        if source.has_test_set:
            test_counts = {c: len(list((layout.test_pp / c).iterdir())) for c in d.class_names}
        else:   # test tách từ cùng nguồn -> thư mục test = đúng phần "test" của split
            test_counts = materialize_subset(split, "test", layout.train_pp, layout.test_pp)

        budget = compute_budget(split, cfg.selection.pool_mult)
        log.info("Thiểu số = %s (%d) | đa số = %s (%d) | cần chọn %d | pool %d ảnh | %d kênh",
                 budget.minority, budget.n_real[budget.minority], budget.majority,
                 budget.n_real[budget.majority], budget.n_select, budget.pool_size, channels)
        write_json_atomic(layout.dataset_card, build_dataset_card(cfg, split, channels, budget, test_counts))
        record_stage(layout, cfg, stage, channels=channels)
        return cls(cfg, layout, split, budget, channels)

    @property
    def train_paths(self) -> dict[str, list[str]]:
        return split_paths(self.layout.train_pp, self.split, "train")

    @property
    def val_paths(self) -> dict[str, list[str]]:
        return split_paths(self.layout.train_pp, self.split, "val")

    def class_idx(self, cls: str) -> int:
        return self.cfg.data.classes[cls]
