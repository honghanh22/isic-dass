"""Dataset card: mô tả bộ dữ liệu sau tiền xử lý và chia — đi kèm mọi kết quả để tái lập / báo cáo."""

from __future__ import annotations

from ..config import Config
from .splits import ClassBudget, Split, split_hash


def build_dataset_card(cfg: Config, split: Split, channels: int, budget: ClassBudget,
                       test_counts: dict[str, int]) -> dict:
    d = cfg.data
    counts = {subset: {c: len(parts.get(subset, [])) for c, parts in split.items()} for subset in ("train", "val")}
    counts["test"] = dict(test_counts)
    return {
        "name": d.name,
        "classes": d.classes,
        "channels": channels,
        "color": "grayscale" if channels == 1 or d.force_grayscale else "rgb",
        "force_grayscale": d.force_grayscale,
        "img_size": d.img_size,
        "source": d.source.type,
        "test_set": "separate" if d.source.has_test_set else "held out from the same pool",
        "preprocess": {"crop_dark_border": d.preprocess.crop_dark_border, "resize": d.preprocess.resize},
        "split": {"type": d.split.type, "val": d.split.val, "test": d.split.test,
                  "group_regex": d.split.group_regex or None, "seed": cfg.seed, "sha1": split_hash(split)},
        "counts": counts,
        "minority": budget.minority,
        "majority": budget.majority,
        "n_select": budget.n_select,
        "imbalance_ratio_train": round(budget.n_real[budget.majority] / max(1, budget.n_real[budget.minority]), 3),
    }
