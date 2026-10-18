"""Ghép thư mục train/val cho từng phương pháp (thiết kế 2.0.0). Val / test luôn 100 % ảnh thật.

Train của mỗi phương pháp = ảnh thật theo split + (tuỳ phương pháp) bản sao ảnh thật lớp thiểu số (B2, ROS) hoặc ảnh
sinh lớp thiểu số đã chọn (M1–M6). Class weight (B1) và augmentation (B3) là thiết lập khi train, không đổi dữ liệu.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .splits import Split

log = logging.getLogger(__name__)

SYNTH_PREFIX = "synth_"
DUP_PREFIX = "dup_"          # bản sao ảnh THẬT lớp thiểu số (B2, random oversampling)
_MARKER = ".complete.json"


def oversample_indices(n_real: int, n_extra: int, seed: int) -> list[int]:
    """Chỉ số các ảnh thật lớp thiểu số được nhân bản thêm, tổng cộng `n_extra` bản sao (B2).

    Chia đều nhất có thể: mọi ảnh được lặp q = n_extra // n_real lần, cộng thêm r = n_extra % n_real ảnh (chọn
    ngẫu nhiên theo `seed`, không lặp) lặp thêm 1 lần -> số lần xuất hiện của các ảnh chênh nhau tối đa 1.
    """
    if n_extra <= 0:
        return []
    if n_real <= 0:
        raise ValueError("Không có ảnh thật lớp thiểu số để nhân bản")
    q, r = divmod(n_extra, n_real)
    extra = np.random.default_rng(seed).choice(n_real, size=r, replace=False)
    return [i for i in range(n_real) for _ in range(q)] + sorted(int(i) for i in extra)


@dataclass
class VariantSpec:
    """Đầu vào ghép tập train của một phương pháp."""
    synth: list[str] = field(default_factory=list)        # ảnh sinh lớp thiểu số (M1–M6)
    extra_real: list[str] = field(default_factory=list)   # bản sao ảnh thật lớp thiểu số (B2)
    class_weight: bool = False                            # B1
    augment: bool = False                                 # B3


@dataclass
class Variant:
    tag: str
    dir: Path
    method: str
    class_weight: bool
    augment: bool


def variant_tag(method: str, pool_mult: float) -> str:
    return f"{method}_k{pool_mult:g}"


def _signature(split: Split, minority: str, synth: list[str], extra_real: list[str]) -> str:
    data = {"split": {c: {s: parts[s] for s in ("train", "val")} for c, parts in split.items()},
            "minority": minority, "synth": synth, "extra_real": extra_real}
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def assemble_variant(out_dir: str | Path, train_pp_dir: str | Path, split: Split, minority: str,
                     synth_paths: list[str] | tuple[str, ...] = (),
                     extra_real: list[str] | tuple[str, ...] = ()) -> dict[str, dict[str, int]]:
    """out_dir/{train,val}/<lớp>/: ảnh thật theo split (không lấy phần test) + ảnh sinh vào
    train/<minority>/synth_XXXXX.png + bản sao ảnh thật vào train/<minority>/dup_XXXXX_<tên gốc>.

    Bỏ qua nếu thư mục đã được ghép với đúng cùng đầu vào (đánh dấu bằng file .complete.json).
    """
    out_dir = Path(out_dir)
    synth_paths, extra_real = list(synth_paths), list(extra_real)
    signature = _signature(split, minority, [Path(p).name for p in synth_paths], [Path(p).name for p in extra_real])
    marker = out_dir / _MARKER
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature:
        return json.loads(marker.read_text())["counts"]

    shutil.rmtree(out_dir, ignore_errors=True)
    for subset in ["train", "val"]:
        for label, parts in split.items():
            dst = out_dir / subset / label
            dst.mkdir(parents=True, exist_ok=True)
            for f in parts[subset]:
                shutil.copy2(Path(train_pp_dir) / label / f, dst / f)
    for i, src in enumerate(synth_paths):
        shutil.copy2(src, out_dir / "train" / minority / f"{SYNTH_PREFIX}{i:05d}.png")
    for i, src in enumerate(extra_real):
        shutil.copy2(src, out_dir / "train" / minority / f"{DUP_PREFIX}{i:05d}_{Path(src).name}")

    counts = {s: {c: len(list((out_dir / s / c).iterdir())) for c in split} for s in ["train", "val"]}
    marker.write_text(json.dumps({"signature": signature, "counts": counts}))
    return counts


def prepare_variants(variants_root: str | Path, train_pp_dir: str | Path, split: Split, minority: str,
                     specs: dict[str, VariantSpec], pool_mult: float) -> dict[str, Variant]:
    """Ghép thư mục cho mọi phương pháp trong `specs` (giữ thứ tự)."""
    variants = {}
    for method, spec in specs.items():
        tag = variant_tag(method, pool_mult)
        out_dir = Path(variants_root) / tag
        counts = assemble_variant(out_dir, train_pp_dir, split, minority, spec.synth, spec.extra_real)
        log.info("%s: %s%s%s", tag, counts, " | class weight" if spec.class_weight else "",
                 " | augmentation" if spec.augment else "")
        variants[tag] = Variant(tag=tag, dir=out_dir, method=method, class_weight=spec.class_weight,
                                augment=spec.augment)
    return variants


def assert_clean_eval_sets(variants: dict[str, Variant], test_dir: str | Path, class_names: list[str]) -> None:
    """Val và test phải 100 % ảnh thật."""
    for tag, v in variants.items():
        for c in class_names:
            bad = [f.name for f in (v.dir / "val" / c).iterdir() if f.name.startswith((SYNTH_PREFIX, DUP_PREFIX))]
            assert not bad, f"{tag}/val/{c}: có {len(bad)} ảnh sinh / nhân bản"
    for c in class_names:
        bad = [f.name for f in (Path(test_dir) / c).iterdir() if f.name.startswith((SYNTH_PREFIX, DUP_PREFIX))]
        assert not bad, f"test/{c}: có {len(bad)} ảnh sinh / nhân bản"
