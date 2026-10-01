"""Ghép thư mục train/val cho từng biến thể (ảnh thật theo split + ảnh sinh đã chọn). Val / test luôn 100 % ảnh thật."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .splits import Split

log = logging.getLogger(__name__)

SYNTH_PREFIX = "synth_"
_MARKER = ".complete.json"


@dataclass
class Variant:
    tag: str
    dir: Path
    method: str
    lam: str
    feature_space: str
    class_weight: bool


def variant_tag(method: str, pool_mult: float) -> str:
    return f"{method}_p{pool_mult:g}"


def _signature(split: Split, minority: str, synth_paths: list[str], majority: str | None,
               majority_synth: list[str]) -> str:
    data = {"split": {c: {s: parts[s] for s in ("train", "val")} for c, parts in split.items()},
            "minority": minority, "synth": synth_paths}
    if majority_synth:
        data.update(majority=majority, majority_synth=majority_synth)
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def assemble_variant(out_dir: str | Path, train_pp_dir: str | Path, split: Split, minority: str,
                     synth_paths: list[str], majority: str | None = None,
                     majority_synth: list[str] | tuple[str, ...] = ()) -> dict[str, dict[str, int]]:
    """out_dir/{train,val}/<lớp>/: ảnh thật theo split (không lấy phần test) + ảnh sinh vào
    train/<minority>/synth_XXXXX.png (và train/<majority>/ nếu có `majority_synth`).

    Bỏ qua nếu thư mục đã được ghép với đúng cùng đầu vào (đánh dấu bằng file .complete.json).
    """
    out_dir = Path(out_dir)
    majority_synth = list(majority_synth)
    if majority_synth and majority is None:
        raise ValueError("majority_synth cần tên lớp đa số")
    signature = _signature(split, minority, [Path(p).name for p in synth_paths], majority,
                           [Path(p).name for p in majority_synth])
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
    for label, srcs in [(minority, synth_paths), (majority, majority_synth)]:
        for i, src in enumerate(srcs):
            shutil.copy2(src, out_dir / "train" / label / f"{SYNTH_PREFIX}{i:05d}.png")

    counts = {s: {c: len(list((out_dir / s / c).iterdir())) for c in split} for s in ["train", "val"]}
    marker.write_text(json.dumps({"signature": signature, "counts": counts}))
    return counts


def prepare_variants(variants_root: str | Path, train_pp_dir: str | Path, split: Split, minority: str,
                     selections: dict[str, list[str]], pool_mult: float, lam: str,
                     class_weight_methods: set[str] | tuple[str, ...], majority: str | None = None,
                     majority_synth: dict[str, list[str]] | None = None) -> dict[str, Variant]:
    """`selections`: {phương pháp: [ảnh sinh lớp thiểu số]}; `majority_synth`: {phương pháp: [ảnh sinh lớp đa số]}.

    Phương pháp trong `class_weight_methods` (baseline, biến thể có ảnh sinh ở cả hai lớp) train với class weight.
    """
    majority_synth = majority_synth or {}
    variants = {}
    for method, synth_paths in selections.items():
        tag = variant_tag(method, pool_mult)
        out_dir = Path(variants_root) / tag
        counts = assemble_variant(out_dir, train_pp_dir, split, minority, synth_paths, majority,
                                  majority_synth.get(method, ()))
        log.info("%s: %s", tag, counts)
        variants[tag] = Variant(tag=tag, dir=out_dir, method=method, lam=lam, feature_space="Ev+Ed",
                                class_weight=method in class_weight_methods)
    return variants


def assert_clean_eval_sets(variants: dict[str, Variant], test_dir: str | Path, class_names: list[str]) -> None:
    """Val và test phải 100 % ảnh thật."""
    for tag, v in variants.items():
        for c in class_names:
            bad = [f.name for f in (v.dir / "val" / c).iterdir() if f.name.startswith(SYNTH_PREFIX)]
            assert not bad, f"{tag}/val/{c}: có {len(bad)} ảnh sinh"
    for c in class_names:
        bad = [f.name for f in (Path(test_dir) / c).iterdir() if f.name.startswith(SYNTH_PREFIX)]
        assert not bad, f"test/{c}: có {len(bad)} ảnh sinh"
