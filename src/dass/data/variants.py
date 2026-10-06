"""Ghép thư mục train/val cho từng biến thể (ảnh thật theo split + ảnh sinh đã chọn, hoặc ảnh thật nhân bản cho
M0b). Val / test luôn 100 % ảnh thật."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .splits import Split

log = logging.getLogger(__name__)

SYNTH_PREFIX = "synth_"
DUP_PREFIX = "dup_"          # bản sao ảnh THẬT lớp thiểu số (M0b, oversampling)
_MARKER = ".complete.json"


def oversample_indices(n_real: int, n_extra: int, seed: int) -> list[int]:
    """Chỉ số các ảnh thật lớp thiểu số được nhân bản thêm, tổng cộng `n_extra` bản sao (M0b).

    Chia đều nhất có thể: mọi ảnh được lặp q = n_extra // n_real lần, cộng thêm r = n_extra % n_real ảnh (chọn
    ngẫu nhiên theo `seed`, không lặp) lặp thêm 1 lần -> số lần xuất hiện của các ảnh chênh nhau tối đa 1
    (ít biến thiên hơn lấy mẫu có hoàn lại, nhưng vẫn là random oversampling tới đúng 1 : 1).
    """
    if n_extra <= 0:
        return []
    if n_real <= 0:
        raise ValueError("Không có ảnh thật lớp thiểu số để nhân bản")
    q, r = divmod(n_extra, n_real)
    extra = np.random.default_rng(seed).choice(n_real, size=r, replace=False)
    return [i for i in range(n_real) for _ in range(q)] + sorted(int(i) for i in extra)


def synth_budget(n_select: int, fraction: float, pool_mult: float) -> tuple[int, int]:
    """(số ảnh sinh cần chọn, số ảnh ĐẦU của pool được dùng) khi chỉ `fraction` phần cần bù là ảnh sinh (giữ đúng k)."""
    n_synth = int(round(n_select * fraction))
    return n_synth, int(np.ceil(pool_mult * n_synth))


def balanced_synth_count(ratio: float, n_majority_real: int) -> int:
    """M8: số ảnh sinh thêm vào MỖI lớp = q × số ảnh thật lớp đa số."""
    return int(round(ratio * n_majority_real))


def ros_fill(selections: dict[str, list[str]], real_minority: list[str], n_select: int, seed: int,
             skip: set[str] | tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Ảnh THẬT lớp thiểu số nhân bản bù cho các biến thể có ít hơn `n_select` ảnh sinh (`selection.synth_fraction`
    < 1) -> mọi biến thể cùng tổng số ảnh. Biến thể không có ảnh sinh (M0, M0b) hoặc trong `skip` thì bỏ qua. Cùng
    `seed` -> mọi biến thể dùng CÙNG tập ảnh nhân bản (khác biệt chỉ còn ở ảnh sinh)."""
    out = {}
    for method, synth in selections.items():
        n_extra = n_select - len(synth)
        if synth and n_extra > 0 and method not in skip:
            out[method] = [real_minority[i] for i in oversample_indices(len(real_minority), n_extra, seed)]
    return out


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
               majority_synth: list[str], extra_real: list[str]) -> str:
    data = {"split": {c: {s: parts[s] for s in ("train", "val")} for c, parts in split.items()},
            "minority": minority, "synth": synth_paths}
    if majority_synth:
        data.update(majority=majority, majority_synth=majority_synth)
    if extra_real:
        data.update(extra_real=extra_real)
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def assemble_variant(out_dir: str | Path, train_pp_dir: str | Path, split: Split, minority: str,
                     synth_paths: list[str], majority: str | None = None,
                     majority_synth: list[str] | tuple[str, ...] = (),
                     extra_real: list[str] | tuple[str, ...] = ()) -> dict[str, dict[str, int]]:
    """out_dir/{train,val}/<lớp>/: ảnh thật theo split (không lấy phần test) + ảnh sinh vào
    train/<minority>/synth_XXXXX.png (và train/<majority>/ nếu có `majority_synth`) + bản sao ảnh thật lớp thiểu số
    vào train/<minority>/dup_XXXXX_<tên gốc> (`extra_real`, M0b).

    Bỏ qua nếu thư mục đã được ghép với đúng cùng đầu vào (đánh dấu bằng file .complete.json).
    """
    out_dir = Path(out_dir)
    majority_synth, extra_real = list(majority_synth), list(extra_real)
    if majority_synth and majority is None:
        raise ValueError("majority_synth cần tên lớp đa số")
    signature = _signature(split, minority, [Path(p).name for p in synth_paths], majority,
                           [Path(p).name for p in majority_synth], [Path(p).name for p in extra_real])
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
    for i, src in enumerate(extra_real):
        shutil.copy2(src, out_dir / "train" / minority / f"{DUP_PREFIX}{i:05d}_{Path(src).name}")

    counts = {s: {c: len(list((out_dir / s / c).iterdir())) for c in split} for s in ["train", "val"]}
    marker.write_text(json.dumps({"signature": signature, "counts": counts}))
    return counts


def prepare_variants(variants_root: str | Path, train_pp_dir: str | Path, split: Split, minority: str,
                     selections: dict[str, list[str]], pool_mult: float, lam: str,
                     class_weight_methods: set[str] | tuple[str, ...], majority: str | None = None,
                     majority_synth: dict[str, list[str]] | None = None,
                     extra_real: dict[str, list[str]] | None = None) -> dict[str, Variant]:
    """`selections`: {phương pháp: [ảnh sinh lớp thiểu số]}; `majority_synth`: {phương pháp: [ảnh sinh lớp đa số]};
    `extra_real`: {phương pháp: [ảnh thật lớp thiểu số được nhân bản thêm]} (M0b).

    Phương pháp trong `class_weight_methods` (baseline, biến thể có ảnh sinh ở cả hai lớp) train với class weight.
    """
    majority_synth, extra_real = majority_synth or {}, extra_real or {}
    variants = {}
    for method, synth_paths in selections.items():
        tag = variant_tag(method, pool_mult)
        out_dir = Path(variants_root) / tag
        counts = assemble_variant(out_dir, train_pp_dir, split, minority, synth_paths, majority,
                                  majority_synth.get(method, ()), extra_real.get(method, ()))
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
