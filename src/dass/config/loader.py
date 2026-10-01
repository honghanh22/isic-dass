"""Nạp cấu hình: nhiều file YAML (mỗi file có thể kế thừa `_base_`) áp theo thứ tự, rồi `--set`, rồi `--tag`.

    dass -c configs/experiments/brain_tumor_dass.yaml -c configs/experiments/smoke.yaml \
         --set selection.gamma=0.25 --tag gamma025

Giá trị lá (kể cả dict như `data.classes`) được THAY THẾ, không gộp; nhóm (dataclass) được áp đệ quy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .schema import Config, validate

BASE_KEY = "_base_"


def _apply(obj: Any, data: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Khoá cấu hình không hợp lệ: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} phải là một nhóm (dict), nhận {type(value).__name__}")
            _apply(current, value, f"{prefix}{key}.")
        else:
            # PyYAML đọc '1e-5' thành chuỗi -> ép về kiểu của giá trị mặc định
            if isinstance(current, float) and isinstance(value, (int, str)) and not isinstance(value, bool):
                value = float(value)
            setattr(obj, key, value)


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: nội dung YAML phải là một dict")
    return data


def apply_file(cfg: Config, path: str | Path, _stack: tuple[Path, ...] = ()) -> None:
    """Áp một file YAML: các file `_base_` (đường dẫn tương đối so với file này) trước, nội dung file sau."""
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"Kế thừa vòng tròn: {' -> '.join(map(str, (*_stack, path)))}")
    data = _read_yaml(path)
    bases = data.pop(BASE_KEY, [])
    for base in [bases] if isinstance(bases, str) else bases:
        apply_file(cfg, path.parent / base, (*_stack, path))
    _apply(cfg, data)


def parse_override(item: str) -> dict[str, Any]:
    """'generator.batch=32' -> {'generator': {'batch': 32}}. Giá trị parse bằng YAML (số, bool, list, ...)."""
    import yaml

    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"Ghi đè phải có dạng section.key=value, nhận: {item!r}")
    parts = key.strip().split(".")
    node: dict[str, Any] = {}
    cursor = node
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = yaml.safe_load(raw)
    return node


def load_config(paths: list[str | Path] | tuple[str | Path, ...] = (), overrides: list[str] | tuple[str, ...] = (),
                tag: str | None = None, check: bool = True) -> Config:
    cfg = Config()
    for p in paths:
        apply_file(cfg, p)
    for item in overrides:
        _apply(cfg, parse_override(item))
    if tag:
        cfg.paths.base_run_tag = cfg.paths.run_tag
        cfg.paths.run_tag = f"{cfg.paths.run_tag}_{tag}"
    if check:
        validate(cfg)
    return cfg
