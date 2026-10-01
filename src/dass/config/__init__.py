"""Cấu hình: schema (dataclass + kiểm tra), loader (YAML `_base_` + `--set` + `--tag`), paths (Layout)."""

from .loader import load_config, parse_override
from .paths import Layout
from .schema import Config, validate

__all__ = ["Config", "Layout", "load_config", "parse_override", "validate"]
