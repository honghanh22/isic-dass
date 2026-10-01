"""Spectral mitigation (Dong et al., CVPR 2022) và power-profile detector đã bị loại bỏ hoàn toàn."""

import re

import pytest
from conftest import CONFIGS, ROOT

from dass.config import load_config

FORBIDDEN = re.compile(r"mitigat|spectral|radial_power|power_profile|dong_|\bDong\b|CVPR2022|SDN|PDC", re.IGNORECASE)


def _files():
    yield from (ROOT / "src").rglob("*.py")
    yield from CONFIGS.rglob("*.yaml")
    yield ROOT / "notebooks" / "colab_pipeline.ipynb"


def test_no_references_in_code_configs_or_runner():
    hits = []
    for path in _files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize("override", ["frequency.apply_spectral_mitigation=true", "frequency.mitigation_mode=sdn",
                                      "generator.apply_spectral_mitigation=true"])
def test_old_flags_are_rejected_clearly(override):
    with pytest.raises(KeyError, match="không hợp lệ"):
        load_config([CONFIGS / "experiments" / "brain_tumor_dass.yaml"], [override])


def test_modules_are_gone():
    assert not (ROOT / "src" / "dass" / "frequency").exists()
    with pytest.raises(ImportError):
        __import__("dass.frequency.mitigation")
