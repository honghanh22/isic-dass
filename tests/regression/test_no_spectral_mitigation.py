"""Spectral mitigation (Dong et al., CVPR 2022) và power-profile detector đã bị loại bỏ hoàn toàn."""

import json
import re

import pytest
from conftest import CONFIGS, ROOT

from dass.config import load_config

FORBIDDEN = re.compile(r"mitigat|spectral|radial_power|power_profile|dong_|\bDong\b|CVPR2022|SDN|PDC", re.IGNORECASE)


def _sources():
    """(nhãn, dòng) của code / config và MÃ NGUỒN các ô notebook (không quét output: ảnh base64 khớp ngẫu nhiên)."""
    for path in [*(ROOT / "src").rglob("*.py"), *CONFIGS.rglob("*.yaml"), ROOT / "scripts" / "build_colab_notebook.py"]:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            yield f"{path.relative_to(ROOT)}:{i}", line
    nb = json.loads((ROOT / "notebooks" / "colab_pipeline.ipynb").read_text(encoding="utf-8"))
    for c, cell in enumerate(nb["cells"]):
        for i, line in enumerate("".join(cell["source"]).splitlines(), 1):
            yield f"colab_pipeline.ipynb cell {c}:{i}", line


def test_no_references_in_code_configs_or_runner():
    hits = [f"{where}: {line.strip()}" for where, line in _sources() if FORBIDDEN.search(line)]
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
