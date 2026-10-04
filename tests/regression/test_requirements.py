"""requirements.txt / requirements-dev.txt phải khớp pyproject.toml (nguồn sự thật): không thiếu gói, cùng ràng buộc
phiên bản; torch / tensorflow cố ý KHÔNG có trong requirements.txt (Colab cài sẵn đúng bản CUDA)."""

import re

import pytest
from conftest import ROOT

tomllib = pytest.importorskip("tomllib")          # Python >= 3.11


def _name(spec: str) -> str:
    return re.split(r"[<>=!~\s\[]", spec.strip(), maxsplit=1)[0].lower()


def _requirements(path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec and not spec.startswith("-"):
            out[_name(spec)] = spec.replace(" ", "")
    return out


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_runtime_requirements_match_pyproject():
    proj, req = _pyproject(), _requirements(ROOT / "requirements.txt")
    for spec in proj["dependencies"]:
        assert req.get(_name(spec)) == spec.replace(" ", ""), spec      # cùng gói, cùng ràng buộc phiên bản
    extras = proj["optional-dependencies"]
    for spec in extras["gan"] + extras["tf"]:
        if _name(spec) in {"torch", "tensorflow"}:
            assert _name(spec) not in req, f"{spec}: Colab cài sẵn, không đưa vào requirements.txt"
        else:
            assert _name(spec) in req, spec


def test_dev_requirements_match_pyproject():
    proj, req = _pyproject(), _requirements(ROOT / "requirements-dev.txt")
    for spec in proj["dependencies"] + proj["optional-dependencies"]["dev"]:
        assert req.get(_name(spec)) == spec.replace(" ", ""), spec
