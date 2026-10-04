""".gitignore không được chặn nhầm mã nguồn / cấu hình / test (đã từng xảy ra: "data/" chặn src/dass/data/)."""

import pytest
from conftest import ROOT

dulwich = pytest.importorskip("dulwich")


def test_no_source_file_is_gitignored():
    from dulwich.ignore import IgnoreFilterManager
    from dulwich.repo import Repo

    if not (ROOT / ".git").exists():
        pytest.skip("không phải git repo")
    ignore = IgnoreFilterManager.from_repo(Repo(str(ROOT)))
    ignored = []
    for top in ["src", "configs", "tests", "scripts", "docs"]:
        for path in (ROOT / top).rglob("*"):
            rel = path.relative_to(ROOT).as_posix()
            if "__pycache__" in rel or rel.endswith(".pyc") or ".egg-info" in rel:   # sản phẩm build (pip install -e .)
                continue
            if ignore.is_ignored(rel + ("/" if path.is_dir() else "")):
                ignored.append(rel)
    assert not ignored, f"Bị .gitignore chặn: {ignored}"
