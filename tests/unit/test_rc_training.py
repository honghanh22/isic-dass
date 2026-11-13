import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from dass.data.variants import assemble_variant, class_source_groups, count_real_train
from dass.evaluation.classification import max_mcc_threshold
from dass.evaluation.statistics import paired_bootstrap_auc


def test_max_mcc_threshold_beats_every_other_candidate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 200), 0, 1)
    t = max_mcc_threshold(y, p)
    best = matthews_corrcoef(y, p >= t)
    assert all(matthews_corrcoef(y, p >= c) <= best + 1e-12 for c in np.unique(p))


def test_max_mcc_threshold_tie_prefers_higher_sensitivity():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert max_mcc_threshold(y, p) == pytest.approx(0.8)        # mọi t ∈ (0.2, 0.8] cùng MCC = 1 -> t lớn nhất


def test_bootstrap_ap_metric():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(80), np.ones(20)].astype(int)
    good, bad = y + rng.normal(0, 0.5, 100), rng.random(100)
    obs, lo, hi, pv = paired_bootstrap_auc(y, good, bad, 300, 0, metric="pr_auc")
    assert obs > 0 and lo > 0 and pv < 0.05
    with pytest.raises(KeyError):
        paired_bootstrap_auc(y, good, bad, 10, 0, metric="nope")


@pytest.fixture
def split_dirs(tmp_path):
    pp = tmp_path / "pp"
    split = {}
    for c, n in (("neg", 6), ("pos", 2)):
        (pp / c).mkdir(parents=True)
        files = [f"{c}{i}.png" for i in range(n)]
        for f in files:
            (pp / c / f).write_bytes(b"x")
        split[c] = {"train": files[:-1], "val": files[-1:], "test": []}
    synth = {}
    for c in ("neg", "pos"):
        d = tmp_path / "pool" / c
        d.mkdir(parents=True)
        synth[c] = []
        for i in range(3):
            (d / f"s{i}.png").write_bytes(b"y")
            synth[c].append(str(d / f"s{i}.png"))
    return tmp_path, pp, split, synth


def test_assemble_synth_by_class_and_groups(split_dirs):
    root, pp, split, synth = split_dirs
    out = root / "v"
    counts = assemble_variant(out, pp, split, "pos", synth_by_class=synth)
    assert counts["train"] == {"neg": 5 + 3, "pos": 1 + 3} and counts["val"] == {"neg": 1, "pos": 1}
    groups = class_source_groups(out / "train", ["neg", "pos"])
    assert [(g[0], g[1], len(g[2])) for g in groups] == [(0, "real", 5), (0, "synth", 3), (1, "real", 1),
                                                       (1, "synth", 3)]
    assert count_real_train(out / "train", ["neg", "pos"]) == 6
    with pytest.raises(ValueError):
        assemble_variant(root / "w", pp, split, "pos", synth_paths=synth["pos"], synth_by_class=synth)


def test_groups_reject_missing_source(split_dirs):
    root, pp, split, synth = split_dirs
    out = root / "v2"
    assemble_variant(out, pp, split, "pos", synth_by_class={"pos": synth["pos"]})
    with pytest.raises(ValueError, match="rỗng"):
        class_source_groups(out / "train", ["neg", "pos"])
