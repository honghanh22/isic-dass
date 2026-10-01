import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dass.evaluation import reporting
from dass.evaluation.aggregate import compare_to_baseline, load_all_runs, summary_stats
from dass.evaluation.classification import binary_metrics
from dass.evaluation.generative import compute_diversity, compute_fid, compute_ssim, kid_from_features, kid_with_std
from dass.evaluation.statistics import paired_bootstrap_auc


def test_binary_metrics_known_values():
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.6]))
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["g_mean"] == pytest.approx(0.5) and m["roc_auc"] == pytest.approx(0.75)


def test_paired_bootstrap(rng):
    y = np.r_[np.ones(40), np.zeros(60)].astype(int)
    p = rng.random(100)
    assert paired_bootstrap_auc(y, p, p, 50, 0) == (0.0, 0.0, 0.0, 1.0)
    obs, lo, _, pv = paired_bootstrap_auc(y, y + rng.normal(0, 0.3, 100), p, 200, 0)
    assert obs > 0.3 and lo > 0 and pv < 0.05


def _naive_kid(real, fake, m):
    n = real.shape[1]
    k = lambda a, b: (a @ b.T / n + 1) ** 3  # noqa: E731
    kxx, kyy, kxy = k(fake, fake), k(real, real), k(fake, real)
    return ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()


def test_kid_matches_unbiased_mmd(rng):
    real, fake = rng.normal(size=(30, 8)), rng.normal(1, 1, size=(30, 8))
    assert kid_from_features(real, fake, 3, 30, 0) == pytest.approx(_naive_kid(real, fake, 30))
    mean, std = kid_with_std(real, fake, 3, 30, 0)
    assert mean == pytest.approx(_naive_kid(real, fake, 30)) and std == pytest.approx(0, abs=1e-12)


def test_kid_orders_distributions(rng):
    real = rng.normal(size=(200, 16))
    near, far = rng.normal(size=(200, 16)), rng.normal(0.5, 1, size=(200, 16))
    assert kid_from_features(real, near, 10, 100, 0) < kid_from_features(real, far, 10, 100, 0)


def test_fid_and_diversity(rng):
    a = rng.normal(size=(100, 4))
    assert compute_fid(a, a) == pytest.approx(0, abs=1e-6)
    assert compute_diversity(np.tile([[1.0, 0.0]], (5, 1))) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("channels,mode", [(1, "L"), (3, "RGB")])
def test_ssim_on_native_channels(tmp_path, rng, channels, mode):
    paths = []
    for i in range(4):
        shape = (16, 16) if channels == 1 else (16, 16, 3)
        Image.fromarray(rng.integers(0, 255, shape, dtype=np.uint8)).save(tmp_path / f"{i}.png")
        paths.append(str(tmp_path / f"{i}.png"))
    assert -1 <= compute_ssim(paths, channels, n_pairs=5) <= 1


def _save_run(path, model, method, seed, y, p, lam=None):
    extra = {} if lam is None else {"lam": lam}
    np.savez(path, y_val=y, p_val=p, y_test=y, p_test=p, model=model, method=method, k=4.0, seed=seed,
             best_epoch=3, **extra)


@pytest.fixture
def runs_dir(tmp_path, rng):
    y = np.r_[np.ones(20), np.zeros(30)].astype(int)
    for seed in [1, 2]:
        _save_run(tmp_path / f"R__M0__s{seed}.npz", "ResNet50", "M0_real_only", seed, y, rng.random(50))
        _save_run(tmp_path / f"R__M6__s{seed}.npz", "ResNet50", "M6_dass", seed, y, y + rng.normal(0, 0.3, 50),
                  lam="v1_d1")
    return tmp_path


def test_aggregate_roundtrip(runs_dir):
    runs, probs = load_all_runs(runs_dir)
    assert len(runs) == 4 and set(runs["lam"]) == {"-", "v1_d1"}     # .npz cũ không có lam vẫn đọc được
    summary = summary_stats(runs)
    assert set(summary["n_seeds"]) == {2} and {"roc_auc_mean", "roc_auc_std", "f1_mean"} <= set(summary.columns)
    cmp = compare_to_baseline(probs, "M0_real_only", n_boot=100, seed=0)
    assert list(cmp["method"]) == ["M6_dass"] and cmp["delta_auc"].iloc[0] > 0


def test_reporting_writes_csv_json_tex(runs_dir, tmp_path):
    summary = summary_stats(load_all_runs(runs_dir)[0])
    table, bold = reporting.classification_table(summary)
    paths = reporting.write_table(tmp_path / "tables", "classification", table, summary, "Kết quả", latex=True,
                                  bold=bold)
    assert [p.suffix for p in paths] == [".csv", ".json", ".tex"]
    assert "±" in pd.read_csv(paths[0])["AUC"].iloc[0]
    records = json.loads(paths[1].read_text(encoding="utf-8"))
    assert isinstance(records[0]["roc_auc_mean"], float)                     # json = số thô
    tex = paths[2].read_text(encoding="utf-8")
    assert r"\toprule" in tex and r"\textbf{" in tex and r"M6\_dass" in tex  # in đậm giá trị tốt nhất, escape "_"


def test_generation_table_bolds_lower_kid():
    q = pd.DataFrame([
        {"set": "selected", "method": "M1_random", "n": 10, "kid": 0.02, "kid_std": 0.001, "fid": 30.0,
         "diversity": 0.3, "ssim": 0.2, "auc_real_vs_synth": 0.9},
        {"set": "selected", "method": "M6_dass", "n": 10, "kid": 0.01, "kid_std": 0.001, "fid": 25.0,
         "diversity": 0.35, "ssim": 0.18, "auc_real_vs_synth": 0.8}])
    table, bold = reporting.generation_table(q)
    assert table["KID×1e3"].iloc[1].startswith("10.00") and bold[(1, "KID×1e3")] and bold[(1, "Diversity")]


def test_dataset_table():
    card = {"classes": {"neg": 0, "pos": 1},
            "counts": {"train": {"neg": 7, "pos": 2}, "val": {"neg": 1, "pos": 1}, "test": {"neg": 2, "pos": 1}}}
    t = reporting.dataset_table(card)
    assert list(t["Total"]) == [9, 2, 3]
