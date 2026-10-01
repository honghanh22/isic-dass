import numpy as np
import pytest

from isic_dass.evaluation.aggregate import compare_to_baseline, load_all_runs, mean_std_table
from isic_dass.evaluation.metrics import binary_metrics
from isic_dass.evaluation.quality import compute_diversity, compute_fid
from isic_dass.evaluation.stats import paired_bootstrap_auc, separability_auc
from isic_dass.gan.metrics import kid_from_features


def test_binary_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.1, 0.6])
    m = binary_metrics(y, p)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["g_mean"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_paired_bootstrap_identical_predictions(rng):
    y = np.r_[np.ones(20), np.zeros(30)].astype(int)
    p = rng.random(50)
    obs, lo, hi, pv = paired_bootstrap_auc(y, p, p, n_boot=50, seed=0)
    assert obs == 0 and lo == 0 and hi == 0 and pv == 1.0


def test_paired_bootstrap_detects_better_model(rng):
    y = np.r_[np.ones(40), np.zeros(60)].astype(int)
    good = y + rng.normal(0, 0.3, 100)
    bad = rng.random(100)
    obs, lo, _, pv = paired_bootstrap_auc(y, good, bad, n_boot=200, seed=0)
    assert obs > 0.3 and lo > 0 and pv < 0.05


def test_separability_auc(rng):
    a = rng.normal(0, 1, (60, 4))
    assert separability_auc(a, a + 5, seed=0) > 0.99
    assert abs(separability_auc(a[:30], a[30:], seed=0) - 0.5) < 0.25


def _naive_kid(real, fake, m):
    """KID không lấy mẫu con: MMD² không chệch với kernel đa thức bậc 3."""
    n = real.shape[1]
    k = lambda a, b: (a @ b.T / n + 1) ** 3  # noqa: E731
    kxx, kyy, kxy = k(fake, fake), k(real, real), k(fake, real)
    return ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()


def test_kid_matches_unbiased_mmd_when_subset_is_full(rng):
    real, fake = rng.normal(size=(30, 8)), rng.normal(1, 1, size=(30, 8))
    assert kid_from_features(real, fake, num_subsets=3, max_subset_size=30, seed=0) == pytest.approx(
        _naive_kid(real, fake, 30))


def test_kid_orders_distributions(rng):
    real = rng.normal(size=(200, 16))
    near, far = rng.normal(size=(200, 16)), rng.normal(0.5, 1, size=(200, 16))
    assert kid_from_features(real, near, 10, 100, 0) < kid_from_features(real, far, 10, 100, 0)


def test_fid_and_diversity(rng):
    a = rng.normal(size=(100, 4))
    assert compute_fid(a, a) == pytest.approx(0, abs=1e-6)
    assert compute_diversity(np.tile([[1.0, 0.0]], (5, 1))) == pytest.approx(0, abs=1e-9)


def _save_run(path, model, method, seed, y, p, lam=None):
    extra = {} if lam is None else {"lam": lam}
    np.savez(path, y_val=y, p_val=p, y_test=y, p_test=p, model=model, method=method, k=4.0, seed=seed,
             best_epoch=3, **extra)


def test_aggregate_roundtrip(tmp_path, rng):
    y = np.r_[np.ones(20), np.zeros(30)].astype(int)
    for seed in [1, 2]:
        _save_run(tmp_path / f"R__M0__s{seed}.npz", "ResNet50", "M0_real_only", seed, y, rng.random(50))
        _save_run(tmp_path / f"R__M6__s{seed}.npz", "ResNet50", "M6_dass", seed, y,
                  y + rng.normal(0, 0.3, 50), lam="v1_d1")
    runs, probs = load_all_runs(tmp_path)
    assert len(runs) == 4
    assert set(runs["lam"]) == {"-", "v1_d1"}      # file cũ không có lam vẫn đọc được
    summary = mean_std_table(runs)
    assert set(summary["n_seeds"]) == {2}
    cmp = compare_to_baseline(probs, "M0_real_only", n_boot=100, seed=0)
    assert list(cmp["method"]) == ["M6_dass"] and cmp["n_seeds"].iloc[0] == 2
    assert cmp["delta_auc"].iloc[0] > 0
