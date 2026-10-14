"""Chỉ số của tập ảnh sinh được chọn: disease margin, đa dạng (E_v), coverage / density (Naeem et al., 2020)."""

import numpy as np
import pandas as pd
import pytest

from dass.evaluation import reporting
from dass.evaluation.selection_quality import (
    coverage_density,
    knn_radii,
    nn_within,
    pairwise_diversity,
    selection_quality,
)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_coverage_is_one_for_copies_and_zero_for_far_samples():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8))
    assert coverage_density(real, real.copy(), k=5)["coverage"] == 1.0
    far = coverage_density(real, real + 100.0, k=5)
    assert far["coverage"] == 0.0 and far["density"] == 0.0


def test_coverage_rewards_spread_over_collapse():
    """Cùng số ảnh: tập trải đều theo phân phối thật phủ nhiều hơn tập dồn vào một chỗ (mode collapse)."""
    rng = np.random.default_rng(1)
    real = rng.normal(size=(400, 4))
    spread = rng.normal(size=(100, 4))
    collapsed = rng.normal(scale=0.05, size=(100, 4))            # dồn quanh tâm
    assert coverage_density(real, spread)["coverage"] > 10 * coverage_density(real, collapsed)["coverage"]


def test_knn_radii_excludes_self():
    pts = np.array([[0.0], [1.0], [3.0]])
    assert np.allclose(knn_radii(pts, 1), [1.0, 1.0, 4.0])      # bình phương khoảng cách tới láng giềng gần nhất


def test_diversity_and_nearest_neighbour():
    same = np.ones((5, 3))
    assert pairwise_diversity(same) == pytest.approx(0.0, abs=1e-9) and nn_within(same) == pytest.approx(0.0, abs=1e-9)
    ortho = np.eye(4)
    assert pairwise_diversity(ortho) == pytest.approx(1.0) and nn_within(ortho) == pytest.approx(1.0)


def test_selection_quality_rows_and_margins():
    rng = np.random.default_rng(2)
    n_pool = 30
    emb = {"pool_names": np.array([f"synth_{i:05d}.png" for i in range(n_pool)]),
           "z_v_pool": _unit(rng.normal(size=(n_pool, 6))), "z_d_pool": _unit(rng.normal(size=(n_pool, 6))),
           "z_v_real__pos": _unit(rng.normal(size=(12, 6))), "z_d_real__pos": _unit(rng.normal(size=(12, 6))),
           "z_v_real__neg": _unit(rng.normal(size=(40, 6))), "z_d_real__neg": _unit(rng.normal(size=(40, 6))),
           "score__M_v": rng.normal(size=n_pool), "score__M_d": np.linspace(-1, 1, n_pool)}
    sel = {"M0_real_only": [], "M1_random": [f"synth_{i:05d}.png" for i in range(0, 20)],
           "M6_dass": [f"synth_{i:05d}.png" for i in range(10, 30)]}
    df = selection_quality(emb, sel, "pos", k=3)
    assert list(df["set"]) == ["reference", "pool", "selected", "selected"]
    m6 = df[df.method == "M6_dass"].iloc[0]
    assert m6["n"] == 20 and m6["md_mean"] == pytest.approx(emb["score__M_d"][10:30].mean())
    assert m6["md_pos_share"] == pytest.approx((emb["score__M_d"][10:30] > 0).mean())
    assert 0 <= m6["coverage_d"] <= 1 and 0 <= m6["coverage_v"] <= 1
    table, bold = reporting.selection_quality_table(df)
    assert "Coverage (E_d)" in table and bold.get((3, "M_d mean"))            # M6 có M_d cao hơn M1
    with pytest.raises(ValueError, match="embeddings"):
        selection_quality(emb, {"M1_random": ["khong_co.png"]}, "pos")


def test_generation_table_shows_coverage_when_present():
    q = pd.DataFrame([{"set": "selected", "method": m, "n": 10, "kid": 0.02, "kid_std": 0.001, "fid": 30.0,
                       "diversity": 0.3, "ssim": 0.2, "auc_real_vs_synth": np.nan, "coverage": c, "density": 1.0}
                      for m, c in [("M1_random", 0.4), ("M6_dass", 0.6)]])
    table, bold = reporting.generation_table(q)
    assert list(table["Coverage"]) == ["0.400", "0.600"] and bold.get((1, "Coverage"))
