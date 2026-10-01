import numpy as np
import pytest

from isic_dass.selection.dass import (
    METHODS,
    diversity_only_select,
    greedy_dass_select,
    jaccard_matrix,
    select_all_methods,
)
from isic_dass.selection.scoring import compute_pool_scores, margin_score, topk_similarity
from isic_dass.utils import l2_normalize, minmax


def test_minmax():
    np.testing.assert_allclose(minmax([2, 4, 6]), [0, 0.5, 1])
    np.testing.assert_allclose(minmax([3, 3]), [0.5, 0.5])


def test_topk_similarity_uses_k_nearest():
    ref = l2_normalize(np.array([[1.0, 0], [0, 1.0], [-1.0, 0]]))
    cand = np.array([[1.0, 0]])
    np.testing.assert_allclose(topk_similarity(cand, ref, topk=1), [1.0])
    np.testing.assert_allclose(topk_similarity(cand, ref, topk=2), [0.5])
    np.testing.assert_allclose(topk_similarity(cand, ref, topk=99), [0.0])   # k bị chặn bởi số ảnh thật


def test_margin_score_prefers_minority_side():
    pos, neg = np.array([[1.0, 0]]), np.array([[0, 1.0]])
    m, s_pos, s_neg = margin_score(np.array([[1.0, 0], [0, 1.0]]), pos, neg, lam=1.0, topk=1)
    assert m[0] > m[1]
    np.testing.assert_allclose(s_pos, [1, 0])
    np.testing.assert_allclose(s_neg, [0, 1])


def test_greedy_without_diversity_equals_ranking(rng):
    z = l2_normalize(rng.normal(size=(50, 8)))
    base = rng.random(50)
    assert greedy_dass_select(z, base, 10, gamma=0.0) == list(np.argsort(-base)[:10])


def test_greedy_returns_unique_indices(rng):
    z = l2_normalize(rng.normal(size=(40, 8)))
    sel = greedy_dass_select(z, rng.random(40), 25, gamma=0.5)
    assert len(sel) == len(set(sel)) == 25
    assert greedy_dass_select(z, rng.random(40), 0, gamma=0.5) == []
    assert len(greedy_dass_select(z, rng.random(40), 100, gamma=0.5)) == 40


def test_diversity_selection_spreads_out():
    # hai cụm chặt: diversity-only phải lấy từ cả hai cụm trước khi lấy thêm trong một cụm
    a = np.tile([1.0, 0.0], (10, 1)) + np.random.default_rng(0).normal(0, 0.01, (10, 2))
    b = np.tile([0.0, 1.0], (10, 1)) + np.random.default_rng(1).normal(0, 0.01, (10, 2))
    z = l2_normalize(np.vstack([a, b]))
    sel = diversity_only_select(z, 2)
    assert {i < 10 for i in sel} == {True, False}


@pytest.fixture
def scored_pool(rng):
    z_v_pool, z_d_pool = l2_normalize(rng.normal(size=(60, 16))), l2_normalize(rng.normal(size=(60, 16)))
    real = {c: l2_normalize(rng.normal(size=(20, 16))) for c in ["benign", "malignant"]}
    scores = compute_pool_scores(z_v_pool, real, z_d_pool, real, "malignant", "benign", 1.0, 1.0, 5)
    return scores, z_v_pool


def test_select_all_methods(scored_pool):
    scores, z_v_pool = scored_pool
    sel = select_all_methods(scores, z_v_pool, n_select=15, alpha=1, beta=1, gamma=0.5, seed=0)
    assert tuple(sel) == METHODS
    assert sel["M0_real_only"] == []
    for m in METHODS[1:]:
        assert len(sel[m]) == len(set(sel[m])) == 15, m
    assert sel == select_all_methods(scores, z_v_pool, 15, 1, 1, 0.5, seed=0)   # tái lập được


def test_jaccard_matrix(scored_pool):
    scores, z_v_pool = scored_pool
    J = jaccard_matrix(select_all_methods(scores, z_v_pool, 15, 1, 1, 0.5, seed=0))
    assert "M0_real_only" not in J.index
    np.testing.assert_allclose(np.diag(J.values.astype(float)), 1.0)
