import numpy as np
import pytest

from dass.selection import (
    BASELINES,
    METHODS,
    SELECTION_METHODS,
    compute_pool_scores,
    greedy_dass_select,
    jaccard_matrix,
    margin_score,
    select_all_methods,
    topk_similarity,
)
from dass.selection.strategies import diversity_only_select
from dass.utils import l2_normalize, minmax


def test_minmax():
    np.testing.assert_allclose(minmax([2, 4, 6]), [0, 0.5, 1])
    np.testing.assert_allclose(minmax([3, 3]), [0.5, 0.5])


def test_topk_similarity_uses_k_nearest():
    ref = l2_normalize(np.array([[1.0, 0], [0, 1.0], [-1.0, 0]]))
    cand = np.array([[1.0, 0]])
    np.testing.assert_allclose(topk_similarity(cand, ref, topk=1), [1.0])
    np.testing.assert_allclose(topk_similarity(cand, ref, topk=2), [0.5])
    np.testing.assert_allclose(topk_similarity(cand, ref, topk=99), [0.0])


def test_margin_score_prefers_minority_side():
    m, s_pos, s_neg = margin_score(np.array([[1.0, 0], [0, 1.0]]), np.array([[1.0, 0]]), np.array([[0, 1.0]]),
                                   lam=1.0, topk=1)
    assert m[0] > m[1]
    np.testing.assert_allclose(s_pos, [1, 0])
    np.testing.assert_allclose(s_neg, [0, 1])


def test_greedy_without_diversity_equals_ranking(rng):
    z = l2_normalize(rng.normal(size=(50, 8)))
    base = rng.random(50)
    assert greedy_dass_select(z, base, 10, gamma=0.0) == list(np.argsort(-base)[:10])


def test_greedy_returns_unique_indices(rng):
    z = l2_normalize(rng.normal(size=(40, 8)))
    assert len(set(greedy_dass_select(z, rng.random(40), 25, gamma=0.5))) == 25
    assert greedy_dass_select(z, rng.random(40), 0, gamma=0.5) == []
    assert len(greedy_dass_select(z, rng.random(40), 100, gamma=0.5)) == 40


def test_diversity_selection_spreads_out():
    a = np.tile([1.0, 0.0], (10, 1)) + np.random.default_rng(0).normal(0, 0.01, (10, 2))
    b = np.tile([0.0, 1.0], (10, 1)) + np.random.default_rng(1).normal(0, 0.01, (10, 2))
    sel = diversity_only_select(l2_normalize(np.vstack([a, b])), 2)
    assert {i < 10 for i in sel} == {True, False}


@pytest.fixture
def scored_pool(rng):
    z_v, z_d = l2_normalize(rng.normal(size=(60, 16))), l2_normalize(rng.normal(size=(60, 16)))
    real = {c: l2_normalize(rng.normal(size=(20, 16))) for c in ["benign", "malignant"]}
    return compute_pool_scores(z_v, real, z_d, real, "malignant", "benign", 1.0, 1.0, 5), z_v


def test_select_all_methods_is_m1_to_m6(scored_pool):
    scores, z_v = scored_pool
    sel = select_all_methods(scores, z_v, n_select=15, alpha=1, beta=1, gamma=0.5, seed=0)
    assert tuple(sel) == SELECTION_METHODS
    for m in SELECTION_METHODS:
        assert len(sel[m]) == len(set(sel[m])) == 15, m
    assert sel == select_all_methods(scores, z_v, 15, 1, 1, 0.5, seed=0)       # tái lập được


def test_method_codes_follow_the_paper_tables():
    """Nhóm 1: B0–B3 (+ B4 = M1, B5 = M6); nhóm 2: M1–M6 theo đúng thứ tự trong bài (M4 = diversity, M5 = M_v + M_d)."""
    assert BASELINES == ("B0_imbalanced", "B1_class_weight", "B2_ros", "B3_augment")
    assert SELECTION_METHODS == ("M1_random", "M2_visual", "M3_disease", "M4_diversity", "M5_visual_disease",
                                 "M6_dass")
    assert METHODS == (*BASELINES, *SELECTION_METHODS)


def test_ablation_variants_match_their_definition(scored_pool):
    from dass.selection.strategies import top_n

    scores, z_v = scored_pool
    sel = select_all_methods(scores, z_v, 15, 1, 1, 0.5, seed=0)
    base = minmax(scores["M_v"]) + minmax(scores["M_d"])
    assert sel["M2_visual"] == top_n(scores["M_v"], 15) and sel["M3_disease"] == top_n(scores["M_d"], 15)
    assert sel["M4_diversity"] == diversity_only_select(z_v, 15)
    assert sel["M5_visual_disease"] == top_n(base, 15)
    assert sel["M6_dass"] == greedy_dass_select(z_v, base, 15, 0.5)


def test_jaccard_matrix(scored_pool):
    scores, z_v = scored_pool
    J = jaccard_matrix(select_all_methods(scores, z_v, 15, 1, 1, 0.5, seed=0))
    assert list(J.index) == list(SELECTION_METHODS)
    np.testing.assert_allclose(np.diag(J.values.astype(float)), 1.0)
