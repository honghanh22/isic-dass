import numpy as np
import pytest

from dass.selection import (
    BOTH_CLASSES,
    METHODS,
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


def test_select_all_methods_is_m0_to_m6_by_default(scored_pool):
    scores, z_v = scored_pool
    sel = select_all_methods(scores, z_v, n_select=15, alpha=1, beta=1, gamma=0.5, seed=0)
    assert tuple(sel) == METHODS[:-1] and sel["M0_real_only"] == []
    for m in METHODS[1:-1]:
        assert len(sel[m]) == len(set(sel[m])) == 15, m
    assert sel == select_all_methods(scores, z_v, 15, 1, 1, 0.5, seed=0)       # tái lập được


def test_both_classes_variant_is_optional(scored_pool):
    scores, z_v = scored_pool
    sel = select_all_methods(scores, z_v, 15, 1, 1, 0.5, seed=0, both_classes=True)
    assert sel[BOTH_CLASSES] == sel["M6_dass"]


def test_jaccard_matrix(scored_pool):
    scores, z_v = scored_pool
    J = jaccard_matrix(select_all_methods(scores, z_v, 15, 1, 1, 0.5, seed=0))
    assert "M0_real_only" not in J.index
    np.testing.assert_allclose(np.diag(J.values.astype(float)), 1.0)





def test_crossfit_folds_and_margin(rng):
    from dass.selection import crossfit_folds, crossfit_margin, margin_score

    f = crossfit_folds(11, 3, 0)
    assert sorted(np.bincount(f)) == [3, 4, 4] and set(f) == {0, 1, 2}
    zp, pos, neg = (l2_normalize(rng.normal(size=(n, 6))) for n in (10, 7, 9))
    m, sp, sn = crossfit_margin([(zp, pos, neg), (zp, pos, neg)], 1.0, 3)
    assert np.allclose(m, margin_score(zp, pos, neg, 1.0, 3)[0])      # hai phần giống nhau -> bằng margin thường
