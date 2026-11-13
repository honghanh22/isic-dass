import numpy as np
import pytest

from dass.selection.rc import (
    RC_DASS,
    RC_RANDOM,
    RC_SELECTION_METHODS,
    RC_TMU,
    RCParams,
    class_margin,
    coverage,
    gate,
    greedy_select,
    l2n,
    margin_rank,
    margin_stats,
    median_bandwidth,
    nc_check,
    nc_threshold,
    nearest,
    select_class,
    tmu,
    tmu_center,
    topk_ids,
)


def _objective(E, u, A, sel, alpha, h):
    k = len(sel)
    return alpha / k * u[sel].sum() + (1 - alpha) * coverage(A, E[sel], h) if k else 0.0


def _brute_greedy(E, u, A, k, alpha, h):
    """Greedy đầy đủ (không lazy), cùng ngân sách K trong hệ số utility."""
    sel: list[int] = []
    for _ in range(k):
        best, best_g = None, -np.inf
        base = (1 - alpha) * coverage(A, E[sel], h) if sel else 0.0
        for i in range(len(E)):
            if i in sel:
                continue
            g = alpha / k * u[i] + (1 - alpha) * coverage(A, E[sel + [i]], h) - base
            if g > best_g + 1e-12:
                best, best_g = i, g
        sel.append(best)
    return sel


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_lazy_greedy_matches_brute_force(alpha):
    rng = np.random.default_rng(1)
    E = l2n(rng.normal(size=(30, 5)))
    A = l2n(rng.normal(size=(40, 5)))
    u = rng.random(30)
    out = greedy_select(E, u, A, 8, alpha, h=0.7)
    assert out.order == _brute_greedy(E, u, A, 8, alpha, 0.7)
    total = sum(out.gain_utility) + sum(out.gain_coverage)
    assert total == pytest.approx(_objective(E, u, A, np.array(out.order), alpha, 0.7))


def test_greedy_gains_are_diminishing_and_tie_break_smallest_id():
    E = np.zeros((4, 2))                      # bốn ứng viên giống hệt nhau
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = greedy_select(E, np.zeros(4), A, 3, 0.0, h=1.0)
    assert out.order == [0, 1, 2]
    assert out.gain_coverage[0] > 0 and out.gain_coverage[1] == pytest.approx(0.0)


def test_greedy_prefers_uncovered_region():
    A = np.array([[0.0, 0], [0.1, 0], [5.0, 0], [5.1, 0]])
    E = np.array([[0.05, 0], [0.06, 0], [5.05, 0]])
    out = greedy_select(E, np.array([1.0, 1.0, 0.0]), A, 2, 0.3, h=0.5)
    assert set(out.order) == {0, 2}             # không lấy hai ảnh cùng vùng


def test_alpha_one_is_topk_on_utility():
    rng = np.random.default_rng(0)
    E, u = rng.normal(size=(20, 3)), rng.random(20)
    assert greedy_select(E, u, E, 5, 1.0, h=1.0).order == topk_ids(u, 5)


def test_class_margin_and_gate():
    z = np.array([2.0, -1.0, 0.0])
    np.testing.assert_allclose(class_margin(z, 1), z)
    np.testing.assert_allclose(class_margin(z, 0), -z)
    np.testing.assert_array_equal(gate(class_margin(z, 1)), [True, False, True])
    assert not gate(np.array([np.nan]))[0]


def test_tmu_peak_at_center():
    st = margin_stats(np.array([1.0, 2.0, 3.0]))
    real = np.array([1.0, 2.0, 3.0])
    mu = tmu_center(real, st, 50)
    t = tmu(np.array([2.0, 3.0, 10.0]), st, mu, 1.0)
    assert t[0] == pytest.approx(1.0) and t[0] > t[1] > t[2] > 0
    with pytest.raises(ValueError):
        tmu_center(np.array([-1.0, -2.0]), st, 50)


def test_margin_rank_ties_and_range():
    r = margin_rank(np.array([3.0, 1.0, 1.0, 5.0]))
    np.testing.assert_allclose(r, [2 / 3, 1 / 6, 1 / 6, 1.0])


def test_median_bandwidth_deterministic_and_positive():
    x = np.random.default_rng(0).normal(size=(300, 4))
    assert median_bandwidth(x, max_pairs=500, seed=3) == median_bandwidth(x, max_pairs=500, seed=3)
    assert median_bandwidth(np.zeros((5, 2))) > 0


def test_nc_threshold_excludes_self():
    x = np.array([[0.0], [1.0], [3.0]])
    tau, d = nc_threshold(x, quantile=0.0)
    np.testing.assert_allclose(d, [1, 1, 2])
    assert tau == pytest.approx(1.0)
    dist, idx = nearest(np.array([[2.9]]), x)
    assert idx[0] == 2 and dist[0] == pytest.approx(0.1)


def test_nc_check_flags_and_keep():
    train = np.array([[0.0, 0], [10.0, 0]])
    pool = np.array([[0.01, 0], [5.0, 0], [5.02, 0], [20.0, 0]])
    r = nc_check(pool, train, tau=0.1, pool_hashes=["a", "b", "c", "t1"], train_hashes=["t1"])
    np.testing.assert_array_equal(r.pixel_duplicate, [False, False, False, True])
    np.testing.assert_array_equal(r.suspect, [True, False, False, False])
    assert r.pool_duplicate_of.tolist() == [-1, -1, 1, -1]
    np.testing.assert_array_equal(r.keep(), [True, True, False, False])           # nghi vấn chưa xác minh -> giữ
    np.testing.assert_array_equal(r.keep({0}), [False, True, False, False])
    with pytest.raises(ValueError):
        r.keep({1})                                                               # không phải nghi vấn


def test_select_class_shared_pool_and_budget():
    rng = np.random.default_rng(0)
    pool_emb, real_emb = rng.normal(size=(60, 6)), rng.normal(size=(20, 6))
    pool_logit = rng.normal(0.5, 1.0, size=60)
    real_logit = rng.normal(1.0, 1.0, size=20)
    keep = np.ones(60, dtype=bool)
    keep[:5] = False
    res = select_class(pool_emb=pool_emb, pool_logit=pool_logit, real_emb=real_emb, real_logit=real_logit, cls=1,
                       nc_keep=keep, k_target=10, params=RCParams(), seed=7)
    elig = set(res.eligible.tolist())
    assert not elig & set(range(5))
    assert all(pool_logit[list(elig)] >= 0)
    for m in RC_SELECTION_METHODS:
        assert len(res.selections[m]) == res.budget == 10
        assert len(set(res.selections[m])) == 10 and set(res.selections[m]) <= elig
    assert res.selections[RC_TMU] == [int(res.eligible[j]) for j in topk_ids(res.tmu_values, 10)]
    again = select_class(pool_emb=pool_emb, pool_logit=pool_logit, real_emb=real_emb, real_logit=real_logit, cls=1,
                         nc_keep=keep, k_target=10, params=RCParams(), seed=7)
    assert again.selections[RC_RANDOM] == res.selections[RC_RANDOM]
    assert again.selections[RC_DASS] == res.selections[RC_DASS]


def test_select_class_shortfall_reduces_budget():
    rng = np.random.default_rng(1)
    res = select_class(pool_emb=rng.normal(size=(12, 4)), pool_logit=np.r_[np.ones(4), -np.ones(8)],
                       real_emb=rng.normal(size=(10, 4)), real_logit=np.ones(10), cls=1,
                       nc_keep=np.ones(12, dtype=bool), k_target=6, params=RCParams(), seed=0)
    assert res.budget == 4 and res.shortfall == 2
    assert all(len(v) == 4 for v in res.selections.values())
