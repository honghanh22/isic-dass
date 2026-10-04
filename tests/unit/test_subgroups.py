import numpy as np
import pandas as pd
import pytest

from dass.evaluation import reporting
from dass.evaluation import subgroups as sg
from dass.evaluation.aggregate import add_holm, compare_to_baseline, load_all_runs
from dass.evaluation.statistics import holm_adjust
from dass.utils import save_npz_atomic

CLASSES = ["negative", "positive"]


def test_holm_adjust_known_values():
    adj = holm_adjust([0.01, 0.04, 0.03, np.nan])
    # sắp xếp: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> đơn điệu: 0.06; NaN giữ nguyên
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06]) and np.isnan(adj[3])
    assert holm_adjust([0.5, 0.9]).max() <= 1.0


@pytest.fixture
def view_runs(tmp_path, rng):
    """Test 40 ảnh: dương phần lớn AP, âm phần lớn PA. M0 = chỉ nhìn tư thế (shortcut); M6 = nhận ra bệnh thật."""
    n_pos, n_neg = 16, 24
    files = [f"negative/n{i:02d}.png" for i in range(n_neg)] + [f"positive/p{i:02d}.png" for i in range(n_pos)]
    y = np.r_[np.zeros(n_neg), np.ones(n_pos)].astype(int)
    view = {f"n{i:02d}": ("AP" if i < 6 else "PA") for i in range(n_neg)}          # âm: 25 % AP
    view |= {f"p{i:02d}": ("AP" if i < 12 else "PA") for i in range(n_pos)}         # dương: 75 % AP
    is_ap = np.array([view[f.split("/")[1][:-4]] == "AP" for f in files], dtype=float)
    meta = tmp_path / "dicom_metadata.csv"
    pd.DataFrame({"image_id": list(view), "ViewPosition": list(view.values())}).to_csv(meta, index=False)
    pred = tmp_path / "predictions"
    for seed in (1, 2):
        noise = rng.normal(0, 0.01, len(y))
        for method, p in [("M0_real_only", is_ap + noise), ("M6_dass", y + 0.3 * is_ap + rng.normal(0, 0.3, len(y)))]:
            save_npz_atomic(pred / f"R__{method}__s{seed}.npz", y_val=y, p_val=p, y_test=y, p_test=p,
                            test_files=np.array(files), model="ResNet50", method=method, k=1.5, seed=seed,
                            best_epoch=1, lam="v1_d1", augment=False, class_weight=method == "M0_real_only")
    return pred, meta, files, y


def test_subgroup_auc_exposes_view_shortcut(view_runs):
    pred, meta, files, y = view_runs
    _, probs = load_all_runs(pred)
    assert all(v[2] == files for v in probs.values())                   # tên ảnh test đọc lại từ .npz
    attr = sg.load_attribute(meta, "ViewPosition")
    resolved = sg.resolve_test_files(probs, None, CLASSES)
    runs = sg.subgroup_runs(probs, resolved, attr, "ViewPosition")
    m0 = runs[runs["method"] == "M0_real_only"].groupby("group")["roc_auc"].mean()
    m6 = runs[runs["method"] == "M6_dass"].groupby("group")["roc_auc"].mean()
    assert set(m0.index) == {"AP", "PA"} and (m0 < 0.75).all()          # trong từng tư thế: shortcut vô dụng
    assert (m6 > 0.85).all()                                            # nhận ra bệnh thật: vẫn tốt trong nhóm
    summary = sg.subgroup_summary(runs)
    assert {"roc_auc_mean", "roc_auc_std", "n", "n_pos", "group"} <= set(summary.columns)
    cmp = sg.subgroup_comparisons(probs, resolved, attr, "ViewPosition", "M0_real_only", [], 100, 0)
    assert set(cmp["group"]) == {"AP", "PA"} and (cmp["delta_auc"] > 0).all() and "p_holm" in cmp
    ref = sg.attribute_only_auc(y, files, attr, "ViewPosition").set_index("group")
    assert ref.loc["AP", "share_in_positive"] == 0.75 and ref.loc["AP", "share_in_negative"] == 0.25
    assert ref.loc["AP", "auc_attribute_only"] == pytest.approx(0.75)  # (0.75 + 0.75) / 2


def test_test_files_rebuilt_from_split_for_old_npz(view_runs):
    pred, meta, files, y = view_runs
    _, probs = load_all_runs(pred)
    old = {k: (v[0], v[1], None) for k, v in probs.items()}               # .npz cũ: không có test_files
    split = {"negative": {"test": [f.split("/")[1] for f in files[:24]][::-1]},   # thứ tự trong JSON không quan trọng
             "positive": {"test": [f.split("/")[1] for f in files[24:]]}}
    fallback = sg.test_files_from_split(split, CLASSES)
    assert fallback == files
    assert sg.resolve_test_files(old, fallback, CLASSES) == {k: files for k in old}
    wrong = {k: (v[0][::-1], v[1], None) for k, v in old.items()}          # nhãn không khớp thứ tự -> không đoán bừa
    assert sg.resolve_test_files(wrong, fallback, CLASSES) == {}
    assert sg.test_files_from_split({"negative": {"train": []}, "positive": {"train": []}}, CLASSES) is None


def test_attribute_share_detects_amplified_view(rng):
    """Ảnh sinh 'trông giống AP' hơn ảnh dương thật -> tỉ lệ dự đoán cao hơn."""
    d = 8
    ap_dir = np.r_[1.0, np.zeros(d - 1)]

    def emb(n, p_ap):
        ap = rng.random(n) < p_ap
        return rng.normal(0, 0.3, (n, d)) + np.outer(ap, ap_dir) * 2, ap

    z_neg, ap_neg = emb(200, 0.25)
    z_pos, ap_pos = emb(200, 0.6)
    attr = {f"n{i}": "AP" if a else "PA" for i, a in enumerate(ap_neg)} | \
           {f"p{i}": "AP" if a else "PA" for i, a in enumerate(ap_pos)}
    names = {"negative": [f"n{i}.png" for i in range(200)], "positive": [f"p{i}.png" for i in range(200)]}
    z_pool, _ = emb(300, 0.9)
    pool_names = [f"synth_{i:05d}.png" for i in range(300)]
    sel = {"M0_real_only": [], "M1_random": pool_names[:150]}
    df, auc = sg.attribute_share_table({"negative": z_neg, "positive": z_pos}, names, z_pool, pool_names, sel,
                                       attr, "ViewPosition", "positive", "negative", seed=0)
    assert auc > 0.9 and set(df["value"]) == {"AP"}
    real_pos = df[df["method"] == "real positive (train)"].iloc[0]
    assert real_pos["true_share"] == pytest.approx(ap_pos.mean()) and abs(real_pos["predicted_share"] - 0.6) < 0.1
    pool = df[df["method"] == "all candidates"].iloc[0]["predicted_share"]
    assert pool > real_pos["predicted_share"] + 0.15                     # khuếch đại tư thế AP
    assert "M0_real_only" not in set(df["method"]) and "M1_random" in set(df["method"])


def test_reporting_subgroup_and_significance_columns(view_runs):
    pred, meta, files, _ = view_runs
    _, probs = load_all_runs(pred)
    attr = sg.load_attribute(meta, "ViewPosition")
    resolved = sg.resolve_test_files(probs, None, CLASSES)
    table, bold = reporting.subgroup_table(sg.subgroup_summary(sg.subgroup_runs(probs, resolved, attr,
                                                                                  "ViewPosition")))
    assert set(table["Subgroup"]) == {"ViewPosition=AP", "ViewPosition=PA"} and "AUC" in table and bold
    cmp = add_holm(compare_to_baseline(probs, "M0_real_only", 50, 0))
    sig = reporting.significance_table(cmp)
    assert {"p (Holm)", "ΔAUC per seed"} <= set(sig.columns) and "±" in sig["ΔAUC per seed"].iloc[0]
    sub_sig = reporting.significance_table(sg.subgroup_comparisons(probs, resolved, attr, "ViewPosition",
                                                                   "M0_real_only", [], 50, 0))
    assert list(sub_sig["Subgroup"]) == ["ViewPosition=AP", "ViewPosition=PA"]
    assert reporting.fmt_mean_std(0.8, float("nan")) == "0.800"         # 1 seed: không in "± 0.000"
