"""Đánh giá theo nhóm con của một thuộc tính ảnh (ví dụ tư thế chụp AP / PA của RSNA) — chỉ ĐO, không đổi cách train.

- AUC trong từng nhóm: mọi ảnh trong nhóm có cùng thuộc tính nên thuộc tính không còn giúp xếp hạng -> đo khả năng
  nhận ra bệnh, tách khỏi shortcut (Oakden-Rayner et al. 2020; Janizek et al. 2020 với tư thế AP / PA).
- Mốc "chỉ dùng thuộc tính": AUC khi dự đoán nhãn CHỈ bằng thuộc tính (ví dụ ảnh AP = dương).
- Tỉ lệ thuộc tính trong ảnh sinh: probe logistic trên đặc trưng E_v của ảnh thật (train) dự đoán thuộc tính, áp lên
  pool và ảnh mỗi phương pháp chọn -> ảnh sinh / cách chọn ảnh có khuếch đại thuộc tính gắn với bệnh không.

Thuộc tính lấy từ CSV metadata (`image_id`, <cột>), ví dụ `dicom_metadata.csv` do nguồn DICOM ghi. numpy / sklearn thuần.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .aggregate import GROUP_COLS, RunKey, add_holm, compare_pairs, compare_to_baseline, mark_family, summary_stats
from .classification import binary_metrics

SUBGROUP_METRICS = ["roc_auc", "pr_auc"]
_MISSING = {"", "nan", "none"}


def load_attribute(metadata_csv: str | Path, column: str) -> dict[str, str]:
    """{tên ảnh không đuôi: giá trị thuộc tính}; cột không có -> {}."""
    df = pd.read_csv(metadata_csv, dtype=str)
    if column not in df or "image_id" not in df:
        return {}
    return {Path(str(i)).stem: str(v).strip() for i, v in zip(df["image_id"], df[column])
            if str(v).strip().lower() not in _MISSING}


def test_files_from_split(split: dict, class_names: list[str]) -> list[str] | None:
    """Thứ tự ảnh test lúc dự đoán (`image_dataset_from_directory`, shuffle = False): lớp theo thứ tự cấu hình, tên
    file sắp xếp tăng dần — dùng cho .npz cũ chưa lưu `test_files`. Split không có phần test -> None."""
    if not all("test" in split.get(c, {}) for c in class_names):
        return None
    return [f"{c}/{f}" for c in class_names for f in sorted(split[c]["test"])]


def resolve_test_files(probs: dict[RunKey, tuple], fallback: list[str] | None,
                       class_names: list[str]) -> dict[RunKey, list[str]]:
    """Tên ảnh test của từng lần chạy: lấy trong .npz; nếu thiếu thì dùng `fallback` khi nhãn khớp đúng thứ tự."""
    out = {}
    fb_labels = None if fallback is None else np.array([class_names.index(f.split("/")[0]) for f in fallback])
    for key, (y, _, files, *_) in probs.items():
        if files is not None and len(files) == len(y):
            out[key] = list(files)
        elif fb_labels is not None and np.array_equal(fb_labels, y):
            out[key] = list(fallback)
    return out


def _restrict(probs: dict[RunKey, tuple], files: dict[RunKey, list[str]], attr: dict[str, str],
              value: str) -> dict[RunKey, tuple]:
    """Chỉ giữ ảnh test có thuộc tính = `value`, sắp theo tên ảnh (mọi lần chạy cùng thứ tự để so sánh ghép cặp)."""
    out = {}
    for key, fs in files.items():
        y, p = probs[key][0], probs[key][1]
        order = np.argsort(fs, kind="stable")
        keep = [i for i in order if attr.get(Path(fs[i]).stem) == value]
        out[key] = (y[keep], p[keep], [fs[i] for i in keep])
    return out


def subgroup_values(files: dict[RunKey, list[str]], attr: dict[str, str]) -> list[str]:
    return sorted({attr[Path(f).stem] for fs in files.values() for f in fs if Path(f).stem in attr})


def subgroup_runs(probs: dict[RunKey, tuple], files: dict[RunKey, list[str]], attr: dict[str, str], column: str,
                  threshold: float = 0.5) -> pd.DataFrame:
    """Metric từng lần chạy trong từng nhóm (nhóm phải có đủ hai lớp)."""
    rows = []
    for value in subgroup_values(files, attr):
        for (model, method, lam, k, seed), (y, p, _) in _restrict(probs, files, attr, value).items():
            if len(np.unique(y)) < 2:
                continue
            rows.append({"model": model, "method": method, "lam": lam, "k": k, "seed": seed, "attribute": column,
                         "group": value, "n": len(y), "n_pos": int(y.sum()), **binary_metrics(y, p, threshold)})
    return pd.DataFrame(rows)


def subgroup_summary(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    out = summary_stats(runs, SUBGROUP_METRICS, by=[*GROUP_COLS, "attribute", "group"])
    sizes = runs.groupby([*GROUP_COLS, "attribute", "group"])[["n", "n_pos"]].first().reset_index()
    return out.merge(sizes, on=[*GROUP_COLS, "attribute", "group"])


def subgroup_comparisons(probs: dict[RunKey, tuple], files: dict[RunKey, list[str]], attr: dict[str, str],
                         column: str, baseline: str, pairs: list[list[str]], n_boot: int, seed: int,
                         primary: list[list[str]] | None = None) -> pd.DataFrame:
    """ΔAUC (vs baseline + các cặp bổ sung) trong từng nhóm; `p_holm` hiệu chỉnh trong mỗi model × k × nhóm."""
    parts = []
    for value in subgroup_values(files, attr):
        sub = _restrict(probs, files, attr, value)
        if not any(len(np.unique(v[0])) == 2 for v in sub.values()):
            continue
        cmp = pd.concat([compare_to_baseline(sub, baseline, n_boot, seed), compare_pairs(sub, pairs, n_boot, seed)],
                        ignore_index=True)
        if len(cmp):
            parts.append(cmp.assign(attribute=column, group=value))
    if not parts:
        return pd.DataFrame()
    return add_holm(mark_family(pd.concat(parts, ignore_index=True), primary),
                    family=("model", "k", "attribute", "group"))


def attribute_only_auc(y: np.ndarray, test_files: list[str], attr: dict[str, str], column: str) -> pd.DataFrame:
    """Mốc: AUC khi dự đoán nhãn CHỈ bằng thuộc tính (điểm = 1 nếu ảnh thuộc nhóm, ngược lại 0), mỗi giá trị một
    dòng, kèm tỉ lệ của giá trị đó trong lớp dương / âm của test."""
    vals = np.array([attr.get(Path(f).stem) for f in test_files], dtype=object)
    known = np.array([x is not None for x in vals])
    rows = []
    for value in sorted(set(vals[known])):
        hit = (vals[known] == value).astype(float)
        yk = np.asarray(y)[known]
        rows.append({"attribute": column, "group": value, "n_test": int(known.sum()),
                     "share_in_positive": float(hit[yk == 1].mean()), "share_in_negative": float(hit[yk == 0].mean()),
                     "auc_attribute_only": float(roc_auc_score(yk, hit))})
    return pd.DataFrame(rows)


def attribute_share_table(z_real: dict[str, np.ndarray], real_names: dict[str, list[str]], z_pool: np.ndarray,
                          pool_names: list[str], selections: dict[str, list[str]], attr: dict[str, str], column: str,
                          minority: str, majority: str, seed: int) -> tuple[pd.DataFrame, float | None]:
    """Tỉ lệ thuộc tính (giá trị gắn với lớp thiểu số, ví dụ AP) trong ảnh thật và ảnh sinh, ước lượng bằng probe
    logistic (không class weight -> xác suất đã hiệu chỉnh) trên đặc trưng E_v của ảnh thật train.

    Ảnh thật: dự đoán ngoài fold (5-fold) để so được với tỉ lệ thật trong metadata (kiểm tra probe). Ảnh sinh: probe
    train trên toàn bộ ảnh thật. Trả về (bảng, AUC của probe); thuộc tính không phải nhị phân -> (bảng rỗng, None).
    """
    X, v, cls = [], [], []
    for c in (majority, minority):
        for z, name in zip(z_real[c], real_names[c]):
            if Path(name).stem in attr:
                X.append(z)
                v.append(attr[Path(name).stem])
                cls.append(c)
    values = sorted(set(v))
    if len(values) != 2:
        return pd.DataFrame(), None
    v, cls, X = np.array(v), np.array(cls), np.asarray(X)
    target = max(values, key=lambda val: np.mean(v[cls == minority] == val))     # giá trị gắn với lớp thiểu số
    y = (v == target).astype(int)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    p_cv = cross_val_predict(probe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             method="predict_proba")[:, 1]
    probe.fit(X, y)
    p_pool = probe.predict_proba(z_pool)[:, 1]
    index = {n: i for i, n in enumerate(pool_names)}

    rows = []
    for c, label in [(majority, f"real {majority} (train)"), (minority, f"real {minority} (train)")]:
        m = cls == c
        rows.append({"set": "real", "method": label, "n": int(m.sum()), "true_share": float(y[m].mean()),
                     "predicted_share": float(p_cv[m].mean())})
    rows.append({"set": "pool", "method": "all candidates", "n": len(z_pool), "true_share": np.nan,
                 "predicted_share": float(p_pool.mean())})
    for method, names in selections.items():
        idx = [index[Path(n).name] for n in names if Path(n).name in index]
        if idx:
            rows.append({"set": "selected", "method": method, "n": len(idx), "true_share": np.nan,
                         "predicted_share": float(p_pool[idx].mean())})
    df = pd.DataFrame(rows).assign(attribute=column, value=target)
    return df, float(roc_auc_score(y, p_cv))
