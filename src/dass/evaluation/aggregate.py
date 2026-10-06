"""Gộp mọi file dự đoán .npz: metric từng lần chạy, thống kê mean / std qua seed, paired bootstrap so với baseline.

Đọc được cả .npz của notebook cũ (không có trường `lam`).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .classification import KEY_METRICS, binary_metrics
from .statistics import holm_adjust, paired_bootstrap_auc

GROUP_COLS = ["model", "method", "lam", "k"]
RunKey = tuple[str, str, str, float, int]   # (model, method, lam, k, seed)
PROTOCOL_FIELDS = ("augment", "class_weight", "monitor", "synth_fraction", "design", "filter_majority")   # thiết lập train ghi trong .npz (augment /
# class_weight từ 1.4.0, monitor từ 1.12.0, synth_fraction từ 1.16.0, design từ 1.18.0; .npz cũ không có -> None)
_PROTOCOL_TYPES = {"monitor": str, "synth_fraction": float, "design": str}


def read_protocol(d) -> dict[str, bool | str | float | None]:
    """{augment, class_weight, monitor, synth_fraction} của một .npz đã nạp; None nếu .npz cũ không ghi trường đó."""
    return {f: (_PROTOCOL_TYPES.get(f, bool)(d[f]) if f in d.files else None) for f in PROTOCOL_FIELDS}


def protocol_mismatch(npz_path: str | Path, expected: dict[str, bool | str | float]) -> str | None:
    """Mô tả chỗ lệch giữa thiết lập train lưu trong `npz_path` và `expected`; None nếu khớp (hoặc .npz cũ không ghi)."""
    with np.load(npz_path, allow_pickle=True) as d:
        stored = read_protocol(d)
    diff = [f"{f}: đã lưu {stored[f]}, cấu hình hiện tại {v}" for f, v in expected.items()
            if stored.get(f) is not None and stored[f] != v]
    return "; ".join(diff) or None


def archive_superseded_run(npz_path: str | Path, weights_path: str | Path, pred_root: str | Path,
                           weights_root: str | Path) -> Path:
    """Chuyển (KHÔNG xoá) .npz và trọng số của một lần chạy theo giao thức cũ sang
    `pred_root/<giao thức>/` và `weights_root/<giao thức>/` (ví dụ `augment-True__class_weight-False`), để train lại theo
    cấu hình hiện tại mà không trộn hai giao thức trong thư mục dự đoán. Trùng tên thì thêm hậu tố, không ghi đè."""
    npz_path, weights_path = Path(npz_path), Path(weights_path)
    with np.load(npz_path, allow_pickle=True) as d:
        label = "__".join(f"{k}-{v}" for k, v in read_protocol(d).items() if v is not None)

    def move(src: Path, root: str | Path) -> Path:
        dst = Path(root) / label / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = 1
        while dst.exists():
            dst = dst.with_name(f"{src.name}.{n}")
            n += 1
        shutil.move(str(src), str(dst))
        return dst

    if weights_path.exists():
        move(weights_path, weights_root)
    return move(npz_path, pred_root)


def check_single_protocol(runs: pd.DataFrame) -> None:
    """Một thư mục dự đoán chỉ được chứa MỘT giao thức train (augmentation, tiêu chí chọn epoch) — trộn thì mean ± std
    và bootstrap sai."""
    for col, what in (("augment", "có và không có augmentation"), ("monitor", "chọn epoch theo tiêu chí khác nhau"),
                      ("synth_fraction", "tỉ lệ ảnh sinh khác nhau"), ("design", "thiết kế tập train khác nhau")):
        if col in runs and len(set(runs[col].dropna())) > 1:
            raise ValueError(f"Thư mục dự đoán trộn lần chạy {what} -> tách bằng --tag / run_tag mới")


def load_all_runs(pred_dir: str | Path, threshold: float = 0.5) -> tuple[pd.DataFrame, dict[RunKey, tuple]]:
    """(bảng metric từng lần chạy, {RunKey: (y_test, p_test, tên ảnh test | None)}). Tên ảnh test (`lớp/tệp`) có
    trong .npz từ 1.9.0; .npz cũ -> None (dựng lại từ split khi cần, xem `evaluation.subgroups`)."""
    rows, probs = [], {}
    for path in sorted(Path(pred_dir).glob("*.npz")):
        with np.load(path, allow_pickle=True) as d:
            model, method, k, seed = str(d["model"]), str(d["method"]), float(d["k"]), int(d["seed"])
            lam = str(d["lam"]) if "lam" in d.files else "-"
            y_val, p_val, y_test, p_test = d["y_val"], d["p_val"], d["y_test"], d["p_test"]
            test_files = [str(f) for f in d["test_files"]] if "test_files" in d.files else None
            row = {"model": model, "method": method, "lam": lam, "k": k, "seed": seed,
                   "best_epoch": int(d["best_epoch"]), "n_test": len(y_test),
                   "val_roc_auc": float(roc_auc_score(y_val, p_val)),
                   "train_roc_auc": (float(roc_auc_score(d["y_train"], d["p_train"])) if "p_train" in d.files
                                     else float("nan")), **read_protocol(d)}
        row.update(binary_metrics(y_test, p_test, threshold))
        rows.append(row)
        probs[(model, method, lam, k, seed)] = (y_test, p_test, test_files)
    return pd.DataFrame(rows), probs


def summary_stats(runs: pd.DataFrame, metrics: list[str] = KEY_METRICS,
                  by: list[str] | tuple[str, ...] = tuple(GROUP_COLS)) -> pd.DataFrame:
    """Một dòng mỗi nhóm `by` (mặc định model, method, lam, k): <metric>_mean, <metric>_std, n_seeds (số liệu thô).
    Chỉ 1 seed -> std = NaN (bảng chỉ in mean), không ghi 0 gây hiểu nhầm là không dao động."""
    g = runs.groupby(list(by))
    mean, std = g[metrics].mean(), g[metrics].std(ddof=1)
    out = pd.concat([mean.add_suffix("_mean"), std.add_suffix("_std")], axis=1)
    out["n_seeds"] = g.size()
    ordered = [c for m in metrics for c in (f"{m}_mean", f"{m}_std")] + ["n_seeds"]
    return out[ordered].reset_index()


def _runs_of(probs: dict[RunKey, tuple], model: str, k: float, method: str,
             lam: str | None = None) -> dict[int, tuple]:
    return {s: v for (m, me, lm, kk, s), v in probs.items()
            if m == model and me == method and kk == k and (lam is None or lm == lam)}


def _compare(model: str, k: float, method: str, lam: str, cur: dict[int, tuple], ref_name: str,
             ref: dict[int, tuple], n_boot: int, seed: int) -> dict | None:
    """ΔAUC = AUC(method) − AUC(ref), hai cách:

    - `delta_auc`, CI, `p_value`: paired bootstrap trên xác suất TRUNG BÌNH qua các seed chung (ensemble) — đo dao
      động do mẫu test, không đo dao động giữa các lần train;
    - `delta_auc_seed_mean` ± `delta_auc_seed_std`: ΔAUC của từng seed (ghép cặp theo seed) — đo dao động giữa các
      lần train; khớp với chênh lệch của bảng mean ± std.
    """
    common = sorted(set(ref) & set(cur))
    if not common:
        return None
    y = ref[common[0]][0]
    if not all(np.array_equal(y, cur[s][0]) and np.array_equal(y, ref[s][0]) for s in common):
        raise ValueError(f"Thứ tự nhãn test không khớp: {model}/{method} vs {ref_name}")
    p_ref = np.mean([ref[s][1] for s in common], axis=0)
    p_cur = np.mean([cur[s][1] for s in common], axis=0)
    obs, lo, hi, pv = paired_bootstrap_auc(y, p_cur, p_ref, n_boot, seed)
    per_seed = [roc_auc_score(y, cur[s][1]) - roc_auc_score(y, ref[s][1]) for s in common]
    return {"model": model, "k": k, "method": method, "lam": lam, "vs": ref_name, "n_seeds": len(common),
            "delta_auc": obs, "ci95_low": lo, "ci95_high": hi, "p_value": pv,
            "delta_auc_seed_mean": float(np.mean(per_seed)),
            "delta_auc_seed_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")}


def mark_family(cmp: pd.DataFrame, primary: list[list[str]] | None) -> pd.DataFrame:
    """Cột `family`: "primary" nếu (method, vs) là giả thuyết chính khai báo trước, ngược lại "exploratory"."""
    if cmp.empty:
        return cmp
    keys = {tuple(p) for p in (primary or [])}
    return cmp.assign(family=["primary" if (m, v) in keys else "exploratory" for m, v in zip(cmp["method"], cmp["vs"])])


def add_holm(cmp: pd.DataFrame, family: list[str] | tuple[str, ...] = ("model", "k")) -> pd.DataFrame:
    """Cột `p_holm`: p hiệu chỉnh Holm trong mỗi họ kiểm định (mặc định mỗi model × k). Có cột `family`
    (`mark_family`) -> họ chính và họ khám phá được hiệu chỉnh RIÊNG."""
    if cmp.empty:
        return cmp
    family = [*family, "family"] if "family" in cmp and "family" not in family else list(family)
    out = cmp.copy()
    out["p_holm"] = np.nan
    for _, idx in out.groupby(list(family)).groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"].to_numpy())
    return out


def compare_to_baseline(probs: dict[RunKey, tuple], baseline: str, n_boot: int, seed: int) -> pd.DataFrame:
    """ΔAUC của từng (phương pháp, lambda) so với baseline, dùng xác suất trung bình qua các seed chung."""
    rows = []
    for model, k in sorted({(m, kk) for m, _, _, kk, _ in probs}):
        base = _runs_of(probs, model, k, baseline)
        if not base:
            continue
        combos = sorted({(me, lm) for m, me, lm, kk, _ in probs if m == model and kk == k and me != baseline})
        for method, lam in combos:
            row = _compare(model, k, method, lam, _runs_of(probs, model, k, method, lam), baseline, base, n_boot,
                           seed)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def compare_pairs(probs: dict[RunKey, tuple], pairs: list[list[str]], n_boot: int, seed: int) -> pd.DataFrame:
    """ΔAUC cho từng cặp [phương pháp, đối chứng] (ví dụ M6 vs M0b, M6 vs M1), trong mỗi (model, k).

    Cặp nào thiếu dự đoán của một trong hai phía (chưa train) thì bỏ qua.
    """
    rows = []
    for model, k in sorted({(m, kk) for m, _, _, kk, _ in probs}):
        for method, ref_name in pairs:
            ref = _runs_of(probs, model, k, ref_name)
            for lam in sorted({lm for m, me, lm, kk, _ in probs if m == model and kk == k and me == method}):
                row = _compare(model, k, method, lam, _runs_of(probs, model, k, method, lam), ref_name, ref,
                               n_boot, seed)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)
