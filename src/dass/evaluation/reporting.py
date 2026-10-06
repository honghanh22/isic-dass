"""Xuất bảng cho bài báo: mỗi bảng 3 định dạng.

- `<tên>.csv`  — đã định dạng (mean ± std), dán thẳng vào bảng biểu; tên phương pháp dạng chữ thường (bỏ LaTeX).
- `<tên>.json` — số liệu thô (records, mã phương pháp nội bộ), cho xử lý tiếp bằng code.
- `<tên>.tex`  — LaTeX booktabs, in đậm giá trị tốt nhất mỗi cột trong từng nhóm; `$...$` trong tên được giữ làm công thức.

Tên hiển thị (`evaluation.method_labels`) chỉ dùng khi xuất bảng: mã nội bộ trong .npz / selections.json không đổi.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .classification import KEY_METRICS, METRIC_LABELS

LOWER_IS_BETTER = {"kid", "fid", "auc_real_vs_synth", "ssim"}
SET_LABELS = {"real val vs real train": "Real (val vs train)", "all candidates": "All candidates (pool)"}
TEXT_COLUMNS = {"Model", "Family", "Group", "Method", "vs", "Set", "Subset", "Subgroup", "Attribute"}   # căn trái trong .tex


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    """'0.812 ± 0.010'; std không có (chỉ 1 seed) -> chỉ in mean."""
    if _isnan(mean):
        return "–"
    return f"{mean:.{digits}f}" if _isnan(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def _escape_text(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("±", r"$\pm$")]:
        text = text.replace(a, b)
    return text


def _latex_escape(s: object) -> str:
    """Escape ký tự đặc biệt của LaTeX, trừ các đoạn công thức `$...$` (giữ nguyên)."""
    parts = str(s).split("$")
    if len(parts) % 2 == 0:              # số `$` lẻ -> không phải công thức, escape toàn bộ
        return _escape_text(str(s)).replace("$", r"\$")
    return "".join(_escape_text(p) if i % 2 == 0 else f"${p}$" for i, p in enumerate(parts))


def plain_label(s: object) -> object:
    """Tên cho .csv: bỏ ký hiệu LaTeX trong `$...$` — 'Filter ($S_{\\text{div}}$)' -> 'Filter (S_div)'."""
    if not isinstance(s, str) or "$" not in s:
        return s
    return re.sub(r"\\text\{([^}]*)\}", r"\1", s).replace("$", "").replace("{", "").replace("}", "")


def method_label(method: str, labels: dict[str, str] | None) -> str:
    return (labels or {}).get(method, method)


CLASS_WEIGHT_SUFFIX = " (class-weighted)"


def labels_for_runs(labels: dict[str, str] | None, runs: pd.DataFrame | None) -> dict[str, str]:
    """Tên hiển thị khớp với cách đã train: thêm " (class-weighted)" cho phương pháp mà MỌI lần chạy đều có class weight
    (đọc từ trường `class_weight` trong .npz). Nhờ đó bảng của cấu hình không class weight (ISIC v9, RSNA) không bị ghi sai.
    """
    out = dict(labels or {})
    if runs is None or "class_weight" not in runs:
        return out
    for method, flags in runs.groupby("method")["class_weight"]:
        flags = flags.dropna()
        if len(flags) and bool(flags.all()) and not out.get(method, method).endswith(CLASS_WEIGHT_SUFFIX):
            out[method] = out.get(method, method) + CLASS_WEIGHT_SUFFIX
    return out


def method_rank(method: str, labels: dict[str, str] | None) -> tuple[int, str]:
    """Thứ tự hàng: theo thứ tự trong `labels`, phương pháp không có tên hiển thị xếp sau (theo mã)."""
    order = list(labels or {})
    return (order.index(method), "") if method in order else (len(order), method)


def method_group(method: str, groups: dict[str, list[str]] | None) -> str:
    return next((g for g, members in (groups or {}).items() if method in members), "")


def to_latex(table: pd.DataFrame, caption: str, label: str, bold: dict[tuple[int, str], bool] | None = None) -> str:
    """Bảng booktabs; `bold[(hàng, cột)] = True` -> in đậm ô đó."""
    bold = bold or {}
    cols = list(table.columns)
    align = "".join("l" if c in TEXT_COLUMNS else "c" for c in cols)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(rf"\textbf{{{_latex_escape(c)}}}" for c in cols) + r" \\", r"\midrule"]
    for i, (_, row) in enumerate(table.iterrows()):
        cells = []
        for c in cols:
            cell = _latex_escape(row[c])
            cells.append(rf"\textbf{{{cell}}}" if bold.get((i, c)) else cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_table(out_dir: str | Path, name: str, formatted: pd.DataFrame, raw: pd.DataFrame, caption: str,
                latex: bool = True, bold: dict[tuple[int, str], bool] | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.csv", out_dir / f"{name}.json"]
    formatted.apply(lambda col: col.map(plain_label)).to_csv(paths[0], index=False)
    records = json.loads(raw.replace({np.nan: None}).to_json(orient="records"))
    paths[1].write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    if latex:
        paths.append(out_dir / f"{name}.tex")
        paths[2].write_text(to_latex(formatted, caption, f"tab:{name}", bold), encoding="utf-8")
    return paths


def _best_rows(values: pd.Series, groups: pd.Series, lower_is_better: bool) -> set[int]:
    best = set()
    for _, idx in values.groupby(groups).groups.items():
        sub = values.loc[idx].dropna()
        if len(sub):
            target = sub.min() if lower_is_better else sub.max()
            best |= set(sub.index[sub == target])
    return best


def _sort_by_method(df: pd.DataFrame, labels: dict[str, str] | None, first: str) -> pd.DataFrame:
    """Sắp hàng theo (`first`, thứ tự phương pháp trong `labels`)."""
    ranked = df.assign(_rank=df["method"].map(lambda m: method_rank(m, labels)))
    return ranked.sort_values([first, "_rank"], kind="stable").drop(columns="_rank")


def classification_table(summary: pd.DataFrame, metrics: list[str] = KEY_METRICS[:8],
                         labels: dict[str, str] | None = None,
                         groups: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Bảng chính: model × phương pháp, mean ± std qua seed; in đậm giá trị tốt nhất của từng model.

    `labels`: mã -> tên hiển thị (và thứ tự hàng); `groups`: tên nhóm -> [mã] (thêm cột Group).
    """
    df = _sort_by_method(summary, labels, "model").reset_index(drop=True)
    out = pd.DataFrame({"Model": df["model"]})
    if groups:
        out["Group"] = [method_group(m, groups) for m in df["method"]]
    out["Method"] = [method_label(m, labels) for m in df["method"]]
    out["Seeds"] = df["n_seeds"]
    bold = {}
    for m in metrics:
        out[METRIC_LABELS[m]] = [fmt_mean_std(a, b) for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
        for i in _best_rows(df[f"{m}_mean"], df["model"], lower_is_better=False):
            bold[(i, METRIC_LABELS[m])] = True
    return out, bold


def _subgroup_label(attribute: object, group: object) -> str:
    return f"{attribute}={group}"


def significance_table(cmp: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Mỗi model (và mỗi nhóm con nếu có): các phép so sánh với M0 trước, rồi các cặp bổ sung (vs ROS, vs Unfiltered
    GAN). ΔAUC (ensemble 3 seed) + CI + p bootstrap; ΔAUC theo từng seed (mean ± std) và p Holm nếu có."""
    sub = "group" in cmp
    df = cmp.assign(_vs=cmp["vs"].map(lambda m: method_rank(m, labels)),
                    _m=cmp["method"].map(lambda m: method_rank(m, labels)),
                    _f=cmp["family"].map({"primary": 0}).fillna(1) if "family" in cmp else 0)
    keys = ["model", "attribute", "group", "_f", "_vs", "_m"] if sub else ["model", "_f", "_vs", "_m"]
    df = df.sort_values(keys, kind="stable").reset_index(drop=True)
    out = pd.DataFrame({"Model": df["model"]})
    if sub:
        out["Subgroup"] = [_subgroup_label(a, g) for a, g in zip(df["attribute"], df["group"])]
    if "family" in df:   # giả thuyết chính (Holm riêng) trước, khám phá sau
        out["Family"] = ["Primary" if f == "primary" else "Exploratory" for f in df["family"]]
    out["Method"] = [method_label(m, labels) for m in df["method"]]
    out["vs"] = [method_label(m, labels) for m in df["vs"]]
    out["Seeds"] = df["n_seeds"]
    out["ΔAUC"] = [f"{d:+.4f}" for d in df["delta_auc"]]
    out["95% CI"] = [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(df["ci95_low"], df["ci95_high"])]
    out["p"] = [f"{p:.3f}" for p in df["p_value"]]
    if "p_holm" in df:
        out["p (Holm)"] = ["–" if _isnan(p) else f"{p:.3f}" for p in df["p_holm"]]
    if "delta_auc_seed_mean" in df:
        out["ΔAUC per seed"] = [("–" if _isnan(m) else f"{m:+.4f}" if _isnan(s) else f"{m:+.4f} ± {s:.4f}")
                                for m, s in zip(df["delta_auc_seed_mean"], df["delta_auc_seed_std"])]
    return out


def subgroup_table(summary: pd.DataFrame, labels: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """AUC / PR-AUC (mean ± std qua seed) của từng model × nhóm con × phương pháp; in đậm tốt nhất mỗi model × nhóm."""
    df = summary.assign(_rank=summary["method"].map(lambda m: method_rank(m, labels)))
    df = df.sort_values(["model", "attribute", "group", "_rank"], kind="stable").reset_index(drop=True)
    sub = [_subgroup_label(a, g) for a, g in zip(df["attribute"], df["group"])]
    out = pd.DataFrame({"Model": df["model"], "Subgroup": sub,
                        "N (pos)": [f"{n} ({p})" for n, p in zip(df["n"], df["n_pos"])],
                        "Method": [method_label(m, labels) for m in df["method"]], "Seeds": df["n_seeds"]})
    bold = {}
    keys = df["model"] + "|" + pd.Series(sub)
    for m in ("roc_auc", "pr_auc"):
        out[METRIC_LABELS[m]] = [fmt_mean_std(a, b) for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
        for i in _best_rows(df[f"{m}_mean"], keys, lower_is_better=False):
            bold[(i, METRIC_LABELS[m])] = True
    return out, bold


def subgroup_reference_table(ref: pd.DataFrame) -> pd.DataFrame:
    """Mốc 'chỉ dùng thuộc tính': AUC khi dự đoán nhãn chỉ bằng thuộc tính, tỉ lệ thuộc tính trong lớp dương / âm."""
    return pd.DataFrame({"Subgroup": [_subgroup_label(a, g) for a, g in zip(ref["attribute"], ref["group"])],
                         "Share in positives": [f"{v:.3f}" for v in ref["share_in_positive"]],
                         "Share in negatives": [f"{v:.3f}" for v in ref["share_in_negative"]],
                         "AUC (attribute only)": [f"{v:.3f}" for v in ref["auc_attribute_only"]]})


def subgroup_share_table(share: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Tỉ lệ thuộc tính (ví dụ AP) trong ảnh thật và trong ảnh sinh mỗi phương pháp chọn (probe trên E_v)."""
    names = [SET_LABELS.get(m, method_label(m, labels)) for m in share["method"]]
    return pd.DataFrame({"Attribute": [_subgroup_label(a, v) for a, v in zip(share["attribute"], share["value"])],
                         "Set": share["set"], "Method": names, "N": share["n"],
                         "True share": ["–" if _isnan(v) else f"{v:.3f}" for v in share["true_share"]],
                         "Predicted share": [f"{v:.3f}" for v in share["predicted_share"]]})


def generation_table(quality: pd.DataFrame, labels: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """KID ×10³ (mean ± std), FID (tham khảo), đa dạng, SSIM nội bộ, AUC thật-vs-sinh, chỉ số học thuộc (không tô đậm:
    tốt nhất là ≈ mốc ảnh thật, không phải nhỏ / lớn nhất; `generation_quality` cũ không có cột này -> "–")."""
    df = quality.reset_index(drop=True)
    names = [SET_LABELS.get(m, method_label(m, labels)) for m in df["method"]]
    out = pd.DataFrame({"Set": df["set"], "Method": names, "N": df["n"],
                        "KID×1e3": [fmt_mean_std(m * 1e3, s * 1e3, 2) for m, s in zip(df["kid"], df["kid_std"])],
                        "FID (ref.)": [f"{v:.1f}" if pd.notna(v) else "–" for v in df["fid"]],
                        "Diversity": [f"{v:.3f}" if pd.notna(v) else "–" for v in df["diversity"]],
                        "SSIM": [f"{v:.3f}" if pd.notna(v) else "–" for v in df["ssim"]],
                        "AUC real/synth": [f"{v:.3f}" if pd.notna(v) else "–" for v in df["auc_real_vs_synth"]]})
    for col, label in [("nn_dist_ratio", "NN ratio"), ("near_copy_rate", "Near-copy")]:
        out[label] = [f"{v:.3f}" if pd.notna(v) else "–" for v in df.get(col, pd.Series(np.nan, index=df.index))]
    bold = {}
    groups = df["set"]
    for col, label in [("kid", "KID×1e3"), ("fid", "FID (ref.)"), ("diversity", "Diversity"), ("ssim", "SSIM"),
                       ("auc_real_vs_synth", "AUC real/synth")]:
        for i in _best_rows(df[col], groups, lower_is_better=col in LOWER_IS_BETTER):
            bold[(i, label)] = True
    return out, bold


SPLIT_AUC = (("train_roc_auc", "Train AUC"), ("val_roc_auc", "Val AUC"), ("roc_auc", "Test AUC"))


def split_auc_table(runs: pd.DataFrame, labels: dict[str, str] | None = None,
                    groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """AUC trên ảnh THẬT của train / val / test (mean ± std qua seed) cạnh nhau + khoảng cách train − test (học thuộc).
    Train AUC chỉ có khi .npz có p_train (từ 1.18.0); thiếu -> "–"."""
    cols = [c for c, _ in SPLIT_AUC if c in runs]
    g = runs.groupby(["model", "method"])
    mean, std = g[cols].mean(), g[cols].std(ddof=1)
    df = mean.reset_index().assign(_rank=lambda d: d["method"].map(lambda m: method_rank(m, labels)))
    df = df.sort_values(["model", "_rank"], kind="stable").reset_index(drop=True)
    out = pd.DataFrame({"Model": df["model"]})
    if groups:
        out["Group"] = [method_group(m, groups) for m in df["method"]]
    out["Method"] = [method_label(m, labels) for m in df["method"]]
    out["Seeds"] = [int(g.size()[(mo, me)]) for mo, me in zip(df["model"], df["method"])]
    for c, name in SPLIT_AUC:
        out[name] = ([fmt_mean_std(mean.loc[(mo, me), c], std.loc[(mo, me), c]) for mo, me in
                      zip(df["model"], df["method"])] if c in cols else "–")
    if "train_roc_auc" in cols:
        out["Train − Test"] = [("–" if _isnan(mean.loc[(mo, me), "train_roc_auc"])
                                else f"{mean.loc[(mo, me), 'train_roc_auc'] - mean.loc[(mo, me), 'roc_auc']:+.3f}")
                               for mo, me in zip(df["model"], df["method"])]
    return out


def dataset_table(card: dict) -> pd.DataFrame:
    rows = []
    for subset in ("train", "val", "test"):
        counts = card["counts"].get(subset, {})
        rows.append({"Subset": subset, **{c: counts.get(c, 0) for c in card["classes"]},
                     "Total": sum(counts.values())})
    return pd.DataFrame(rows)
