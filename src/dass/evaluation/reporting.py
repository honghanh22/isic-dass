"""Xuất bảng cho bài báo: mỗi bảng 3 định dạng.

- `<tên>.csv`  — đã định dạng (mean ± std), dán thẳng vào bảng biểu.
- `<tên>.json` — số liệu thô (records), cho xử lý tiếp bằng code.
- `<tên>.tex`  — LaTeX booktabs, in đậm giá trị tốt nhất mỗi cột trong từng nhóm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .classification import KEY_METRICS, METRIC_LABELS

LOWER_IS_BETTER = {"kid", "fid", "auc_real_vs_synth", "ssim"}


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "–"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _latex_escape(s: object) -> str:
    text = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("±", r"$\pm$")]:
        text = text.replace(a, b)
    return text


def to_latex(table: pd.DataFrame, caption: str, label: str, bold: dict[tuple[int, str], bool] | None = None) -> str:
    """Bảng booktabs; `bold[(hàng, cột)] = True` -> in đậm ô đó."""
    bold = bold or {}
    cols = list(table.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 2 + "c" * (len(cols) - 2) + "}", r"\toprule",
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
    formatted.to_csv(paths[0], index=False)
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


def classification_table(summary: pd.DataFrame, metrics: list[str] = KEY_METRICS[:8]) -> tuple[pd.DataFrame, dict]:
    """Bảng chính: model × phương pháp, mean ± std qua seed; in đậm giá trị tốt nhất của từng model."""
    df = summary.sort_values(["model", "method"]).reset_index(drop=True)
    out = pd.DataFrame({"Model": df["model"], "Method": df["method"], "Seeds": df["n_seeds"]})
    bold = {}
    for m in metrics:
        out[METRIC_LABELS[m]] = [fmt_mean_std(a, b) for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
        for i in _best_rows(df[f"{m}_mean"], df["model"], lower_is_better=False):
            bold[(i, METRIC_LABELS[m])] = True
    return out, bold


def significance_table(cmp: pd.DataFrame) -> pd.DataFrame:
    df = cmp.sort_values(["model", "method"]).reset_index(drop=True)
    return pd.DataFrame({
        "Model": df["model"], "Method": df["method"], "vs": df["vs"], "Seeds": df["n_seeds"],
        "ΔAUC": [f"{d:+.4f}" for d in df["delta_auc"]],
        "95% CI": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(df["ci95_low"], df["ci95_high"])],
        "p": [f"{p:.3f}" for p in df["p_value"]],
    })


def generation_table(quality: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """KID ×10³ (mean ± std), FID (tham khảo), đa dạng, SSIM nội bộ, AUC thật-vs-sinh."""
    df = quality.reset_index(drop=True)
    out = pd.DataFrame({"Set": df["set"], "Method": df["method"], "N": df["n"],
                        "KID×1e3": [fmt_mean_std(m * 1e3, s * 1e3, 2) for m, s in zip(df["kid"], df["kid_std"])],
                        "FID (ref.)": [f"{v:.1f}" if pd.notna(v) else "–" for v in df["fid"]],
                        "Diversity": [f"{v:.3f}" if pd.notna(v) else "–" for v in df["diversity"]],
                        "SSIM": [f"{v:.3f}" if pd.notna(v) else "–" for v in df["ssim"]],
                        "AUC real/synth": [f"{v:.3f}" if pd.notna(v) else "–" for v in df["auc_real_vs_synth"]]})
    bold = {}
    groups = df["set"]
    for col, label in [("kid", "KID×1e3"), ("fid", "FID (ref.)"), ("diversity", "Diversity"), ("ssim", "SSIM"),
                       ("auc_real_vs_synth", "AUC real/synth")]:
        for i in _best_rows(df[col], groups, lower_is_better=col in LOWER_IS_BETTER):
            bold[(i, label)] = True
    return out, bold


def dataset_table(card: dict) -> pd.DataFrame:
    rows = []
    for subset in ("train", "val", "test"):
        counts = card["counts"].get(subset, {})
        rows.append({"Subset": subset, **{c: counts.get(c, 0) for c in card["classes"]},
                     "Total": sum(counts.values())})
    return pd.DataFrame(rows)
