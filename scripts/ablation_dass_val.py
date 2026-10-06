"""Ablation DASS chọn trên VAL (không dự đoán test): E_d có gây hại không, λ và γ nên là bao nhiêu.

    python scripts/ablation_dass_val.py -c configs/server/rsna_pneumonia.yaml --tag srv

Thiết kế KHAI BÁO TRƯỚC (ghi trong code, không đổi sau khi xem kết quả):
- Chỉ biến thể M6 (S = M̃_v + β·M̃_d + γ·S̃_div, α = 1), cùng pool / split / giao thức train của lần chạy, model
  `--model` (mặc định EfficientNetV2B0), MỘT seed (`--seed`, mặc định 2026).
- Lưới 18 cấu hình: E_d ∈ {none (β = 0), plain (E_d hiện tại), crossfit (K = `--folds`, mỗi E_d so với ảnh nó KHÔNG
  học)} × λ ∈ {0; 0,5; 1} (λ_v = λ_d = λ) × γ ∈ {0; 0,5}.
- Chỉ dự đoán VAL; test KHÔNG được chạm tới. Kết quả: `results_<run>/ablation_val/<cấu hình>.json` + bảng
  `ablation_val/summary.csv`.
- Quy tắc chọn: val AUC cao nhất; nếu hơn cấu hình mặc định (plain, λ = 1, γ = 0,5) dưới 0,003 -> giữ mặc định.
- Chạy lại được: cấu hình đã có json thì bỏ qua (bị ngắt giữa chừng -> chạy lại lệnh).
Điểm S± của E_v / E_d thường lấy từ `embeddings.npz` của `select`; E_d cross-fit train một lần, điểm lưu vào
`ablation_val/crossfit_scores.npz`. Mọi margin tuyến tính theo λ nên tính lại được với mọi λ từ S⁺ và S⁻.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dass.config import load_config
from dass.utils import minmax, setup_logging, write_json_atomic

log = logging.getLogger("ablation_val")

E_D_MODES = ("none", "plain", "crossfit")
LAMBDAS = (0.0, 0.5, 1.0)
GAMMAS = (0.0, 0.5)
DEFAULT = ("plain", 1.0, 0.5)
MIN_GAIN = 0.003


def config_name(ed: str, lam: float, gamma: float) -> str:
    return f"ed-{ed}__lam{lam:g}__gamma{gamma:g}"


def select_m6(scores: dict[str, np.ndarray], z_v_pool: np.ndarray, n: int, ed: str, lam: float,
              gamma: float) -> list[int]:
    """M6 với β = 0 (ed = none) hoặc 1; E_d thường hay cross-fit tuỳ khoá điểm truyền vào."""
    from dass.selection.strategies import greedy_dass_select, top_n

    base = minmax(scores["S_v_pos"] - lam * scores["S_v_neg"])
    if ed != "none":
        pre = "cf_" if ed == "crossfit" else ""
        base = base + minmax(scores[f"{pre}S_d_pos"] - lam * scores[f"{pre}S_d_neg"])
    return greedy_dass_select(z_v_pool, base, n, gamma) if gamma > 0 else top_n(base, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default="EfficientNetV2B0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    setup_logging()

    from dass.data.variants import Variant, assemble_variant
    from dass.engine.trainer import predict_dir, train_classifier
    from dass.pipeline.context import Context
    from dass.pipeline.pool import resolve_candidate_pool
    from dass.pipeline.stages import init_tensorflow
    from dass.pipeline.stages.select import _crossfit_md

    cfg = load_config([Path(c) for c in args.configs], tag=args.tag,
                      overrides=["classifier.save_weights_to_drive=false"])
    init_tensorflow(cfg)
    ctx = Context.create(cfg, "ablation_val")
    lay, b = ctx.layout, ctx.budget
    out_dir = lay.results_dir / "ablation_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final

    with np.load(lay.embeddings_npz, allow_pickle=True) as e:
        if [str(n) for n in e["pool_names"]] != [Path(p).name for p in pool]:
            raise RuntimeError("embeddings.npz không khớp pool hiện tại -> chạy lại `select`")
        z_v_pool = e["z_v_pool"]
        scores = {k: e[f"score__{k}"] for k in ("S_v_pos", "S_v_neg", "S_d_pos", "S_d_neg")}

    cf_npz = out_dir / "crossfit_scores.npz"
    if not cf_npz.exists():
        cfg.encoder.e_d_crossfit_folds = args.folds
        cf = _crossfit_md(cfg, ctx, pool, lay.clf_dir, train_if_missing=True)
        np.savez(cf_npz, S_d_pos=cf["S_d_pos"], S_d_neg=cf["S_d_neg"], folds=args.folds)
    with np.load(cf_npz) as c:
        scores.update(cf_S_d_pos=c["S_d_pos"], cf_S_d_neg=c["S_d_neg"])

    for ed, lam, gamma in itertools.product(E_D_MODES, LAMBDAS, GAMMAS):
        name = config_name(ed, lam, gamma)
        res_json = out_dir / f"{name}.json"
        if res_json.exists():
            log.info("[bỏ qua, đã có] %s", name)
            continue
        idx = select_m6(scores, z_v_pool, b.n_select, ed, lam, gamma)
        vdir = lay.variants.parent / "ablation_val" / name
        assemble_variant(vdir, lay.train_pp, ctx.split, b.minority, [pool[i] for i in idx])
        variant = Variant(tag=f"abl_{name}", dir=vdir, method="M6_dass", lam=f"v{lam:g}_d{lam:g}",
                          feature_space="Ev" if ed == "none" else "Ev+Ed", class_weight=False)
        log.info("=== %s | %s | seed %d ===", args.model, name, args.seed)
        model, preprocess, best_epoch = train_classifier(cfg, lay, args.model, variant, args.seed, ctx.channels)
        y_val, p_val, _ = predict_dir(model, preprocess, vdir / "val", cfg, ctx.channels)   # CHỈ val
        auc = float(roc_auc_score(y_val, p_val))
        write_json_atomic(res_json, {"config": name, "e_d": ed, "lambda": lam, "gamma": gamma,
                                     "val_auc": auc, "best_epoch": best_epoch, "seed": args.seed,
                                     "model": args.model, "n_selected": len(idx),
                                     "selected": sorted(Path(pool[i]).name for i in idx)})
        log.info("val AUC = %.4f | %s", auc, name)
        shutil.rmtree(vdir, ignore_errors=True)
        del model

    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("ed-*.json"))]
    df = pd.DataFrame(rows).drop(columns=["selected"]).sort_values("val_auc", ascending=False)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(df.round(4).to_string(index=False))
    if len(df) == len(E_D_MODES) * len(LAMBDAS) * len(GAMMAS):
        best = df.iloc[0]
        default = df[df["config"] == config_name(*DEFAULT)].iloc[0]
        keep_default = best["val_auc"] - default["val_auc"] < MIN_GAIN
        choice = default if keep_default else best
        write_json_atomic(out_dir / "decision.json", {
            "rule": f"val AUC cao nhất; hơn mặc định < {MIN_GAIN} -> giữ mặc định",
            "best": best["config"], "best_val_auc": best["val_auc"],
            "default": default["config"], "default_val_auc": default["val_auc"],
            "chosen": choice["config"], "kept_default": bool(keep_default)})
        print(f"\nChọn: {choice['config']} (val AUC {choice['val_auc']:.4f}; "
              f"mặc định {default['val_auc']:.4f}{' — giữ mặc định' if keep_default else ''})")


if __name__ == "__main__":
    main()
