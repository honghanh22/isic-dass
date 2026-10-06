"""Chất lượng ảnh sinh theo nghĩa PHÂN LOẠI: classifier chỉ học từ ảnh thật (M0) chấm ảnh sinh của từng lớp.

    python scripts/synthetic_label_fidelity.py -c configs/server/rsna_pneumonia.yaml --tag srv --model EfficientNetV2B0
    python scripts/synthetic_label_fidelity.py -c configs/server/isic2016.yaml --tag srv --judge ed
    python scripts/synthetic_label_fidelity.py -c configs/server/isic2016.yaml --tag srv --model ResNet50 \
        --judge-run v11_srv

- Người chấm (chỉ học từ ảnh thật): `--judge m0` (mặc định) = M0 (`M0_real_only`) của `--model`, mọi seed có trọng số
  trong `checkpoints_<--judge-run hoặc run hiện tại>/classifiers`, xác suất lấy trung bình qua seed; `--judge ed` = E_d
  (`encoder.e_d_model`, classifier đầy đủ train trên ảnh thật). E_d là chính bộ mã hoá DASS dùng để chọn ảnh -> dòng
  "selected M3 / M4 / M6" của người chấm E_d thiên lệch (vòng tròn); các dòng khác không bị ảnh hưởng.
- Tập được chấm: ảnh THẬT (train / test, mỗi lớp) làm mốc; ảnh SINH lớp thiểu số (pool của lần chạy) và lớp đa số
  (sinh thêm `--n-majority` ảnh, cùng GAN, seed `gen_seed + 1`, như pool M7); tập được chọn của M1 / M4 / M6.
- Chỉ số: xác suất dương trung bình, tỉ lệ dự đoán dương ở ngưỡng 0,5, và "AUC nhãn" = AUC giữa ảnh sinh dương và
  ảnh sinh âm (ảnh sinh có giữ đúng khác biệt giữa hai lớp như ảnh thật không; so với AUC test thật của cùng M0).
- Ghi `results_<run>/metrics/synthetic_label_fidelity.csv`. Cần GPU (sinh ảnh lớp đa số + suy luận).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dass.config import load_config
from dass.utils import setup_logging

log = logging.getLogger("label_fidelity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default="EfficientNetV2B0")
    ap.add_argument("--n-majority", type=int, default=1000)
    ap.add_argument("--judge", choices=("m0", "ed"), default="m0")
    ap.add_argument("--judge-run", default=None, help="run_tag chứa trọng số M0 (mặc định: run hiện tại)")
    args = ap.parse_args()
    setup_logging()

    from dass.data.loaders import configure_gpu, paths_dataset
    from dass.models.classifiers import build_model
    from dass.pipeline.context import Context
    from dass.pipeline.pool import load_selections, resolve_candidate_pool, resolve_pool
    from dass.pipeline.stages import save_metrics

    cfg = load_config([Path(c) for c in args.configs], tag=args.tag)
    configure_gpu()
    ctx = Context.create(cfg, "label_fidelity")
    lay, b = ctx.layout, ctx.budget
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    maj_pool = resolve_pool(ctx, b.majority, args.n_majority, cfg.generator.gen_seed + 1, True).final
    selections = load_selections(ctx, pool)

    test_dir = Path(lay.test_pp)
    sets = {
        f"real train {b.minority}": ctx.train_paths[b.minority],
        f"real train {b.majority}": ctx.train_paths[b.majority],
        f"real test {b.minority}": sorted(str(p) for p in (test_dir / b.minority).iterdir()),
        f"real test {b.majority}": sorted(str(p) for p in (test_dir / b.majority).iterdir()),
        f"synth {b.minority} (pool)": pool,
        f"synth {b.majority}": maj_pool,
        **{f"synth {b.minority} selected {m}": selections[m] for m in ("M1_random", "M4_visual_disease", "M6_dass")
           if selections.get(m)},
    }

    if args.judge == "ed":
        judge_model, weights = cfg.encoder.e_d_model, [lay.e_d_ckpt]
    else:
        clf_dir = (Path(cfg.paths.drive_root) / f"checkpoints_{args.judge_run}" / "classifiers" if args.judge_run
                   else lay.clf_dir)
        judge_model = args.model
        weights = sorted(clf_dir.glob(f"{args.model}__M0_real_only_p*__s*.weights.h5"))
    if not weights or not all(w.exists() for w in weights):
        raise FileNotFoundError(f"Không có trọng số người chấm: {weights}")
    probs = {k: [] for k in sets}
    for w in weights:
        model, _, preprocess = build_model(judge_model, cfg.classifier)
        model.load_weights(str(w))
        log.info("Người chấm: %s", w.name)
        for k, paths in sets.items():
            ds = paths_dataset(paths, cfg.classifier.size, ctx.channels, preprocess, cfg.classifier.batch_size)
            probs[k].append(model.predict(ds, verbose=0).ravel())
    p = {k: np.mean(v, axis=0) for k, v in probs.items()}   # trung bình qua seed

    def pair_auc(pos: str, neg: str) -> float:
        y = np.r_[np.ones(len(p[pos])), np.zeros(len(p[neg]))]
        return float(roc_auc_score(y, np.r_[p[pos], p[neg]]))

    rows = [{"set": k, "n": len(v), "label": b.minority if b.minority in k else b.majority,
             "mean_p_positive": float(v.mean()), "frac_pred_positive": float((v >= 0.5).mean()),
             "median_p_positive": float(np.median(v))} for k, v in p.items()]
    df = pd.DataFrame(rows)
    aucs = pd.DataFrame([
        {"set": "AUC nhãn: real test", "auc": pair_auc(f"real test {b.minority}", f"real test {b.majority}")},
        {"set": "AUC nhãn: synth (pool vs majority)", "auc": pair_auc(f"synth {b.minority} (pool)",
                                                                      f"synth {b.majority}")},
        {"set": "real test minority vs synth minority", "auc": pair_auc(f"real test {b.minority}",
                                                                         f"synth {b.minority} (pool)")},
    ])
    judge = f"{judge_model}_Ed" if args.judge == "ed" else f"{judge_model}_M0" + (f"_{args.judge_run}" if args.judge_run
                                                                                     else "")
    out = pd.concat([df, aucs], ignore_index=True).assign(judge=judge, n_weights=len(weights))
    name = "synthetic_label_fidelity" + ("" if judge == "EfficientNetV2B0_M0" else f"__{judge}")
    save_metrics(lay, name, out)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
