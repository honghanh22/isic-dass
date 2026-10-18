"""Kiểm tra trực tiếp đường tắt "dấu vết GAN -> lớp thiểu số": classifier ĐÃ train chấm ảnh SINH của lớp ĐA SỐ.

    python scripts/shortcut_probe.py -c configs/server/pcam.yaml --tag srv --model ResNet50

- Classifier: mọi phương pháp B0–B3, M1–M6 của lần chạy (`--model`, mọi seed có trọng số; xác suất trung bình qua
  seed). Không train gì. B0–B3 chưa từng thấy ảnh sinh -> mốc so sánh.
- Ảnh được chấm: ảnh THẬT lớp đa số / thiểu số của tập VAL (không dùng test: test chỉ được dự đoán một lần); ảnh SINH
  lớp đa số (pool riêng `pool_<lớp>_probe`, `--n-majority` ảnh, cùng GAN, seed gen_seed + `--seed-offset`; không
  phương pháp nào train trên ảnh sinh lớp đa số); ảnh sinh lớp thiểu số trong pool mà DASS (M6) không chọn.
- Dấu hiệu đường tắt: ảnh sinh lớp ĐA SỐ có xác suất lớp thiểu số cao hơn hẳn ảnh thật lớp đa số (`shift` =
  TB p(sinh đa số) − TB p(thật đa số); `pos_synth_majority` so với `pos_real_majority`; `auc_synth_vs_real_majority`
  > 0,5 rõ), và mức này lớn hơn nhiều so với B0–B3.
- Ghi `results_<run>/metrics/shortcut_probe_<model>.csv`. Cần GPU (suy luận).
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

log = logging.getLogger("shortcut_probe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-majority", type=int, default=1000)
    ap.add_argument("--seed-offset", type=int, default=2, help="seed ảnh sinh lớp đa số = gen_seed + offset")
    args = ap.parse_args()
    setup_logging()

    from dass.data.loaders import configure_gpu, paths_dataset
    from dass.models.classifiers import build_model
    from dass.pipeline.context import Context
    from dass.pipeline.pool import load_selections, resolve_candidate_pool, resolve_pool
    from dass.pipeline.stages import save_metrics
    from dass.selection import DASS, METHODS

    cfg = load_config([Path(c) for c in args.configs], args.overrides, tag=args.tag)
    configure_gpu()
    ctx = Context.create(cfg, "shortcut_probe")
    lay, b = ctx.layout, ctx.budget
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)
    chosen = set(selections[DASS])
    sets = {
        "real_majority": ctx.val_paths[b.majority],
        "synth_majority": resolve_pool(ctx, b.majority, args.n_majority, cfg.generator.gen_seed + args.seed_offset,
                                        True, label="_probe").final,
        "real_minority": ctx.val_paths[b.minority],
        "synth_minority_unseen": [p for p in pool if p not in chosen],
    }
    log.info("Số ảnh: %s", {k: len(v) for k, v in sets.items()})

    rows = []
    for method in METHODS:
        weights = sorted(lay.clf_dir.glob(f"{args.model}__{method}_k*__s*.weights.h5"))
        if not weights:
            log.warning("Không có trọng số %s -> bỏ qua", method)
            continue
        probs = {k: [] for k in sets}
        for w in weights:
            model, _, preprocess = build_model(args.model, cfg.classifier)
            model.load_weights(str(w))
            for k, paths in sets.items():
                ds = paths_dataset(paths, cfg.classifier.size, ctx.channels, preprocess, cfg.classifier.batch_size)
                probs[k].append(model.predict(ds, verbose=0).ravel())
        p = {k: np.mean(v, axis=0) for k, v in probs.items()}
        y = np.r_[np.ones(len(p["synth_majority"])), np.zeros(len(p["real_majority"]))]
        row = {"method": method, "n_seeds": len(weights)}
        for k, v in p.items():
            row[f"p_{k}"] = float(v.mean())
            row[f"pos_{k}"] = float((v >= 0.5).mean())
        row["shift"] = row["p_synth_majority"] - row["p_real_majority"]
        row["auc_synth_vs_real_majority"] = float(roc_auc_score(y, np.r_[p["synth_majority"], p["real_majority"]]))
        rows.append(row)
        log.info("%s: p(thật đa số) %.3f | p(sinh đa số) %.3f | shift %+.3f | AUC %.3f", method,
                 row["p_real_majority"], row["p_synth_majority"], row["shift"], row["auc_synth_vs_real_majority"])
    df = pd.DataFrame(rows).assign(model=args.model)
    save_metrics(lay, f"shortcut_probe_{args.model}", df)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
