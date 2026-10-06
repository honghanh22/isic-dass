"""Kiểm tra trực tiếp đường tắt "dấu vết GAN -> lớp thiểu số": classifier ĐÃ train chấm ảnh SINH của lớp ĐA SỐ.

    python scripts/shortcut_probe.py -c configs/server/rsna_pneumonia.yaml --tag srv

- Classifier: M0, M0b (chưa từng thấy ảnh sinh -> đối chứng), M1, M6, M8 (nếu có) của lần chạy (`--model`, mọi seed có trọng số;
  xác suất trung bình qua seed). Không train gì.
- Ảnh được chấm: ảnh THẬT lớp đa số / thiểu số của tập VAL (không dùng test: test chỉ được dự đoán một lần); ảnh SINH
  lớp đa số (pool riêng `pool_<lớp>_probe`, `--n-majority` ảnh, cùng GAN, seed gen_seed + `--seed-offset` = 2 — KHÁC
  seed + 1 mà M7 / M8 dùng để train, nên không biến thể nào đã thấy; trước 1.17.0 dùng seed + 1); ảnh sinh lớp thiểu
  số không được M6 lẫn M8 chọn.
- Dấu hiệu đường tắt: với M6 (và M1), ảnh sinh lớp ĐA SỐ có xác suất lớp thiểu số cao hơn hẳn ảnh thật lớp đa số
  (`shift` = TB p(sinh đa số) − TB p(thật đa số); `auc_synth_vs_real_majority` > 0,5 rõ), và mức này lớn hơn nhiều so
  với M0 / M0b (không thấy ảnh sinh khi train -> chỉ phản ánh khác biệt ảnh, không phải điều đã học).
- Ghi `results_<run>/metrics/shortcut_probe.csv`. Cần GPU (suy luận).
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
METHODS = ("M0_real_only", "M0b_real_oversample", "M0c_real_oversample_matched", "M1_random", "M6_dass",
           "M7_dass_both_classes",
           "M8r_ros_balanced_random", "M8_ros_balanced_synth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--model", default="EfficientNetV2B0")
    ap.add_argument("--n-majority", type=int, default=1000)
    ap.add_argument("--seed-offset", type=int, default=2, help="seed ảnh sinh lớp đa số = gen_seed + offset")
    args = ap.parse_args()
    setup_logging()

    from dass.data.loaders import configure_gpu, paths_dataset
    from dass.models.classifiers import build_model
    from dass.pipeline.context import Context
    from dass.pipeline.pool import load_selections, resolve_candidate_pool, resolve_pool
    from dass.pipeline.stages import save_metrics

    cfg = load_config([Path(c) for c in args.configs], args.overrides, tag=args.tag)
    configure_gpu()
    ctx = Context.create(cfg, "shortcut_probe")
    lay, b = ctx.layout, ctx.budget
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)
    chosen = {p for m in ("M6_dass", "M8r_ros_balanced_random", "M8_ros_balanced_synth") for p in selections.get(m, [])}
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
        weights = sorted(lay.clf_dir.glob(f"{args.model}__{method}_p*__s*.weights.h5"))
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
    save_metrics(lay, "shortcut_probe", df)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
