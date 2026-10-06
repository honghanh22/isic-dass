"""Train MỘT biến thể (theo `selections.json` của lần chạy) với MỘT seed, CHỈ đánh giá trên VAL (không chạm test).

    python scripts/train_variant_val.py -c configs/server/rsna_pneumonia.yaml --tag srv_rc \\
        --set generator.match_resize_chain=true --method M6_dass --seed 2026

Dùng để thử một thay đổi (ví dụ cách xử lý ảnh sinh) trên val trước khi khoá cấu hình. Trọng số lưu như lần chạy
chính (`checkpoints_<run>/classifiers/<model>__<biến thể>__s<seed>.weights.h5`) để `scripts/shortcut_probe.py` dùng
lại được; val AUC ghi vào `results_<run>/val_only/<model>__<biến thể>__s<seed>.json`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.metrics import roc_auc_score

from dass.config import load_config
from dass.utils import setup_logging, write_json_atomic

log = logging.getLogger("train_variant_val")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--method", default="M6_dass")
    ap.add_argument("--model", default="EfficientNetV2B0")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    setup_logging()

    from dass.data.variants import Variant, assemble_variant, variant_tag
    from dass.engine.trainer import predict_dir, train_classifier
    from dass.pipeline.context import Context
    from dass.pipeline.stages import init_tensorflow
    from dass.pipeline.stages.train import variant_inputs

    cfg = load_config([Path(c) for c in args.configs], args.overrides, tag=args.tag)
    init_tensorflow(cfg)
    ctx = Context.create(cfg, "train_variant_val")
    lay, b, sel = ctx.layout, ctx.budget, cfg.selection
    selections, majority_synth, extra_real, majority_extra_real, cw_methods = variant_inputs(cfg, ctx)   # như `train`
    synth = selections[args.method]
    tag = variant_tag(args.method, sel.pool_mult)
    vdir = lay.variants / tag
    assemble_variant(vdir, lay.train_pp, ctx.split, b.minority, synth, b.majority,
                     majority_synth.get(args.method, ()), extra_real.get(args.method, ()),
                     majority_extra_real.get(args.method, ()))
    variant = Variant(tag=tag, dir=vdir, method=args.method, lam=f"v{sel.lambda_v:g}_d{sel.lambda_d:g}",
                      feature_space="Ev+Ed", class_weight=args.method in cw_methods)
    model, preprocess, best_epoch = train_classifier(cfg, lay, args.model, variant, args.seed, ctx.channels)
    y_val, p_val, _ = predict_dir(model, preprocess, vdir / "val", cfg, ctx.channels)     # CHỈ val
    auc = float(roc_auc_score(y_val, p_val))
    out = lay.results_dir / "val_only" / f"{args.model}__{tag}__s{args.seed}.json"
    write_json_atomic(out, {"method": args.method, "model": args.model, "seed": args.seed, "val_auc": auc,
                            "best_epoch": best_epoch, "n_synth": len(synth),
                            "n_real_duplicates": len(extra_real.get(args.method, ())),
                            "n_majority_duplicates": len(majority_extra_real.get(args.method, ())), "run_tag": cfg.paths.run_tag,
                            "overrides": args.overrides})
    log.info("val AUC = %.4f -> %s", auc, out)


if __name__ == "__main__":
    main()
