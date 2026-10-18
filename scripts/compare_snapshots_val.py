"""So hai snapshot GAN bằng lợi ích cho classifier, CHỈ trên val (không chạm test).

    python scripts/compare_snapshots_val.py -c configs/server/pcam.yaml --tag srv --kimg 1000 1500 \\
        --model ConvNeXtTiny --seeds 2026 2027

Với mỗi snapshot `snapshots/snapshot-XXXXXXkimg.pkl`: sinh g ảnh lớp thiểu số (g = n_đa số − n_thiểu số, seed
`generator.gen_seed`, ψ như pool) — tương đương M1 (chọn ngẫu nhiên) — ghép tập train "Balanced GAN" (ảnh thật + g ảnh
sinh lớp thiểu số), train classifier theo đúng giao thức của lần chạy, rồi dự đoán VAL: ROC-AUC, PR-AUC, sensitivity /
specificity ở ngưỡng 0,5. Cùng seed sinh ảnh và cùng seed classifier cho mọi snapshot (ghép cặp).
Kết quả từng (snapshot, seed): `results_<run>/snapshot_val/<model>__<kimg>kimg__s<seed>.json` — đã có thì bỏ qua
(chạy lại được sau khi bị ngắt; classifier dở dang train tiếp theo epoch).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dass.config import load_config
from dass.utils import read_json, setup_logging, write_json_atomic

log = logging.getLogger("compare_snapshots_val")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--kimg", type=int, nargs="+", required=True)
    ap.add_argument("--model", default="ConvNeXtTiny")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026])
    args = ap.parse_args()
    setup_logging()

    from dass.data.variants import Variant, assemble_variant
    from dass.engine.trainer import predict_dir, train_classifier
    from dass.models.generator.patches import ensure_stylegan_repo
    from dass.models.generator.sampler import generate_images
    from dass.pipeline.context import Context
    from dass.pipeline.stages import init_tensorflow

    cfg = load_config([Path(c) for c in args.configs], tag=args.tag)
    init_tensorflow(cfg)
    ctx = Context.create(cfg, "compare_snapshots_val")
    lay, b, g = ctx.layout, ctx.budget, cfg.generator
    out_dir = lay.results_dir / "snapshot_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_stylegan_repo(lay.sg2_repo)

    for kimg in args.kimg:
        todo = [s for s in args.seeds if not (out_dir / f"{args.model}__{kimg}kimg__s{s}.json").exists()]
        if not todo:
            log.info("[bỏ qua, đã có] %d kimg", kimg)
            continue
        pkl = lay.gan_snapshots_dir / f"snapshot-{kimg:06d}kimg.pkl"
        synth_dir = lay.candidates / f"snapshot_{kimg}kimg_{b.minority}_n{b.n_select}"
        synth = sorted(str(p) for p in synth_dir.glob("*.png")) if synth_dir.exists() else []
        if len(synth) != b.n_select:
            synth = generate_images(lay.sg2_repo, pkl, synth_dir, ctx.class_idx(b.minority), b.n_select, g.trunc_psi,
                                    g.gen_seed, ctx.channels, g.channel_tolerance, cfg.data.force_grayscale)
        vdir = lay.variants / f"snapshot_{kimg}kimg_random"
        log.info("%d kimg: %s", kimg, assemble_variant(vdir, lay.train_pp, ctx.split, b.minority, synth))
        variant = Variant(tag=f"snapshot{kimg}kimg_random", dir=vdir, method=f"snapshot_{kimg}kimg",
                          class_weight=False, augment=False)
        for seed in todo:
            log.info("=== %s | %d kimg | seed %d ===", args.model, kimg, seed)
            model, preprocess, best_epoch, _ = train_classifier(cfg, lay, args.model, variant, seed, ctx.channels)
            y, p, _ = predict_dir(model, preprocess, vdir / "val", cfg, ctx.channels)      # CHỈ val
            q = p >= cfg.evaluation.threshold
            res = {"kimg": kimg, "seed": seed, "model": args.model, "best_epoch": best_epoch,
                   "val_auc": float(roc_auc_score(y, p)), "val_pr_auc": float(average_precision_score(y, p)),
                   "val_sensitivity": float(q[y == 1].mean()), "val_specificity": float((~q[y == 0]).mean()),
                   "n_synth": len(synth)}
            write_json_atomic(out_dir / f"{args.model}__{kimg}kimg__s{seed}.json", res)
            log.info("val AUC = %.4f | PR-AUC %.4f | sens %.3f | spec %.3f", res["val_auc"], res["val_pr_auc"],
                     res["val_sensitivity"], res["val_specificity"])
            del model

    rows = [read_json(p) for p in sorted(out_dir.glob(f"{args.model}__*kimg__s*.json"))]
    print("\nkimg  seed   val AUC  PR-AUC  sens   spec")
    for r in sorted(rows, key=lambda r: (r["kimg"], r["seed"])):
        print(f"{r['kimg']:5d} {r['seed']:5d}  {r['val_auc']:.4f}  {r['val_pr_auc']:.4f}  "
              f"{r['val_sensitivity']:.3f}  {r['val_specificity']:.3f}")
    for k in sorted({r["kimg"] for r in rows}):
        v = [r["val_auc"] for r in rows if r["kimg"] == k]
        print(f"{k} kimg: val AUC TB {np.mean(v):.4f} ({len(v)} seed)")


if __name__ == "__main__":
    main()
