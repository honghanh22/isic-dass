"""Chấm lại MỌI snapshot GAN đã lưu theo luật chọn hiện tại của config rồi dựng lại `gan_state.json`.

    python scripts/rescore_gan_snapshots.py -c configs/server/pcam.yaml --tag srv

Dùng khi đổi luật chọn snapshot (`generator.select_classes` / `select_ema_alpha`) giữa chừng: GAN chỉ train tiếp khi
lịch sử dùng đúng một luật (`trainer.check_selection_rule`). Cần `generator.keep_snapshots: true`.
- Mỗi snapshot `snapshots/snapshot-XXXXXXkimg.pkl` (tới `cum_kimg` hiện tại) được chấm lại bằng đúng hàm của stage
  `gan` (cùng E_d, cùng seed sinh ảnh, ảnh thật TRAIN) -> điểm từng lớp; điểm chọn = trung bình các lớp được chọn.
- Phát lại luật làm mượt + dừng sớm theo thứ tự kimg; nếu luật mới đã đủ patience ở một mốc thì GAN coi như dừng ở
  đó (các snapshot sau vẫn được giữ nhưng không xét).
- `best.pkl` = snapshot tốt nhất theo luật mới; `latest.pkl` không đổi (train tiếp từ đúng chỗ đã dừng).
- Bản `gan_state.json` cũ được giữ lại thành `gan_state_before_rescore_<thời điểm>.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from dass.config import load_config
from dass.utils import copy_atomic, setup_logging, write_json_atomic

log = logging.getLogger("rescore_gan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()
    setup_logging()

    from dass.models.generator.patches import ensure_stylegan_repo
    from dass.models.generator.trainer import default_state, load_state, update_selection
    from dass.pipeline.context import Context
    from dass.pipeline.stages.gan import build_scorer

    cfg = load_config([Path(c) for c in args.configs], args.overrides, tag=args.tag)
    g = cfg.generator
    if not g.keep_snapshots:
        raise SystemExit("Cần generator.keep_snapshots: true (không có snapshot để chấm lại)")
    ctx = Context.create(cfg, "rescore_gan")
    lay = ctx.layout
    ensure_stylegan_repo(lay.sg2_repo)
    old = load_state(lay.gan_state_json)
    snaps = sorted((int(p.stem.split("-")[1].removesuffix("kimg")), p)
                   for p in lay.gan_snapshots_dir.glob("snapshot-*kimg.pkl"))
    snaps = [(k, p) for k, p in snaps if k <= old["cum_kimg"]]
    missing = sorted({h["cum_kimg"] for h in old["history"]} - {k for k, _ in snaps})
    if missing:
        raise SystemExit(f"Thiếu snapshot cho các mốc {missing} kimg -> không chấm lại được đầy đủ")

    scorer, info = build_scorer(cfg, ctx)
    keep = {k: old.get(k) for k in ("cum_kimg", "data_name", "split_sha1")}
    state = {**default_state(), **keep, **info}
    best_pkl = None
    for kimg, pkl in snaps:
        raw, per_class = scorer(pkl)
        improved, smoothed = update_selection(state, raw, kimg, g)
        if improved:
            best_pkl = pkl
        state["history"].append({"cum_kimg": kimg, "kid": raw, "improved": bool(improved), "per_class": per_class,
                                 **({"smoothed": smoothed} if g.select_ema_alpha < 1 else {})})
        log.info("%5d kimg | điểm %.3f (mượt %.3f) | %s%s", kimg, raw, smoothed,
                 {c: round(v, 2) for c, v in per_class.items()}, "  <-- tốt nhất" if improved else "")
        if state["bad_count"] >= g.patience:
            state.update(finished=True, stop_reason=f"early stopping tại {kimg} kimg (best {state['best_cum_kimg']} "
                                                    "kimg, chấm lại theo luật mới)")
            break

    backup = lay.gan_state_json.with_name(f"gan_state_before_rescore_{time.strftime('%Y%m%d_%H%M%S')}.json")
    shutil.copy2(lay.gan_state_json, backup)
    copy_atomic(best_pkl, lay.gan_best_pkl)
    write_json_atomic(lay.gan_state_json, state)
    print(json.dumps({k: state[k] for k in ("best_cum_kimg", "best_kid", "bad_count", "finished", "stop_reason",
                                            "select_classes")}, ensure_ascii=False, indent=1))
    print(f"state cũ -> {backup.name}; best.pkl <- {best_pkl.name}")


if __name__ == "__main__":
    main()
