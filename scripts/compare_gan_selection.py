"""So hai thước đo chọn snapshot GAN trên CÙNG các snapshot đã lưu (chỉ đọc, không đổi gan_state.json / best.pkl).

    python scripts/compare_gan_selection.py -c configs/server/pcam.yaml --tag srv

- Thước đo đang dùng: lấy từ `gan_state.json` (vd FID-E_d lớp thiểu số, làm mượt EMA).
- Thước đo đối chiếu: luật cũ của ISIC / RSNA — KID trên Inception-v3, lớp thiểu số, 1.000 ảnh sinh (seed 123), so với
  toàn bộ ảnh train lớp thiểu số, không làm mượt; phát lại cùng luật dừng sớm (min_kimg, patience, min_rel_delta).
- Ghi `results_<run>/metrics/gan_selection_comparison.csv` (mỗi snapshot một dòng: KID, FID-E_d, hạng) + hình
  `results_<run>/gan/selection_comparison.png`; in snapshot mỗi luật chọn và tương quan hạng Spearman.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import pandas as pd

from dass.config import load_config
from dass.utils import setup_logging

log = logging.getLogger("compare_gan_selection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", dest="configs", action="append", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    setup_logging()

    from scipy.stats import spearmanr

    from dass.models.generator.patches import ensure_stylegan_repo
    from dass.models.generator.trainer import default_state, load_state, update_selection
    from dass.pipeline.context import Context
    from dass.pipeline.stages.gan import build_scorer

    cfg = load_config([Path(c) for c in args.configs], tag=args.tag)
    ctx = Context.create(cfg, "compare_gan_selection")
    lay = ctx.layout
    ensure_stylegan_repo(lay.sg2_repo)
    state = load_state(lay.gan_state_json)
    hist = {h["cum_kimg"]: h for h in state["history"]}
    snaps = sorted((int(p.stem.split("-")[1].removesuffix("kimg")), p)
                   for p in lay.gan_snapshots_dir.glob("snapshot-*kimg.pkl"))
    snaps = [(k, p) for k, p in snaps if k in hist]

    # luật cũ: KID-Inception lớp thiểu số, 1.000 ảnh sinh, không làm mượt (các tham số khác giữ nguyên)
    old = dataclasses.replace(cfg.generator, select_metric="kid_inception", select_classes="minority",
                              select_ema_alpha=1.0, kid_n_gen=1000)
    scorer, _ = build_scorer(dataclasses.replace(cfg, generator=old), ctx)
    st, stop = default_state(), None
    rows = []
    for kimg, pkl in snaps:
        kid, _ = scorer(pkl)
        improved = False
        if stop is None:            # sau khi luật cũ đã dừng: vẫn chấm để vẽ đủ đường, nhưng không đổi lựa chọn
            improved, _ = update_selection(st, kid, kimg, old)
            st["history"].append({"cum_kimg": kimg, "kid": kid})
            if st["bad_count"] >= old.patience:
                stop = kimg
        h = hist[kimg]
        rows.append({"cum_kimg": kimg, "kid_inception": kid, "kid_improved": improved,
                     "fid_ed_raw": h["kid"], "fid_ed_smoothed": h.get("smoothed", h["kid"]),
                     **{f"fid_ed_{c}": v for c, v in h.get("per_class", {}).items()}})
        log.info("%5d kimg | KID-Inception %.5f%s | FID-E_d %.2f (mượt %.2f)", kimg, kid,
                 "  <-- tốt nhất (luật cũ)" if improved else "", h["kid"], h.get("smoothed", h["kid"]))
    df = pd.DataFrame(rows)
    rho = spearmanr(df["kid_inception"], df["fid_ed_raw"]).statistic
    rho_late = spearmanr(df.loc[df.cum_kimg >= 400, "kid_inception"],
                         df.loc[df.cum_kimg >= 400, "fid_ed_raw"]).statistic
    lay.metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(lay.metrics_dir / "gan_selection_comparison.csv", index=False)
    print(df.round(5).to_string(index=False))
    print(f"\nLuật cũ (KID-Inception, không làm mượt) chọn: {st['best_cum_kimg']} kimg | luật đang dùng: "
          f"{state['best_cum_kimg']} kimg | luật cũ dừng sớm tại: {stop or 'chưa dừng'} kimg")
    print(f"Spearman(KID-Inception, FID-E_d lớp thiểu số): mọi snapshot {rho:+.3f} | từ 400 kimg {rho_late:+.3f}")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df.cum_kimg, df.kid_inception * 1e3, "o-", color="tab:blue", label="KID-Inception ×1e3 (luật cũ)")
    ax1.set_xlabel("kimg")
    ax1.set_ylabel("KID-Inception ×1e3", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df.cum_kimg, df.fid_ed_smoothed, "s-", color="tab:orange", label="FID-E_d mượt (luật đang dùng)")
    ax2.set_ylabel("FID-E_d (mượt)", color="tab:orange")
    ax1.axvline(st["best_cum_kimg"], color="tab:blue", ls="--")
    ax2.axvline(state["best_cum_kimg"], color="tab:orange", ls=":")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    fig.legend(loc="upper right")
    ax1.set_title(f"Hai thước đo chọn snapshot (lớp thiểu số) — Spearman {rho:+.2f}")
    fig.tight_layout()
    out = lay.gan_fig_dir / "selection_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    log.info("Hình: %s", out)


if __name__ == "__main__":
    main()
