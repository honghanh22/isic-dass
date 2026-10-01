"""Giao diện dòng lệnh: `isic-dass [--config FILE] [--set section.key=value ...] <stage> [tuỳ chọn]`.

Ví dụ:
    isic-dass --config configs/default.yaml prepare
    isic-dass --config configs/default.yaml gan-train
    isic-dass --config configs/default.yaml train --model ResNet50 --seeds 2026 2027
    isic-dass --config configs/default.yaml --set classifier.monitor=val_auc train --model DenseNet121
"""

from __future__ import annotations

import argparse
import json
import logging

from .classify.models import MODEL_NAMES
from .config import load_config
from .utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isic-dass", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="file YAML cấu hình (mặc định: giá trị trong config.py)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="ghi đè cấu hình, ví dụ --set gan.batch=32 (dùng nhiều lần được)")
    parser.add_argument("-v", "--verbose", action="store_true", help="log mức DEBUG")
    sub = parser.add_subparsers(dest="stage", required=True, metavar="<stage>")

    sub.add_parser("show-config", help="in cấu hình đã hợp nhất")
    sub.add_parser("prepare", help="copy ảnh từ Drive, tiền xử lý PNG, tách val cố định")

    p = sub.add_parser("gan-setup", help="clone + vá StyleGAN2-ADA, biên dịch plugin CUDA")
    p.add_argument("--no-reset", action="store_true", help="không đưa repo về nguyên bản trước khi vá")
    p.add_argument("--skip-verify", action="store_true", help="bỏ qua bước biên dịch / kiểm tra plugin CUDA")
    p.add_argument("--clear-ext-cache", action="store_true", help="xoá ~/.cache/torch_extensions trước")

    p = sub.add_parser("gan-train", help="train GAN có early stopping theo KID (tự resume)")
    p.add_argument("--fresh-start", action="store_true", help="XOÁ trạng thái GAN trên Drive và train lại từ đầu")
    p.add_argument("--dry-run", action="store_true", help="chỉ kiểm tra dataset + cấu hình, không train")

    sub.add_parser("gan-report", help="đường KID và ảnh mẫu của snapshot tốt nhất")
    sub.add_parser("generate", help="sinh (hoặc khôi phục) candidate pool cho lớp thiểu số")
    sub.add_parser("frequency", help="phân tích fingerprint miền tần số (tuỳ chọn)")
    sub.add_parser("select", help="nhúng E_v/E_d, chấm điểm, chọn ảnh cho M0–M6, chẩn đoán")

    p = sub.add_parser("train", help="train classifier trên mọi biến thể")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--seeds", type=int, nargs="+", help="mặc định: classifier.seeds trong config")

    sub.add_parser("aggregate", help="gộp dự đoán, bảng mean ± std, paired bootstrap")
    sub.add_parser("quality", help="chất lượng tập ảnh sinh được chọn")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    import matplotlib

    matplotlib.use("Agg")   # stage chỉ lưu hình ra file

    cfg = load_config(args.config, args.overrides)
    from . import pipeline as pl

    stage = args.stage
    if stage == "show-config":
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    elif stage == "prepare":
        pl.stage_prepare(cfg)
    elif stage == "gan-setup":
        pl.stage_gan_setup(cfg, reset=not args.no_reset, verify=not args.skip_verify, clear_cache=args.clear_ext_cache)
    elif stage == "gan-train":
        pl.stage_gan_train(cfg, fresh_start=args.fresh_start, dry_run=args.dry_run)
    elif stage == "gan-report":
        pl.stage_gan_report(cfg)
    elif stage == "generate":
        pl.stage_generate(cfg)
    elif stage == "frequency":
        pl.stage_frequency(cfg)
    elif stage == "select":
        pl.stage_select(cfg)
    elif stage == "train":
        pl.stage_train(cfg, args.model, args.seeds)
    elif stage == "aggregate":
        pl.stage_aggregate(cfg)
    elif stage == "quality":
        pl.stage_quality(cfg)


if __name__ == "__main__":
    main()
