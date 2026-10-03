"""Giao diện dòng lệnh `dass`.

    dass -c configs/experiments/brain_tumor_dass.yaml run                      # toàn bộ thực nghiệm
    dass -c configs/experiments/isic2016_dass.yaml prepare                     # một stage
    dass -c configs/experiments/isic2016_dass.yaml train --model resnet50 --seeds 2026 2027
    dass -c configs/experiments/isic2016_dass.yaml -c configs/experiments/smoke.yaml run
    dass -c configs/experiments/isic2016_dass.yaml --set selection.gamma=0.25 --tag gamma025 run --from select

Tuỳ chọn chung (-c, --set, --tag, -v) đặt TRƯỚC tên lệnh.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

from .models.classifiers import MODEL_NAMES
from .utils import setup_logging

STAGE_COMMANDS = ("prepare", "gan", "sample", "fingerprint", "select", "train", "evaluate", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dass", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", dest="configs", action="append", default=[], metavar="YAML",
                        help="file cấu hình (dùng nhiều lần được; file sau ghi đè file trước)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="ghi đè một khoá, ví dụ --set selection.gamma=0.25 (dùng nhiều lần được)")
    parser.add_argument("--tag", help="hậu tố cho paths.run_tag -> thư mục kết quả riêng cho ablation")
    parser.add_argument("-v", "--verbose", action="store_true", help="log mức DEBUG")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<lệnh>")

    sub.add_parser("show-config", help="in cấu hình đã hợp nhất (JSON)")
    sub.add_parser("prepare", help="đọc nguồn, nhận diện số kênh, tiền xử lý, chia, dataset card")
    p = sub.add_parser("label-stats", help="(nguồn DICOM, nhãn MD.ai) thống kê nhãn theo ảnh + hình ảnh mẫu mỗi nhãn")
    p.add_argument("--focus", nargs="*", default=["Exclude"], metavar="NHÃN",
                   help="nhãn cần xem các nhãn đi kèm (mặc định: Exclude)")
    p.add_argument("--samples", type=int, default=4, metavar="N",
                   help="số ảnh mẫu mỗi nhãn -> results_<run_tag>/label_samples.png (0 = không vẽ)")

    p = sub.add_parser("gan-setup", help="clone + vá StyleGAN2-ADA, biên dịch plugin CUDA")
    p.add_argument("--no-reset", action="store_true", help="không đưa repo về nguyên bản trước khi vá")
    p.add_argument("--skip-verify", action="store_true", help="bỏ qua bước biên dịch / kiểm tra plugin CUDA")
    p.add_argument("--clear-ext-cache", action="store_true", help="xoá ~/.cache/torch_extensions trước")

    p = sub.add_parser("gan", help="train GAN (early stopping theo KID, tự resume) + báo cáo")
    p.add_argument("--fresh-start", action="store_true", help="XOÁ trạng thái GAN trên Drive và train lại từ đầu")
    p.add_argument("--dry-run", action="store_true", help="chỉ kiểm tra dataset + cấu hình, không train")

    sub.add_parser("sample", help="sinh (hoặc khôi phục) candidate pool")
    sub.add_parser("fingerprint", help="(tuỳ chọn) fingerprint miền tần số của ảnh sinh — Frank et al.")
    p = sub.add_parser("select", help="E_v / E_d, chấm điểm, chọn ảnh M0–M6, kiểm tra shortcut")
    p.add_argument("--force", action="store_true", help="chọn lại dù đã có selections.json")

    p = sub.add_parser("train", help="train một backbone trên mọi biến thể")
    p.add_argument("--model", required=True, help=f"một trong {', '.join(MODEL_NAMES)} (không phân biệt hoa thường)")
    p.add_argument("--seeds", type=int, nargs="+", help="mặc định: classifier.seeds")

    p = sub.add_parser("evaluate", help="metric phân loại + KID/FID (Inception-v3) + bootstrap -> metrics/")
    p.add_argument("--skip-generative", action="store_true", help="bỏ phần KID / FID (không cần GPU)")

    sub.add_parser("report", help="xuất bảng cho bài báo -> tables/*.csv, *.json, *.tex")

    p = sub.add_parser("run", help="chạy chuỗi stage, mỗi stage một tiến trình riêng")
    p.add_argument("--from", dest="start", default="prepare", choices=STAGE_COMMANDS)
    p.add_argument("--to", dest="end", default="report", choices=STAGE_COMMANDS)
    p.add_argument("--with-fingerprint", action="store_true", help="chạy thêm stage fingerprint")
    p.add_argument("--models", nargs="+", help="mặc định: classifier.models")
    p.add_argument("--seeds", type=int, nargs="+", help="mặc định: classifier.seeds")
    return parser


def _global_args(args: argparse.Namespace) -> list[str]:
    out = []
    for c in args.configs:
        out += ["-c", c]
    for o in args.overrides:
        out += ["--set", o]
    if args.tag:
        out += ["--tag", args.tag]
    if args.verbose:
        out.append("-v")
    return out


def _run_chain(args: argparse.Namespace, cfg) -> None:
    from .pipeline.stages import PIPELINE

    order = list(PIPELINE)
    if args.with_fingerprint:
        order.insert(order.index("select"), "fingerprint")
    if args.start not in order or args.end not in order:
        raise SystemExit("--from / --to phải nằm trong chuỗi stage (thêm --with-fingerprint nếu cần)")
    chain = order[order.index(args.start):order.index(args.end) + 1]
    base = [sys.executable, "-m", "dass", *_global_args(args)]
    for stage in chain:
        if stage == "train":
            for model in args.models or cfg.classifier.models:
                cmd = [*base, "train", "--model", model] + (["--seeds", *map(str, args.seeds)] if args.seeds else [])
                logging.getLogger("dass").info("▶ %s", " ".join(cmd[3:]))
                subprocess.run(cmd, check=True)
        else:
            logging.getLogger("dass").info("▶ %s", stage)
            subprocess.run([*base, stage], check=True)


def _label_stats(cfg, focus: list[str], samples: int) -> None:
    import pandas as pd

    from .config import Layout
    from .data.sources import DicomCsvSource, build_source

    layout = Layout(cfg)
    source = build_source(cfg, layout)
    if not isinstance(source, DicomCsvSource):
        raise SystemExit("label-stats chỉ dùng cho nguồn dicom_csv (nhãn JSON MD.ai)")
    name, table, co = source.label_report(focus)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"Nhãn trong {name} (một ảnh có thể mang nhiều nhãn; lớp cuối cùng: dương = "
              f"{cfg.data.source.positive_labels}):\n{table.to_string()}")
        for label, s in co.items():
            print(f"\nCác nhãn khác trên ảnh có '{label}':\n" + (s.to_string() if len(s) else "  (không có ảnh nào)"))
    if samples > 0:
        from .analysis.figures import plot_label_samples

        plot_label_samples(source.label_samples(samples), table["ảnh"].drop("TỔNG").to_dict(),
                           f"{cfg.data.name}: {samples} ảnh ngẫu nhiên mỗi nhãn — khung đỏ = khung của nhãn ở hàng đó; "
                           "dưới ảnh: lớp cuối cùng · tư thế chụp", layout.label_samples_png)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    import matplotlib

    matplotlib.use("Agg")   # stage chỉ lưu hình ra file

    from .config import load_config

    if not args.configs:
        raise SystemExit("Cần ít nhất một file cấu hình: dass -c configs/experiments/<thực nghiệm>.yaml <lệnh>")
    try:
        cfg = load_config(args.configs, args.overrides, args.tag)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Lỗi cấu hình: {e}") from None
    cmd = args.command

    if cmd == "show-config":
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    elif cmd == "run":
        _run_chain(args, cfg)
    elif cmd == "prepare":
        from .pipeline.stages import prepare
        prepare.run(cfg)
    elif cmd == "label-stats":
        _label_stats(cfg, args.focus, args.samples)
    elif cmd == "gan-setup":
        from .pipeline.stages import gan
        gan.setup(cfg, reset=not args.no_reset, verify=not args.skip_verify, clear_cache=args.clear_ext_cache)
    elif cmd == "gan":
        from .pipeline.stages import gan
        gan.run(cfg, fresh_start=args.fresh_start, dry_run=args.dry_run)
    elif cmd == "sample":
        from .pipeline.stages import sample
        sample.run(cfg)
    elif cmd == "fingerprint":
        from .pipeline.stages import fingerprint
        fingerprint.run(cfg)
    elif cmd == "select":
        from .pipeline.stages import select
        select.run(cfg, force=args.force)
    elif cmd == "train":
        from .pipeline.stages import train
        train.run(cfg, args.model, args.seeds)
    elif cmd == "evaluate":
        from .pipeline.stages import evaluate
        evaluate.run(cfg, skip_generative=args.skip_generative)
    elif cmd == "report":
        from .pipeline.stages import report
        report.run(cfg)


if __name__ == "__main__":
    main()
