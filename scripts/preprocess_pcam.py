"""Tiền xử lý PCam (PatchCamelyon v1) thành MỘT file HDF5 mất cân bằng, đọc lại không cần file gốc.

    python scripts/preprocess_pcam.py --raw /data/honghanh/datasets/PCam/raw/pcamv1 \\
        --out /data/honghanh/datasets/PCam/pcam_imb12.h5

Các bước (cố định, seed --seed):
1. Gộp cả ba split gốc (train / valid / test) thành một tập.
2. Bỏ ảnh trùng hệt (md5 pixel) trên toàn tập gộp, giữ ảnh đầu mỗi nhóm: PCam lấy mẫu lặp lớp dương để đạt
   50 / 50, và vài ảnh trùng hệt nằm ở hai WSI khác nhau.
3. Bỏ ảnh có > --max-white (mặc định 50 %) pixel nền trắng (xám > 220): nền trắng gần như chỉ có ở lớp âm ->
   đường tắt "trắng = không u".
4. Chia theo WSI (một lát cắt chỉ thuộc một phần) với tỉ lệ --fractions (mặc định 70 / 15 / 15): WSI có ảnh dương
   và WSI chỉ có ảnh âm được xáo trộn và chia riêng, gán tham lam để số ảnh dương sẵn có mỗi phần ~ tỉ lệ.
5. Trong mỗi phần lấy n_pos = round(--n-pos × tỉ lệ phần) ảnh dương và n_neg = --ratio × n_pos ảnh âm (mặc định
   3.000 dương tổng, âm : dương = 12), ngẫu nhiên phân tầng theo WSI (mỗi WSI góp theo tỉ lệ số ảnh của nó).
Ảnh giữ nguyên 96×96×3 uint8 (resize khi nạp). File ra: nhóm train / valid / test, mỗi nhóm có x, y, wsi, coord_x,
coord_y, src_split, src_index (vị trí trong file PCam gốc); attrs ghi tham số, số lượng và md5 của từng mảng x.
Kèm `<out>_split.csv` (image_id,split = train | val) cho `split.type: file` của pipeline (nguồn `hdf5`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SRC_SPLITS = ("train", "valid", "test")
OUT_SPLITS = ("train", "valid", "test")
CHUNK = 8192


def scan_split(raw: Path, split: str) -> pd.DataFrame:
    x = h5py.File(raw / f"camelyonpatch_level_2_split_{split}_x.h5", "r")["x"]
    y = h5py.File(raw / f"camelyonpatch_level_2_split_{split}_y.h5", "r")["y"][:].reshape(-1).astype(np.int8)
    meta = pd.read_csv(raw / f"camelyonpatch_level_2_split_{split}_meta.csv")
    md5, white = [], []
    for i in range(0, len(y), CHUNK):
        b = x[i:i + CHUNK]
        md5 += [hashlib.md5(a.tobytes()).hexdigest() for a in b]
        white.append((b.astype(np.float32).mean(-1) > 220).mean((1, 2)))
    return pd.DataFrame({"src_split": split, "src_index": np.arange(len(y)), "y": y, "md5": md5,
                         "white": np.concatenate(white), "wsi": meta["wsi"].astype(str),
                         "coord_x": meta["coord_x"], "coord_y": meta["coord_y"]})


def assign_wsi(df: pd.DataFrame, fractions: list[float], rng: np.random.Generator) -> dict[str, int]:
    """WSI -> phần (0 / 1 / 2). Hai nhóm WSI (có ảnh dương / chỉ ảnh âm) chia riêng; mỗi WSI (thứ tự ngẫu nhiên,
    lớn trước) vào phần đang thiếu nhiều nhất so với chỉ tiêu, đo bằng số ảnh dương (nhóm 1) hoặc âm (nhóm 2)."""
    per = df.groupby("wsi")["y"].agg(n_pos="sum", n="size")
    per["n_neg"] = per["n"] - per["n_pos"]
    out = {}
    for wsis, col in ((per.index[per.n_pos > 0], "n_pos"), (per.index[per.n_pos == 0], "n_neg")):
        w = per.loc[wsis].iloc[rng.permutation(len(wsis))].sort_values(col, ascending=False, kind="stable")
        target = np.array(fractions) * w[col].sum()
        got = np.zeros(len(fractions))
        for name, v in w[col].items():
            k = int(np.argmax(target - got))
            out[name] = k
            got[k] += v
    return out


def stratified_by_wsi(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n ảnh ngẫu nhiên, mỗi WSI góp ~ tỉ lệ số ảnh của nó (làm tròn theo phần dư lớn nhất)."""
    if n > len(df):
        raise SystemExit(f"cần {n} ảnh nhưng chỉ còn {len(df)}")
    counts = df.groupby("wsi").size()
    quota = counts / counts.sum() * n
    take = np.floor(quota).astype(int)
    take[(quota - take).sort_values(ascending=False).index[:n - take.sum()]] += 1
    return pd.concat([g.iloc[rng.choice(len(g), size=take[w], replace=False)]
                      for w, g in df.groupby("wsi") if take[w] > 0])


def write_split_csv(h5_path: Path) -> Path:
    """CSV `image_id,split` (train | val) đi kèm file .h5 cho split.type = file của pipeline; image_id giống tên PNG
    mà nguồn `hdf5` ghi ra (`<src_split>_<src_index:06d>`)."""
    out = h5_path.with_name(h5_path.stem + "_split.csv")
    rows = []
    with h5py.File(h5_path, "r") as f:
        for group, split in (("train", "train"), ("valid", "val")):
            g = f[group]
            rows += [(f"{s}_{int(i):06d}", split) for s, i in zip(g["src_split"][:].astype(str), g["src_index"][:])]
    pd.DataFrame(rows, columns=["image_id", "split"]).to_csv(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-pos", type=int, default=3000, help="tổng số ảnh dương (cả ba phần)")
    ap.add_argument("--ratio", type=int, default=12, help="số ảnh âm / ảnh dương trong mỗi phần")
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--max-white", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split-csv-only", action="store_true", help="chỉ ghi lại CSV chia train / val từ --out có sẵn")
    args = ap.parse_args()
    if args.split_csv_only:
        print("->", write_split_csv(args.out))
        return
    rng = np.random.default_rng(args.seed)

    df = pd.concat([scan_split(args.raw, s) for s in SRC_SPLITS], ignore_index=True)
    stats = {"goc": df.y.value_counts().sort_index().tolist()}
    df = df.drop_duplicates("md5")
    stats["bo_trung"] = df.y.value_counts().sort_index().tolist()
    df = df[df.white <= args.max_white]
    stats["bo_nen_trang"] = df.y.value_counts().sort_index().tolist()
    stats["so_wsi"] = int(df.wsi.nunique())

    part = df.wsi.map(assign_wsi(df, args.fractions, rng))
    n_pos = np.round(np.array(args.fractions) * args.n_pos).astype(int)
    n_pos[0] = args.n_pos - n_pos[1:].sum()          # tổng đúng --n-pos
    chosen = {}
    for k, s in enumerate(OUT_SPLITS):
        d = df[part == k]
        sel = pd.concat([stratified_by_wsi(d[d.y == 0], int(n_pos[k] * args.ratio), rng),
                         stratified_by_wsi(d[d.y == 1], int(n_pos[k]), rng)])
        chosen[s] = sel.sort_values(["src_split", "src_index"])
        stats[s] = {"am": int((sel.y == 0).sum()), "duong": int((sel.y == 1).sum()),
                    "wsi": int(d.wsi.nunique()), "wsi_co_duong": int(d[d.y == 1].wsi.nunique())}

    w = {s: set(chosen[s].wsi) for s in OUT_SPLITS}
    assert not (w["train"] & w["valid"]) and not (w["train"] & w["test"]) and not (w["valid"] & w["test"])

    src = {s: h5py.File(args.raw / f"camelyonpatch_level_2_split_{s}_x.h5", "r")["x"] for s in SRC_SPLITS}
    tmp = args.out.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["description"] = ("PCam v1 đã tiền xử lý: gộp 3 split, bỏ trùng, bỏ nền trắng, chia lại theo WSI, "
                                  "mất cân bằng âm : dương")
        f.attrs["params"] = json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})
        f.attrs["classes"] = json.dumps({"negative": 0, "positive": 1})
        f.attrs["stats"] = json.dumps(stats, ensure_ascii=False)
        for s in OUT_SPLITS:
            sel = chosen[s]
            xs = np.concatenate([src[ss][np.sort(g.src_index.to_numpy())] for ss, g in sel.groupby("src_split", sort=False)])
            g = f.create_group(s)
            g.create_dataset("x", data=xs, chunks=(256, 96, 96, 3), compression="gzip", compression_opts=4)
            g.create_dataset("y", data=sel.y.to_numpy(np.int8))
            g.create_dataset("src_split", data=sel.src_split.to_numpy().astype("S"))
            g.create_dataset("src_index", data=sel.src_index.to_numpy())
            g.create_dataset("coord_x", data=sel.coord_x.to_numpy())
            g.create_dataset("coord_y", data=sel.coord_y.to_numpy())
            g.create_dataset("wsi", data=sel.wsi.to_numpy().astype("S"))
            g.attrs["md5_x"] = hashlib.md5(xs.tobytes()).hexdigest()
    tmp.replace(args.out)
    write_split_csv(args.out)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print("->", args.out)


if __name__ == "__main__":
    main()
