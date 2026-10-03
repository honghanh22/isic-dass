"""Nguồn ảnh DICOM + bảng nhãn CSV (ví dụ RSNA Pneumonia Detection Challenge 2018).

Tự dò cấu trúc thư mục đã tải về:
- ảnh: mọi file `*.dcm` dưới `images_root` (đệ quy); nếu chưa có mà chỉ có file nén (.zip / .tar / .tar.gz / .tgz)
  thì giải nén ra ổ cục bộ (`extract_dir`) rồi tìm lại;
- nhãn: `labels_path`, hoặc (nếu để trống) file CSV đầu tiên dưới `images_root` có đủ cột `id_column` và
  `label_column`. Một ảnh có nhiều dòng (nhiều khung bệnh) -> nhãn = giá trị lớn nhất (có ít nhất một khung -> dương).

Ảnh không có trong bảng nhãn (ví dụ tập test chưa công bố nhãn của cuộc thi) bị bỏ qua. Có thể lấy một tập con phân
tầng theo nhãn (`subset_size`, seed cố định). Mỗi ảnh được chuyển sang PNG xám 1 kênh (đảo nếu MONOCHROME1), lưu
`<dst>/<lớp>/<id>.png`. Thông tin DICOM (tư thế chụp, giới tính, tuổi) ghi ra `metadata_csv` để báo cáo / kiểm tra
shortcut (ví dụ tư thế AP gắn với bệnh nặng). Lần gọi sau đọc lại đánh dấu `.ingest.json`, không quét lại Drive.
"""

from __future__ import annotations

import json
import logging
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..image_io import write_png
from .base import DatasetSource
from .csv_source import normalize_label

log = logging.getLogger(__name__)

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
META_TAGS = ("ViewPosition", "PatientSex", "PatientAge", "PhotometricInterpretation", "Rows", "Columns")
_MARKER = ".ingest.json"


def find_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.name.lower().endswith(suffixes))


def extract_archives(archives: list[Path], out_dir: Path) -> None:
    """Giải nén (bỏ qua file đã giải nén: đánh dấu `<tên>.done`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in archives:
        done = out_dir / f"{a.name}.done"
        if done.exists():
            continue
        log.info("Giải nén %s -> %s ...", a.name, out_dir)
        if a.name.lower().endswith(".zip"):
            with zipfile.ZipFile(a) as z:
                z.extractall(out_dir)
        else:
            with tarfile.open(a) as t:
                try:
                    t.extractall(out_dir, filter="data")        # Python >= 3.12: chặn đường dẫn ra ngoài out_dir
                except TypeError:
                    t.extractall(out_dir)
        done.write_text("ok")


def inventory(roots: list[Path], limit: int = 40) -> str:
    """Tóm tắt nội dung thư mục để chẩn đoán: số file theo đuôi + tên các file không phải .dcm (kèm dung lượng)."""
    by_ext, others = {}, []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            ext = "".join(p.suffixes[-2:]).lower() or "(không đuôi)"
            by_ext[ext] = by_ext.get(ext, 0) + 1
            if ext != ".dcm":
                others.append(f"  {p.relative_to(root)}  ({p.stat().st_size / 1e6:.1f} MB)")
    head = "Số file theo đuôi: " + (", ".join(f"{e}: {n}" for e, n in sorted(by_ext.items())) or "(thư mục rỗng)")
    tail = others[:limit] + ([f"  … và {len(others) - limit} file khác"] if len(others) > limit else [])
    return head + "\nCác file không phải .dcm:\n" + ("\n".join(tail) or "  (không có)")


def search_label_csv(roots: list[Path], id_column: str, label_column: str) -> tuple[Path | None, list]:
    """CSV đầu tiên có đủ hai cột; kèm danh sách (CSV, cột) đã xem để báo lỗi."""
    seen = []
    for root in roots:
        if not root.is_dir():
            continue
        for csv in find_files(root, (".csv",)):
            try:
                cols = list(pd.read_csv(csv, nrows=0).columns)
            except Exception:                                # file CSV hỏng / không phải bảng
                continue
            seen.append((csv, cols))
            if id_column in cols and label_column in cols:
                return csv, seen
    return None, seen


def find_label_csv(root: Path, id_column: str, label_column: str) -> Path:
    csv, seen = search_label_csv([root], id_column, label_column)
    if csv is not None:
        return csv
    listing = "\n".join(f"  {c.relative_to(root)}: {cols}" for c, cols in seen) or "  (không có file CSV nào)"
    raise FileNotFoundError(f"Không tìm thấy CSV có cột '{id_column}' và '{label_column}' dưới {root}. CSV tìm thấy:\n"
                            f"{listing}\n{inventory([root])}\n"
                            "Khai báo đúng data.source.train_labels / id_column / label_column.")


def read_dicom_labels(csv: Path, id_column: str, label_column: str, classes: dict[str, int]) -> pd.Series:
    """Series id -> tên lớp. Nhiều dòng cùng id (nhiều khung) -> lấy nhãn lớn nhất."""
    df = pd.read_csv(csv, usecols=[id_column, label_column])
    df[id_column] = df[id_column].astype(str).str.strip()
    per_id = df.groupby(id_column)[label_column].max()
    return per_id.map(lambda v: normalize_label(v, classes)).sort_index()


def stratified_subset(labels: pd.Series, size: int, seed: int) -> pd.Series:
    """Tập con `size` ảnh giữ nguyên tỉ lệ lớp (tất định theo seed). size <= 0 hoặc >= tổng -> giữ tất cả."""
    if size <= 0 or size >= len(labels):
        return labels
    rng = np.random.default_rng(seed)
    frac = size / len(labels)
    keep = []
    for _, ids in sorted(labels.groupby(labels).groups.items()):
        ids = sorted(ids)
        n = int(round(len(ids) * frac))
        keep += [ids[i] for i in sorted(rng.choice(len(ids), size=n, replace=False))]
    return labels.loc[sorted(keep)]


def dicom_to_uint8(ds) -> np.ndarray:
    """Mảng điểm ảnh -> uint8 (H, W); MONOCHROME1 (trắng = giá trị thấp) được đảo; ảnh > 8 bit co giãn min-max."""
    arr = ds.pixel_array.astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Chỉ hỗ trợ DICOM 2D một kênh, nhận shape {arr.shape}")
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    if arr.min() < 0 or arr.max() > 255:
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


class DicomCsvSource(DatasetSource):
    def __init__(self, classes: dict[str, int], images_root: Path, labels_path: Path | None, id_column: str,
                 label_column: str, subset_size: int, seed: int, extract_dir: Path, metadata_csv: Path):
        super().__init__(classes)
        self.images_root, self.labels_path = Path(images_root), labels_path
        self.id_column, self.label_column = id_column, label_column
        self.subset_size, self.seed = subset_size, seed
        self.extract_dir, self.metadata_csv = Path(extract_dir), Path(metadata_csv)

    @property
    def has_test_set(self) -> bool:
        return False                      # test tách từ cùng nguồn (split stratified)

    def _extract(self) -> bool:
        """Giải nén mọi file nén dưới `images_root` ra `extract_dir` (một lần mỗi server). True nếu có file nén."""
        archives = find_files(self.images_root, ARCHIVE_SUFFIXES)
        if archives:
            extract_archives(archives, self.extract_dir)
        return bool(archives)

    def _labels_csv(self) -> Path:
        """CSV nhãn: khai báo sẵn, hoặc tự tìm trong thư mục; chưa thấy thì giải nén các file nén rồi tìm lại."""
        if self.labels_path is not None:
            return self.labels_path
        csv, seen = search_label_csv([self.images_root], self.id_column, self.label_column)
        if csv is None and self._extract():
            csv, seen = search_label_csv([self.images_root, self.extract_dir], self.id_column, self.label_column)
        if csv is not None:
            return csv
        listing = "\n".join(f"  {c.name}: {cols}" for c, cols in seen) or "  (không có file CSV nào)"
        raise FileNotFoundError(
            f"Không tìm thấy CSV nhãn có cột '{self.id_column}' và '{self.label_column}' (đã tìm cả trong file nén).\n"
            f"CSV tìm thấy:\n{listing}\n{inventory([self.images_root, self.extract_dir])}\n"
            "Khai báo đúng data.source.train_labels / id_column / label_column.")

    def _signature(self) -> dict:
        return {"labels": str(self.labels_path or "auto"), "id_column": self.id_column,
                "label_column": self.label_column, "subset_size": self.subset_size, "seed": self.seed}

    def ingest(self, subset: str, dst_root: str | Path) -> dict[str, int]:
        if subset != "train":
            raise ValueError("Nguồn DICOM chỉ có tập train (test tách bằng split)")
        dst_root = Path(dst_root)
        marker = dst_root / _MARKER
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get("signature") == self._signature():
                return saved["counts"]
            raise ValueError(f"Ảnh đã chuyển ở {dst_root} theo thiết lập khác ({saved.get('signature')}). "
                             "Dùng run_tag / data.name mới khi đổi nhãn hoặc subset_size.")

        csv = self._labels_csv()
        labels = read_dicom_labels(csv, self.id_column, self.label_column, self.classes)
        log.info("Nhãn từ %s: %d ảnh %s", csv.name, len(labels), labels.value_counts().to_dict())
        labels = stratified_subset(labels, self.subset_size, self.seed)
        if self.subset_size:
            log.info("Tập con phân tầng (seed %d): %d ảnh %s", self.seed, len(labels), labels.value_counts().to_dict())

        dicoms = find_files(self.images_root, (".dcm",))
        if not dicoms and self._extract():
            dicoms = find_files(self.extract_dir, (".dcm",))
        if not dicoms:
            raise FileNotFoundError(f"Không có file .dcm dưới {self.images_root} (kể cả sau khi giải nén).\n"
                                    f"{inventory([self.images_root, self.extract_dir])}")
        index = {p.stem: p for p in dicoms}
        log.info("Tìm thấy %d file DICOM", len(index))

        import pydicom

        for label in self.classes:
            (dst_root / label).mkdir(parents=True, exist_ok=True)
        counts, missing, meta = dict.fromkeys(self.classes, 0), [], []
        for image_id, label in tqdm(labels.items(), total=len(labels), desc="DICOM -> PNG"):
            src = index.get(image_id)
            if src is None:
                missing.append(image_id)
                continue
            dst = dst_root / label / f"{image_id}.png"
            ds = pydicom.dcmread(src, stop_before_pixels=dst.exists())
            if not dst.exists():
                write_png(dicom_to_uint8(ds), dst)
            counts[label] += 1
            meta.append({"image_id": image_id, "label": label,
                         **{t: getattr(ds, t, None) for t in META_TAGS}})
        if missing:
            raise FileNotFoundError(f"{len(missing)} ảnh có nhãn nhưng không có file DICOM (ví dụ {missing[:3]}) "
                                    f"dưới {self.images_root}")

        meta = pd.DataFrame(meta)
        self.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        meta.astype(str).to_csv(self.metadata_csv, index=False)
        if "ViewPosition" in meta:
            view = pd.crosstab(meta["label"], meta["ViewPosition"].astype(str), normalize="index").round(3)
            log.info("Tư thế chụp theo lớp (tỉ lệ; AP gắn với bệnh nặng -> nguy cơ shortcut):\n%s", view.to_string())
        marker.write_text(json.dumps({"signature": self._signature(), "counts": counts}))
        return counts
