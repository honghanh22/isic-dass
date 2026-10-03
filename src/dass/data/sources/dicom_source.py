"""Nguồn ảnh DICOM + nhãn (ví dụ RSNA Pneumonia Detection Challenge 2018).

Tự dò cấu trúc thư mục đã tải về (bỏ qua thư mục kết quả `checkpoints_*`, `results_*` ở mọi cấp):
- ảnh: mọi file `*.dcm` dưới `images_root` (đệ quy, kể cả cây Study/Series/SOP.dcm của MD.ai); nếu thiếu thì giải nén
  các file .zip / .tar / .tar.gz / .tgz ra ổ cục bộ (`extract_dir`) rồi tìm lại. Bản giải nén cục bộ được ưu tiên (đọc
  nhanh hơn Drive).
- nhãn, theo thứ tự:
  1. CSV kiểu Kaggle (`labels_path`, hoặc CSV có cột `id_column` + `label_column`): nhiều dòng cùng mã -> nhãn lớn nhất;
  2. JSON xuất từ MD.ai (bản tải từ trang RSNA, có khoá `labelGroups`): ảnh dương nếu có ít nhất một chú thích mang tên
     trong `positive_labels` (ví dụ "Lung Opacity"), ngược lại âm. Mã ảnh = SOPInstanceUID (hoặc Series / Study UID).
- mapping sang NIH ChestX-ray8 (JSON có tên ảnh dạng `00000013_005.png`): lấy mã bệnh nhân NIH (8 chữ số đầu). Với
  `one_per_patient`, giữ đúng MỘT ảnh cho mỗi bệnh nhân (chọn ngẫu nhiên theo seed) -> không rò rỉ bệnh nhân giữa
  train / val / test.

Ảnh không có nhãn bị bỏ qua. Có thể lấy tập con phân tầng theo nhãn (`subset_size`, seed). Mỗi ảnh được chuyển sang PNG
xám 1 kênh (đảo nếu MONOCHROME1), lưu `<dst>/<lớp>/<mã ảnh>.png`. Thông tin DICOM (tư thế chụp, giới tính, tuổi) và mã
bệnh nhân NIH ghi ra `metadata_csv`. Lần gọi sau đọc lại đánh dấu `.ingest.json`, không quét lại Drive.
"""

from __future__ import annotations

import json
import logging
import re
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
SOURCE_SUFFIXES = (".dcm", ".csv", ".json") + ARCHIVE_SUFFIXES
# drive_root vừa có thể chứa dữ liệu gốc vừa chứa kết quả của pipeline (Layout): bỏ qua các thư mục kết quả khi tìm dữ
# liệu nguồn, để không nhặt nhầm CSV số liệu hay file pool .zip.
OUTPUT_DIR_PREFIXES = ("checkpoints_", "results_")
META_TAGS = ("ViewPosition", "PatientSex", "PatientAge", "PhotometricInterpretation", "Rows", "Columns")
UID_KEYS = ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID")
NIH_IMAGE = re.compile(r"(\d{8})_\d{3}\.png")          # tên ảnh NIH ChestX-ray8: <mã bệnh nhân>_<số thứ tự>.png
MAX_JSON_MB = 500
_MARKER = ".ingest.json"


# ----------------------------------------------------------------------------------------------- tìm file / thư mục
def glob_escape(name: str) -> str:
    """Escape ký tự đặc biệt của glob ([, ], *, ?) trong tên thư mục."""
    return "".join(f"[{c}]" if c in "[]*?" else c for c in name)


def _in_outputs(p: Path, root: Path) -> bool:
    """Nằm trong thư mục kết quả của pipeline (`checkpoints_*`, `results_*`) ở BẤT KỲ cấp nào dưới `root` — thư mục kết
    quả có thể nằm bên trong thư mục dữ liệu (ví dụ `RSNA Pneumonia/Result_Pneumonia/checkpoints_rsna_v1/`)."""
    return any(part.startswith(OUTPUT_DIR_PREFIXES) for part in p.relative_to(root).parts[:-1])


def find_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """File có đuôi `suffixes` dưới `root` (đệ quy), bỏ qua thư mục kết quả của pipeline."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.name.lower().endswith(suffixes) and not _in_outputs(p, root))


def _has_files(root: Path) -> bool:
    """Có dữ liệu nguồn (.dcm / .csv / .json / file nén) ngoài các thư mục kết quả của pipeline."""
    return bool(find_files(root, SOURCE_SUFFIXES))


def _listing(root: Path, limit: int = 15) -> str:
    items = sorted(root.iterdir())
    lines = [f"  {p.name}{'/' if p.is_dir() else f'  ({p.stat().st_size / 1e6:.1f} MB)'}" for p in items[:limit]]
    if len(items) > limit:
        lines.append(f"  … và {len(items) - limit} mục khác")
    return "\n".join(lines) or "  (rỗng)"


def _ext(p: Path) -> str:
    name = p.name.lower()
    return next((s for s in (".tar.gz",) if name.endswith(s)), p.suffix.lower() or "(không đuôi)")


def inventory(roots: list[Path], limit: int = 40) -> str:
    """Tóm tắt nội dung thư mục để chẩn đoán: số file theo đuôi + tên các file không phải .dcm (kèm dung lượng)."""
    by_ext, others = {}, []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or _in_outputs(p, root):
                continue
            ext = _ext(p)
            by_ext[ext] = by_ext.get(ext, 0) + 1
            if ext != ".dcm":
                others.append(f"  {p.relative_to(root)}  ({p.stat().st_size / 1e6:.1f} MB)")
    head = "Số file theo đuôi: " + (", ".join(f"{e}: {n}" for e, n in sorted(by_ext.items())) or "(thư mục rỗng)")
    tail = others[:limit] + ([f"  … và {len(others) - limit} file khác"] if len(others) > limit else [])
    return head + "\nCác file không phải .dcm:\n" + ("\n".join(tail) or "  (không có)")


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


# ------------------------------------------------------------------------------------------------------------ nhãn
def search_label_csv(roots: list[Path], id_column: str, label_column: str) -> tuple[Path | None, list]:
    """CSV đầu tiên có đủ hai cột; kèm danh sách (CSV, cột) đã xem để báo lỗi."""
    seen = []
    for root in roots:
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


def _walk_dicts(obj):
    """Mọi dict lồng trong một cấu trúc JSON (duyệt không đệ quy)."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            yield o
            stack.extend(v for v in o.values() if isinstance(v, (dict, list)))
        elif isinstance(o, list):
            stack.extend(v for v in o if isinstance(v, (dict, list)))


def is_mdai_annotations(obj) -> bool:
    return isinstance(obj, dict) and "labelGroups" in obj


def _iter_mdai(obj: dict):
    """(mã ảnh, tên nhãn, chú thích gốc) của mọi chú thích có mã ảnh và nhãn hợp lệ trong JSON xuất từ MD.ai.

    Mã ảnh = StudyInstanceUID (mỗi ca chụp X-quang ngực của RSNA có đúng một ảnh): chú thích cấp ảnh (khung "Lung
    Opacity") và cấp ca chụp ("Normal", …) được gộp về cùng một khoá. Thiếu Study UID thì dùng Series / SOP UID.
    """
    names = {lab["id"]: str(lab.get("name", "")).strip()
             for g in obj.get("labelGroups", []) for lab in g.get("labels", []) if "id" in lab}
    anns = [a for ds in obj.get("datasets", []) for a in ds.get("annotations", [])] + list(obj.get("annotations", []))
    for a in anns:
        key = next((a[k] for k in reversed(UID_KEYS) if a.get(k)), None)      # Study -> Series -> SOP
        name = names.get(a.get("labelId"))
        if key is not None and name is not None:
            yield key, name, a


def mdai_annotations(obj: dict) -> list[tuple[str, str]]:
    """JSON xuất từ MD.ai -> [(mã ảnh, tên nhãn)], mỗi chú thích một phần tử."""
    return [(key, name) for key, name, _ in _iter_mdai(obj)]


def mdai_boxes(obj: dict) -> dict[str, list[tuple[str, tuple[float, float, float, float]]]]:
    """Mã ảnh -> [(tên nhãn, (x, y, rộng, cao))] của các chú thích dạng khung (toạ độ điểm ảnh của ảnh gốc)."""
    out: dict[str, list] = {}
    for key, name, a in _iter_mdai(obj):
        d = a.get("data")
        if isinstance(d, dict) and all(isinstance(d.get(k), (int, float)) for k in ("x", "y", "width", "height")):
            out.setdefault(key, []).append((name, (d["x"], d["y"], d["width"], d["height"])))
    return out


def sample_per_label(pairs: list[tuple[str, str]], n: int, seed: int) -> dict[str, list[str]]:
    """Mỗi tên nhãn: tối đa `n` mã ảnh ngẫu nhiên (tất định theo seed); nhãn xếp theo số ảnh giảm dần."""
    df = pd.DataFrame(pairs, columns=["image_id", "name"]).drop_duplicates()
    counts = df["name"].value_counts()
    rng = np.random.default_rng(seed)
    out = {}
    for name in sorted(counts.index, key=lambda k: (-counts[k], k)):
        ids = sorted(df.loc[df["name"] == name, "image_id"])
        out[name] = [ids[i] for i in sorted(rng.choice(len(ids), size=min(n, len(ids)), replace=False))]
    return out


def read_mdai_labels(obj: dict, positive_labels: list[str], classes: dict[str, int]) -> tuple[pd.Series, dict]:
    """JSON xuất từ MD.ai -> (Series mã ảnh -> tên lớp, {tên nhãn: số chú thích}). Ảnh dương nếu có ít nhất một chú
    thích mang tên trong `positive_labels`, ngược lại âm."""
    positive = {n.strip().lower() for n in positive_labels}
    by_idx = {i: c for c, i in classes.items()}
    per_image, usage = {}, {}
    for key, name in mdai_annotations(obj):
        usage[name] = usage.get(name, 0) + 1
        per_image[key] = per_image.get(key, False) or name.lower() in positive
    if not per_image:
        raise ValueError(f"JSON MD.ai không có chú thích dùng được (khoá cấp đầu: {list(obj)[:10]}, "
                         f"{len(obj.get('labelGroups', []))} nhóm nhãn)")
    labels = pd.Series({k: by_idx[1] if v else by_idx[0] for k, v in per_image.items()}).sort_index()
    if (labels == by_idx[1]).sum() == 0:
        raise ValueError(f"Không có ảnh dương: không nhãn nào trùng positive_labels={positive_labels}. "
                         f"Nhãn có trong file: {usage}")
    return labels, usage


def label_image_table(pairs: list[tuple[str, str]], labels: pd.Series, class_names: list[str],
                      patients: dict[str, str] | None = None) -> pd.DataFrame:
    """Thống kê theo từng tên nhãn: số chú thích, số ẢNH mang nhãn (nhiều khung / nhiều bác sĩ trên cùng một ảnh chỉ tính
    một lần), % trên tổng số ảnh có nhãn, số ảnh đó theo lớp cuối cùng, số bệnh nhân NIH. Dòng cuối "TỔNG"."""
    df = pd.DataFrame(pairs, columns=["image_id", "name"])
    images = df.drop_duplicates()
    images = images.assign(cls=images["image_id"].map(labels))
    table = pd.crosstab(images["name"], images["cls"]).reindex(columns=class_names, fill_value=0)
    table.insert(0, "chú thích", df["name"].value_counts())
    table.insert(1, "ảnh", images["name"].value_counts())
    table.insert(2, "% ảnh", (100 * table["ảnh"] / len(labels)).round(2))
    total = {"chú thích": len(df), "ảnh": len(labels), "% ảnh": 100.0, **labels.value_counts().to_dict()}
    if patients:
        table["bệnh nhân"] = images.assign(p=images["image_id"].map(patients)).groupby("name")["p"].nunique()
        total["bệnh nhân"] = labels.index.map(patients).dropna().nunique()
    table = table.sort_values(["ảnh", "chú thích"], ascending=False)
    table.loc["TỔNG"] = pd.Series(total)
    counts = [c for c in table.columns if c != "% ảnh"]
    table[counts] = table[counts].astype(int)
    table.index.name, table.columns.name = "nhãn", None
    return table


def cooccurring_labels(pairs: list[tuple[str, str]], focus: str) -> pd.Series:
    """Ảnh mang nhãn `focus`: số ảnh có thêm từng nhãn khác; "(chỉ có nhãn này)" = ảnh không có nhãn nào khác."""
    df = pd.DataFrame(pairs, columns=["image_id", "name"]).drop_duplicates()
    is_focus = df["name"].str.lower() == focus.strip().lower()
    ids = set(df.loc[is_focus, "image_id"])
    other = df[df["image_id"].isin(ids) & ~is_focus]
    out = other["name"].value_counts()
    alone = len(ids) - other["image_id"].nunique()
    if alone:
        out["(chỉ có nhãn này)"] = alone
    out.index.name = None
    return out.rename(f"số ảnh có '{focus}'")


def read_nih_patients(obj) -> dict[str, str]:
    """Mapping RSNA -> NIH ChestX-ray8: {UID (SOP / Series / Study): mã bệnh nhân NIH 8 chữ số}."""
    out = {}
    for d in _walk_dicts(obj):
        nih = None
        for v in d.values():
            m = NIH_IMAGE.search(v) if isinstance(v, str) else None
            if m:
                nih = m.group(1)
                break
        if nih:
            for k in UID_KEYS:
                if isinstance(d.get(k), str):
                    out[d[k]] = nih
    return out


def one_image_per_patient(labels: pd.Series, patients: dict[str, str], seed: int) -> pd.Series:
    """Giữ đúng MỘT ảnh cho mỗi bệnh nhân (ngẫu nhiên theo seed, tất định); ảnh không có mã bệnh nhân giữ nguyên."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for image_id in sorted(labels.index):
        groups.setdefault(patients.get(image_id, f"__{image_id}"), []).append(image_id)
    keep = [ids[int(rng.integers(len(ids)))] if len(ids) > 1 else ids[0] for _, ids in sorted(groups.items())]
    return labels.loc[sorted(keep)]


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


def expand_patients(patients: dict[str, str], files: list[Path]) -> dict[str, str]:
    """Mapping có thể chỉ ghi một loại UID: dùng cây thư mục Study/Series/SOP.dcm để mọi UID của cùng một ảnh đều trỏ
    tới cùng mã bệnh nhân."""
    out = dict(patients)
    for p in files:
        keys = (p.parent.parent.name, p.parent.name, p.stem)
        nih = next((patients[k] for k in keys if k in patients), None)
        if nih is not None:
            out.update(dict.fromkeys(keys, nih))
    return out


def dicom_index(files: list[Path]) -> dict[str, Path]:
    """Mã -> file: tên file (patientId của Kaggle / SOPInstanceUID của MD.ai) và hai thư mục cha (Series, Study UID)."""
    index = {}
    for p in files:
        for key in (p.parent.parent.name, p.parent.name, p.stem):       # cấp chi tiết nhất ghi sau cùng (ưu tiên)
            index[key] = p
    return index


# --------------------------------------------------------------------------------------------------------- nguồn
class DicomCsvSource(DatasetSource):
    def __init__(self, classes: dict[str, int], images_root: Path, labels_path: Path | None, id_column: str,
                 label_column: str, subset_size: int, seed: int, extract_dir: Path, metadata_csv: Path,
                 positive_labels: list[str] | tuple[str, ...] = (), one_per_patient: bool = False):
        super().__init__(classes)
        self.images_root, self.labels_path = Path(images_root), labels_path
        self.id_column, self.label_column = id_column, label_column
        self.subset_size, self.seed = subset_size, seed
        self.extract_dir, self.metadata_csv = Path(extract_dir), Path(metadata_csv)
        self.positive_labels, self.one_per_patient = list(positive_labels), one_per_patient
        self._diag, self._files = None, None          # cache cho lệnh chẩn đoán label-stats

    @property
    def has_test_set(self) -> bool:
        return False                      # test tách từ cùng nguồn (split stratified)

    def _check_root(self) -> None:
        """Thư mục dữ liệu phải có file. Rỗng / không tồn tại mà có đúng MỘT bản trùng tên do Google Drive tạo
        (`<tên> (1)`, `<tên> (2)`, …) chứa dữ liệu -> đọc dữ liệu từ bản đó (kết quả vẫn ghi vào thư mục cấu hình).
        Còn lại -> báo lỗi, liệt kê thư mục cha và nội dung các bản trùng tên."""
        root = self.images_root
        if _has_files(root):
            return
        parent = root.parent
        duplicates = sorted(p for p in parent.glob(f"{glob_escape(root.name)} (*)") if p.is_dir()) \
            if parent.is_dir() else []
        with_data = [p for p in duplicates if _has_files(p)]
        if len(with_data) == 1:
            log.warning("%s không có dữ liệu -> đọc dữ liệu từ %s (bản trùng tên do Google Drive tạo). Kết quả vẫn "
                        "ghi vào %s.", root.name, with_data[0].name, root.name)
            self.images_root = with_data[0]
            return
        siblings = sorted(p.name + ("/" if p.is_dir() else "") for p in parent.iterdir()) if parent.is_dir() else []
        state = "rỗng (chưa có file nào — dữ liệu chưa tải lên xong / chưa đồng bộ?)" if root.is_dir() \
            else "KHÔNG tồn tại (tên thư mục khác?)"
        dup_info = "".join(f"\nNội dung {p.name}/ (tối đa 15 mục):\n" + _listing(p) for p in duplicates)
        raise FileNotFoundError(f"Thư mục dữ liệu {root} {state}.\nNội dung {parent}:\n"
                                + ("\n".join(f"  {s}" for s in siblings) or "  (không đọc được / rỗng)")
                                + dup_info
                                + "\nĐưa dữ liệu vào đúng thư mục (ví dụ tải bằng Kaggle API) hoặc sửa paths.drive_root.")

    def _extract(self) -> bool:
        """Giải nén mọi file nén dưới `images_root` ra `extract_dir` (một lần mỗi server). True nếu có file nén."""
        archives = find_files(self.images_root, ARCHIVE_SUFFIXES)
        if archives:
            extract_archives(archives, self.extract_dir)
        return bool(archives)

    def _read_jsons(self, roots: list[Path]) -> tuple[tuple[dict, Path] | None, dict[str, str]]:
        """((JSON MD.ai đầu tiên, đường dẫn) hoặc None, mapping bệnh nhân NIH) tìm được dưới `roots`."""
        patients: dict[str, str] = {}
        mdai = None
        for root in roots:
            for path in find_files(root, (".json",)):
                if path.stat().st_size > MAX_JSON_MB * 1e6:
                    continue
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if is_mdai_annotations(obj):
                    mdai = mdai or (obj, path)
                else:
                    found = read_nih_patients(obj)
                    if found:
                        patients.update(found)
                        log.info("Mapping NIH từ %s: %d mã ảnh -> %d bệnh nhân", path.name, len(found),
                                 len(set(found.values())))
        return mdai, patients

    def _scan_labels(self, roots: list[Path]) -> tuple[pd.Series | None, str, dict[str, str]]:
        """(nhãn, mô tả nguồn nhãn, mapping bệnh nhân NIH) tìm được dưới `roots`."""
        mdai, patients = self._read_jsons(roots)
        csv = self.labels_path if self.labels_path is not None else \
            search_label_csv(roots, self.id_column, self.label_column)[0]
        if csv is not None:
            return read_dicom_labels(csv, self.id_column, self.label_column, self.classes), csv.name, patients
        if mdai is not None:
            labels, usage = read_mdai_labels(mdai[0], self.positive_labels, self.classes)
            log.info("Nhãn MD.ai %s — số chú thích theo nhãn: %s (dương = %s)", mdai[1].name, usage,
                     self.positive_labels)
            return labels, mdai[1].name, patients
        return None, "", patients

    def _labels(self) -> tuple[pd.Series, str, dict[str, str]]:
        """Nhãn: CSV kiểu Kaggle hoặc JSON MD.ai; chưa thấy thì giải nén các file nén rồi tìm lại."""
        labels, source, patients = self._scan_labels([self.images_root])
        if labels is None and self._extract():
            labels, source, patients = self._scan_labels([self.images_root, self.extract_dir])
        if labels is not None:
            return labels, source, patients
        seen = search_label_csv([self.images_root, self.extract_dir], self.id_column, self.label_column)[1]
        listing = "\n".join(f"  {c.name}: {cols}" for c, cols in seen) or "  (không có file CSV nào)"
        raise FileNotFoundError(
            f"Không tìm thấy nhãn: không có CSV có cột '{self.id_column}' + '{self.label_column}', cũng không có JSON "
            f"xuất từ MD.ai (đã tìm cả trong file nén).\nCSV tìm thấy:\n{listing}\n"
            f"{inventory([self.images_root, self.extract_dir])}\n"
            "Khai báo đúng data.source.train_labels / id_column / label_column.")

    # ------------------------------------------------------------------ chẩn đoán nhãn MD.ai (chỉ đọc, không ghi gì)
    def _mdai(self) -> tuple[dict, str, pd.Series, dict[str, str]]:
        """(JSON MD.ai, tên file, nhãn theo ảnh, mapping bệnh nhân NIH), đọc một lần cho mỗi đối tượng."""
        if self._diag is None:
            self._check_root()
            mdai, patients = self._read_jsons([self.images_root])
            if mdai is None and self._extract():
                mdai, patients = self._read_jsons([self.images_root, self.extract_dir])
            if mdai is None:
                raise FileNotFoundError(f"Không có JSON nhãn MD.ai dưới {self.images_root}: thống kê theo tên nhãn chỉ "
                                        "có với bản tải từ trang RSNA (MD.ai); CSV kiểu Kaggle chỉ có cột 0 / 1.")
            obj, path = mdai
            labels, _ = read_mdai_labels(obj, self.positive_labels, self.classes)
            self._diag = obj, path.name, labels, patients
        return self._diag

    def _dicom_files(self) -> list[Path]:
        if self._files is None:
            self._files = find_files(self.images_root, (".dcm",)) + find_files(self.extract_dir, (".dcm",))
        return self._files

    def label_report(self, focus: list[str] | tuple[str, ...] = ()) -> tuple[str, pd.DataFrame, dict[str, pd.Series]]:
        """Thống kê nhãn theo ẢNH -> (tên file nhãn, `label_image_table`, {nhãn trong `focus`: `cooccurring_labels`})."""
        obj, name, labels, patients = self._mdai()
        if patients and not set(labels.index) <= set(patients):
            patients = expand_patients(patients, self._dicom_files())
        pairs = mdai_annotations(obj)
        table = label_image_table(pairs, labels, list(self.classes), patients)
        return name, table, {f: cooccurring_labels(pairs, f) for f in focus}

    def label_samples(self, n: int, size: int = 256) -> dict[str, list[tuple[np.ndarray, list, str]]]:
        """Mỗi tên nhãn: tối đa `n` ảnh ngẫu nhiên (seed) -> [(ảnh uint8 thu về size × size, khung của CHÍNH nhãn đó
        theo toạ độ đã thu nhỏ, chú thích "lớp cuối cùng · tư thế chụp")]. Để vẽ hình, không ghi gì."""
        import pydicom
        from PIL import Image

        obj, _, labels, _ = self._mdai()
        index, boxes = dicom_index(self._dicom_files()), mdai_boxes(obj)
        out = {}
        for name, ids in sample_per_label(mdai_annotations(obj), n, self.seed).items():
            row = []
            for image_id in ids:
                if image_id not in index:
                    continue
                ds = pydicom.dcmread(index[image_id])
                arr = dicom_to_uint8(ds)
                sy, sx = size / arr.shape[0], size / arr.shape[1]
                img = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
                own = [(x * sx, y * sy, w * sx, h * sy) for lab, (x, y, w, h) in boxes.get(image_id, []) if lab == name]
                row.append((img, own, f"{labels[image_id]} · {getattr(ds, 'ViewPosition', '?')}"))
            out[name] = row
        return out

    def _signature(self) -> dict:
        return {"labels": str(self.labels_path or "auto"), "id_column": self.id_column,
                "label_column": self.label_column, "positive_labels": self.positive_labels,
                "one_per_patient": self.one_per_patient, "subset_size": self.subset_size, "seed": self.seed}

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

        self._check_root()
        labels, source, patients = self._labels()
        log.info("Nhãn từ %s: %d ảnh %s", source, len(labels), labels.value_counts().to_dict())

        files = find_files(self.images_root, (".dcm",))
        if not set(labels.index) <= set(dicom_index(files)) and self._extract():
            files += find_files(self.extract_dir, (".dcm",))
        elif self.extract_dir.is_dir():
            files += find_files(self.extract_dir, (".dcm",))       # bản giải nén cục bộ (ghi sau -> được ưu tiên)
        if not files:
            raise FileNotFoundError(f"Không có file .dcm dưới {self.images_root} (kể cả sau khi giải nén).\n"
                                    f"{inventory([self.images_root, self.extract_dir])}")
        index = dicom_index(files)
        log.info("Tìm thấy %d file DICOM", len(files))
        patients = expand_patients(patients, files)

        if self.one_per_patient:
            if patients:
                n_before = len(labels)
                labels = one_image_per_patient(labels, patients, self.seed)
                log.info("Một ảnh / bệnh nhân NIH (seed %d): %d -> %d ảnh %s", self.seed, n_before, len(labels),
                         labels.value_counts().to_dict())
            else:
                log.warning("one_per_patient = true nhưng không tìm thấy mapping bệnh nhân NIH -> không lọc được; "
                            "nhiều ảnh của cùng một bệnh nhân có thể rơi vào cả train và test")
        labels = stratified_subset(labels, self.subset_size, self.seed)
        if self.subset_size:
            log.info("Tập con phân tầng (seed %d): %d ảnh %s", self.seed, len(labels), labels.value_counts().to_dict())

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
            meta.append({"image_id": image_id, "label": label, "nih_patient": patients.get(image_id),
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
