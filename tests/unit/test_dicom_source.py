import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pydicom = pytest.importorskip("pydicom")

from dass.data.sources.dicom_source import (  # noqa: E402
    DicomCsvSource,
    dicom_to_uint8,
    find_label_csv,
    stratified_subset,
)

CLASSES = {"negative": 0, "positive": 1}


def write_dicom(path, pixels: np.ndarray, view: str = "PA", photometric: str = "MONOCHROME2") -> None:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.ViewPosition, ds.PatientSex, ds.PatientAge = view, "F", "045Y"
    ds.PixelData = pixels.astype(np.uint8).tobytes()
    ds.save_as(path, enforce_file_format=True)


@pytest.fixture
def rsna_like(tmp_path):
    """Thư mục giống RSNA: ảnh .dcm + stage_2_train_labels.csv (nhiều dòng / ảnh dương) + 1 ảnh không nhãn."""
    root = tmp_path / "RSNA Pneumonia"
    img_dir = root / "stage_2_train_images"
    img_dir.mkdir(parents=True)
    rows = []
    for i in range(12):
        pid = f"p{i:02d}"
        target = 1 if i < 3 else 0
        write_dicom(img_dir / f"{pid}.dcm", np.full((8, 8), 10 * i, np.uint8), view="AP" if target else "PA")
        rows += [{"patientId": pid, "x": None, "Target": target}] * (2 if target else 1)   # ảnh dương: 2 khung
    write_dicom(img_dir / "unlabeled.dcm", np.zeros((8, 8), np.uint8))                    # test cuộc thi: không nhãn
    pd.DataFrame(rows).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame({"a": [1]}).to_csv(root / "other.csv", index=False)
    return root


def _source(root, tmp_path, subset=0, labels=None):
    return DicomCsvSource(CLASSES, root, labels, "patientId", "Target", subset, 2026, tmp_path / "extract",
                          tmp_path / "meta" / "dicom_metadata.csv")


def test_ingest_converts_labelled_dicoms_to_gray_png(rsna_like, tmp_path):
    dst = tmp_path / "raw"
    counts = _source(rsna_like, tmp_path).ingest("train", dst)
    assert counts == {"negative": 9, "positive": 3}                        # 2 khung / ảnh dương -> vẫn 1 ảnh
    assert not (dst / "negative" / "unlabeled.png").exists()               # ảnh không có nhãn bị bỏ qua
    with Image.open(dst / "negative" / "p05.png") as im:
        assert im.mode == "L" and np.asarray(im)[0, 0] == 50               # PNG xám 1 kênh, giữ nguyên giá trị
    meta = pd.read_csv(tmp_path / "meta" / "dicom_metadata.csv")
    assert set(meta.loc[meta["label"] == "positive", "ViewPosition"]) == {"AP"}


def test_ingest_resyncs_local_images_when_settings_change(rsna_like, tmp_path):
    dst = tmp_path / "raw"
    full = _source(rsna_like, tmp_path).ingest("train", dst)
    sub = _source(rsna_like, tmp_path, subset=4).ingest("train", dst)          # cùng runtime, thiết lập khác
    pngs = lambda: sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*.png"))  # noqa: E731
    assert sum(sub.values()) == 4 and len(pngs()) == 4                       # ảnh của thiết lập cũ bị xoá
    assert _source(rsna_like, tmp_path).ingest("train", dst) == full and len(pngs()) == 12
    for p in (rsna_like / "stage_2_train_images").iterdir():
        p.unlink()                                                         # lần sau không cần đọc lại Drive
    assert _source(rsna_like, tmp_path).ingest("train", dst) == full


def test_ingest_extracts_zip_when_no_dicom(rsna_like, tmp_path):
    img_dir = rsna_like / "stage_2_train_images"
    with zipfile.ZipFile(rsna_like / "images.zip", "w") as z:
        for p in img_dir.iterdir():
            z.write(p, arcname=f"stage_2_train_images/{p.name}")
    for p in img_dir.iterdir():
        p.unlink()
    counts = _source(rsna_like, tmp_path).ingest("train", tmp_path / "raw")
    assert sum(counts.values()) == 12 and (tmp_path / "extract" / "images.zip.done").exists()


def test_labels_and_images_only_inside_archives(rsna_like, tmp_path):
    """Trường hợp tải từ trang RSNA chưa giải nén: cả CSV nhãn lẫn ảnh nằm trong file nén."""
    with zipfile.ZipFile(rsna_like / "challenge.zip", "w") as z:
        for p in (rsna_like / "stage_2_train_images").iterdir():
            z.write(p, arcname=f"images/{p.name}")
        z.write(rsna_like / "stage_2_train_labels.csv", arcname="stage_2_train_labels.csv")
    for p in list((rsna_like / "stage_2_train_images").iterdir()) + [rsna_like / "stage_2_train_labels.csv"]:
        p.unlink()
    counts = _source(rsna_like, tmp_path).ingest("train", tmp_path / "raw")
    assert counts == {"negative": 9, "positive": 3}


def test_no_labels_reports_inventory(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "annotations.json").write_text("{}")
    write_dicom(root / "a.dcm", np.zeros((4, 4), np.uint8))
    with pytest.raises(FileNotFoundError) as err:
        _source(root, tmp_path).ingest("train", tmp_path / "raw")
    msg = str(err.value)
    assert ".dcm: 1" in msg and ".json: 1" in msg and "annotations.json" in msg


def test_missing_or_empty_root_lists_parent(tmp_path):
    parent = tmp_path / "ColabData"
    (parent / "RSNA_Pneumonia").mkdir(parents=True)                      # tên thật khác tên trong cấu hình
    with pytest.raises(FileNotFoundError, match="KHÔNG tồn tại") as err:
        _source(parent / "RSNA Pneumonia", tmp_path).ingest("train", tmp_path / "raw")
    assert "RSNA_Pneumonia/" in str(err.value)
    with pytest.raises(FileNotFoundError, match="rỗng"):
        _source(parent / "RSNA_Pneumonia", tmp_path).ingest("train", tmp_path / "raw")


def test_drive_duplicate_folder_is_used_when_configured_one_has_no_data(rsna_like, tmp_path):
    """Google Drive tạo "RSNA Pneumonia (1)"; thư mục cấu hình chỉ có thư mục kết quả của pipeline (kể cả file)."""
    dup = rsna_like.parent / "RSNA Pneumonia (1)"
    rsna_like.rename(dup)
    outputs = rsna_like / "checkpoints_rsna_v1" / "data"
    outputs.mkdir(parents=True)
    (outputs / "dicom_metadata.csv").write_text("image_id,label\n")          # kết quả không được coi là dữ liệu
    (outputs / "pool_positive_n10_c1.zip").write_bytes(b"")
    counts = _source(rsna_like, tmp_path).ingest("train", tmp_path / "raw")
    assert counts == {"negative": 9, "positive": 3}


def test_outputs_are_ignored_when_searching_source_files(tmp_path):
    from dass.data.sources.dicom_source import find_files
    (tmp_path / "results_x").mkdir()
    (tmp_path / "results_x" / "m.csv").write_text("a\n")
    (tmp_path / "labels.csv").write_text("a\n")
    assert [p.name for p in find_files(tmp_path, (".csv",))] == ["labels.csv"]
    nested = tmp_path / "Result_Pneumonia" / "checkpoints_rsna_v1" / "data"     # kết quả nằm TRONG thư mục dữ liệu
    nested.mkdir(parents=True)
    (nested / "pool_positive_n10_c1.zip").write_bytes(b"")
    (nested / "dicom_metadata.csv").write_text("a\n")
    assert [p.name for p in find_files(tmp_path, (".csv", ".zip"))] == ["labels.csv"]


def _mdai_fixture(root):
    """Bản tải từ trang RSNA: ảnh MD.ai (Study/Series/SOP.dcm) + nhãn JSON MD.ai + mapping NIH."""
    imgs = root / "mdai_rsna_project_x_images_2018"
    studies = [f"1.2.{i}" for i in range(8)]
    for i, st in enumerate(studies):
        d = imgs / st / f"{st}.9"
        d.mkdir(parents=True)
        write_dicom(d / f"{st}.9.1.dcm", np.full((4, 4), i, np.uint8), view="AP" if i < 3 else "PA")
    labels = {"labelGroups": [{"id": "G1", "labels": [{"id": "L_op", "name": "Lung Opacity"},
                                                      {"id": "L_nn", "name": "No Lung Opacity / Not Normal"},
                                                      {"id": "L_n", "name": "Normal"}]}],
              "datasets": [{"id": "D1", "annotations": []}]}
    anns = labels["datasets"][0]["annotations"]
    for st in studies[:3]:            # dương: 2 khung cấp ảnh (có SOP)
        anns += [{"StudyInstanceUID": st, "SeriesInstanceUID": f"{st}.9", "SOPInstanceUID": f"{st}.9.1",
                  "labelId": "L_op", "data": {"x": 1}}] * 2
    for st in studies[3:7]:           # âm: chú thích cấp ca chụp (chỉ có Study UID)
        anns.append({"StudyInstanceUID": st, "labelId": "L_n" if st != studies[6] else "L_nn", "data": None})
    (root / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json").write_text(json.dumps(labels))
    mapping = [{"SOPInstanceUID": f"{st}.9.1", "img": f"0000000{min(i, 5)}_00{i}.png"} for i, st in enumerate(studies)]
    (root / "pneumonia-challenge-dataset-mappings_2018.json").write_text(json.dumps(mapping))
    return studies


def test_mdai_export_labels_and_one_image_per_patient(tmp_path):
    root = tmp_path / "RSNA Pneumonia (1)"
    studies = _mdai_fixture(root)
    src = DicomCsvSource(CLASSES, root, None, "patientId", "Target", 0, 2026, tmp_path / "extract",
                         tmp_path / "meta.csv", positive_labels=["Lung Opacity"], one_per_patient=True)
    counts = src.ingest("train", tmp_path / "raw")
    # 3 dương (0–2) + 4 âm (3–6); ảnh 7 không có nhãn -> bỏ; ảnh 5 và 6 cùng bệnh nhân NIH 00000005 -> giữ 1
    assert counts == {"negative": 3, "positive": 3}
    assert {p.stem for p in (tmp_path / "raw" / "positive").iterdir()} == set(studies[:3])
    meta = pd.read_csv(tmp_path / "meta.csv")
    assert meta["nih_patient"].astype(str).str.zfill(8).nunique() == len(meta)       # mỗi bệnh nhân đúng 1 ảnh


def _add_exclude(root, studies, targets):
    """Thêm nhãn "Exclude" cho các ảnh `targets` (chỉ số trong `studies`)."""
    path = root / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
    obj = json.loads(path.read_text())
    obj["labelGroups"][0]["labels"].append({"id": "L_ex", "name": "Exclude"})
    obj["datasets"][0]["annotations"] += [{"StudyInstanceUID": studies[i], "labelId": "L_ex"} for i in targets]
    path.write_text(json.dumps(obj))
    return path


def test_exclude_labels_drop_images_before_one_per_patient(tmp_path):
    root = tmp_path / "data"
    studies = _mdai_fixture(root)
    _add_exclude(root, studies, [1, 3, 5])        # 1 dương, 2 âm (5 cùng bệnh nhân với 6 -> giữ 6)
    src = DicomCsvSource(CLASSES, root, None, "patientId", "Target", 0, 2026, tmp_path / "extract",
                         tmp_path / "meta.csv", positive_labels=["Lung Opacity"], one_per_patient=True,
                         exclude_labels=["Exclude"])
    counts = src.ingest("train", tmp_path / "raw")
    assert counts == {"negative": 2, "positive": 2}
    kept = {p.stem for p in (tmp_path / "raw").rglob("*.png")}
    assert kept == {studies[i] for i in (0, 2, 4, 6)}
    assert json.loads((tmp_path / "raw" / ".ingest.json").read_text())["signature"]["exclude_labels"] == ["Exclude"]

    _, table, _ = src.label_report()
    assert table.loc["Exclude", "bị loại"] == 3 and table.loc["TỔNG", "bị loại"] == 3
    assert table.loc["TỔNG", "ảnh"] == 7                                      # bảng = nhãn TRƯỚC khi loại


def test_exclude_labels_need_mdai_names(rsna_like, tmp_path):
    src = DicomCsvSource(CLASSES, rsna_like, None, "patientId", "Target", 0, 2026, tmp_path / "extract",
                         tmp_path / "meta.csv", exclude_labels=["Exclude"])
    with pytest.raises(ValueError, match="MD.ai"):
        src.ingest("train", tmp_path / "raw")


def test_mdai_wrong_positive_label_lists_available_labels(tmp_path):
    root = tmp_path / "data"
    _mdai_fixture(root)
    src = DicomCsvSource(CLASSES, root, None, "patientId", "Target", 0, 2026, tmp_path / "extract",
                         tmp_path / "meta.csv", positive_labels=["Pneumonia"])
    with pytest.raises(ValueError, match="Lung Opacity"):
        src.ingest("train", tmp_path / "raw")


def test_label_report_counts_images_not_annotations(tmp_path):
    root = tmp_path / "data"
    studies = _mdai_fixture(root)
    path = _add_exclude(root, studies, [0, 0, 4])                             # 2 bác sĩ đánh dấu ảnh 0
    src = DicomCsvSource(CLASSES, root, None, "patientId", "Target", 0, 2026, tmp_path / "extract",
                         tmp_path / "meta.csv", positive_labels=["Lung Opacity"])
    name, table, co = src.label_report(["Exclude"])
    assert name == path.name
    assert table.loc["Exclude", ["chú thích", "ảnh", "negative", "positive", "bệnh nhân"]].tolist() == [3, 2, 1, 1, 2]
    assert table.loc["Lung Opacity", ["chú thích", "ảnh"]].tolist() == [6, 3]
    assert table.loc["TỔNG", ["ảnh", "negative", "positive", "bệnh nhân"]].tolist() == [7, 4, 3, 6]   # ảnh 5, 6 cùng BN
    assert table.loc["Exclude", "% ảnh"] == round(100 * 2 / 7, 2)
    assert co["Exclude"].to_dict() == {"Lung Opacity": 1, "Normal": 1}
    assert not (tmp_path / "meta.csv").exists() and not (tmp_path / "extract").exists()      # chỉ đọc, không ghi


def test_label_samples_draw_own_boxes_scaled(tmp_path):
    root = tmp_path / "data"
    _mdai_fixture(root)
    path = root / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
    obj = json.loads(path.read_text())
    for a in obj["datasets"][0]["annotations"]:
        if a["labelId"] == "L_op":
            a["data"] = {"x": 1, "y": 0, "width": 2, "height": 3}             # ảnh gốc 4 × 4
    path.write_text(json.dumps(obj))
    src = DicomCsvSource(CLASSES, root, None, "patientId", "Target", 0, 2026, tmp_path / "extract",
                         tmp_path / "meta.csv", positive_labels=["Lung Opacity"])
    rows = src.label_samples(n=2, size=8)
    assert list(rows) == ["Lung Opacity", "Normal", "No Lung Opacity / Not Normal"]     # theo số ảnh giảm dần
    assert [len(v) for v in rows.values()] == [2, 2, 1]
    img, boxes, caption = rows["Lung Opacity"][0]
    assert img.shape == (8, 8) and img.dtype == np.uint8
    assert boxes == [(2.0, 0.0, 4.0, 6.0)] * 2 and caption == "positive · AP"           # 2 khung, co theo 4 -> 8
    assert rows["Normal"][0][1] == [] and rows["Normal"][0][2].startswith("negative")
    assert src.label_samples(n=2, size=8)["Normal"][1][2] == rows["Normal"][1][2]        # tất định theo seed

    from dass.analysis.figures import plot_label_samples
    out = tmp_path / "fig" / "label_samples.png"
    plot_label_samples(rows, {"Lung Opacity": 3}, "test", out)
    assert out.stat().st_size > 0


def test_subset_is_stratified_and_deterministic():
    labels = pd.Series(["positive"] * 20 + ["negative"] * 80, index=[f"id{i:03d}" for i in range(100)])
    a = stratified_subset(labels, 50, seed=1)
    assert a.value_counts().to_dict() == {"negative": 40, "positive": 10}
    assert a.index.equals(stratified_subset(labels, 50, seed=1).index)
    assert not a.index.equals(stratified_subset(labels, 50, seed=2).index)
    assert stratified_subset(labels, 0, seed=1).equals(labels)


def test_monochrome1_is_inverted_and_wide_range_rescaled():
    class Fake:
        def __init__(self, arr, photometric):
            self.pixel_array, self.PhotometricInterpretation = arr, photometric
    inv = dicom_to_uint8(Fake(np.array([[0, 255]], np.uint8), "MONOCHROME1"))
    assert inv.tolist() == [[255, 0]]
    dim = Fake(np.array([[10, 100]], np.uint8), "MONOCHROME1")           # ảnh tối: không có điểm 255
    dim.BitsStored = 8
    assert dicom_to_uint8(dim).tolist() == [[245, 155]]                  # đảo theo 2^8 − 1, không theo max của ảnh
    wide = dicom_to_uint8(Fake(np.array([[0, 4095]], np.uint16), "MONOCHROME2"))
    assert wide.tolist() == [[0, 255]] and wide.dtype == np.uint8


def test_missing_label_csv_lists_candidates(tmp_path):
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    with pytest.raises(FileNotFoundError, match="x.csv"):
        find_label_csv(tmp_path, "patientId", "Target")


def test_marker_records_signature(rsna_like, tmp_path):
    dst = tmp_path / "raw"
    _source(rsna_like, tmp_path, subset=8).ingest("train", dst)
    saved = json.loads((dst / ".ingest.json").read_text())
    assert saved["signature"]["subset_size"] == 8 and sum(saved["counts"].values()) == 8
