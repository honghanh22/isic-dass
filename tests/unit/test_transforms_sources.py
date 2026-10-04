import numpy as np
import pytest
from PIL import Image

from dass.data.sources import CsvSource, FolderSource, build_source, normalize_label
from dass.data.sources.folder_source import copy_class_folders
from dass.data.transforms import content_ratio_stats, crop_dark_border, pad_to_square, preprocess_image, preprocess_tree

C2I = {"benign": 0, "malignant": 1}


# ------------------------------------------------------------------------------------------ transforms
def test_crop_dark_border_only_crops_large_borders():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 30:70] = 200
    assert crop_dark_border(img).shape == (20, 40, 3)
    img2 = np.zeros((100, 100, 1), dtype=np.uint8)
    img2[10:90, 10:90] = 200
    assert crop_dark_border(img2).shape == (100, 100, 1)


def test_pad_to_square_centers_image():
    sq = pad_to_square(np.full((4, 10, 1), 255, dtype=np.uint8))
    assert sq.shape == (10, 10, 1) and sq[3:7].min() == 255 and sq[:3].max() == 0


@pytest.mark.parametrize("channels,mode", [(1, "L"), (3, "RGB")])
def test_preprocess_keeps_channels(tmp_path, channels, mode):
    src = tmp_path / "a.png"
    Image.fromarray(np.full((40, 80, 3), 200, dtype=np.uint8)).save(src)
    preprocess_image(src, tmp_path / "b.png", 32, channels, crop=False, resize_mode="pad_square")
    out = Image.open(tmp_path / "b.png")
    assert out.mode == mode and out.size == (32, 32)
    arr = np.asarray(out)
    assert arr[0].max() == 0 and arr[16].min() > 150                  # dải đen trên, nội dung giữa


def test_preprocess_tree_redoes_wrong_channel_files(tmp_path, gray_tree):
    dst = tmp_path / "pp"
    (dst / "negative").mkdir(parents=True)
    stale = dst / "negative" / "negative_00.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(stale)          # ảnh RGB từ bản cũ
    preprocess_tree(gray_tree, dst, 16, ["negative", "positive"], channels=1)
    assert {Image.open(p).mode for p in dst.rglob("*.png")} == {"L"}


def test_preprocess_tree_removes_images_without_source(tmp_path, gray_tree):
    """Ảnh gốc đổi theo thiết lập nguồn (ví dụ tập con DICOM khác) -> ảnh đã xử lý của thiết lập cũ bị xoá."""
    dst = tmp_path / "pp"
    preprocess_tree(gray_tree, dst, 16, ["negative", "positive"], channels=1)
    gone = sorted((gray_tree / "negative").iterdir())[0]
    gone.unlink()
    preprocess_tree(gray_tree, dst, 16, ["negative", "positive"], channels=1)
    assert not (dst / "negative" / f"{gone.stem}.png").exists()
    assert {p.stem for p in (dst / "negative").iterdir()} == {p.stem for p in (gray_tree / "negative").iterdir()}


def test_content_ratio_stats(tmp_path):
    paths = []
    for i in range(3):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[0:5] = 200
        Image.fromarray(img).save(tmp_path / f"{i}.png")
        paths.append(str(tmp_path / f"{i}.png"))
    assert content_ratio_stats(paths)["median"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------------- sources
@pytest.mark.parametrize("raw,expected", [("benign", "benign"), (0, "benign"), ("1.0", "malignant"),
                                          (" Malignant ", "malignant")])
def test_normalize_label(raw, expected):
    assert normalize_label(raw, C2I) == expected


def test_normalize_label_rejects_unknown():
    for bad in ["nevus", 2, "0.5"]:
        with pytest.raises(ValueError):
            normalize_label(bad, C2I)


def test_csv_source(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    for name in ["a.jpg", "b.jpg", "c.png"]:
        (src / name).write_bytes(b"x")
    (tmp_path / "gt.csv").write_text("a,0\nb,1.0\nc.png,benign\nmissing,0\n")
    s = CsvSource(C2I, src, tmp_path / "gt.csv")
    assert not s.has_test_set
    counts = s.ingest("train", tmp_path / "dst")
    assert counts == {"benign": 2, "malignant": 1}
    assert (tmp_path / "dst" / "benign" / "c.png").exists()
    with pytest.raises(ValueError):
        s.ingest("test", tmp_path / "dst2")


def test_csv_source_with_header(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    (src / "x.png").write_bytes(b"x")
    (tmp_path / "gt.csv").write_text("image,diagnosis,extra\nx,malignant,foo\n")
    CsvSource(C2I, src, tmp_path / "gt.csv", header=True, image_ext=".png").ingest("train", tmp_path / "d")
    assert (tmp_path / "d" / "malignant" / "x.png").exists()


def test_folder_source_with_class_dirs(gray_tree, tmp_path):
    s = FolderSource({"neg": 0, "pos": 1}, gray_tree, class_dirs={"neg": "negative", "pos": "positive"})
    assert s.ingest("train", tmp_path / "dst") == {"neg": 12, "pos": 5}
    with pytest.raises(FileNotFoundError):
        copy_class_folders(gray_tree, tmp_path / "x", ["negative", "nevus"])


def test_build_source_dispatch():
    from conftest import CONFIGS

    from dass.config import Layout, load_config
    from dass.data.sources import DicomCsvSource

    folders = ["data.source.type=folders", "data.source.train_labels=''", "data.source.test_labels=''"]
    for overrides, kind in [([], CsvSource), (folders, FolderSource)]:
        cfg = load_config([CONFIGS / "experiments" / "isic2016_dass.yaml"], overrides)
        src = build_source(cfg, Layout(cfg))
        assert isinstance(src, kind) and src.has_test_set           # ISIC có tập test riêng
    cfg = load_config([CONFIGS / "experiments" / "rsna_pneumonia_dass.yaml"])
    assert isinstance(build_source(cfg, Layout(cfg)), DicomCsvSource)
