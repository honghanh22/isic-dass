"""Đường dẫn artefact trên Drive: ISIC dùng lại GAN cũ (checkpoints_v5), RSNA dùng GAN mới (checkpoints_rsna, gắn với
split rsna_v2); kết quả của cấu hình hiện tại ghi vào thư mục MỚI (ISIC v10, RSNA rsna_v2) — không đụng kết quả cũ."""

from pathlib import Path

from conftest import CONFIGS

from dass.config import Layout, load_config

ISIC_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/ColabData/ISBI2016_ISIC_Part3")


def _layout(name: str, tag: str | None = None) -> Layout:
    return Layout(load_config([CONFIGS / "experiments" / name], tag=tag))


def test_isic_artifact_paths():
    lay = _layout("isic2016_dass.yaml")
    assert lay.gan_best_pkl == ISIC_ROOT / "checkpoints_v5/stylegan2ada/best.pkl"
    assert lay.gan_state_json == ISIC_ROOT / "checkpoints_v5/stylegan2ada/gan_state.json"
    assert lay.split_json == ISIC_ROOT / "checkpoints_v10/data/real_val_split.json"
    assert lay.selections_json == ISIC_ROOT / "checkpoints_v10/data/selections.json"
    assert lay.pred_dir == ISIC_ROOT / "results_v10/predictions"
    assert lay.clf_dir == ISIC_ROOT / "checkpoints_v10/classifiers"
    assert not {"v7", "v8", "v9"} & set(lay.results_dir.name.split("_"))   # không ghi vào kết quả / cấu hình cũ
    assert lay.drive_train_images == ISIC_ROOT / "ISBI2016_ISIC_Part3_Training_Data"
    assert lay.drive_test_labels == ISIC_ROOT / "ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
    assert lay.expected_split is None


def test_rsna_artifact_paths():
    """Dữ liệu gốc (RSNA Pneumonia) tách khỏi kết quả (RSNA Pneumonia/Result_Pneumonia)."""
    data = Path("/content/drive/MyDrive/Colab Notebooks/ColabData/RSNA Pneumonia")
    out = data / "Result_Pneumonia"
    lay = _layout("rsna_pneumonia_dass.yaml")
    assert lay.gan_best_pkl == out / "checkpoints_rsna/stylegan2ada/best.pkl"       # GAN mới, riêng cho RSNA
    assert lay.results_dir == out / "results_rsna_v2"
    assert lay.drive_train_images == data and lay.drive_train_labels is None        # tự tìm .dcm và CSV nhãn
    assert lay.dicom_metadata_csv == out / "checkpoints_rsna_v2/data/dicom_metadata.csv"
    assert "rsna_pneumonia" in lay.train_raw.parts
    # GAN rsna train trên split của rsna_v2 -> lần chạy nào dùng GAN này cũng phải trùng đúng split đó
    assert lay.expected_split == out / "checkpoints_rsna_v2/data/real_val_split.json"
    assert lay.expected_split.parent == lay.split_json.parent           # chính split của run hiện tại (rsna_v2)
    assert _layout("rsna_pneumonia_dass.yaml", tag="aug").expected_split == lay.expected_split   # --tag: vẫn kiểm
    v3 = Layout(load_config([CONFIGS / "experiments" / "rsna_pneumonia_dass.yaml"], ["paths.run_tag=rsna_v3"]))
    assert v3.split_json != lay.split_json and v3.expected_split == lay.expected_split   # run mới: so với split v2


def test_channels_match_the_gan_of_each_dataset():
    """ISIC: ảnh màu, GAN v5 3 kênh. RSNA: ảnh xám 1 kênh, GAN rsna 1 kênh (pool có hậu tố _c1)."""
    isic = load_config([CONFIGS / "experiments" / "isic2016_dass.yaml"])
    rsna = load_config([CONFIGS / "experiments" / "rsna_pneumonia_dass.yaml"])
    assert isic.data.force_grayscale is False and rsna.data.force_grayscale is False
    assert rsna.data.channels == 1 and rsna.paths.gan_tag == "rsna"


def test_local_dirs_are_separated_per_dataset():
    a, b = _layout("isic2016_dass.yaml"), _layout("rsna_pneumonia_dass.yaml")
    assert a.train_pp != b.train_pp and "isic2016" in a.train_pp.parts and "rsna_pneumonia" in b.train_pp.parts
