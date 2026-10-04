"""Đường dẫn artefact trên Drive: dùng lại GAN cũ (ISIC: checkpoints_v5; Brain Tumor: checkpoints_bt + split tham chiếu
real_split.json), còn kết quả của cấu hình hiện tại (k = 1,5, có augmentation, M0 có class weight) ghi vào thư mục MỚI
(ISIC v10, Brain bt_v5) — không đụng kết quả cũ (ISIC v7 / v9; Brain bt_v3 / bt_v4)."""

from pathlib import Path

from conftest import CONFIGS

from dass.config import Layout, load_config

ISIC_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/ColabData/ISBI2016_ISIC_Part3")
BT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/ColabData/BrainTumor_GAN")


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


def test_brain_tumor_artifact_paths():
    lay = _layout("brain_tumor_dass.yaml")
    assert lay.gan_best_pkl == BT_ROOT / "checkpoints_bt/stylegan2ada/best.pkl"
    assert lay.expected_split == BT_ROOT / "checkpoints_bt/data/real_split.json"
    assert lay.results_dir == BT_ROOT / "results_bt_v5"            # tách khỏi results_bt, results_bt_v3, results_bt_v4
    assert lay.drive_train_images == Path("/content/drive/MyDrive/Colab Notebooks/ColabData/Brain_Tumor_Dataset")
    assert lay.drive_test_images is None and lay.drive_train_labels is None


def test_rsna_artifact_paths():
    """Dữ liệu gốc (RSNA Pneumonia) tách khỏi kết quả (RSNA Pneumonia/Result_Pneumonia), như Brain Tumor."""
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


def test_brain_tumor_uses_three_channels_like_reused_gan():
    """GAN checkpoints_bt được train trên ảnh 3 kênh -> dữ liệu Brain Tumor cũng 3 kênh (pool không có hậu tố _c1)."""
    cfg = load_config([CONFIGS / "experiments" / "brain_tumor_dass.yaml"])
    assert cfg.data.channels == 3 and cfg.paths.gan_tag == "bt"
    assert cfg.data.force_grayscale is True        # loại nhiễu màu JPEG chỉ có ở lớp negative
    isic = load_config([CONFIGS / "experiments" / "isic2016_dass.yaml"])
    assert isic.data.force_grayscale is False      # dermoscopy là ảnh màu thật


def test_local_dirs_are_separated_per_dataset():
    a, b = _layout("isic2016_dass.yaml"), _layout("brain_tumor_dass.yaml")
    assert a.train_pp != b.train_pp and "isic2016" in a.train_pp.parts and "brain_tumor" in b.train_pp.parts
