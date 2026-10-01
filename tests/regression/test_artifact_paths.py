"""Đường dẫn artefact trên Drive phải giữ nguyên để dùng lại GAN và kết quả cũ (ISIC: checkpoints_v5 / v7;
Brain Tumor: checkpoints_bt + split tham chiếu real_split.json)."""

from pathlib import Path

from conftest import CONFIGS

from dass.config import Layout, load_config

ISIC_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/ColabData/ISBI2016_ISIC_Part3")
BT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/ColabData/BrainTumor_GAN")


def _layout(name: str) -> Layout:
    return Layout(load_config([CONFIGS / "experiments" / name]))


def test_isic_artifact_paths():
    lay = _layout("isic2016_dass.yaml")
    assert lay.gan_best_pkl == ISIC_ROOT / "checkpoints_v5/stylegan2ada/best.pkl"
    assert lay.gan_state_json == ISIC_ROOT / "checkpoints_v5/stylegan2ada/gan_state.json"
    assert lay.split_json == ISIC_ROOT / "checkpoints_v7/data/real_val_split.json"
    assert lay.selections_json == ISIC_ROOT / "checkpoints_v7/data/selections.json"
    assert lay.pred_dir == ISIC_ROOT / "results_v7/predictions"
    assert lay.clf_dir == ISIC_ROOT / "checkpoints_v7/classifiers"
    assert lay.drive_train_images == ISIC_ROOT / "ISBI2016_ISIC_Part3_Training_Data"
    assert lay.drive_test_labels == ISIC_ROOT / "ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
    assert lay.expected_split is None


def test_brain_tumor_artifact_paths():
    lay = _layout("brain_tumor_dass.yaml")
    assert lay.gan_best_pkl == BT_ROOT / "checkpoints_bt/stylegan2ada/best.pkl"
    assert lay.expected_split == BT_ROOT / "checkpoints_bt/data/real_split.json"
    assert lay.results_dir == BT_ROOT / "results_bt_v3"            # tách khỏi results_bt của notebook v1
    assert lay.drive_train_images == Path("/content/drive/MyDrive/Colab Notebooks/ColabData/Brain_Tumor_Dataset")
    assert lay.drive_test_images is None and lay.drive_train_labels is None


def test_brain_tumor_uses_three_channels_like_reused_gan():
    """GAN checkpoints_bt được train trên ảnh 3 kênh -> dữ liệu Brain Tumor cũng 3 kênh (pool không có hậu tố _c1)."""
    cfg = load_config([CONFIGS / "experiments" / "brain_tumor_dass.yaml"])
    assert cfg.data.channels == 3 and cfg.paths.gan_tag == "bt"


def test_local_dirs_are_separated_per_dataset():
    a, b = _layout("isic2016_dass.yaml"), _layout("brain_tumor_dass.yaml")
    assert a.train_pp != b.train_pp and "isic2016" in a.train_pp.parts and "brain_tumor" in b.train_pp.parts
