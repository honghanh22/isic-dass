"""Cấu hình pipeline.

- `Config`: các dataclass lồng nhau theo nhóm (paths, data, gan, ...), nạp từ YAML và ghi đè bằng `section.key=value`.
- `Layout`: mọi đường dẫn suy ra từ `Config`. Drive = lưu lâu dài; local (/content) = mất khi runtime Colab reset.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    drive_root: str = "/content/drive/MyDrive/Colab Notebooks/ColabData/ISBI2016_ISIC_Part3"
    local_root: str = "/content/local_data"
    external_root: str = "/content"  # nơi clone các repo bên ngoài (StyleGAN2-ADA, GANDCTAnalysis, ...)
    gan_tag: str = "v5"              # checkpoint GAN nằm ở checkpoints_<gan_tag>/ (dùng lại GAN đã train)
    run_tag: str = "v7"              # dữ liệu, classifier và kết quả nằm ở *_<run_tag>/


@dataclass
class DataConfig:
    img_size: int = 256
    class_to_idx: dict[str, int] = field(default_factory=lambda: {"benign": 0, "malignant": 1})
    val_fraction: float = 0.15

    @property
    def class_names(self) -> list[str]:
        return list(self.class_to_idx)


@dataclass
class GanConfig:
    cfg: str = "paper256"
    batch: int = 16
    gamma: float = 1.0
    mirror: bool = True
    aug_target: float = 0.6
    max_kimg: int = 3000
    kimg_per_snap: int = 100
    min_kimg: int = 400           # chưa đếm patience trước mốc này
    patience: int = 5             # số snapshot liên tiếp KID không cải thiện thì dừng
    min_rel_delta: float = 0.02   # KID phải giảm ít nhất 2 % mới tính là cải thiện
    kid_n_gen: int = 1000
    kid_subsets: int = 50
    kid_seed: int = 123
    gen_seed: int = 777
    trunc_psi: float = 1.0
    workers: int = 2
    poll_seconds: int = 30


@dataclass
class FrequencyConfig:
    harmonize_channels: Any = "auto"   # "auto" | true | false
    match_resize_chain: bool = False
    apply_spectral_mitigation: bool = False
    mitigation_mode: str = "sdn+pdc"   # "sdn" | "pdc" | "sdn+pdc"
    fingerprint_n_max: int = 400
    fingerprint_epochs: int = 15


@dataclass
class SelectionConfig:
    pool_mult: float = 4.0   # pool = pool_mult × số ảnh cần chọn
    sim_topk: int = 5        # S⁺, S⁻ = trung bình cosine với TOP-K ảnh thật gần nhất
    lambda_v: float = 1.0
    lambda_d: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5


@dataclass
class EncoderConfig:
    e_d_model: str = "DenseNet121"   # nên KHÁC model báo cáo downstream
    e_d_seed: int = 4242
    e_d_head_epochs: int = 5
    e_d_ft_epochs: int = 20
    embed_batch_size: int = 32


@dataclass
class ClassifierConfig:
    size: int = 224
    batch_size: int = 16
    head_epochs: int = 5
    head_lr: float = 1e-3
    ft_epochs: int = 30
    ft_lr: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = 0.3
    early_stop_patience: int = 8
    seeds: list[int] = field(default_factory=lambda: [2026, 2027, 2028])
    monitor: str = "val_macro_recall"   # hoặc "val_auc"
    mixed_precision: bool = False
    save_weights_to_drive: bool = True
    vit_preset: str = "vit_base_patch16_224_imagenet"
    swin_preset: str = "hf://keras/swin_tiny_patch4_window7_224"


@dataclass
class EvaluationConfig:
    n_bootstrap: int = 2000
    baseline_method: str = "M0_real_only"
    ssim_pairs: int = 200


@dataclass
class Config:
    seed: int = 2026
    save_figures: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gan: GanConfig = field(default_factory=GanConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _apply(obj: Any, data: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Khoá cấu hình không hợp lệ: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} phải là một nhóm (dict), nhận {type(value).__name__}")
            _apply(current, value, f"{prefix}{key}.")
        else:
            # PyYAML đọc '1e-5' thành chuỗi -> ép về kiểu của giá trị mặc định
            if isinstance(current, float) and isinstance(value, (int, str)) and not isinstance(value, bool):
                value = float(value)
            setattr(obj, key, value)


def parse_override(item: str) -> dict[str, Any]:
    """'gan.batch=32' -> {'gan': {'batch': 32}}. Giá trị được parse bằng YAML (số, bool, list, ...)."""
    import yaml

    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"Ghi đè phải có dạng section.key=value, nhận: {item!r}")
    parts = key.strip().split(".")
    node: dict[str, Any] = {}
    cursor = node
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = yaml.safe_load(raw)
    return node


def load_config(path: str | Path | None = None, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    cfg = Config()
    if path:
        import yaml

        with open(path, encoding="utf-8") as f:
            _apply(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        _apply(cfg, parse_override(item))
    return cfg


class Layout:
    """Toàn bộ đường dẫn của một lần chạy. Tên file giữ tương thích với notebook v5/v7 để dùng lại artefact trên Drive."""

    def __init__(self, cfg: Config):
        p, size = cfg.paths, cfg.data.img_size

        drive = Path(p.drive_root)
        self.drive_train_img_dir = drive / "ISBI2016_ISIC_Part3_Training_Data"
        self.drive_train_gt_csv = drive / "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
        self.drive_test_img_dir = drive / "ISBI2016_ISIC_Part3_Test_Data"
        self.drive_test_gt_csv = drive / "ISBI2016_ISIC_Part3_Test_GroundTruth.csv"

        self.results_dir = drive / f"results_{p.run_tag}"
        self.pred_dir = self.results_dir / "predictions"
        self.freq_fig_dir = self.results_dir / "frequency_analysis"
        self.selection_fig_dir = self.results_dir / "selection_figures"
        self.gan_fig_dir = self.results_dir / "gan"
        self.gan_dir = drive / f"checkpoints_{p.gan_tag}" / "stylegan2ada"
        self.clf_dir = drive / f"checkpoints_{p.run_tag}" / "classifiers"
        self.data_dir = drive / f"checkpoints_{p.run_tag}" / "data"

        local = Path(p.local_root)
        self.train_raw = local / "raw" / "train"
        self.test_raw = local / "raw" / "test"
        self.train_pp = local / f"pp{size}_png" / "train"
        self.test_pp = local / f"pp{size}_png" / "test"
        self.gan_dataset_zip = local / "gan_dataset" / f"isic2016_train_cond_{size}.zip"
        self.gan_runs = local / "gan_runs"
        self.candidates = local / f"candidates_{p.run_tag}"
        self.variants = local / f"variants_{p.run_tag}"
        self.clf_ckpt = local / f"classifier_ckpt_{p.run_tag}"
        self.real_only = local / f"real_only_{p.run_tag}"

        ext = Path(p.external_root)
        self.sg2_repo = ext / "stylegan2-ada-pytorch"
        self.frank_repo = ext / "GANDCTAnalysis"
        self.dong_dir = ext / "dong_cvpr2022"

        self.split_json = self.data_dir / "real_val_split.json"
        self.selections_json = self.data_dir / "selections.json"
        self.embeddings_npz = self.data_dir / "embeddings.npz"
        self.gan_state_json = self.gan_dir / "gan_state.json"
        self.gan_best_pkl = self.gan_dir / "best.pkl"
        self.gan_latest_pkl = self.gan_dir / "latest.pkl"

    def makedirs(self) -> None:
        for d in [self.results_dir, self.pred_dir, self.gan_dir, self.clf_dir, self.data_dir,
                  self.train_raw, self.test_raw, self.train_pp, self.test_pp, self.gan_dataset_zip.parent,
                  self.gan_runs, self.candidates, self.variants, self.clf_ckpt, self.real_only]:
            d.mkdir(parents=True, exist_ok=True)
