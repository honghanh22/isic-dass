"""Schema cấu hình: dataclass lồng nhau theo nhóm + kiểm tra ràng buộc.

Giá trị mặc định ở đây = `configs/_base_/*.yaml` (test `test_base_yaml_matches_schema_defaults` bảo đảm).
Mọi thứ phụ thuộc bộ dữ liệu (`paths`, `data`) KHÔNG có mặc định hợp lệ — phải khai báo trong `configs/datasets/`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

METHOD_LABELS = {
    "M0_real_only": "Imbalanced Baseline",          # report tự thêm " (class-weighted)" nếu đã train có class weight
    "M0b_real_oversample": "Random Oversampling (ROS)",
    "M1_random": "Unfiltered GAN (Random Selection)",
    "M2_visual": "Visual-only Filter ($M_v$)",
    "M3_disease": "Disease-only Filter ($M_d$)",
    "M5_diversity": r"Diversity-only Filter ($S_{\text{div}}$)",
    "M4_visual_disease": "Dual-Margin Filter ($M_v + M_d$)",
    "M6_dass": "DASS (Ours)",
}
METHOD_GROUPS = {
    "Real Data Baselines": ("M0_real_only", "M0b_real_oversample"),
    "Generative Augmentation (StyleGAN2-ADA)": ("M1_random", "M2_visual", "M3_disease", "M5_diversity",
                                                "M4_visual_disease", "M6_dass"),
}
SOURCE_TYPES = ("csv", "folders", "dicom_csv")
AUGMENT_PROFILES = ("rotation_invariant", "upright")
SPLIT_TYPES = ("holdout_val", "stratified", "file")
RESIZE_MODES = ("stretch", "pad_square")
CHANNEL_MODES = ("auto", 1, 3)


@dataclass
class PathsConfig:
    drive_root: str = ""             # thư mục trên Drive chứa artefact (checkpoint, kết quả) của bộ dữ liệu
    local_root: str = "/content/local_data"
    external_root: str = "/content"  # nơi clone repo ngoài (StyleGAN2-ADA, GANDCTAnalysis)
    gan_tag: str = "v1"              # checkpoint GAN: checkpoints_<gan_tag>/stylegan2ada
    run_tag: str = "v1"              # dữ liệu / classifier / kết quả: *_<run_tag>/
    base_run_tag: str = ""           # TỰ ĐIỀN khi dùng --tag: run_tag của lần chạy chính (trước khi thêm hậu tố)


@dataclass
class SourceConfig:
    """Nguồn ảnh. Đường dẫn tương đối so với `paths.drive_root` (hoặc tuyệt đối).

    - `csv`: `*_images` là thư mục ảnh phẳng, `*_labels` là CSV (cột 1 image_id, cột 2 nhãn).
    - `folders`: `*_images` chứa các thư mục lớp `<*_images>/<thư mục lớp>/`; không dùng `*_labels`.
    - `dicom_csv`: `train_images` là thư mục chứa file `.dcm` (tìm đệ quy; tự giải nén .zip / .tar nếu cần),
      `train_labels` là CSV nhãn (để trống -> tự tìm CSV có cột `id_column` và `label_column`).
    `test_images: ""` -> không có tập test riêng (khi đó split phải là `stratified` hoặc `file` có phần test).
    """

    type: str = "folders"
    train_images: str = ""
    train_labels: str = ""
    test_images: str = ""
    test_labels: str = ""
    class_dirs: dict[str, str] = field(default_factory=dict)   # folders: tên lớp -> tên thư mục (nếu khác)
    header: bool = False             # csv: file nhãn có dòng tiêu đề
    image_ext: str = ".jpg"          # csv: đuôi ảnh khi image_id không kèm đuôi
    id_column: str = ""              # dicom_csv: cột mã ảnh (= tên file .dcm không đuôi), ví dụ patientId
    label_column: str = ""           # dicom_csv: cột nhãn (0 / 1 hoặc tên lớp); nhiều dòng cùng mã -> lấy lớn nhất
    subset_size: int = 0             # dicom_csv: > 0 -> tập con phân tầng theo nhãn (seed toàn cục), 0 = tất cả
    positive_labels: list[str] = field(default_factory=list)   # dicom_csv + nhãn MD.ai: tên nhãn = lớp dương
    one_per_patient: bool = False    # dicom_csv: giữ 1 ảnh / bệnh nhân (cần mapping NIH) -> không rò rỉ bệnh nhân
    exclude_labels: list[str] = field(default_factory=list)    # dicom_csv + nhãn MD.ai: loại ảnh mang nhãn này

    @property
    def has_test_set(self) -> bool:
        return bool(self.test_images)


@dataclass
class PreprocessConfig:
    crop_dark_border: bool = False   # cắt viền đen khi nội dung < 50 % một cạnh (ảnh dermoscopy)
    resize: str = "stretch"          # stretch: resize thẳng | pad_square: đệm đen thành ảnh vuông rồi resize


@dataclass
class SplitConfig:
    type: str = "stratified"         # holdout_val | stratified | file
    val: float = 0.15                # tỉ lệ val (stratified: tính trên phần còn lại sau khi tách test)
    test: float = 0.0                # stratified: tỉ lệ test tách từ cùng nguồn
    group_regex: str = ""            # stratified: regex lấy mã bệnh nhân từ tên file -> chia theo nhóm
    file: str = ""                   # file: CSV `image_id,split`
    expected: str = ""               # JSON split phải trùng (ví dụ split mà GAN dùng lại đã train)


@dataclass
class DataConfig:
    name: str = ""                   # tên ngắn của bộ dữ liệu -> thư mục cục bộ riêng
    channels: Any = "auto"           # auto | 1 | 3 — giữ nguyên số kênh gốc xuyên suốt pipeline
    force_grayscale: bool = False    # chuyển về xám (luminance) rồi lưu `channels` kênh BẰNG NHAU — cho ảnh y tế xám
                                     # lưu dạng RGB có nhiễu màu JPEG (tránh manh mối màu giả gắn với nhãn)
    img_size: int = 256
    # augmentation khi train (classifier, E_d): rotation_invariant = lật ngang + dọc, xoay ±180° (dermoscopy: tổn thương
    # không có hướng); upright = lật ngang, xoay ±10° (ảnh có hướng giải phẫu cố định, ví dụ X-quang ngực)
    augment_profile: str = "rotation_invariant"
    classes: dict[str, int] = field(default_factory=dict)   # tên lớp -> chỉ số (nhị phân: 0, 1)
    source: SourceConfig = field(default_factory=SourceConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    @property
    def class_names(self) -> list[str]:
        return list(self.classes)


@dataclass
class GeneratorConfig:
    """StyleGAN2-ADA có điều kiện + early stopping theo KID của lớp thiểu số."""

    cfg: str = "paper256"
    batch: int = 16
    gamma: float = 1.0
    mirror: bool = True
    aug_target: float = 0.6
    max_kimg: int = 3000
    kimg_per_snap: int = 100
    min_kimg: int = 400              # chưa đếm patience trước mốc này
    patience: int = 5                # số snapshot liên tiếp KID không cải thiện thì dừng
    min_rel_delta: float = 0.02      # KID phải giảm ít nhất 2 % mới tính là cải thiện
    kid_n_gen: int = 1000
    kid_subsets: int = 50
    kid_seed: int = 123
    gen_seed: int = 777
    trunc_psi: float = 1.0
    workers: int = 2
    poll_seconds: int = 30
    channel_tolerance: int = 0       # GAN 3 kênh cho dữ liệu 1 kênh: chênh lệch kênh tối đa (mức xám) khi gộp
    match_resize_chain: bool = False  # cho ảnh sinh đi qua cùng chuỗi nội suy với ảnh thật


@dataclass
class AnalysisConfig:
    fingerprint_n_max: int = 400
    fingerprint_epochs: int = 15


@dataclass
class SelectionConfig:
    pool_mult: float = 1.5           # k: pool = k × số ảnh cần chọn (chung mọi bộ dữ liệu)
    sim_topk: int = 5                # S⁺, S⁻ = trung bình cosine với TOP-K ảnh thật gần nhất
    lambda_v: float = 1.0
    lambda_d: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    oversample_variant: bool = True      # thêm M0b (nhân bản ảnh thật lớp thiểu số lên 1 : 1) — baseline oversampling
    both_classes_variant: bool = False   # thêm M7 (ảnh sinh ở cả hai lớp) — tuỳ chọn, ngoài M0–M6 chuẩn


@dataclass
class EncoderConfig:
    e_d_model: str = "DenseNet121"   # nên KHÁC model báo cáo downstream
    e_d_seed: int = 4242
    e_d_head_epochs: int = 5
    e_d_ft_epochs: int = 20
    embed_batch_size: int = 32
    # Dùng lại E_d đã train của lần chạy khác thay vì train mới: "" = E_d của chính lần chạy này;
    # "base" = E_d của lần chạy chính (run_tag trước --tag) — dùng cho ablation để E_d giống hệt nhau;
    # tên run_tag khác = E_d của lần chạy đó. Khi dùng lại mà chưa có file -> báo lỗi, không train lại.
    e_d_from_run: str = ""


@dataclass
class ClassifierConfig:
    models: list[str] = field(default_factory=lambda: ["EfficientNetV2B0", "ResNet50", "DenseNet121",
                                                        "ConvNeXtTiny", "ViT-B16", "SwinT"])
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
    monitor: str = "val_macro_recall"
    augment: bool = True             # augmentation khi train classifier (mọi biến thể, chỉ tập train)
    baseline_class_weight: bool = True   # M0 có class weight (false: baseline mất cân bằng thật, như v9 / bt_v4)
    mixed_precision: bool = False
    save_weights_to_drive: bool = True
    vit_preset: str = "hf://keras/vit_base_patch16_224_imagenet"
    swin_preset: str = "hf://keras/swin_tiny_patch4_window7_224"


@dataclass
class EvaluationConfig:
    threshold: float = 0.5           # ngưỡng cố định — không dò trên val / test
    baseline_method: str = "M0_real_only"
    # paired bootstrap ΔAUC bổ sung, ngoài "mọi phương pháp vs baseline": [phương pháp, đối chứng]
    comparisons: list[list[str]] = field(default_factory=lambda: [["M6_dass", "M0b_real_oversample"],
                                                                  ["M6_dass", "M1_random"]])
    # Tên hiển thị trong bảng bài báo (mã nội bộ giữ nguyên trong file .npz / selections.json); thứ tự = thứ tự
    # hàng. `$...$` được giữ làm công thức trong .tex, bỏ ký hiệu LaTeX trong .csv.
    method_labels: dict[str, str] = field(default_factory=lambda: dict(METHOD_LABELS))
    method_groups: dict[str, list[str]] = field(default_factory=lambda: {g: list(m) for g, m in METHOD_GROUPS.items()})
    n_bootstrap: int = 2000
    kid_subsets: int = 50
    kid_subset_size: int = 1000
    ssim_pairs: int = 200
    latex: bool = True


@dataclass
class Config:
    seed: int = 2026
    deterministic: bool = False
    save_figures: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate(cfg: Config) -> None:
    """Kiểm tra các ràng buộc pipeline giả định. Báo lỗi sớm, trước khi tốn GPU."""
    d, s, sp = cfg.data, cfg.data.source, cfg.data.split
    errors = []
    from ..models.classifiers import resolve_model_name  # không import TF

    for m in [*cfg.classifier.models, cfg.encoder.e_d_model]:
        try:
            resolve_model_name(m)
        except KeyError as e:
            errors.append(str(e).strip("'\""))
    if not cfg.classifier.seeds:
        errors.append("classifier.seeds không được rỗng")
    if not cfg.selection.pool_mult >= 1:
        errors.append(f"selection.pool_mult (k) phải >= 1 (pool không được nhỏ hơn số ảnh cần chọn), nhận "
                      f"{cfg.selection.pool_mult}")
    for pair in cfg.evaluation.comparisons:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(m, str) for m in pair)):
            errors.append(f"evaluation.comparisons: mỗi phần tử phải là [phương pháp, đối chứng], nhận {pair!r}")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in cfg.evaluation.method_labels.items()):
        errors.append("evaluation.method_labels phải là {mã phương pháp: tên hiển thị}")
    if not all(isinstance(v, list) for v in cfg.evaluation.method_groups.values()):
        errors.append("evaluation.method_groups phải là {tên nhóm: [mã phương pháp, ...]}")
    if cfg.encoder.e_d_from_run == "base" and not cfg.paths.base_run_tag:
        errors.append("encoder.e_d_from_run = base dùng cho ablation: cần --tag (để tách thư mục khỏi lần chạy chính)")
    if not cfg.paths.drive_root:
        errors.append("paths.drive_root chưa khai báo (cấu hình dataset)")
    if not d.name:
        errors.append("data.name chưa khai báo")
    if len(d.classes) != 2 or sorted(d.classes.values()) != [0, 1]:
        errors.append(f"data.classes phải có đúng 2 lớp với chỉ số 0 và 1 (bài toán nhị phân), nhận {d.classes}")
    if d.channels not in CHANNEL_MODES:
        errors.append(f"data.channels phải là một trong {CHANNEL_MODES}, nhận {d.channels!r}")
    if s.type not in SOURCE_TYPES:
        errors.append(f"data.source.type phải là một trong {SOURCE_TYPES}, nhận {s.type!r}")
    if not s.train_images:
        errors.append("data.source.train_images chưa khai báo")
    if s.type == "csv" and (not s.train_labels or (s.has_test_set and not s.test_labels)):
        errors.append("data.source.type = csv cần train_labels (và test_labels khi có test_images)")
    if s.type == "folders" and (s.train_labels or s.test_labels):
        errors.append("data.source.type = folders lấy nhãn từ thư mục lớp, không dùng *_labels")
    if s.type == "dicom_csv" and (not s.id_column or not s.label_column or s.has_test_set or s.subset_size < 0):
        errors.append("data.source.type = dicom_csv cần id_column, label_column, subset_size >= 0 và không có "
                      "test_images (test tách bằng split)")
    if s.exclude_labels and s.type != "dicom_csv":
        errors.append("data.source.exclude_labels chỉ dùng cho nguồn dicom_csv (nhãn JSON MD.ai)")
    overlap = {n.strip().lower() for n in s.exclude_labels} & {n.strip().lower() for n in s.positive_labels}
    if overlap:
        errors.append(f"data.source.exclude_labels trùng positive_labels: {sorted(overlap)}")
    if d.augment_profile not in AUGMENT_PROFILES:
        errors.append(f"data.augment_profile phải là một trong {AUGMENT_PROFILES}, nhận {d.augment_profile!r}")
    if d.preprocess.resize not in RESIZE_MODES:
        errors.append(f"data.preprocess.resize phải là một trong {RESIZE_MODES}, nhận {d.preprocess.resize!r}")
    if sp.type not in SPLIT_TYPES:
        errors.append(f"data.split.type phải là một trong {SPLIT_TYPES}, nhận {sp.type!r}")
    if sp.type in ("holdout_val", "stratified") and not 0 < sp.val < 1:
        errors.append(f"data.split.val phải trong (0, 1), nhận {sp.val}")
    if sp.type == "holdout_val" and (not s.has_test_set or sp.test):
        errors.append("data.split.type = holdout_val cần tập test riêng (source.test_images) và split.test = 0")
    if sp.type == "stratified" and (s.has_test_set or not 0 < sp.test < 1):
        errors.append("data.split.type = stratified tách test từ cùng nguồn: cần source.test_images = \"\" "
                      "và split.test trong (0, 1)")
    if sp.type == "file" and not sp.file:
        errors.append("data.split.type = file cần data.split.file")
    if errors:
        raise ValueError("Cấu hình không hợp lệ:\n  - " + "\n  - ".join(errors))
