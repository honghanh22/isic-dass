# Kiến trúc

## Các tầng và chiều phụ thuộc

```
                 cli.py ──► pipeline/stages/*  (điều phối, mỗi stage một tiến trình)
                                   │
         ┌──────────────┬──────────┼──────────────┬───────────────┬──────────────┐
         ▼              ▼          ▼              ▼               ▼              ▼
      data/          models/   selection/    evaluation/      analysis/      engine/
  (DATA LOADER)  (BACKBONE)  (SELECTION)    (METRICS)      (chẩn đoán)   (train classifier)
         │              │          │              │               │              │
         └──────────────┴──────────┴──── config/ (schema, loader, Layout) ── utils.py
```

- Tầng dưới **không** import tầng trên. `selection/` và `evaluation/` là numpy thuần, không TF / torch.
- TensorFlow chỉ có trong `data/loaders.py`, `models/encoders`, `engine/`, `analysis/fingerprint.py`
  (import trong hàm). PyTorch chỉ có trong `models/generator/*` và chỉ import trong hàm hoặc ở tiến trình con.
- `config/paths.py` (`Layout`) là nơi duy nhất định nghĩa đường dẫn artefact.

## Ba điểm khác biệt giữa các bộ dữ liệu — đều nằm trong `data/`

| Điểm | Module | Lựa chọn |
|---|---|---|
| Đọc nguồn | `data/sources/` | `csv` (thư mục phẳng + CSV nhãn), `folders` (thư mục theo lớp, `class_dirs` ánh xạ tên) |
| Tiền xử lý | `data/transforms.py` | `crop_dark_border`, `resize: stretch | pad_square` |
| Chia | `data/splits/` | `holdout_val` (có test riêng), `stratified` (tách test, `group_regex`), `file` |

Mọi thứ khác (GAN, DASS, classifier, metric) dùng chung 100 %, cùng công thức.

## Số kênh ảnh

`data/image_io.py` là nơi duy nhất đọc / ghi ảnh:

```
ảnh gốc ──detect_channels──► channels (1 | 3)
   │
   ├─ preprocess ─► PNG "L" (1 kênh) | "RGB" (3 kênh)          ← lưu trên đĩa: đúng số kênh gốc
   ├─ GAN dataset (zip) ─► StyleGAN2-ADA train generator 1 | 3 kênh
   ├─ sample ─► ảnh sinh lưu đúng số kênh; GAN 3 kênh + dữ liệu 1 kênh: gộp CHỈ KHI 3 kênh giống hệt (ngược lại dừng)
   └─ mạng pretrain ImageNet / Inception-v3: nhân bản 1 → 3 kênh trên bộ nhớ, ngay trước mạng
```

Không còn bước nào biến đổi ảnh theo từng kênh (nguồn gốc lỗi "ảnh xám có màu" của spectral mitigation, đã loại bỏ).

## Luồng dữ liệu của một thực nghiệm

```
prepare   nguồn ─► raw/ ─► pp{size}_png/ ─► split (+ test tách ra nếu cần) ─► dataset_card.json
gan       pp train ─► zip ─► StyleGAN2-ADA (KID early stopping) ─► best.pkl
sample    best.pkl ─► pool_<lớp>_<tag>.zip (Drive)
select    E_v, E_d ─► M_v, M_d ─► M0–M6 ─► selections.json, embeddings.npz, shortcut_check
train     variants/<M>_p<k>/{train,val} ─► <model>__<variant>__s<seed>.npz (dự đoán val + test)
evaluate  .npz ─► metrics/classification_*; Inception-v3 ─► metrics/generation_quality
report    metrics/ ─► tables/*.csv | *.json | *.tex
```

## Mở rộng

| Muốn | Làm |
|---|---|
| Bộ dữ liệu mới (nhị phân) | `configs/datasets/<tên>.yaml` (từ `_template.yaml`) + `configs/experiments/<tên>_dass.yaml` |
| Nguồn dữ liệu kiểu mới | lớp con `DatasetSource` trong `data/sources/` + nhánh trong `build_source` |
| Cách chia mới (k-fold, …) | hàm trong `data/splits/` + nhánh trong `build_split` + loại trong `SPLIT_TYPES` |
| Backbone mới | builder trong `models/classifiers/__init__.py` (`MODEL_BUILDERS`) |
| Biến thể chọn ảnh mới | `selection/strategies.py` (`METHODS`, `select_all_methods`) + test |
| Tham số mới | dataclass trong `config/schema.py` **và** `configs/_base_/*.yaml` (test đối chiếu hai nơi) |
| Stage mới | `pipeline/stages/<tên>.py` + lệnh trong `cli.py` + cell trong `scripts/build_colab_notebook.py` |
