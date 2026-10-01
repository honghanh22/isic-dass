# Changelog

## 1.0.0 — kiến trúc chuẩn bài báo, package `dass`

### Cấu trúc
- Package `isic_dass` → **`dass`**, lệnh CLI `isic-dass` → **`dass`**.
- Tách thành các tầng: `data/` (sources, transforms, splits, image_io, loaders), `models/` (generator, encoders, classifiers), `selection/`, `evaluation/`, `analysis/`, `engine/`, `pipeline/` (context, pool, stages/), `config/`.
- Cấu hình chia ba lớp: `configs/_base_/` (dùng chung), `configs/datasets/` (chỉ phần phụ thuộc dữ liệu), `configs/experiments/` (kế thừa qua `_base_`). `-c` dùng được nhiều lần; thêm `--tag` cho ablation.
- `data.source` (`csv` | `folders`), `data.preprocess`, `data.split` (`holdout_val` | `stratified` | `file`) thay cho các cờ phẳng của 0.7.
- Stage mới: `evaluate` (số thô → `metrics/`), `report` (bảng → `tables/*.csv|json|tex`), `run --from/--to` (chạy cả chuỗi, mỗi stage một tiến trình). Đổi tên: `gan-train` → `gan` (kèm báo cáo), `generate` → `sample`, `frequency` → `fingerprint`, `aggregate`/`quality` → `evaluate` + `report`.
- `run_manifest.json` (config đã hợp nhất, phiên bản thư viện, từng stage) và `dataset_card.json` (số kênh, số ảnh, SHA-1 của split).
- `select` không ghi đè `selections.json` đã có, trừ khi dùng `--force`.

### Loại bỏ Spectral Mitigation (Dong et al., CVPR 2022)
- Xoá module mitigation (SDN / PDC, tải mã nguồn của tác giả), power-profile detector, các khoá config `apply_spectral_mitigation` / `mitigation_mode`, và cell tương ứng trong notebook archive (ISIC cell 28–29, Brain Tumor cell 26–29).
- Xoá bước chuẩn hoá kênh màu của ảnh sinh (`harmonize`): không còn cần vì số kênh được giữ đúng từ gốc. Chuỗi resize được giữ thành `generator.match_resize_chain`.
- Khoá cũ trong `--set` giờ báo `Lỗi cấu hình: Khoá cấu hình không hợp lệ` thay vì lỗi tên biến.

### Giữ nguyên số kênh (sửa lỗi "ảnh xám bị thành ảnh màu")
- `data/image_io.py` là nơi duy nhất đọc / ghi ảnh. `data.channels: auto` tự nhận diện ảnh xám / màu.
- Ảnh xám lưu PNG "L" ở mọi bước: tiền xử lý, dataset GAN, ảnh sinh, tập train của classifier.
- Mạng pretrain ImageNet / Inception-v3 nhận ảnh 3 kênh bằng cách nhân bản trên bộ nhớ, ngay trước mạng.
- Dùng lại GAN Brain Tumor 3 kênh: ảnh sinh chỉ được gộp về 1 kênh khi 3 kênh giống hệt nhau (`generator.channel_tolerance = 0`); khác nhau thì dừng với `ChannelMismatchError`. Pool 1 kênh có hậu tố `_c1`.
- `preprocess` tự xử lý lại ảnh cũ bị sai số kênh. Kết quả Brain Tumor ghi vào `run_tag: bt_v3`.

### Đánh giá
- KID / FID của tập ảnh được chọn chuyển sang **Inception-v3** (trước đây tính trên đặc trưng E_v). KID báo cáo mean ± std, FID chỉ để tham khảo; thêm hàng tham chiếu "real val vs real train" và "all candidates".
- SSIM tính trên đúng số kênh gốc.
- Sửa lỗi: số ảnh nguồn CSV đếm cả ảnh có trong CSV nhưng không có trên đĩa.

### Công thức
Thống nhất theo ISIC v5 cho mọi bộ dữ liệu: M0–M6. M7 vẫn có nhưng tắt mặc định (kể cả Brain Tumor).

## 0.7.0 — Brain Tumor chuẩn hoá theo pipeline ISIC v5

Notebook `brain_tumor_stylegan2ada_dass_pipeline_v1.ipynb` (đã chuyển vào `notebooks/archive/`) được thay bằng
`configs/brain_tumor.yaml` chạy trên cùng package, **cùng công thức với ISIC v5**:

| | Brain Tumor v1 (cũ) | Chuẩn hoá (mới) |
|---|---|---|
| Điểm chọn | CDS = max S_same − 0.5·max S_other, chỉ E_v | M = S⁺ − λS⁻ (top-K), E_v + E_d, min-max |
| DASS | CDS + 0.5·D | α·M̃_v + β·M̃_d + γ·S̃_div |
| Biến thể | M1 unfiltered, M3 S_same, M4 CDS | M1 random, M2 visual, M3 disease, M4 visual+disease |
| Pool | k = 1.5 (pool lồng nhau 1–3) | pool_mult = 4 |
| Chọn epoch | val_auc | val_macro_recall |
| M0 | không class weight | class weight |
| Metric | — | thêm g_mean |

Giữ lại từ Brain Tumor (thành tuỳ chọn chung):
- `data.test_images: ""` + `data.test_fraction`: tự tách test từ cùng nguồn (phân tầng, cùng thuật toán v1 → split trùng khớp).
- `data.group_regex`: chia theo mã bệnh nhân. `data.resize_mode: pad_square`: đệm vuông giữ tỉ lệ giải phẫu.
- `data.class_folders`: tên thư mục khác tên lớp (`Positive` → `positive`).
- `selection.both_classes_variant`: biến thể **M7** — ảnh sinh ở cả hai lớp + class weight (chống shortcut).
- Sao lưu log + ảnh mẫu GAN lên Drive (`stylegan2ada/logs_and_samples/`), mọi bộ dữ liệu.
- Thống kê tỉ lệ vùng nội dung ở stage `prepare` (gợi ý có nên crop).

An toàn khi dùng lại GAN đã train (`gan_tag: bt`): `data.expected_split` so split mới với `checkpoints_bt/data/real_split.json`;
khác nhau → dừng (GAN có thể đã thấy val/test).

Sửa lỗi: spectral mitigation chạy từng kênh làm ảnh xám (MRI) có màu trở lại (chênh lệch kênh 0 → 0,17 ở v1, thành
shortcut mới). Khi ảnh thật là ảnh xám, ảnh sau mitigation được đưa về xám (`*_mitigated_<mode>_gray_*.zip`).

## 0.6.0 — dữ liệu và cách chia cấu hình được

- `data.*` mới: `name`, `train_images`, `train_labels`, `test_images`, `test_labels`, `labels_header`, `image_ext`,
  `crop_dark_border`, `split_method` (`random` | `file`), `split_file`. Mặc định giữ nguyên ISIC 2016.
- Nguồn dữ liệu dạng thư mục theo lớp (`*_labels: ""`) hoặc CSV nhãn; nhãn nhận theo `class_to_idx` (tên lớp hoặc chỉ số).
- `split_method: file`: đọc cách chia từ CSV `image_id,split`.
- Split đã lưu được so với cấu hình hiện tại: đổi cách chia / `seed` / `val_fraction` mà giữ `run_tag` -> báo lỗi.
- Kiểm tra cấu hình: đúng 2 lớp (chỉ số 0, 1), `split_method` hợp lệ.
- Dữ liệu cục bộ chuyển sang `/content/local_data/<data.name>/` (lần chạy đầu sẽ copy lại từ Drive).
- Tài liệu `docs/WORKFLOW.md`; mẫu `configs/example_custom_data.yaml`.

## 0.5.0 — tách notebook thành package

Mã nguồn chuyển từ `isic2016_stylegan2ada_dass_pipeline_v5.ipynb` (70 cell, dùng biến global) sang package
`isic_dass` với CLI theo stage. Thuật toán, siêu tham số và tên artefact trên Drive giữ nguyên.

### Sửa lỗi của notebook
- `evaluate_snapshot_kid` dùng `CFG.MALIGNANT_IDX` (không tồn tại) → dùng chỉ số lớp thiểu số tính từ dữ liệu.
- `CFG.K_RUN` (không tồn tại) ở các cell train ResNet50 / DenseNet121 / ConvNeXt / Swin → dùng `selection.pool_mult` thống nhất.
- `CFG.MITIGATION_MODE` chưa định nghĩa → `frequency.mitigation_mode` (mặc định `sdn+pdc`).
- Cell chất lượng ảnh sinh dùng tên cũ (`filtering_results_k`, `real_malignant_emb`, `n_train_malignant_real`, `info["emb"]`) → stage `quality` đọc `embeddings.npz` + `selections.json`.
- Hàm gộp kết quả bị định nghĩa hai lần (cell 67/68) → giữ bản có cột `lam`, vẫn đọc được file `.npz` cũ.
- Cell 5b (E_d) gọi hàm được định nghĩa ở cell sau → phụ thuộc giờ là import tường minh.
- `StyleGanTrainer` báo lỗi thay vì lặp vô hạn nếu `train.py` kết thúc mà không có snapshot mới.

### Loại bỏ
- Bước 7d (pool ảnh biến đổi kiểu CosSIF): tham chiếu `CLASS_BUDGET`, `LOCAL_TRANSFORMED`, `plot_image_grid` không tồn tại và kết quả không được dùng ở bước ghép tập.
- Các cell debug (xoá cache torch_extensions → `gan-setup --clear-ext-cache`; cell kiểm tra biến global "OK/THIẾU").

### Thay đổi hành vi (nhỏ)
- Split train/val được **đọc lại** từ `real_val_split.json` nếu đã có (trước đây tạo lại mỗi lần — cùng seed nên cùng kết quả) và được kiểm tra khớp với ảnh trên đĩa.
- Thư mục biến thể chỉ ghép lại khi đầu vào đổi (file `.complete.json`).
- SSIM nội bộ tính bằng `skimage` (cửa sổ Gauss σ = 1.5) thay cho `tf.image.ssim` — cùng định nghĩa, sai khác số học nhỏ.
- Khi hai ứng viên có điểm bằng nhau tuyệt đối, thứ tự chọn có thể khác notebook (sắp xếp ổn định).
- Hình được lưu ra file thay vì `plt.show()`; notebook Colab hiển thị lại từ Drive.
- Dữ liệu cục bộ ở `/content/local_data/candidates_<run_tag>`; zip dataset GAN đổi tên thành `isic2016_train_cond_256.zip` (chỉ cục bộ).
