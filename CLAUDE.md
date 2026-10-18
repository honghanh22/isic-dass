# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này.

## Tổng quan

Mã nguồn chính thức cho bài báo: StyleGAN2-ADA có điều kiện + chọn ảnh sinh DASS cho phân loại ảnh y tế mất cân bằng (nhị phân). Các benchmark: ISIC 2016 (RGB), RSNA Pneumonia (X-quang ngực, DICOM, 1 kênh, GAN `gan_tag: rsna`) và PCam (mô bệnh học H&E, 128 px, file HDF5 đã tiền xử lý, âm : dương = 6), chạy chung một package `dass`, **cùng công thức DASS** và **cùng thiết kế thí nghiệm (2.0.0)**; chỉ khác nhau ở `configs/datasets/*.yaml`. Brain Tumor đã bị **loại khỏi dự án và bài báo** (người dùng quyết định, 1.10.0: AUC chạm trần khoảng 0,99, không phân biệt được phương pháp); kết quả cũ của Brain vẫn nằm trên Drive, không xoá.

Tài liệu:
- [README.md](README.md): tiếng Anh, viết cho reviewer / cộng đồng.
- [docs/GUIDE.md](docs/GUIDE.md): tổng quan cho người mới, gồm bài toán, phương pháp, từng bước và cách chạy. Cập nhật file này khi công thức, tham số hoặc quy trình thay đổi.
- [docs/METHOD.md](docs/METHOD.md): phương pháp của bài báo viết lại từ code (công thức, tham số, giao thức đánh giá, giới hạn). Cập nhật khi phương pháp thay đổi.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): kiến trúc các tầng.
- [docs/WORKFLOW.md](docs/WORKFLOW.md): cách làm việc hằng ngày.
- [docs/RESULTS_FORMAT.md](docs/RESULTS_FORMAT.md): định dạng kết quả.

Quy ước ngôn ngữ: chú thích, docstring, log và `docs/` viết bằng **tiếng Việt**; tên hàm / biến và README bằng tiếng Anh.

`notebooks/archive/` là notebook gốc (ISIC v5), chỉ để tham khảo; **không chạy, không dùng làm nguồn sự thật**.

## Lệnh

```bash
pip install -e ".[dev]"            # cục bộ, không cần GPU
pytest                             # unit + regression, ~10 s; không cần TF / torch / GPU
ruff check src tests scripts
python -m compileall -q src        # kiểm tra cú pháp các module TF / torch không import được cục bộ
dass -c configs/experiments/rsna_pneumonia_dass.yaml show-config
python scripts/build_colab_notebook.py   # sinh lại notebooks/colab_pipeline.ipynb
```

Các stage `gan*`, `sample`, `fingerprint`, `select`, `train` và `evaluate` (phần KID) cần Colab có GPU cùng Drive đã mount, nên **không chạy được trên máy Windows cục bộ**.

## Kiến trúc (chi tiết: docs/ARCHITECTURE.md)

- Các tầng: `data/` (Data Loader), `models/` (generator, encoders, classifiers), `selection/` (DASS), `evaluation/` (metric + bảng), `analysis/` (chẩn đoán), `engine/` (train classifier), `pipeline/` (context + stages), `config/`.
- **Mọi khác biệt giữa bộ dữ liệu nằm trong `data/`:** `sources/` (csv | folders | dicom_csv | hdf5), `transforms.py`, augmentation (`data.augment_profile`), `splits/` (holdout_val | stratified | file). Không thêm nhánh `if dataset == ...` ở chỗ khác; khác biệt mới phải thành tuỳ chọn config có mặc định.
- `config/schema.py` là dataclass. Giá trị mặc định = `configs/_base_/*.yaml` (có test đối chiếu); khoá lạ thì báo lỗi. `config/paths.py` (`Layout`) là **nơi duy nhất** định nghĩa đường dẫn.
- Quy tắc import:
  - `selection/`, `evaluation/`, `data/` (trừ `loaders.py`), `config/` là numpy / sklearn / PIL thuần.
  - TensorFlow chỉ có ở `data/loaders.py`, `models/encoders`, `engine/`, và trong hàm của `analysis/fingerprint.py`.
  - torch chỉ có trong hàm của `models/generator/*` và ở tiến trình con `_sample_worker.py`.
  - Stage import module nặng bên trong hàm `run`.
- StyleGAN2-ADA không được vendor vào repo: `models/generator/patches.py` clone repo NVlabs rồi áp `PATCHES` (idempotent).

## Bất biến (không được phá vỡ — đa số có test regression)

- **Giữ số kênh gốc:** mọi đọc / ghi ảnh đi qua `data/image_io.py`. Ảnh xám lưu PNG "L" và được giữ 1 kênh. Chỉ nhân bản 1 → 3 kênh trên bộ nhớ, ngay trước mạng pretrain (`data/loaders.to_backbone_input`, `models/generator/inception`). Không bước nào được biến đổi ảnh theo từng kênh. GAN 3 kênh dùng cho dữ liệu 1 kênh chỉ được gộp kênh khi 3 kênh giống hệt nhau (`generator.channel_tolerance`).
- Tuỳ chọn chung `force_grayscale` (hiện không bộ dữ liệu nào dùng): ảnh xám lưu RGB được chuyển về luminance rồi lưu 3 kênh bằng nhau, cho cả ảnh thật và ảnh sinh (tránh nhiễu màu JPEG gắn với một lớp). Mọi phép chuyển xám phải dùng đúng luminance của PIL (`convert("L")`), giống nhau cho ảnh thật và ảnh sinh.
- **Spectral mitigation (Dong et al.) và power-profile đã bị loại bỏ**; `tests/regression/test_no_spectral_mitigation.py` chặn việc đưa lại.
- Val / test 100 % ảnh thật. Val chỉ dùng để chọn epoch (`val_auc`). Test dùng ngưỡng cố định 0,5 và chỉ được dự đoán một lần.
- `data/splits/stratified.py` phải giữ **đúng** thuật toán đã khoá trong `tests/regression/test_split_reproduction.py` (ISIC: split của notebook v5, GAN `v5`; RSNA: split `rsna_v2`, GAN `rsna` gắn qua `split.expected`). Không đổi thứ tự gọi RNG. `split.expected` được kiểm tra khi dùng lại GAN.
- **Thiết kế thí nghiệm 2.0.0** (người dùng quyết định, 2026-10-18; thiết kế cũ đã bị XOÁ khỏi code, tái lập kết
  quả cũ bằng commit cũ). Nhóm 1 (bảng `comparison`): B0 Imbalanced, B1 class-weighted loss, B2 ROS (1 : 1), B3
  augmentation trên dữ liệu mất cân bằng, B4 = M1, B5 = M6. Nhóm 2 (bảng `ablation`): M1 Random, M2 M_v, M3 M_d,
  M4 Diversity/Coverage, M5 M_v + M_d, M6 DASS. "Balanced GAN" (M1–M6) = ảnh thật + g ảnh sinh **chỉ ở lớp thiểu số**
  (g = n_đa số − n_thiểu số, cân bằng 1 : 1), chọn từ CÙNG pool ⌈k·g⌉, k = 2. Ảnh sinh chỉ ở một lớp -> nguy cơ
  đường tắt "trông giống ảnh GAN -> bệnh": `scripts/shortcut_probe.py` là BẮT BUỘC và phải báo cáo. Họ giả thuyết chính
  (khai báo trước): **H1** M6 vs M1, **H2** M6 vs B2, **H3** M6 vs B0; còn lại là khám phá. Augmentation CHỈ ở B3,
  class weight CHỈ ở B1 (`Variant.augment` / `Variant.class_weight`, ghi trong .npz). Mã nội bộ = mã trong bảng (B0_…,
  M1_… M6_…); tên hiển thị ở `evaluation.method_labels` / `result_tables`. Classifier: ResNet50, ConvNeXt-Tiny, Swin-T
  (Swin từ KerasHub nhưng CÙNG head GAP + dropout + Dense với hai CNN), 3 seed. Lần chạy: ISIC `v14`, RSNA `rsna_v5`,
  PCam `pcam6_v1` (server thêm `_srv`; GAN `pcam6`, E_d train trong lần chạy).
- **Tài liệu và mô tả chỉ trình bày thiết kế hiện tại** (người dùng yêu cầu, 2026-10-15): `docs/`, README, chú thích
  trong `configs/`, chữ trong notebook không nhắc thiết kế tập train cũ, các biến thể cũ hay các lần chạy cũ. Code vẫn
  giữ các nhánh cũ (`selection.design: minority_only`, profile `configs/experiments/design_a.yaml`, các tuỳ chọn mở
  rộng mặc định tắt) để tái lập kết quả cũ trên Drive: không xoá, không phá (có test). CHANGELOG giữ nguyên lịch sử.
- Công thức chọn ảnh: M1–M6 với `S_DASS = α·M̃_v + β·M̃_d + γ·S̃_div`. Không đổi **mã** các biến thể đã có (mã nằm
  trong `.npz` trên Drive). `.npz` ghi `design`; trộn hai thiết kế trong một thư mục dự đoán bị chặn.
- **Classifier** (mọi bộ dữ liệu, không ghi đè trong `configs/datasets/`, có test): chọn epoch theo `val_auc`;
  augmentation chỉ ở B3, class weight chỉ ở B1. E_d luôn train có augmentation + class weight (thành phần của DASS);
  ADA của GAN giữ nguyên. RSNA còn `generator.mirror: false` (X-quang không đối xứng trái / phải; có test).
- **k (`pool_mult`) = 2 và mọi tham số DASS / E_d giống nhau giữa các bộ dữ liệu.** Không ghi đè trong
  `configs/datasets/` (có test). k đã đổi nhiều lần theo quyết định của người dùng (4 → 3 → 2 → 1,5 → 2 ở 2.0.0), có lần
  sau khi đã xem kết quả test — phải nêu trong bài báo và báo cáo mọi cấu hình đã chạy.
- KID / FID tính trên Inception-v3 của StyleGAN2-ADA; KID là chỉ số chính, FID chỉ để tham khảo.
- Chỉ số chọn snapshot / dừng sớm GAN (`generator.select_metric`) so với ảnh thật của **train**, không bao giờ dùng val (val chỉ để chọn epoch classifier; có test `tests/regression/test_gan_kid_reference.py`). Mặc định (ISIC, RSNA): KID-Inception lớp thiểu số, không làm mượt. PCam (2.1.0, người dùng quyết định, khai báo trước khi train GAN): FID trên đặc trưng **chính E_d** của lần chạy (cùng file với M_d), chấm cả hai lớp, gộp bằng **max** (lớp kém nhất) để dừng sớm; thiết kế chỉ thêm ảnh sinh vào lớp dương dùng FID lớp dương (phát lại trên lịch sử); làm mượt EMA α = 0,5 (Xue et al. 2021), cùng luật dừng sớm; dùng chung E_d cho hai bước là giới hạn phải nêu trong bài.
- PCam: nguồn `hdf5` đọc MỘT file đã tiền xử lý (`scripts/preprocess_pcam.py`: gộp 3 split, bỏ trùng, bỏ nền trắng, chia theo WSI 70 / 15 / 15, âm : dương = 12, rồi tập con 6 : 1 `pcam_imb6.h5`) + CSV chia train / val đi kèm; ảnh 128 × 128.
- **Tách bộ dữ liệu** (có test `tests/regression/test_dataset_isolation.py`): ổ tạm `<local_root>/<data.name>` thuộc
  đúng một nguồn (`.owner.json`; trùng tên khác nguồn -> dừng); GAN train từ 1.10.1 ghi `data_name` + `split_sha1` vào
  `gan_state.json` và mọi stage dùng GAN kiểm tra (GAN cũ ISIC `v5` / RSNA `rsna` chưa ghi -> dựa vào
  `split.expected` / test regression). Lớp thiểu số nên là lớp 1 (metric tính cho lớp 1; khác thì cảnh báo).
- Kiểm định: paired bootstrap ΔAUC trên ensemble seed + ΔAUC theo từng seed + `p_holm` (Holm trong mỗi model). RSNA có
  phân tích nhóm con theo tư thế (`evaluation.subgroup_columns: [ViewPosition]`, đăng ký trước kết quả test): AUC
  trong nhóm AP / PA, mốc chỉ dùng tư thế, tỉ lệ "trông giống AP" của ảnh sinh. `.npz` lưu `test_files` và
  `code_version`; `run_manifest.json` lưu commit + config theo từng stage.

## Tương thích artefact trên Drive

Các tên sau giữ nguyên (`tests/regression/test_artifact_paths.py`):
- `checkpoints_<gan_tag>/stylegan2ada/{best,latest}.pkl`, `gan_state.json`
- `checkpoints_<run_tag>/data/{real_val_split.json, selections.json, embeddings.npz, dataset_card.json}`
- `pool_<lớp>_from<kimg>kimg_n<N>[_c1].zip`
- `Ed_<model>_s<seed>.weights.h5`
- `results_<run_tag>/predictions/<model>__<variant>__s<seed>.npz`

Đổi định dạng thì sửa cả code đọc lẫn code ghi, và ghi vào CHANGELOG. Không bao giờ xoá artefact trên Drive; muốn tách lần chạy thì dùng `--tag` hoặc `run_tag` mới.

## Vòng làm việc với người dùng (VS Code + extension Colab)

- Người dùng chạy cell trong `notebooks/colab_pipeline.ipynb` (kernel là Colab) rồi lưu bằng Ctrl+S. **Đọc output trực tiếp từ file `.ipynb`**, không bắt người dùng copy log.
- Notebook lấy code từ GitHub (`REPO_URL` trong cell lấy mã nguồn; repo public `honghanh22/isic-dass`). Sau khi sửa `src/` hoặc `configs/` và chạy `pytest`, **phải commit và push** thì Colab mới thấy thay đổi. Máy người dùng không có Git: dùng `dulwich` (đã cài qua pip) với credential `git:https://github.com` trong Windows Credential Manager, và không bao giờ in token ra. Sau đó nhắc người dùng chạy lại cell lấy mã nguồn (`git pull`) rồi chạy lại cell cần chạy.
- Không đưa logic vào notebook. Muốn sửa notebook thì sửa `scripts/build_colab_notebook.py` rồi chạy lại script.
