# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này.

## Tổng quan

Mã nguồn chính thức cho bài báo: StyleGAN2-ADA có điều kiện + chọn ảnh sinh DASS cho phân loại ảnh y tế mất cân bằng (nhị phân). Các benchmark: ISIC 2016 (RGB) và RSNA Pneumonia (X-quang ngực, DICOM, 1 kênh, GAN mới `gan_tag: rsna`), chạy chung một package `dass` và **cùng công thức DASS**; chỉ khác nhau ở `configs/datasets/*.yaml`. Brain Tumor đã bị **loại khỏi dự án và bài báo** (người dùng quyết định, 1.10.0: AUC chạm trần khoảng 0,99, không phân biệt được phương pháp); kết quả cũ của Brain vẫn nằm trên Drive, không xoá.

Tài liệu:
- [README.md](README.md): tiếng Anh, viết cho reviewer / cộng đồng.
- [docs/GUIDE.md](docs/GUIDE.md): tổng quan cho người mới, gồm bài toán, phương pháp, từng bước và cách chạy. Cập nhật file này khi công thức, tham số hoặc quy trình thay đổi.
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
- **Mọi khác biệt giữa bộ dữ liệu nằm trong `data/`:** `sources/` (csv | folders | dicom_csv), `transforms.py`, augmentation (`data.augment_profile`), `splits/` (holdout_val | stratified | file). Không thêm nhánh `if dataset == ...` ở chỗ khác; khác biệt mới phải thành tuỳ chọn config có mặc định.
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
- Val / test 100 % ảnh thật. Val chỉ dùng để chọn epoch (`val_macro_recall`). Test dùng ngưỡng cố định 0,5 và chỉ được dự đoán một lần.
- `data/splits/stratified.py` phải giữ **đúng** thuật toán đã khoá trong `tests/regression/test_split_reproduction.py` (ISIC: split của notebook v5, GAN `v5`; RSNA: split `rsna_v2`, GAN `rsna` gắn qua `split.expected`). Không đổi thứ tự gọi RNG. `split.expected` được kiểm tra khi dùng lại GAN.
- Công thức chuẩn: M0–M6 với `S_DASS = α·M̃_v + β·M̃_d + γ·S̃_div`, cộng baseline oversampling **M0b**
  (`M0b_real_oversample`: nhân bản ảnh thật lớp thiểu số lên 1 : 1, không class weight; ghép ở bước `train`, không nằm
  trong `selections.json`). M7 là tuỳ chọn, mặc định tắt. Không đổi **mã** các biến thể đã có (mã nằm trong `.npz` trên
  Drive); tên trong bài báo là **tên hiển thị** `evaluation.method_labels` (Imbalanced Baseline, Random Oversampling
  (ROS), Unfiltered GAN (Random Selection), Visual-only / Disease-only / Diversity-only Filter, Dual-Margin Filter,
  DASS (Ours)), chỉ áp dụng khi xuất bảng. Kiểm định: mọi biến thể vs M0, cộng `evaluation.comparisons` (M6 vs M0b,
  M6 vs M1).
- **Classifier không can thiệp dữ liệu, cho MỌI bộ dữ liệu** (người dùng quyết định, 1.11.0):
  `classifier.augment: false` và `classifier.baseline_class_weight: false` trong `configs/_base_/classifier.yaml`
  (M0 = baseline mất cân bằng thật; ROS là baseline cân bằng bằng ảnh thật). Không ghi đè trong `configs/datasets/`
  (có test: classifier của ISIC và RSNA giống hệt). Lần chạy hiện tại: ISIC `v11`, RSNA `rsna_v2`. Cấu hình cũ giữ
  nguyên trên Drive, báo cáo riêng: ISIC `v10` / `v7` (có augmentation, M0 có class weight), ISIC `v9` (cả hai tắt,
  k = 2). E_d luôn train có augmentation + class weight (thành phần của DASS), độc lập với hai tuỳ chọn trên; ADA
  của GAN giữ nguyên. RSNA còn `generator.mirror: false` (X-quang không đối xứng trái / phải; có test). Khi người
  dùng nói "không can thiệp / không tăng cường", hỏi rõ phạm vi: augmentation, class weight của M0, hay cả hai.
- **k (`pool_mult`) = 1,5 và mọi tham số DASS / E_d giống nhau giữa các bộ dữ liệu.** Không ghi đè trong `configs/datasets/` (có test). k đã đổi theo quyết định của người dùng: 4 (ISIC `v7`) → 3 → 2 (ISIC `v9`) → 1,5 (ISIC `v10` / `v11`, RSNA `rsna_v2`, hiện tại); lý do ghi nhận: chọn càng gắt thì tập DASS càng lệch phân phối ảnh thật. Cấu hình đổi sau khi đã xem kết quả test — phải nêu trong bài báo và báo cáo mọi cấu hình đã chạy. Không chạy ablation theo k. Mọi kết quả cũ giữ nguyên trên Drive; ISIC `v8` chưa từng chạy. Tuỳ chọn `encoder.e_d_from_run` (dùng lại E_d) vẫn có sẵn cho các ablation sau này.
- KID / FID tính trên Inception-v3 của StyleGAN2-ADA; KID là chỉ số chính, FID chỉ để tham khảo.
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
