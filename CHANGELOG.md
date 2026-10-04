# Changelog

## 1.9.0 — đóng các điểm còn mở sau khi rà soát toàn bộ code
Không đổi cách train, cách chọn ảnh hay bất kỳ kết quả nào đã có; chỉ thêm phép đo, truy vết và sửa lỗi nhỏ.

**Thống kê (`evaluate`)**
- `significance_vs_baseline`: thêm `delta_auc_seed_mean` / `delta_auc_seed_std` (ΔAUC ghép cặp theo seed, khớp bảng
  mean ± std; bootstrap trên ensemble chỉ đo dao động do mẫu test) và `p_holm` (Holm–Bonferroni trong mỗi model × k).
  Bảng `significance` thêm cột `p (Holm)`, `ΔAUC per seed`.
- `summary_stats`: chỉ 1 seed -> std = NaN, bảng in mean (trước đây "± 0.000" dễ hiểu nhầm). `summary_stats(by=…)`.

**Phân tích nhóm con (RSNA, đăng ký trước kết quả test)**
- Tuỳ chọn `evaluation.subgroup_columns` (mặc định rỗng; RSNA: `[ViewPosition]` từ `dicom_metadata.csv`). Module
  `evaluation/subgroups.py` (numpy / sklearn): AUC / PR-AUC trong từng nhóm AP / PA, ΔAUC trong nhóm (Holm trong mỗi
  model × nhóm), mốc AUC chỉ dùng tư thế, tỉ lệ "trông giống AP" trong ảnh dương thật / pool / ảnh mỗi phương pháp
  chọn (probe logistic trên E_v từ `embeddings.npz`). Metrics `classification_by_subgroup[_runs]`,
  `significance_by_subgroup`, `subgroup_reference`, `subgroup_share` + bảng cùng tên. Chạy trong `evaluate`, không GPU.
- `train` lưu `test_files` (tên ảnh test `lớp/tệp` đúng thứ tự dự đoán) trong `.npz`; `build_dataset` giữ
  `file_paths`. `.npz` cũ: dựng lại thứ tự từ split (lớp theo cấu hình, tên tăng dần), chỉ khi nhãn khớp.

**Truy vết** (nguyên nhân sự cố EfficientNetV2B0 train bằng code cũ)
- `utils.code_version()` (commit git, `-dirty` nếu sửa chưa commit). Đầu mỗi `train` in một dòng: commit, phiên bản,
  augment, profile, phương pháp có class weight, seeds. `.npz` lưu `code_version`, `dass_version`.
- `run_manifest.json`: `stages.<stage>` lưu riêng commit, phiên bản và config của từng stage (trước đây config cấp
  trên cùng bị ghi đè bởi stage gần nhất).

**Lỗi nhỏ**
- DICOM MONOCHROME1: đảo theo 2^BitsStored − 1 thay vì giá trị lớn nhất của từng ảnh (RSNA là MONOCHROME2, không ảnh
  hưởng).
- KID dừng sớm của GAN dùng `generator.trunc_psi` như lúc sinh pool (đang là 1,0 nên không đổi kết quả).
- `load_all_runs` đóng file `.npz`; giá trị trong `probs` thêm phần tử thứ ba (tên ảnh test | None).
- Chú thích bảng `.tex` bằng tiếng Anh (cùng ngôn ngữ với tên phương pháp).

**Tài liệu**: GUIDE (phiên bản, ba bộ dữ liệu, RQ6, mục 5.2 Holm + ΔAUC theo seed, mục 5.5 nhóm con, mối đe doạ:
shortcut tư thế, tắt augmentation không trung lập với ROS, M7 / phân tích độ nhạy kèm lệnh chạy riêng), CLAUDE.md,
README, RESULTS_FORMAT, notebook (ô evaluate in thêm các bảng nhóm con).

## 1.8.4 — RSNA: GAN gắn với split đã train
- `configs/datasets/rsna_pneumonia.yaml`: `data.split.expected: checkpoints_rsna_v2/data/real_val_split.json` (như
  Brain Tumor). GAN `rsna` train trên phần train của split `rsna_v2`; mọi lần chạy dùng GAN này (run_tag mới như
  `rsna_v3`, hoặc `--tag`) phải có đúng split đó, nếu không `Context` dừng với lỗi "GAN dùng lại có thể đã thấy ảnh
  val/test" -> chặn rò rỉ val / test qua GAN. Với chính `rsna_v2`: split so với chính nó, không đổi gì.

## 1.8.3 — `train --archive-mismatched`
- Dự đoán `.npz` đã có nhưng train theo giao thức khác cấu hình hiện tại (augmentation / class weight) vẫn bị chặn như
  trước; thêm cờ `--archive-mismatched` để **chuyển** (không xoá) `.npz` sang
  `results_<run>/predictions_superseded/<giao thức>/` và trọng số trên Drive sang
  `checkpoints_<run>/classifiers_superseded/<giao thức>/`, rồi train lại. Trùng tên thì thêm hậu tố, không ghi đè.
  Thư mục `predictions/` vì thế chỉ chứa một giao thức; `evaluate` không đọc thư mục superseded.
- Lý do: RSNA `rsna_v2` có một phần EfficientNetV2B0 đã train CÓ augmentation (trước khi Colab lấy bản 1.8.2); các
  kết quả đó được giữ lại trong thư mục superseded.

## 1.8.2 — RSNA: classifier không augmentation
- `configs/datasets/rsna_pneumonia.yaml`: `classifier.augment: false` (người dùng quyết định, trước khi train
  classifier RSNA). Mọi biến thể của RSNA cùng không augmentation nên vẫn so sánh công bằng; khác `v10` / `bt_v5` (có
  augmentation) -> nêu trong bài báo. Class weight cho M0 giữ nguyên (bật); E_d vẫn train có augmentation; ADA của
  GAN không đổi. ISIC / Brain Tumor không đổi. Test: classifier RSNA chỉ khác ISIC ở `augment`.

## 1.8.1 — RSNA: GAN không học bản lật ngang
- `configs/datasets/rsna_pneumonia.yaml`: `generator.mirror: false`. X-quang ngực không đối xứng trái / phải; với
  `mirror: true`, khoảng một nửa ảnh sinh sẽ có tim bên phải và chữ L / R bị ngược. Mọi tham số GAN khác giống ISIC /
  Brain Tumor (có test). Chốt trước khi train GAN RSNA lần đầu; ISIC / Brain Tumor không đổi. Classifier vẫn lật ngang
  ngẫu nhiên khi train (profile `upright`), như nhau cho mọi phương pháp.
- Notebook (`scripts/build_colab_notebook.py`): `show()` / `table()` đọc lại thư mục kết quả từ cấu hình mỗi lần gọi;
  trước đây `RESULTS` chỉ tính một lần nên sau khi đổi run_tag vẫn hiện hình của run cũ.

## 1.8.0 — RSNA `rsna_v2`: loại ảnh "Exclude", dùng toàn bộ dữ liệu
- Tuỳ chọn mới `data.source.exclude_labels` (mặc định rỗng; chỉ cho `dicom_csv` + nhãn MD.ai): ảnh có chú thích mang
  tên này bị loại TRƯỚC bước 1 ảnh / bệnh nhân. Không được trùng `positive_labels`.
- RSNA: `exclude_labels: [Exclude]` (106 ảnh: chụp nghiêng, ổ bụng, ảnh hỏng; 75 ảnh không có nhãn lâm sàng nào và
  trước đây bị xếp vào lớp âm), `subset_size: 0` (toàn bộ khoảng 12,2 nghìn ảnh thay vì 6.000), `run_tag: rsna_v2`.
  `rsna_v1` chỉ chạy đến `prepare`, giữ nguyên trên Drive.
- Ảnh cục bộ được đồng bộ khi thiết lập nguồn đổi trong cùng runtime: nguồn DICOM chuyển ảnh thiếu và xoá ảnh thừa
  (trước đây báo lỗi); `preprocess_tree` xoá ảnh đã xử lý không còn ảnh gốc. Tránh việc split mới lẫn ảnh của thiết lập
  cũ. Chỉ đụng bản cục bộ (/content); kết quả của run_tag cũ vẫn được bảo vệ bởi split lưu trên Drive.
- `label-stats`: cột "bị loại" khi có `exclude_labels`; bớt khoảng trắng phía trên hình ảnh mẫu.

## 1.7.2 — `label-stats` vẽ ảnh mẫu mỗi nhãn
- `label-stats --samples N` (mặc định 4; 0 = không vẽ): mỗi tên nhãn MD.ai một hàng gồm N ảnh ngẫu nhiên (seed), khung
  đỏ = khung của chính nhãn đó, dưới ảnh ghi lớp cuối cùng và tư thế chụp -> `results_<run_tag>/label_samples.png`
  (`Layout.label_samples_png`). Notebook có thêm ô gọi lệnh này sau `prepare`.

## 1.7.1 — lệnh `label-stats` (thống kê nhãn MD.ai theo ảnh)
- `dass -c <cấu hình> label-stats [--focus Exclude …]`: với nhãn JSON MD.ai, in theo từng tên nhãn số chú thích, số
  ảnh (nhiều khung / nhiều bác sĩ trên một ảnh chỉ tính một lần), % trên tổng số ảnh, số ảnh theo lớp cuối cùng và số
  bệnh nhân NIH; kèm các nhãn đi cùng của nhãn trong `--focus`. Chỉ đọc JSON, không chuyển ảnh, không ghi gì.
  Không đổi cách gán nhãn.

## 1.7.0 — RSNA: đọc bản phát hành MD.ai của trang RSNA, 1 ảnh / bệnh nhân
- Nguồn DICOM đọc được bản tải từ trang RSNA (xuất từ MD.ai): ảnh `Study/Series/SOP.dcm`, nhãn JSON
  `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` (dương = có chú thích trong `positive_labels`, ví dụ
  "Lung Opacity"; âm = các nhãn còn lại). Mã ảnh = StudyInstanceUID (gộp chú thích cấp ảnh và cấp ca chụp). Vẫn đọc
  được CSV kiểu Kaggle như trước.
- Mapping NIH (`pneumonia-challenge-dataset-mappings_2018.json`, tên ảnh `<mã bệnh nhân>_<số>.png`): lấy mã bệnh nhân
  NIH; `source.one_per_patient: true` giữ đúng 1 ảnh / bệnh nhân (seed) -> không rò rỉ bệnh nhân giữa train / val /
  test. Mã bệnh nhân ghi vào `dicom_metadata.csv`.
- Sửa thống kê đuôi file trong thông báo lỗi (tên file SOP UID có nhiều dấu chấm).

## 1.6.3 — RSNA: tách dữ liệu gốc và kết quả
- `configs/datasets/rsna_pneumonia.yaml`: kết quả ghi vào `RSNA Pneumonia/Result_Pneumonia` (`drive_root`), dữ liệu gốc
  đọc từ `RSNA Pneumonia` (`train_images`, đường dẫn tuyệt đối) — như Brain Tumor.
- Nguồn DICOM bỏ qua thư mục kết quả (`checkpoints_*`, `results_*`) ở MỌI cấp (thư mục kết quả có thể nằm bên trong
  thư mục dữ liệu).

## 1.6.2 — RSNA: thư mục trùng tên do Google Drive tạo
- Nguồn DICOM: thư mục cấu hình không có dữ liệu nguồn mà có đúng một bản `<tên> (N)` chứa dữ liệu -> đọc dữ liệu từ bản
  đó (cảnh báo trong log); kết quả vẫn ghi vào thư mục cấu hình. Nếu không, lỗi in nội dung các bản trùng tên.
- Khi tìm dữ liệu nguồn (.dcm / .csv / file nén) bỏ qua thư mục kết quả của pipeline (`checkpoints_*`, `results_*`):
  không nhặt nhầm CSV số liệu hay file pool .zip.

## 1.6.1 — RSNA: tìm nhãn cả trong file nén
- Nguồn DICOM: chưa thấy CSV nhãn thì giải nén các file .zip / .tar rồi tìm lại (trước đây chỉ giải nén khi thiếu ảnh).
- Không tìm thấy nhãn / ảnh -> lỗi in số file theo đuôi và tên các file không phải .dcm (chẩn đoán cấu trúc dữ liệu đã
  tải, ví dụ nhãn ở dạng JSON).

## 1.6.0 — bộ dữ liệu RSNA Pneumonia (DICOM)
- Nguồn mới `data.source.type: dicom_csv` (`data/sources/dicom_source.py`): tự tìm file `.dcm` (đệ quy; tự giải nén
  .zip / .tar ra ổ cục bộ nếu chưa giải nén) và CSV nhãn (cột `id_column` + `label_column`; nhiều dòng cùng mã -> nhãn
  lớn nhất). Ảnh không có nhãn (test của cuộc thi) bị bỏ qua. Chuyển sang PNG xám 1 kênh (đảo MONOCHROME1, co giãn ảnh
  > 8 bit). Tập con phân tầng `subset_size` (seed toàn cục). Ghi `checkpoints_<run>/data/dicom_metadata.csv` (tư thế
  chụp AP / PA, giới tính, tuổi) và in tỉ lệ AP / PA theo lớp để kiểm tra shortcut. Lần sau đọc đánh dấu
  `.ingest.json`, không quét lại Drive; đổi thiết lập nguồn mà giữ thư mục cũ -> báo lỗi.
- `configs/datasets/rsna_pneumonia.yaml` + `configs/experiments/rsna_pneumonia_dass.yaml`: 1 kênh, GAN mới
  (`gan_tag: rsna`), `run_tag: rsna_v1`, positive = `Target = 1` (đám mờ phổi), negative = còn lại, tập con 6.000 ảnh,
  chia stratified 70 / 15 / 15.
- `data.augment_profile`: `rotation_invariant` (mặc định, như cũ: lật ngang + dọc, xoay ±180°) hoặc `upright` (lật
  ngang, xoay ±10° — ảnh có hướng giải phẫu như X-quang ngực). Áp dụng cho classifier và E_d; RSNA dùng `upright`.
- Phụ thuộc mới: `pydicom`.

## 1.5.1 — sửa lỗi logic (review toàn bộ code)
- **Augmentation:** 5 lớp ngẫu nhiên (lật, xoay, zoom, độ sáng, tương phản) dùng CHUNG một seed. Keras 3 rút số bằng RNG
  không trạng thái theo bộ đếm, nên cả 5 phép biến đổi của một ảnh bị khoá vào cùng một số ngẫu nhiên (ảnh bị lật thì
  luôn tối hơn, giảm tương phản, xoay về một phía). Nay mỗi lớp một seed (`seed`, `seed + 1`, …). Lỗi này có từ notebook
  gốc: **mọi lần chạy cũ có augmentation (ISIC `v7`, Brain `bt_v3`) và mọi E_d** đã dùng augmentation tương quan. Phép
  map augmentation chạy tuần tự (trạng thái seed dùng chung, chạy song song không xác định).
- **Dự đoán `.npz` ghi nguyên tử** (file tạm + đổi tên): Colab ngắt giữa chừng không còn để lại file hỏng bị coi là
  "đã có".
- **Không dùng lại kết quả của giao thức khác:** nếu `.npz` đã có nhưng `augment` / `class_weight` lưu trong file khác
  cấu hình hiện tại -> báo lỗi (trước đây lặng lẽ bỏ qua). `evaluate` báo lỗi nếu một thư mục dự đoán trộn lần chạy có
  và không có augmentation.
- **Tên M0 khớp với cách đã train:** tên gốc là "Imbalanced Baseline"; `report` tự thêm " (class-weighted)" khi mọi lần
  chạy của phương pháp đó có class weight (đọc từ `.npz`). Bảng của `v9` / `bt_v4` không còn bị ghi sai.
- Class weight của baseline xác định bằng hằng `BASELINE`, không phụ thuộc `evaluation.baseline_method` (cài đặt báo cáo).
- `selection.pool_mult` (k) < 1 bị từ chối ngay khi nạp cấu hình.
- `pipeline/pool.py`: đóng file ảnh sau khi đọc kích thước (chỉ khi bật `match_resize_chain`).
- Sửa CLAUDE.md (mục bất biến bị dính chữ ở lần sửa 1.5.0) và checklist trong GUIDE (còn giá trị của 1.4.0).

## 1.5.0 — huấn luyện như ISIC v7, k = 1,5
- **Quay lại thiết lập huấn luyện của ISIC v7** (theo yêu cầu người dùng): `classifier.augment: true` (cùng
  augmentation cho mọi biến thể) và `classifier.baseline_class_weight: true` (M0 có class weight). Tên hiển thị của M0:
  "Imbalanced Baseline (class-weighted)".
- **k = 1,5** (trước: 2). Pool: ISIC 705, Brain Tumor 1.680; mốc Jaccard ngẫu nhiên 0,5.
- Giữ từ các bản sau v7: ROS (M0b), 3 seed, các cặp kiểm định bổ sung, tên hiển thị / nhóm.
- Run tag mới: ISIC **`v10`**, Brain Tumor **`bt_v5`**. Giữ nguyên `v9` / `bt_v4` (1.4.0: k = 2, không augmentation,
  M0 không class weight) để báo cáo riêng.
- Ghi chú sửa lại: mục 1.1.0 ghi "k chọn theo CosSIF ≈ 3,6" — không chính xác; CosSIF (FAGT) chỉ loại 15–25 % ảnh sinh
  (k ≈ 1,2–1,3).

## 1.4.0 — classifier không can thiệp dữ liệu, k = 2, tên phương pháp cho bài báo
- **Không augmentation khi train classifier** (`classifier.augment: false`) cho mọi biến thể. E_d (thành phần của DASS)
  vẫn train có augmentation; ADA của GAN giữ nguyên.
- **Imbalanced Baseline (M0) không class weight** (`classifier.baseline_class_weight: false`): train thẳng trên dữ liệu
  mất cân bằng. Khác biệt giữa các biến thể chỉ còn đến từ dữ liệu train.
- **k = 2** (trước: 3). Lý do: với k = 3, tập DASS lệch xa phân phối ảnh thật (Brain `bt_v3`: KID 0,100 so với pool
  0,029; đa dạng 0,168 so với 0,241). Mốc Jaccard ngẫu nhiên: 1/3.
- **Tên hiển thị trong bảng** (`evaluation.method_labels`, `evaluation.method_groups`): Real Data Baselines (Imbalanced
  Baseline, Random Oversampling (ROS)) và Generative Augmentation (StyleGAN2-ADA) (Unfiltered GAN (Random Selection),
  Visual-only / Disease-only / Diversity-only Filter, Dual-Margin Filter, DASS (Ours)). Bảng có cột `Group`, hàng theo
  thứ tự trên; `$...$` giữ làm công thức trong `.tex`, bỏ LaTeX trong `.csv`. **Mã nội bộ không đổi** (`.npz`,
  `selections.json`) — đổi tên chỉ cần chạy lại `report`.
- `.npz` dự đoán ghi thêm `augment`, `class_weight` (truy vết thiết lập train).
- Run tag mới: ISIC **`v9`**, Brain Tumor **`bt_v4`**. Giữ nguyên `bt_v3` (k = 3, có augmentation, M0 có class weight);
  ISIC `v8` (k = 3) chưa từng chạy.

## 1.3.0 — baseline oversampling M0b và kiểm định bổ sung
- Biến thể mới **`M0b_real_oversample`**: nhân bản ảnh THẬT lớp thiểu số lên 1 : 1 (random oversampling, Buda et al.
  2018), không class weight, cùng augmentation. Cùng số ảnh, cùng số bước train, cùng cách cân bằng với M1–M6, nên
  **M6 vs M0b** đo đúng đóng góp của nội dung ảnh sinh. Ảnh được nhân bản đều nhất có thể
  (`data.variants.oversample_indices`, seed toàn cục): số lần xuất hiện của các ảnh chênh nhau tối đa 1.
- M0b không chọn từ pool nên **không nằm trong `selections.json`**: tập train được ghép ở bước `train`
  (`selection.oversample_variant`, mặc định bật). Không cần chạy lại `select`; các dự đoán `.npz` đã có được giữ nguyên,
  chạy lại ô `train` chỉ train thêm M0b (3 seed mỗi backbone).
- `evaluation.comparisons` (mặc định `[[M6_dass, M0b_real_oversample], [M6_dass, M1_random]]`): paired bootstrap ΔAUC
  bổ sung, ghi chung vào `metrics/significance_vs_baseline` và bảng `tables/significance` (cột `vs` = đối chứng).
- Ma trận thực nghiệm: 8 biến thể × 6 backbone × 3 seed = 144 lần train mỗi bộ dữ liệu.

## Tài liệu (sau 1.2.0)
- `docs/GUIDE.md` viết lại theo cấu trúc bài báo: phát biểu hình thức, câu hỏi nghiên cứu / giả thuyết, thiết kế
  thực nghiệm, phân tích thống kê, các mối đe doạ đến tính hợp lệ, checklist kiểm tra, quy tắc báo cáo. Sửa ba điểm
  sai so với code: E_d có dùng val để chọn epoch; M1–M6 train không có class weight (chỉ M0); `results_bt` là kết quả
  công thức cũ của notebook v1 (không phải k = 4).

## 1.2.0 — k = 3 cho cả hai bộ dữ liệu
- `selection.pool_mult` (k) = **3** (trước: 4), chung cho ISIC và Brain Tumor; DASS giữ 1/3 pool
  (ISIC 1.410, Brain Tumor 3.360 ứng viên).
- ISIC chuyển sang `run_tag: v8` để không lẫn với kết quả k = 4 cũ của notebook v5 (`checkpoints_v7`, `results_v7`
  giữ nguyên). Brain Tumor giữ `bt_v3` (chưa có lần chạy thật nào).
- Bỏ các profile ablation theo k (`configs/experiments/ablation/`) — không chạy ablation. `encoder.e_d_from_run`
  vẫn giữ cho các ablation sau này.

## 1.1.0 — k thống nhất giữa hai bộ dữ liệu, dùng chung E_d cho ablation
- k (`pool_mult`) = 4 cho cả ISIC và Brain Tumor (chọn trước, theo CosSIF ≈ 3.6); test chặn việc ghi đè tham số
  chọn ảnh trong `configs/datasets/`.
- `configs/experiments/ablation/k{2,3,8}.yaml`: độ nhạy theo k, CÙNG file cho cả hai bộ dữ liệu
  (1 backbone × 3 seed, bắt buộc `--tag`).
- `encoder.e_d_from_run`: dùng lại E_d đã train (`base` = lần chạy chính) thay vì train mới; thiếu file -> báo lỗi.
  `paths.base_run_tag` tự điền khi dùng `--tag`. Đường dẫn E_d tập trung ở `Layout.e_d_ckpt`.
- Pillow 13: bỏ tham số `mode` của `Image.fromarray`.

## 1.0.2 — loại manh mối màu giả ở Brain Tumor
- Đo trên dữ liệu thật: 129 / 2000 ảnh **negative** có nhiễu màu JPEG (lệch kênh ≤ 3.1 mức xám), **0 / 400 positive**
  -> "có chút màu" gắn với nhãn negative (và ảnh sinh positive có kênh bằng nhau) = shortcut tiềm ẩn.
- `data.force_grayscale` (Brain Tumor: `true`): chuyển về luminance rồi lưu 3 kênh BẰNG NHAU cho cả ảnh thật
  (tiền xử lý) lẫn ảnh sinh (`_sample_worker`). 2271 / 2400 ảnh không đổi giá trị. Pool có hậu tố `_gray`.
- `preprocess` tự xử lý lại ảnh cũ còn lệch kênh. Ngưỡng nhận diện ảnh xám (`auto`) nới lên 5 mức xám (nhiễu JPEG).

## 1.0.1
- Brain Tumor: `data.channels: 3` (như notebook v1 và GAN `checkpoints_bt` đã train trên ảnh 3 kênh); ảnh xám gốc
  được lưu thành 3 kênh bằng nhau. Chế độ 1 kênh vẫn có cho bộ dữ liệu khác (`channels: 1` / `auto`).
- `.gitignore`: chỉ chặn `/data/` ở gốc (trước đó chặn nhầm `src/dass/data/`); thêm test kiểm tra.

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
