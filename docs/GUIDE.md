# Hướng dẫn tổng quan: bài toán, phương pháp, thiết kế thực nghiệm và cách chạy

Tài liệu dành cho người mới tiếp cận dự án, đồng thời là nguồn tham chiếu khi viết phần *Phương pháp* và *Thực
nghiệm* của bài báo. Mọi công thức, con số và tham số dưới đây khớp với code (`src/dass/`) và cấu hình (`configs/`)
của phiên bản **1.11.0**. Hai bộ dữ liệu: **ISIC 2016** và **RSNA Pneumonia**; k = 1,5 và **cùng giao thức classifier
(không augmentation, M0 không class weight)** cho cả hai. Brain Tumor
đã bị loại khỏi dự án và bài báo (1.10.0: AUC chạm trần khoảng 0,99, không phân biệt được phương pháp; kết quả cũ vẫn
trên Drive). Khi đổi code hoặc cấu hình, cập nhật file này.

**Các cấu hình đã chạy** (kết quả đều giữ trên Drive, không trộn với nhau):

| Run tag | k | Augmentation classifier | Class weight M0 | Ghi chú |
|---|---|---|---|---|
| ISIC `v7` | 4 | có | có | notebook v5, chỉ EfficientNetV2B0, 2 seed, không có ROS |
| ISIC `v9` | 2 | **không** | **không** | EfficientNetV2B0, ResNet50 |
| ISIC `v10` | 1,5 | có | có | cấu hình chính của ISIC trước 1.11.0 |
| **ISIC `v11`** | **1,5** | **không** | **không** | cấu hình chính của ISIC (từ 1.11.0) |
| **RSNA `rsna_v2`** | **1,5** | **không** | **không** | cấu hình chính của RSNA (mục 2.5) |

**Giao thức chung (1.11.0, người dùng quyết định):** classifier **không can thiệp dữ liệu** ở bất kỳ phương pháp nào,
cho cả hai bộ dữ liệu — không augmentation và M0 không class weight. RSNA chạy như vậy từ đầu (augmentation tắt ở
1.8.2, class weight của M0 tắt ở 1.9.1; M0 của ResNet50 lỡ train có class weight được chuyển sang
`predictions_superseded/` và train lại). ISIC chuyển sang giao thức này ở `v11` (cùng GAN `v5`, cùng split, k = 1,5);
`v10` (có augmentation + class weight) giữ nguyên trên Drive để báo cáo riêng. `rsna_v2`: toàn bộ ảnh, đã loại
"Exclude"; `rsna_v1` (tập con 6.000 ảnh) chỉ chạy đến `prepare`.

**Lỗi đã sửa ở 1.5.1:** trước đó 5 phép augmentation dùng chung một seed nên bị tương quan với nhau (xem CHANGELOG). Các lần chạy có augmentation trước 1.5.1 (`v7`) và mọi E_d cũ dùng augmentation tương quan này; `v10` dùng augmentation đã sửa, nên không giống hệt `v7` ở điểm này.

Cấu hình được đổi nhiều lần sau khi đã xem kết quả test. Khi viết bài báo phải nêu rõ điều này, và nên báo cáo kết
quả của mọi cấu hình đã chạy (ví dụ trong phụ lục) thay vì chỉ chọn cấu hình có kết quả đẹp nhất.

**Tên phương pháp:** trong code và file kết quả dùng **mã nội bộ** (M0, M0b, M1–M6); trong bảng bài báo dùng **tên
hiển thị** (`evaluation.method_labels`):

| Nhóm | Mã | Tên trong bài báo |
|---|---|---|
| Real Data Baselines | M0 (`M0_real_only`) | **Imbalanced Baseline** (bảng tự thêm "(class-weighted)" nếu lần chạy có class weight, như ISIC `v10`) |
| | M0b (`M0b_real_oversample`) | **Random Oversampling (ROS)** |
| Generative Augmentation (StyleGAN2-ADA) | M1 (`M1_random`) | **Unfiltered GAN (Random Selection)** |
| | M2 (`M2_visual`) | **Visual-only Filter (M_v)** |
| | M3 (`M3_disease`) | **Disease-only Filter (M_d)** |
| | M5 (`M5_diversity`) | **Diversity-only Filter (S_div)** |
| | M4 (`M4_visual_disease`) | **Dual-Margin Filter (M_v + M_d)** |
| | M6 (`M6_dass`) | **DASS (Ours)** |

Tài liệu liên quan: [ARCHITECTURE.md](ARCHITECTURE.md) (cấu trúc code), [WORKFLOW.md](WORKFLOW.md) (làm việc hằng
ngày), [RESULTS_FORMAT.md](RESULTS_FORMAT.md) (ý nghĩa từng cột kết quả).

**Phần A — Nghiên cứu:** [1. Bài toán](#1-bài-toán) · [2. Dữ liệu](#2-dữ-liệu) · [3. Phương pháp](#3-phương-pháp) ·
[4. Thiết kế thực nghiệm](#4-thiết-kế-thực-nghiệm) · [5. Đánh giá và thống kê](#5-đánh-giá-và-phân-tích-thống-kê) ·
[6. Tính hợp lệ](#6-tính-hợp-lệ-rò-rỉ-shortcut-và-các-mối-đe-doạ) · [7. Hạn chế](#7-hạn-chế-đã-biết)

**Phần B — Thực hành:** [8. Cách chạy](#8-cách-chạy) · [9. Kiểm tra sau mỗi bước](#9-kiểm-tra-sau-mỗi-bước) ·
[10. Kết quả](#10-kết-quả-lưu-ở-đâu-và-đọc-thế-nào) · [11. Quy tắc báo cáo](#11-quy-tắc-báo-cáo) ·
[12. Bảng tham số](#12-bảng-tham-số)

---

# Phần A — Nghiên cứu

## 1. Bài toán

### 1.1 Bối cảnh

**Phân loại ảnh y tế nhị phân khi dữ liệu mất cân bằng lớp.** Lớp bệnh thường có ít ảnh hơn nhiều so với lớp bình
thường, nên classifier thiên về lớp đa số: accuracy cao nhưng **bỏ sót ca bệnh** (sensitivity thấp).

Một hướng xử lý là bù lớp thiểu số bằng **ảnh tổng hợp** do GAN sinh ra. Nhưng không phải ảnh sinh nào cũng có ích.
Ảnh lỗi, ảnh trông giống lớp đa số hoặc ảnh lặp lại nhau có thể làm classifier tệ đi. Vì vậy cần **chọn lọc** ảnh
sinh, và đó là đóng góp của dự án: **DASS**.

### 1.2 Phát biểu hình thức

```
Tập train thật   D = D_maj ∪ D_min,  n_maj = |D_maj| > n_min = |D_min|.   Lớp thiểu số = lớp dương (nhãn 1).
Bộ sinh          G(z, c): StyleGAN2-ADA có điều kiện theo lớp c, train trên D (cả hai lớp).
Ngân sách        n = n_maj − n_min                 số ảnh sinh cần thêm để tập train cân bằng 1 : 1.
Pool ứng viên    P = { G(z_j, min) : j = 1..N },  N = ⌈k · n⌉,  k = pool_mult.
Bài toán chọn    Tìm S ⊂ P, |S| = n, sao cho classifier f train trên D ∪ S (gán nhãn lớp thiểu số)
                 đạt hiệu năng cao nhất trên tập test thật T.
```

Không thể tối ưu trực tiếp hiệu năng trên T, vì T không được phép dùng trong lúc xây dựng mô hình. DASS thay mục tiêu
đó bằng một **tiêu chí đại diện** chỉ tính từ D. Theo tiêu chí này, ảnh sinh tốt là ảnh:

1. **gần lớp thiểu số thật**,
2. **xa lớp đa số thật**, tức nằm rõ về phía lớp bệnh,
3. **không trùng lặp** với những ảnh đã chọn.

Hai tiêu chí đầu được đo trong **hai không gian đặc trưng**: không gian thị giác chung (E_v, pretrain ImageNet) và
không gian nhận biết bệnh (E_d, học có giám sát trên ảnh thật).

### 1.3 Câu hỏi nghiên cứu và giả thuyết

Tám biến thể M0, M0b, M1–M6 ([mục 3.7](#37-các-biến-thể-so-sánh)) được thiết kế để trả lời các câu hỏi sau:

| | Câu hỏi | So sánh | Giả thuyết |
|---|---|---|---|
| RQ1 | Thêm ảnh sinh **không chọn lọc** có giúp classifier không? | M1 vs M0 | (mở, có thể giúp hoặc hại) |
| RQ2 | DASS có tốt hơn baseline chỉ dùng ảnh thật không? | M6 vs M0 | **H1 (chính): AUC(M6) > AUC(M0)** |
| RQ3 | Ảnh sinh có mang thêm thông tin so với **nhân bản ảnh thật** không? (cùng số ảnh, cùng số bước train, cùng cách cân bằng) | M6 vs M0b | **H2: AUC(M6) > AUC(M0b)** |
| RQ4 | Chọn lọc có tốt hơn chọn ngẫu nhiên không? | M6 vs M1 | **H3: AUC(M6) > AUC(M1)** |
| RQ5 | Mỗi thành phần đóng góp gì? | M2 / M3 vs M4 (một hay hai không gian); M4 vs M6 (thêm đa dạng); M5 vs M6 (thêm điểm lề) | H4: M6 ≥ M2, M3, M4, M5 |
| RQ6 | Kết quả có nhất quán giữa các backbone (CNN, Transformer) và giữa các bộ dữ liệu không? | 6 backbone × 2 bộ dữ liệu (ISIC `v11`, RSNA `rsna_v2`; cùng giao thức classifier) | H5: chiều của H1–H3 giữ ở đa số cấu hình |
| RQ7 | Ảnh được chọn khác pool như thế nào (chất lượng, khả năng tạo shortcut)? | KID, AUC thật-vs-sinh, Jaccard | (mô tả) |

**H1, H2, H3** được kiểm định thống kê trực tiếp bằng paired bootstrap ΔAUC: mọi biến thể so với M0, cộng hai cặp
M6 vs M0b và M6 vs M1 (`evaluation.comparisons`). Các so sánh thành phần (RQ5) được đọc từ bảng mean ± std và nên
trình bày là **phân tích thăm dò** ([mục 5.2](#52-phân-tích-thống-kê)).

---

## 2. Dữ liệu

### 2.1 Hai bộ dữ liệu

| | ISIC 2016 Part 3 | RSNA Pneumonia (chi tiết ở mục 2.5) |
|---|---|---|
| Loại ảnh | dermoscopy, RGB | X-quang ngực, DICOM xám 1 kênh |
| Lớp đa số / thiểu số | benign (0) / **malignant (1)** | negative (0) / **positive (1)** = "Lung Opacity" |
| Nguồn | thư mục ảnh `.jpg` + CSV nhãn; có tập test chính thức | DICOM + nhãn JSON MD.ai; không có tập test có nhãn |
| Số ảnh dùng | train 900 (727 / 173), test 379 (304 / 75) | 12.249 (10.738 / 1.511), 1 ảnh mỗi bệnh nhân |
| Tỉ lệ mất cân bằng | ≈ 4,2 : 1 | ≈ 7,1 : 1 |

### 2.2 Tiền xử lý (chạy một lần, áp dụng như nhau cho mọi tập)

| Bước | ISIC 2016 | RSNA Pneumonia |
|---|---|---|
| Đọc ảnh | RGB | DICOM → PNG xám 1 kênh (8 bit, giữ nguyên giá trị) |
| Cắt viền | cắt viền đen: pixel có trung bình kênh > 15 được coi là nội dung; chỉ cắt khi vùng nội dung chiếm **< 50 %** một cạnh | không cắt (tỉ lệ nội dung trung vị 1,00) |
| Đưa về 256 × 256 | kéo giãn (bilinear) | kéo giãn (ảnh gốc đã vuông 1024 × 1024) |
| Lưu | PNG | PNG |

- **Mọi ảnh lưu PNG** (nén không mất dữ liệu), để ảnh thật và ảnh sinh không khác nhau về kiểu nén.
- Tuỳ chọn chung `force_grayscale` (chuyển ảnh xám lưu RGB về luminance, 3 kênh bằng nhau, cho cả ảnh thật và ảnh sinh)
  vẫn có trong code nhưng hiện không bộ dữ liệu nào dùng.
- Khi đưa vào classifier, ảnh được resize về 224 × 224 rồi chuẩn hoá theo yêu cầu của từng backbone.

### 2.3 Cách chia

| | ISIC 2016 | RSNA Pneumonia |
|---|---|---|
| Kiểu chia | `holdout_val`: tách val 15 % từ train (phân tầng theo lớp) | `stratified`: test 15 %, sau đó val 17,6 % phần còn lại, tức **70 / 15 / 15** |
| Train (đa số + thiểu số) | 618 + 148 | 7.522 + 1.059 |
| Val | 109 + 25 | 1.606 + 226 |
| Test | 304 + 75 (chính thức) | 1.610 + 226 |

- Split **tất định** (seed toàn cục 2026), lưu thành JSON. Chạy lại với cách chia khác mà vẫn giữ `run_tag` thì lệnh
  **dừng**, vì kết quả cũ gắn với split cũ.
- GAN chỉ được thấy phần train của split nó đã train. ISIC: split tái lập đúng notebook v5 (test regression
  `tests/regression/test_split_reproduction.py`). RSNA: GAN `rsna` gắn với split `rsna_v2` qua `split.expected`, kiểm
  tra tự động ở mọi stage; thuật toán `stratified` cũng được khoá bằng test regression.
- **Val và test luôn 100 % ảnh thật.**

### 2.4 Ngân sách sinh và ý nghĩa của k

| | ISIC 2016 | RSNA Pneumonia |
|---|---|---|
| n = n_maj − n_min | 618 − 148 = **470** | 7.522 − 1.059 = **6.463** |
| k (`pool_mult`) | 1,5 | 1,5 |
| N = ⌈k · n⌉ (pool) | **705** | **9.695** |
| Tập train sau khi thêm ảnh sinh | 618 : (148 + 470) = **1 : 1** | 7.522 : (1.059 + 6.463) = **1 : 1** |
| Tỉ lệ ảnh sinh trong lớp thiểu số | 76 % | 86 % |

- **n quyết định tỉ lệ lớp cuối cùng** (luôn 1 : 1). **k chỉ quyết định độ chọn lọc:** DASS giữ 1/k pool.
  - k = 1: pool đúng bằng n, nên mọi biến thể M1–M6 chọn **cùng một tập** (không có gì để chọn).
  - k = 1,5: mỗi biến thể giữ 2/3 pool (loại 1/3). k lớn hơn thì chọn lọc gắt hơn, nhưng tập được chọn lệch xa phân
    phối ảnh thật hơn.
- **k = 1,5** (từ phiên bản 1.5.0), dùng chung cho mọi bộ dữ liệu. Lịch sử: 4 (ISIC `v7`) → 3 → 2 (ISIC `v9`) → 1,5.
  Lý do giảm dần: chọn càng gắt thì tập DASS càng lệch phân phối ảnh thật (ở một lần chạy với k = 3: KID của tập chọn
  0,100 so với 0,029 của toàn pool; độ đa dạng 0,168 so với 0,241). CosSIF (FAGT) cũng chỉ loại 15–25 % ảnh sinh (k ≈ 1,2–1,3).
  **Phải nêu trong bài báo** rằng k được đổi sau các lần chạy trước, kèm lý do. Không chạy ablation theo k.
- **Mốc tham chiếu khi đọc Jaccard:** hai tập con ngẫu nhiên độc lập, mỗi tập chiếm 1/k pool, có Jaccard kỳ vọng
  1/(2k − 1) = **0,5** khi k = 1,5.

### 2.5 Chi tiết RSNA Pneumonia (X-quang ngực)

Chọn vì có nguồn gốc rõ ràng, nhãn do bác sĩ gán và hội chẩn, hai lớp cùng nguồn chụp, đủ ảnh lớp thiểu số để train
GAN, và còn chỗ để cải thiện (không chạm trần).

| | RSNA Pneumonia Detection Challenge 2018 |
|---|---|
| Nguồn | RSNA + NIH ChestX-ray8; nhãn do bác sĩ X-quang gán (Shih et al., *Radiology: AI* 2019) |
| Ảnh | DICOM xám 1024 × 1024, **mỗi bệnh nhân một ảnh** |
| Nhãn | positive = có đám mờ phổi (Kaggle: `Target = 1`; bản MD.ai của trang RSNA: chú thích "Lung Opacity" — nhãn cuối cùng sau hội chẩn, file `…annotations-adjudicated-kaggle_2018.json`); negative = *Normal* + *No Lung Opacity / Not Normal* |
| Loại bỏ | 106 ảnh mang nhãn "Exclude" (`exclude_labels`): chụp nghiêng, ổ bụng, ảnh hỏng; 75 ảnh trong số đó không có nhãn lâm sàng nào |
| Bệnh nhân | mapping NIH (`…dataset-mappings_2018.json`) -> giữ **1 ảnh / bệnh nhân** (`one_per_patient`, chọn ngẫu nhiên theo seed), tránh rò rỉ giữa train / val / test |
| Quy mô | **toàn bộ** sau khi lọc (`subset_size: 0`): khoảng 12,2 nghìn ảnh, khoảng 12 % positive. Tỉ lệ dương thấp hơn 23,7 % của 30.000 ảnh vì người có đám mờ được chụp lại nhiều lần (2,7 ảnh / bệnh nhân dương so với 1,35 ở ảnh Normal) |
| Số kênh | **1** (PNG xám); GAN mới sinh thẳng ảnh 1 kênh; nhân bản 1 -> 3 kênh chỉ trên bộ nhớ |
| Chia | test của cuộc thi không có nhãn -> `stratified` 70 / 15 / 15 |
| Augmentation | classifier: **không** (`classifier.augment: false`, chung cho mọi bộ dữ liệu); E_d: profile `upright` — lật ngang, xoay ±10° (không lật dọc, không xoay 180°) |
| GAN | train mới (`gan_tag: rsna`), 1 kênh, cùng cấu hình và cùng cách dừng sớm theo KID; riêng **`mirror: false`** (X-quang ngực không đối xứng trái / phải: lật ngang làm tim nằm bên phải, chữ L / R bị ngược). GAN train trên split của `rsna_v2` -> `split.expected: checkpoints_rsna_v2/data/real_val_split.json`: mọi lần chạy dùng GAN này (run_tag / `--tag` mới) phải trùng đúng split đó |

- Bước `prepare` tự đọc dữ liệu (`data.source.type: dicom_csv`): tìm `.dcm` (tự giải nén nếu cần), tìm CSV nhãn, chuyển
  sang PNG và ghi `checkpoints_rsna_v2/data/dicom_metadata.csv` (tư thế chụp, giới tính, tuổi). Đổi thiết lập nguồn
  trong cùng runtime (ví dụ `subset_size`) thì ảnh cục bộ được đồng bộ lại; kết quả của run_tag cũ được bảo vệ bởi
  split đã lưu trên Drive (khác split -> báo lỗi, phải đổi run_tag).
- Các lần chạy: `rsna_v1` (tập con 6.000 ảnh, chưa loại "Exclude") chỉ chạy đến `prepare`; `rsna_v2` là cấu hình
  chính.
- **Nhiễu nhãn (nêu trong bài):** khoảng 11 % ảnh âm từng được một bác sĩ khoanh "Lung Opacity (… Prob)" và khoảng 12 %
  ảnh dương từng được một bác sĩ ghi "No Lung Opacity / Not Normal"; dùng nhãn cuối cùng sau hội chẩn.
- **Vị trí trên Drive:** dữ liệu gốc ở `ColabData/RSNA Pneumonia` (`source.train_images`, đường dẫn tuyệt đối); kết
  quả ở `ColabData/RSNA Pneumonia/Result_Pneumonia` (`paths.drive_root`). Khi tìm dữ liệu, các thư mục kết quả
  (`checkpoints_*`, `results_*`) được bỏ qua ở mọi cấp; nếu thư mục dữ liệu rỗng mà có đúng một bản `RSNA Pneumonia (1)`
  chứa dữ liệu thì tự đọc từ bản đó.
- **Shortcut cần kiểm tra:** bệnh nhân nặng thường chụp tư thế AP. Log của `prepare` in tỉ lệ AP / PA theo lớp; nên báo
  cáo trong bài. Ở `rsna_v1`: AP chiếm 58,5 % ảnh dương nhưng chỉ 22,7 % ảnh âm, tức đoán theo tư thế đã đạt AUC
  khoảng 0,68.
- **Xem nhãn theo ảnh:** `dass -c configs/experiments/rsna_pneumonia_dass.yaml label-stats` in, cho từng tên nhãn trong
  JSON MD.ai, số chú thích, số ảnh, % trên tổng số ảnh, số ảnh theo lớp cuối cùng, số bệnh nhân; `--focus Exclude Flag`
  in thêm các nhãn đi kèm; `--samples 4` vẽ 4 ảnh ngẫu nhiên mỗi nhãn (khung đỏ = vùng đám mờ, dưới ảnh: lớp · tư thế
  chụp) vào `results_rsna_v2/label_samples.png`; cột "bị loại" = số ảnh bị `exclude_labels` loại. Không đổi dữ liệu
  dùng để train.

---

## 3. Phương pháp

### 3.1 Toàn cảnh pipeline

```
 ảnh gốc ─► [1] prepare ─► ảnh 256×256 PNG + split train / val / test
                              │  chỉ phần TRAIN đi tiếp; val dùng để chọn epoch; test chỉ dùng ở [5] (một lần)
                              ▼
                        [2] gan ──► StyleGAN2-ADA có điều kiện, snapshot có KID lớp thiểu số thấp nhất
                              ▼
                        [3] sample ──► pool N = ⌈k·n⌉ ảnh sinh lớp thiểu số
                              ▼
                 ( [3b] fingerprint — tuỳ chọn: đo dấu vết tần số của ảnh sinh )
                              ▼
                        [4] select ──► E_v, E_d → điểm lề + đa dạng → mỗi biến thể M1…M6 chọn n ảnh
                              ▼
                        [5] train ──► 6 backbone × 8 biến thể × 3 seed (val chọn epoch, test dự đoán 1 lần)
                              ▼
                        [6] evaluate ──► metric phân loại, paired bootstrap ΔAUC, KID / FID
                              ▼
                        [7] report ──► bảng cho bài báo (.csv / .json / .tex)
```

Mỗi bước là một lệnh `dass …` chạy trong tiến trình riêng, lưu artefact lên Google Drive và **chạy lại được** (phần đã
xong được khôi phục hoặc bỏ qua).

### 3.2 Mô hình sinh: StyleGAN2-ADA có điều kiện

- **Mô hình:** StyleGAN2-ADA bản chính thức của NVlabs (PyTorch), cấu hình `paper256`, có điều kiện theo nhãn lớp.
  Code không được chép vào repo: pipeline tự clone repo NVlabs rồi áp 9 bản vá để chạy được trên PyTorch 2.x và
  Python 3.12. Các bản vá chỉ sửa tương thích, không đổi thuật toán.
- **Dữ liệu train:** toàn bộ ảnh **train của cả hai lớp**, kèm nhãn. Lớp đa số giúp GAN học đặc trưng chung (giải
  phẫu, độ tương phản), còn nhãn điều khiển phần khác biệt giữa hai lớp.
- **Siêu tham số:** batch 16, R1 γ = 1, lật ngang (`mirror`), ADA target 0,6. ADA (Adaptive Discriminator
  Augmentation) tự điều chỉnh mức augmentation của discriminator để chống overfit khi ít ảnh.
- **Early stopping theo KID của lớp thiểu số:**
  - Mỗi 100 kimg lưu một snapshot, sinh 1.000 ảnh lớp thiểu số (seed 123) và tính KID so với ảnh thật lớp thiểu số
    của tập train (Inception-v3, 50 tập con).
  - Snapshot được tính là **cải thiện** khi KID < 0,98 × KID tốt nhất trước đó (giảm ít nhất 2 %).
  - Sau 400 kimg, nếu 5 snapshot liên tiếp không cải thiện thì dừng. Tối đa 3.000 kimg.
  - Bước sau dùng **snapshot có KID thấp nhất** (`best.pkl`), không dùng snapshot cuối.
  - **Không** tính KID dừng sớm trên ảnh val: val khi đó vừa chọn GAN vừa chọn epoch classifier (phá quy tắc "val chỉ
    để chọn epoch"), và lớp thiểu số của val còn nhỏ hơn train nên KID nhiễu hơn
    (`tests/regression/test_gan_kid_reference.py`).
- **Hiện trạng:** ISIC **dùng lại GAN đã train** (`checkpoints_v5`, KID tốt nhất 0,02007 tại 1.800 kimg, dừng sớm ở
  2.300 kimg); khi đó lệnh `gan` chỉ báo cáo (đường KID, ảnh mẫu) và không train lại. RSNA train GAN mới (`checkpoints_rsna`, 1 kênh, `mirror: false`): KID tốt nhất 0,0145
  tại 800 kimg, dừng sớm ở 1.300 kimg (khoảng 5,5 giờ trên A100); xác suất ADA chỉ 0–0,03 (đủ dữ liệu, gần như không
  cần tăng cường).

### 3.3 Candidate pool

- Sinh N = ⌈k · n⌉ ảnh lớp thiểu số từ `best.pkl` với `truncation ψ = 1` (không cắt bớt độ đa dạng). Seed cố định
  777, nên pool tái lập được. Pool được nén zip lên Drive.
- Ảnh sinh lưu đúng số kênh của bộ dữ liệu: ISIC 3 kênh; RSNA 1 kênh (generator 1 kênh, pool có hậu tố `_c1`).
- **Mọi biến thể chọn từ cùng một pool**, nên khác biệt giữa các biến thể chỉ đến từ tiêu chí chọn.

### 3.4 Hai không gian đặc trưng

| | E_v (thị giác) | E_d (nhận biết bệnh) |
|---|---|---|
| Kiến trúc | EfficientNet-B0 (pooling trung bình, 1.280 chiều) | DenseNet-121 (pooling trung bình, 1.024 chiều) |
| Trọng số | ImageNet, **đóng băng** | ImageNet, rồi **train có giám sát** |
| Dữ liệu train | — | ảnh **thật** của tập train, nhãn thật, có class weight; val chỉ dùng để chọn epoch |
| Lịch train | — | 5 epoch head (lr 1e-3), rồi 20 epoch fine-tune (lr 1e-5); seed 4242 |
| Vai trò | ảnh có "trông giống" lớp thiểu số về mặt thị giác không | ảnh có mang đặc trưng phân biệt bệnh không |

- Mọi vector nhúng được chuẩn hoá L2, nên tích vô hướng chính là cosine.
- E_d là một mô hình **riêng** (seed riêng), cố ý khác baseline M0. Nếu chọn ảnh bằng chính M0 rồi so với M0, việc chọn
  chỉ củng cố ranh giới của M0.
- E_d **không thấy test, không thấy ảnh sinh**. E_d có dùng val để chọn epoch, giống như mọi classifier.
- E_v (EfficientNet-B0) và backbone classifier EfficientNetV2B0 là **hai mô hình khác nhau**.

### 3.5 Điểm lề

Với mỗi ứng viên x và mỗi không gian (v hoặc d):

```
S⁺(x) = (1/K) · Σ cos(z_x, z_i)    trên K ảnh THẬT lớp thiểu số gần x nhất (tập train)
S⁻(x) = (1/K) · Σ cos(z_x, z_j)    trên K ảnh THẬT lớp đa số gần x nhất (tập train)
M(x)  = S⁺(x) − λ · S⁻(x)          K = 5,  λ_v = λ_d = 1
```

- M cao nghĩa là x gần lớp thiểu số **hơn** lớp đa số.
- Dùng trung bình top-K thay vì max, vì max chỉ phụ thuộc vào một ảnh thật nên rất nhạy với ngoại lai.
- Log in **tương quan M_v–M_d**. Tương quan gần 1 cho thấy hai không gian đo cùng một thứ, khi đó E_d không thêm thông
  tin.

### 3.6 Điểm DASS và thuật toán chọn tham lam

```
Chuẩn hoá:   M̃_v, M̃_d = min-max của M_v, M_d trên TOÀN pool (về [0, 1])
Điểm nền:    b(x) = α · M̃_v(x) + β · M̃_d(x)                         α = β = 1
Đa dạng:     S_div(x | S) = 1 − max_{y ∈ S} cos(z_x^v, z_y^v)        khoảng cách tới ảnh gần nhất đã chọn, trong E_v

S ← { argmax_x b(x) }                                              ảnh đầu tiên: điểm nền cao nhất
lặp đến khi |S| = n:
    với mọi x ∈ P \ S:   score(x) = b(x) + γ · S̃_div(x | S)        S̃_div: min-max trên P \ S, tính lại MỖI vòng
    S ← S ∪ { argmax score }                                        γ = 0,5
```

- γ cân bằng giữa **chất lượng** (điểm nền) và **độ phủ** (đa dạng). Khi γ = 0, thuật toán thành top-n theo b(x)
  (chính là M4).
- S_div được chuẩn hoá lại mỗi vòng vì giá trị của nó giảm dần khi S lớn lên. Nếu không chuẩn hoá lại, ảnh hưởng của
  đa dạng sẽ yếu dần.
- Độ phức tạp O(n · N) phép tích vô hướng. Mỗi vòng chỉ cập nhật khoảng cách tới ảnh vừa được chọn.

### 3.7 Các biến thể so sánh

M1–M6 chọn **đúng n ảnh sinh từ cùng một pool**. M0 và M0b không dùng ảnh sinh:

| ID | Tên trong bài báo | Tiêu chí chọn | Đa dạng | Biến thể này cô lập yếu tố gì |
|---|---|---|---|---|
| M0 | Imbalanced Baseline | không dùng ảnh sinh; dữ liệu thật mất cân bằng (ISIC 4,2 : 1, RSNA 7,1 : 1), **không cân bằng** (không class weight, không lấy mẫu lại) | – | **baseline** |
| M0b | Random Oversampling (ROS) | không dùng ảnh sinh: **nhân bản ảnh thật** lớp thiểu số lên 1 : 1 | – | **baseline oversampling**: cùng số ảnh, cùng số bước train, cùng cách cân bằng với M1–M6 |
| M1 | Unfiltered GAN (Random Selection) | ngẫu nhiên (seed 2026) | – | tác dụng của ảnh sinh khi không chọn lọc |
| M2 | Visual-only Filter (M_v) | M_v | – | chỉ không gian thị giác |
| M3 | Disease-only Filter (M_d) | M_d | – | chỉ không gian bệnh |
| M5 | Diversity-only Filter (S_div) | 0 (k-center greedy, γ = 1) | ✓ | chỉ đa dạng, không có điểm lề |
| M4 | Dual-Margin Filter (M_v + M_d) | α·M̃_v + β·M̃_d | – | kết hợp hai không gian, chưa có đa dạng |
| **M6** | **DASS (Ours)** | **α·M̃_v + β·M̃_d** | **γ = 0,5** | **DASS đầy đủ (phương pháp đề xuất)** |

**Cách M0b nhân bản ảnh** (`data.variants.oversample_indices`, seed 2026): cần thêm n bản sao từ n_min ảnh thật, nên
mỗi ảnh được lặp q = ⌊n / n_min⌋ lần, và r = n mod n_min ảnh (chọn ngẫu nhiên, không lặp) được lặp thêm một lần. Số
lần xuất hiện của các ảnh vì thế chênh nhau tối đa 1. Ví dụ ISIC: 470 = 148 × 3 + 26, nên mỗi ảnh malignant thật xuất
hiện 4 hoặc 5 lần trong tập train; RSNA: 6.463 = 1.059 × 6 + 109, nên mỗi ảnh dương thật xuất hiện 7 hoặc 8 lần (RSNA
không augmentation, nên đây là các bản sao y hệt). M0b không chọn gì từ pool
nên không nằm trong `selections.json`; tập train của M0b được ghép ở bước `train` (bật bằng
`selection.oversample_variant`, mặc định bật). Lý do có M0b: theo Buda et al. (2018), oversampling tới 1 : 1 là cách
xử lý mất cân bằng đơn giản mạnh nhất với CNN, nên **M6 vs M0b** là phép so sánh công bằng nhất cho đóng góp của
**nội dung** ảnh sinh.

M7 (thêm ảnh sinh vào cả lớp đa số) là tuỳ chọn `selection.both_classes_variant`, **mặc định tắt** và không nằm trong
thực nghiệm chính.

### 3.8 Classifier downstream

**Tập train của mỗi biến thể:**
- Lớp đa số: giữ nguyên ảnh thật.
- Lớp thiểu số:
  - M0: chỉ ảnh thật;
  - M0b: ảnh thật cộng n bản sao ảnh thật;
  - M1–M6: ảnh thật cộng n ảnh sinh đã chọn.
- Val và test: giữ nguyên, 100 % ảnh thật.

**Cân bằng lớp:**

| | Số ảnh train: ISIC (`v11`) | RSNA (`rsna_v2`) | Bước mỗi epoch (ISIC / RSNA) | Cân bằng bằng |
|---|---|---|---|---|
| M0 Imbalanced Baseline | 618 : 148 | 7.522 : 1.059 | 48 / 537 | **không cân bằng** |
| M0b (ROS) | 618 : (148 + 470 bản sao) | 7.522 : (1.059 + 6.463 bản sao) | 78 / 941 | dữ liệu (ảnh thật nhắc lại) |
| M1–M6 | 618 : (148 + 470 ảnh sinh) | 7.522 : (1.059 + 6.463 ảnh sinh) | 78 / 941 | dữ liệu (ảnh sinh) |

- **Không can thiệp dữ liệu ở classifier, cho cả hai bộ dữ liệu** (người dùng quyết định, 1.11.0): không augmentation
  (`classifier.augment: false`) và M0 **không class weight** (`classifier.baseline_class_weight: false`), đặt trong
  `configs/_base_/classifier.yaml`, không bộ dữ liệu nào ghi đè. Khác biệt giữa các phương pháp vì thế chỉ nằm ở
  **nội dung tập train**.
- M0 là baseline mất cân bằng thật (ISIC 4,2 : 1, RSNA 7,1 : 1); ROS là baseline cân bằng bằng ảnh thật. Ở ngưỡng 0,5,
  M0 sẽ ít khi đoán dương nên sensitivity thấp; so sánh chính dựa vào AUC (không phụ thuộc ngưỡng).
- M0b và M1–M6 giống hệt nhau về số ảnh, số bước train và cách cân bằng; chỉ khác **nội dung** ảnh thêm vào. Vì vậy
  **M6 vs M0b** là phép so sánh chặt chẽ nhất.
- Cấu hình cũ có augmentation + M0 class weight (ISIC `v10`, như `v7`) chạy lại được bằng `--set classifier.augment=true
  --set classifier.baseline_class_weight=true` kèm `--tag` riêng; khi đó bảng tự đổi tên M0 thành "Imbalanced
  Baseline (class-weighted)".
- Mỗi `.npz` ghi giao thức train (augment, class_weight). Gặp `.npz` của giao thức khác, `train` dừng báo lỗi;
  `train --archive-mismatched` chuyển nó (và trọng số) sang `predictions_superseded/<giao thức>/` (không xoá) rồi train
  lại, nên thư mục `predictions/` luôn chỉ có một giao thức. Ở `rsna_v2`, phần EfficientNetV2B0 lỡ train có
  augmentation (trước bản 1.8.2) và M0 của ResNet50 lỡ train có class weight (trước bản 1.9.1) nằm trong thư mục này.

**Backbone (pretrain ImageNet):**

| Backbone | Họ | Nguồn | Đầu ra |
|---|---|---|---|
| EfficientNetV2B0, ResNet50, DenseNet121, ConvNeXtTiny | CNN | `keras.applications` | pooling trung bình → Dropout 0,3 → Dense(1, sigmoid) |
| ViT-B16, SwinT | Transformer | KerasHub (preset Hugging Face) | đầu phân loại của preset, 1 đầu ra sigmoid |

**Augmentation** của classifier: **tắt** ở cấu hình hiện tại (cả hai bộ dữ liệu). Khi bật (`classifier.augment: true`,
như ISIC `v10`): chỉ tập train, giống hệt nhau cho mọi biến thể, kể cả M0 và M0b; profile `rotation_invariant` = lật
ngang và dọc, xoay tới ±180° (biên phản chiếu), zoom ±10 %, độ sáng ±10 %, tương phản ±10 %, áp dụng trực tuyến (không
làm tăng số ảnh, không đổi tỉ lệ lớp). Augmentation vẫn có ở hai chỗ, độc lập với tuỳ chọn này:
- **ADA của StyleGAN2-ADA** là augmentation cho discriminator khi train GAN;
- **E_d** (encoder bệnh của DASS) luôn train với augmentation và class weight (RSNA dùng profile `upright`: lật ngang,
  xoay ±10°).

Tiền xử lý dữ liệu (cắt viền đen ở ISIC) là làm sạch dữ liệu, áp dụng như nhau cho mọi ảnh, không phải augmentation.

**Huấn luyện 2 giai đoạn** (loss binary cross-entropy, ảnh 224 × 224, batch 16):
1. Đóng băng backbone, train lớp đầu ra **5 epoch** (AdamW, lr 1e-3).
2. Mở toàn bộ, fine-tune **tối đa 30 epoch** (AdamW, lr 1e-5, weight decay 1e-4). Với CNN, BatchNorm luôn ở chế độ
   inference.

**Chọn epoch và dự đoán:**
- Theo dõi `val_macro_recall` = (sensitivity + specificity) / 2 tại ngưỡng 0,5. Lưu checkpoint tốt nhất; dừng sớm sau
  8 epoch không cải thiện.
- Nạp lại checkpoint tốt nhất, rồi **dự đoán test đúng một lần**. Xác suất val và test được lưu vào
  `predictions/<model>__<biến thể>__s<seed>.npz`. Mọi bảng phân loại đều tính từ các file này.

---

## 4. Thiết kế thực nghiệm

### 4.1 Biến số

| Loại | Biến số |
|---|---|
| **Biến độc lập (yếu tố nghiên cứu)** | phương pháp cân bằng / chọn ảnh: M0, M0b, M1–M6 |
| **Yếu tố khối (để kiểm tra tính tổng quát)** | backbone (6), bộ dữ liệu (2) |
| **Lặp lại** | seed classifier: 2026, 2027, 2028 (khởi tạo đầu ra, thứ tự trộn dữ liệu) |
| **Biến phụ thuộc** | metric phân loại trên test ([mục 5.1](#51-metric-phân-loại)); KID và các chỉ số chẩn đoán của tập ảnh được chọn |
| **Biến được kiểm soát** | cùng split, cùng GAN, cùng pool, cùng n, cùng siêu tham số train, **không augmentation và không class weight cho mọi biến thể**, cùng seed cho mọi phương pháp (thiết kế **ghép cặp**), cùng ngưỡng 0,5, cùng tập val / test |

### 4.2 Ma trận thực nghiệm

| | Mỗi bộ dữ liệu | Hai bộ dữ liệu (ISIC + RSNA) |
|---|---|---|
| Lần train classifier | 8 biến thể × 6 backbone × 3 seed = **144** | **288** |
| Lần train E_d | 1 | 2 |
| GAN | ISIC dùng lại `v5`; RSNA train một lần (`rsna`) | 1 |

### 4.3 Siêu tham số được cố định trước

- Tham số DASS (K = 5, λ = 1, α = β = 1, γ = 0,5) lấy theo công thức của notebook ISIC v5. Lịch train E_d và mọi siêu
  tham số classifier được **dùng chung cho mọi bộ dữ liệu và mọi backbone**.
- k = 1,5 (xem [mục 2.4](#24-ngân-sách-sinh-và-ý-nghĩa-của-k)); classifier không augmentation, M0 không class weight
  (1.11.0).
- **Lưu ý:** k, augmentation và class weight đã được đổi qua nhiều phiên bản **sau khi xem kết quả test** (bảng ở đầu
  tài liệu). Đây không còn là thiết lập được chốt trước; phải báo cáo minh bạch mọi cấu hình đã chạy.
- **Không có tham số nào được dò trên test.** Val chỉ dùng để chọn epoch.
- Không dò siêu tham số riêng cho từng backbone. Cách này công bằng giữa các phương pháp, nhưng có thể chưa tối ưu cho
  từng backbone (xem [mục 7](#7-hạn-chế-đã-biết)).

### 4.4 Seed và khả năng tái lập

| Seed | Dùng cho |
|---|---|
| 2026 (`seed` toàn cục) | chia dữ liệu, tập ngẫu nhiên của M1, ảnh được nhân bản thêm của M0b, bootstrap, cross-validation của probe / shortcut, tập con KID và cặp SSIM khi đánh giá |
| 777 (`generator.gen_seed`) | sinh pool lớp thiểu số (778 dùng cho pool lớp đa số của M7) |
| 123 (`generator.kid_seed`) | 1.000 ảnh để tính KID mỗi snapshot GAN |
| 4242 (`encoder.e_d_seed`) | train E_d |
| 2026 / 2027 / 2028 (`classifier.seeds`) | train classifier |

- `deterministic: false`: một số phép toán trên GPU không tất định, nên chạy lại có thể cho số hơi khác. **Nguồn sự
  thật là các file `.npz` dự đoán đã lưu**, không phải một lần chạy lại.
- `results_<run_tag>/run_manifest.json` lưu cấu hình đã hợp nhất, phiên bản `dass`, phiên bản thư viện và lệnh của
  từng bước. Ô lấy mã nguồn trong notebook in ra **commit hash** của code; ghi lại hash này cho bài báo.

---

## 5. Đánh giá và phân tích thống kê

### 5.1 Metric phân loại

Tính trên **test**, lớp dương = lớp thiểu số, **ngưỡng cố định 0,5** (TP, FN, TN, FP theo lớp dương):

| Metric | Công thức / định nghĩa | Ghi chú |
|---|---|---|
| **ROC-AUC** | diện tích dưới đường ROC | **chỉ số chính, được kiểm định**; không phụ thuộc ngưỡng |
| PR-AUC | average precision | nhạy với lớp thiểu số hơn ROC-AUC |
| Sensitivity (recall) | TP / (TP + FN) | tỉ lệ ca bệnh được phát hiện |
| Specificity | TN / (TN + FP) | |
| Balanced accuracy | (Sens + Spec) / 2 | cũng là đại lượng dùng để chọn epoch trên val |
| G-mean | √(Sens · Spec) | |
| MCC | hệ số tương quan Matthews | dùng cả bốn ô của ma trận nhầm lẫn |
| F1, macro-F1, precision, accuracy | | accuracy chỉ để tham khảo, vì gây hiểu lầm khi lớp mất cân bằng |

Độ phân giải của sensitivity trên test: ISIC có 75 ca dương, nên mỗi ca tương ứng ≈ 1,3 điểm %; RSNA có 226 ca
dương, mỗi ca ≈ 0,44 điểm %. Chênh lệch nhỏ hơn mức này chỉ là một ca bệnh.

### 5.2 Phân tích thống kê

- **Tổng hợp qua seed:** mỗi cặp (backbone, phương pháp) được báo cáo là **mean ± std** (độ lệch chuẩn mẫu, ddof = 1)
  qua 3 seed. Độ biến thiên này phản ánh **ngẫu nhiên của quá trình train**.
- **Paired bootstrap ΔAUC** (bảng `significance`, cột `vs` ghi đối chứng):
  - **mọi biến thể so với M0** (`evaluation.baseline_method`);
  - cộng hai cặp bổ sung (`evaluation.comparisons`): **M6 vs M0b** (H2) và **M6 vs M1** (H3).

  Cách tính cho mỗi cặp:
  1. Lấy trung bình xác suất test qua các seed chung, tức **ensemble theo seed** của từng phương pháp.
  2. Lấy mẫu lại tập test **có hoàn lại, phân tầng theo lớp** (ca dương và ca âm được lấy riêng), 2.000 lần. Hai
     phương pháp dùng **cùng một mẫu** ở mỗi lần (ghép cặp).
  3. ΔAUC = AUC(phương pháp) − AUC(đối chứng). Khoảng tin cậy 95 % theo phân vị 2,5–97,5. p hai phía =
     2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0)).
  4. **ΔAUC theo từng seed** (cột `ΔAUC per seed`): AUC(phương pháp, seed s) − AUC(đối chứng, seed s), ghép cặp theo
     seed, báo cáo mean ± std — khớp với chênh lệch của bảng `classification` và phản ánh dao động giữa các lần train.
- **Cách đọc và lưu ý:**
  - ΔAUC (cột chính) trong bảng `significance` là AUC của **ensemble**, nên có thể khác hiệu của các AUC trung bình
    trong bảng `classification`; cột `ΔAUC per seed` là con số tương ứng với bảng đó.
  - Bootstrap đo **độ bất định do mẫu test**, với điều kiện mô hình đã train cố định. Độ bất định do train được phản
    ánh qua std giữa các seed (và cột `ΔAUC per seed`). Báo cáo cả hai.
  - Với 2.000 lần lấy mẫu, p nhỏ nhất khác 0 là 0,001. p = 0 nên ghi là **p < 0,001**.
  - **Đa so sánh:** mỗi backbone có 7 phép so sánh với M0 và 2 cặp bổ sung. Cột **`p (Holm)`** hiệu chỉnh
    Holm–Bonferroni trong cả họ 9 phép của từng backbone (kiểm soát xác suất có ít nhất một kết luận sai). Kết luận
    chính (H1–H3: M6 vs M0, M6 vs M0b, M6 vs M1) nên dựa vào `p (Holm)`; nêu trong bài rằng hiệu chỉnh làm theo từng
    backbone, không gộp 6 backbone.
  - So sánh M6 với M2–M5 (RQ5) **chưa có kiểm định chính thức** (chỉ đọc từ mean ± std). Có thể thêm bằng cách bổ
    sung cặp vào `evaluation.comparisons`, không cần train lại.

### 5.3 Chất lượng ảnh sinh (Inception-v3 của StyleGAN2-ADA, 2.048 chiều)

So với **ảnh thật lớp thiểu số của tập train** (ISIC 148 ảnh, RSNA 1.059 ảnh), tính cho từng tập ảnh được chọn:

| Chỉ số | Định nghĩa | Tốt khi |
|---|---|---|
| **KID** (chỉ số chính) | MMD² không chệch với kernel đa thức k(x, y) = (xᵀy / 2048 + 1)³; 50 tập con, mỗi tập m = min(n_thật, n_sinh, 1.000) ảnh; báo cáo mean ± std | thấp |
| FID (tham khảo) | khoảng cách Fréchet giữa hai phân phối Gauss | thấp; **chệch mạnh khi số ảnh < 2.048** nên chỉ tham khảo |
| Đa dạng | 1 − cosine trung bình giữa các cặp ảnh trong tập | cao |
| SSIM nội bộ | SSIM trung bình của 200 cặp ảnh ngẫu nhiên (cửa sổ Gauss σ = 1,5) | thấp (ít trùng lặp) |
| AUC thật-vs-sinh | hồi quy logistic (chuẩn hoá, C = 0,1, cân bằng lớp), CV phân tầng 5 fold, trên E_v: ảnh thật lớp thiểu số vs tập được chọn | gần 0,5 (khó phân biệt) |

Có hai hàng tham chiếu:
- **`real val vs real train`:** mức nền của metric giữa hai tập ảnh **thật**. Val nhỏ (25 / 59 ảnh), nên m nhỏ và std
  của hàng này lớn.
- **`all candidates`:** toàn bộ pool chưa lọc. So một tập được chọn với hàng này để thấy việc chọn lọc làm KID tăng
  hay giảm.

### 5.4 Chẩn đoán đi kèm

| Chẩn đoán | Đo gì | Bước |
|---|---|---|
| AUC probe | hai lớp bệnh tách nhau đến đâu trong E_v và E_d (logistic, CV 5 fold). Dòng **val** gần khách quan (E_d không học trên val, chỉ dùng val để chọn epoch); dòng train thì lạc quan | `select` |
| Kiểm tra shortcut | AUC tách ảnh thật / ảnh sinh trong E_v cho từng biến thể; hàng tham chiếu là AUC tách hai lớp bệnh thật | `select` |
| Jaccard giữa các biến thể | các tiêu chí có thật sự chọn ra các tập khác nhau không (mốc ngẫu nhiên 0,5 khi k = 1,5) | `select` |
| Hình ảnh | phân tán M_v–M_d, ảnh được chọn và bị loại, ảnh sinh cạnh ảnh thật gần nhất (kiểm tra GAN có "học thuộc" không) | `select` |
| Dấu vân tay tần số (Frank et al., ICML 2020) | ảnh xám → DCT 2 chiều → log → chuẩn hoá → CNN nông và hồi quy; ≤ 400 ảnh mỗi phía, 15 epoch. AUC ≈ 1 nghĩa là ảnh sinh mang dấu vết GAN. **Chỉ đo, không sửa ảnh** | `fingerprint` (tuỳ chọn) |

### 5.5 Phân tích theo nhóm con (RSNA: tư thế chụp AP / PA)

Bật bằng `evaluation.subgroup_columns` (RSNA: `[ViewPosition]`, lấy từ `dicom_metadata.csv`; ISIC không có). **Đăng ký trước** (2026-10-04, trước khi có kết quả test RSNA): báo cáo dù kết quả thế nào. Chạy trong
`evaluate`, không cần GPU, không đổi cách train.

| Bảng | Đo gì | Cách đọc |
|---|---|---|
| `classification_by_subgroup` | AUC, PR-AUC (mean ± std qua seed) trong từng nhóm AP, PA | Trong một nhóm, mọi ảnh cùng tư thế nên tư thế không giúp xếp hạng: đo khả năng nhận ra bệnh tách khỏi shortcut |
| `significance_by_subgroup` | ΔAUC vs M0 và các cặp bổ sung **trong từng nhóm**, p Holm trong mỗi model × nhóm | DASS hơn đối chứng ở **cả hai** nhóm -> lợi thế đến từ bệnh; chỉ hơn ở AUC toàn bộ -> nghi shortcut |
| `subgroup_reference` | AUC khi dự đoán nhãn **chỉ bằng tư thế**, tỉ lệ AP trong lớp dương / âm của test | Mức "ăn điểm nhờ shortcut" (RSNA `rsna_v1`: khoảng 0,68) |
| `subgroup_share` | Tỉ lệ ảnh "trông giống AP" trong ảnh dương thật, toàn pool và ảnh mỗi phương pháp chọn (probe logistic trên E_v của ảnh thật train, không class weight; ảnh thật dùng dự đoán ngoài fold) | Pool cao hơn ảnh dương thật -> GAN khuếch đại tư thế; ảnh DASS chọn cao hơn pool -> cách chọn khuếch đại |

Tên ảnh test lưu trong `.npz` (`test_files`, từ 1.9.0). `.npz` cũ (ví dụ ResNet50 của `rsna_v2` train trước 1.9.0)
được dựng lại thứ tự từ split (lớp theo cấu hình, tên file tăng dần — đúng thứ tự `image_dataset_from_directory`
với shuffle = False), chỉ khi nhãn khớp đúng thứ tự.

---

## 6. Tính hợp lệ: rò rỉ, shortcut và các mối đe doạ

### 6.1 Biện pháp chống rò rỉ

| Nguyên tắc | Cách bảo đảm |
|---|---|
| Không rò rỉ test | GAN, DASS và E_d chỉ dùng ảnh train; test chỉ được đọc ở bước train, sau khi đã chọn epoch |
| Val chỉ để chọn epoch | val không dùng để chọn ngưỡng, chọn k hay chọn phương pháp; val và test 100 % ảnh thật (assert trước khi train) |
| Không dò ngưỡng | ngưỡng cố định 0,5 |
| Test chỉ dự đoán một lần | dự đoán sau khi nạp checkpoint tốt nhất; không có vòng lặp nào dùng kết quả test |
| GAN dùng lại hợp lệ | split phải trùng split mà GAN đã train (RSNA: `split.expected`, kiểm tra tự động ở mọi stage; ISIC: test regression) |
| Không có shortcut định dạng | mọi ảnh lưu PNG, đúng số kênh gốc cho cả ảnh thật và ảnh sinh (ISIC 3 kênh, RSNA 1 kênh) |
| Không rò rỉ bệnh nhân | RSNA: 1 ảnh mỗi bệnh nhân (mapping NIH), nên chia theo ảnh cũng là chia theo bệnh nhân |
| So sánh công bằng | cùng pool, cùng n, cùng seed, cùng quy trình train, không augmentation / class weight cho mọi biến thể; M0b có cùng số ảnh, số bước train và cách cân bằng với M1–M6 |

### 6.2 Các mối đe doạ đến tính hợp lệ

| Loại | Mối đe doạ | Cách xử lý hoặc đo |
|---|---|---|
| Nội tại | Ảnh sinh **chỉ** nằm ở lớp thiểu số, nên đặc điểm "trông như ảnh GAN" tương quan với nhãn trong tập train | Test toàn ảnh thật nên điều này không thổi phồng kết quả test, nhưng có thể làm giảm lợi ích. Đo bằng AUC thật-vs-sinh và fingerprint (RSNA `rsna_v2`: khoảng 0,999, ảnh sinh chiếm khoảng 86 % lớp dương). **M7** (tuỳ chọn, chưa bật) là đối chứng trực tiếp: thêm ảnh sinh vào cả lớp đa số. Chạy như một lần chạy riêng, không đụng `rsna_v2`: `dass … --set selection.both_classes_variant=true --set encoder.e_d_from_run=base --tag m7 run --from sample --models EfficientNetV2B0` (dùng lại E_d của `rsna_v2`; cần GPU để sinh pool lớp đa số; run này train lại mọi biến thể cho backbone đã chọn, nên giới hạn bằng `--models`) |
| Nội tại | **Shortcut tư thế chụp (RSNA):** AP gắn với bệnh nặng (dương ≈ 58 % AP, âm ≈ 23 % AP) | Đo, không sửa dữ liệu: AUC trong từng nhóm AP / PA, mốc chỉ dùng tư thế, tỉ lệ "trông giống AP" của ảnh sinh (mục 5.5) |
| Nội tại | **Classifier không can thiệp dữ liệu** (không augmentation, M0 không class weight; người dùng quyết định, cả hai bộ dữ liệu): công bằng về quy trình nhưng không trung lập về tác động — ROS lặp y hệt mỗi ảnh dương (ISIC 4–5 lần, RSNA 7–8 lần) nên dễ học thuộc hơn, trong khi biến thể GAN có ảnh mới | Nêu trong bài. Kết quả ISIC `v10` (cùng k, có augmentation + class weight) là phân tích độ nhạy sẵn có. RSNA (tuỳ chọn): một backbone có augmentation trong lần chạy riêng, `dass … --set classifier.augment=true --set encoder.e_d_from_run=base --tag aug run --from sample --models EfficientNetV2B0` (`split.expected` bảo đảm cùng split; pool sinh lại từ cùng GAN, cùng seed) |
| Nội tại | M0 có ít bước train hơn (ISIC 48 so với 78, RSNA 537 so với 941 bước mỗi epoch) và không cân bằng, nên M6 vs M0 lẫn tác dụng của số bước / cách cân bằng | **M0b (ROS)** có cùng số ảnh, số bước và cách cân bằng với M6; M6 vs M0b tách riêng tác dụng của nội dung ảnh sinh |
| Nội tại | Cấu hình (k, augmentation, class weight) được đổi sau khi xem kết quả test; Brain Tumor bị loại sau khi thấy AUC chạm trần | Báo cáo mọi cấu hình đã chạy (ISIC `v7`, `v9`, `v10`, `v11`; RSNA `rsna_v2`) và nêu lý do loại Brain Tumor; so `v10` với `v11` (cùng k, chỉ khác augmentation + class weight của M0) cho thấy lợi thế của ảnh sinh phụ thuộc giao thức classifier đến mức nào |
| Nội tại | M1 chỉ có **một** lần rút ngẫu nhiên (seed 2026); M0b cũng chỉ một lần chọn ảnh nhân bản thêm | Độ biến thiên do việc rút ngẫu nhiên không nằm trong std; nêu rõ khi so M6 với M1 / M0b |
| Nội tại | Backbone DenseNet121 có **cùng kiến trúc** với E_d (khác seed, khác lần train) | Diễn giải riêng hàng DenseNet121; so sánh với 5 backbone còn lại |
| Nội tại | KID dùng trong early stopping của GAN so với chính ảnh train mà GAN đã thấy (thưởng cả việc học thuộc); KID của snapshot tốt nhất vì thế hơi lạc quan | Kiểm tra ảnh sinh cạnh ảnh thật gần nhất (học thuộc); so với hàng tham chiếu "real val vs real train" |
| Nội tại | KID dao động lớn khi ít ảnh thật (ISIC 148 ảnh): ngưỡng cải thiện 2 % có thể nằm trong nhiễu, snapshot "tốt nhất" có thể do may mắn | Patience 5 + `min_kimg` 400; xem `kid_history.png` — trên đoạn phẳng các snapshot tương đương, không diễn giải số kimg tốt nhất |
| Cấu trúc | Inception-v3 train trên ImageNet, xa miền ảnh y tế (nhất là X-quang): KID thấp ≠ giống về dấu hiệu bệnh | Nêu trong phần hạn chế; thước đo cuối cùng là kết quả classifier (M1 vs M0, M6 vs M0b) |
| Nội tại | GPU không tất định | Báo cáo từ file `.npz` đã lưu; 3 seed |
| Cấu trúc | Metric tại ngưỡng 0,5 phụ thuộc tỉ lệ lớp lúc train: M0 (mất cân bằng) có xác suất lệch về lớp đa số, nên ở ngưỡng 0,5 sensitivity thấp là do thiết kế, không phải do mô hình kém hơn về khả năng xếp hạng | Chỉ số chính là ROC-AUC (không phụ thuộc ngưỡng); báo cáo thêm balanced accuracy, G-mean, MCC |
| Cấu trúc | Tập thật lớp thiểu số nhỏ ở ISIC (148 ảnh) | KID là chỉ số chính, kèm std; FID chỉ tham khảo |
| Thống kê | Chỉ 3 seed; test ISIC nhỏ (75 ca dương; RSNA 226); 54 so sánh mỗi bộ dữ liệu | [Mục 5.2](#52-phân-tích-thống-kê): báo cáo cả std lẫn CI; p Holm trong từng backbone |
| Ngoại tại | Hai bộ dữ liệu, chỉ bài toán nhị phân, một họ GAN, ảnh 256 px | Nêu trong phần hạn chế |
| Ngoại tại | RSNA: nhãn "Lung Opacity" là nhận định trên ảnh (không phải chẩn đoán viêm phổi xác nhận lâm sàng), có bất đồng giữa bác sĩ khoảng 11–12 % | Dùng nhãn cuối cùng sau hội chẩn; nêu trong phần hạn chế |

---

## 7. Hạn chế đã biết

Các điểm sau nên nêu trong bài báo:

- **Dấu vân tay tần số của GAN:** detector của Frank et al. phân biệt ảnh thật / ảnh sinh với AUC ≈ 1. Test toàn ảnh
  thật nên điều này không làm kết quả lạc quan giả tạo, nhưng có thể làm giảm lợi ích của ảnh sinh.
- **Shortcut tư thế chụp ở RSNA** (AP gắn với bệnh nặng): không sửa dữ liệu, chỉ đo bằng phân tích nhóm con (mục 5.5).
- **Classifier không augmentation** (cả hai bộ dữ liệu): AUC tuyệt đối có thể thấp hơn các công trình dùng
  augmentation; kết luận của bài là về **chênh lệch giữa các phương pháp** trong cùng giao thức.
- **Chỉ hỗ trợ bài toán nhị phân.** Bài toán đa lớp cần mở rộng DASS theo kiểu one-vs-rest.
- **Siêu tham số chung**, không dò riêng cho từng backbone; Transformer có thể chưa đạt mức tối ưu.
- **Kiểm định chính thức** có cho mọi biến thể so với M0 và cho M6 vs M0b, M6 vs M1, kèm p hiệu chỉnh Holm trong từng
  backbone. Các so sánh thành phần (M6 với M2–M5) chưa có kiểm định riêng (thêm được qua `evaluation.comparisons`).

---

# Phần B — Thực hành

## 8. Cách chạy

### 8.1 Môi trường

Google Colab có GPU (khuyến nghị A100 hoặc L4). Có hai cách mở `notebooks/colab_pipeline.ipynb`:

- **VS Code + extension Colab:** mở notebook, chọn *Select Kernel* → *Colab* → *New Colab Server* → *GPU* → *A100* →
  *Python 3 (ipykernel)*. Thư mục dự án phải ở chế độ *Trust*.
- **Trình duyệt:** https://colab.research.google.com/github/honghanh22/isic-dass/blob/main/notebooks/colab_pipeline.ipynb,
  rồi *Runtime* → *Change runtime type* → *A100 GPU*.

Notebook **tự tải code mới nhất từ GitHub**, không cần upload tay. Notebook không chứa logic: mỗi ô chỉ gọi một lệnh
`dass`.

### 8.2 Các ô trong notebook (chạy từ trên xuống bằng Shift + Enter)

| Ô | Việc | Phải thấy |
|---|---|---|
| Mount Drive | đăng nhập Google, chọn *Allow* | `Mounted at /content/drive`, tên GPU |
| Lấy mã nguồn | clone / pull từ GitHub rồi cài đặt | `Phiên bản code: <hash> — …` |
| Chọn thực nghiệm | đặt `EXPERIMENT`, `PROFILE`, `EXTRA` | cấu hình in ra (`run_tag`, `pool_mult`, …) |
| Tiện ích | định nghĩa `show()`, `table()`, `RESULTS` | không lỗi |
| `prepare` | dữ liệu, split | [mục 9](#9-kiểm-tra-sau-mỗi-bước) |
| `gan-setup` | clone + vá StyleGAN2-ADA, biên dịch plugin CUDA | chỉ cần chạy lại khi là **server mới** |
| `gan` | dùng lại GAN, vẽ đường KID | |
| `sample` | sinh pool | `DONE <N> …` |
| `fingerprint` | (tuỳ chọn) dấu vân tay tần số | bảng AUC |
| `select` | E_v, E_d, chọn ảnh, chẩn đoán | `M6_dass: chọn n / N ảnh sinh` |
| 6 ô `train` | mỗi ô một backbone (8 biến thể × 3 seed) | mỗi lần chạy in `test ROC-AUC = …` |
| `evaluate` + `report` | metric và bảng | các bảng `dataset`, `classification`, `significance`, `generation_quality` |

### 8.3 Chạy thử (smoke) và chạy thật

```python
EXPERIMENT = "configs/experiments/rsna_pneumonia_dass.yaml"   # hoặc "configs/experiments/isic2016_dass.yaml"
PROFILE = ""                                                   # "configs/experiments/smoke.yaml" = chạy thử
EXTRA = ""                                                     # ghi đè, ví dụ "--set selection.gamma=0 --tag nodiv"
```

| | Smoke (`PROFILE = "configs/experiments/smoke.yaml"`) | Chạy thật (`PROFILE = ""`) |
|---|---|---|
| Mục đích | kiểm tra mọi bước chạy thông, **không dùng để báo cáo** | kết quả cho bài báo |
| Pool | k = 1,2 | k = 1,5 |
| E_d | 1 + 1 epoch | 5 + 20 epoch |
| Classifier | EfficientNetV2B0, 1 seed, 1 + 1 epoch | 6 backbone, 3 seed, 5 + 30 epoch |
| Thư mục | `checkpoints_smoke/`, `results_smoke/` | `checkpoints_<run_tag>/`, `results_<run_tag>/` |

**Chuyển từ smoke sang chạy thật:** không cần tải lại notebook hay khởi động lại kernel.
1. Sửa `PROFILE = ""`, chạy lại ô chọn thực nghiệm. Kiểm tra cấu hình in ra: RSNA `"run_tag": "rsna_v2"`,
   ISIC `"run_tag": "v11"`; cả hai `"augment": false`, `"baseline_class_weight": false`, `"pool_mult": 1.5`.
2. **Bắt buộc** chạy lại ô tiện ích, vì `RESULTS` được tính từ cấu hình. Nếu không chạy lại, `show()` / `table()` vẫn
   đọc thư mục của smoke.
3. Chạy tiếp từ `prepare` trở xuống.

### 8.4 Thứ tự chạy và thời gian

1. **RSNA:** `prepare` → `gan` → `sample` → `select` → các ô `train` → `evaluate` + `report` (`fingerprint` tuỳ chọn).
   GAN `rsna` đã train xong (KID 0,0145 tại 800 kimg); `prepare` mất khoảng 30 phút trên server mới (chuyển DICOM).
2. **ISIC:** đổi `EXPERIMENT = "configs/experiments/isic2016_dass.yaml"`, chạy lại ô chọn thực nghiệm và ô tiện ích,
   rồi lặp lại đúng chuỗi trên.

Về thời gian:
- Bước `train` chiếm gần hết thời gian: 24 lần train cho mỗi backbone, và Transformer chậm hơn CNN.
- Nên chạy **từng backbone một**. Sau backbone đầu tiên có thể chạy `evaluate` + `report` để kiểm tra pipeline cho ra
  bảng đúng; các bảng được tính lại mỗi lần chạy `evaluate`.
- **Xem kết quả sơ bộ thì được, nhưng không được đổi tham số dựa trên kết quả test.** Nếu buộc phải đổi (ví dụ phát hiện
  lỗi), ghi lại lý do và chạy lại **toàn bộ** với `run_tag` mới.
- A100 tốn nhiều compute unit. Bước train chạy được trên L4: chậm hơn, nhưng thiết lập giống hệt.

### 8.5 Khi Colab ngắt kết nối

| Tình huống | Chạy lại |
|---|---|
| Mất kết nối nhưng server còn | ô mount Drive, ô chọn thực nghiệm, ô tiện ích, rồi ô đang dở |
| Server mới (máy trống) | ô mount Drive, ô lấy mã nguồn, ô chọn thực nghiệm, ô tiện ích, `gan-setup`, rồi ô đang dở |
| Code mới vừa được push | ô lấy mã nguồn (`git pull`), ô chọn thực nghiệm, ô tiện ích, rồi ô cần chạy |

Mọi bước đều chạy lại được: GAN, pool, E_d, `selections.json` và các file `.npz` đã có đều được khôi phục hoặc bỏ qua.
Một ô `train` bị ngắt giữa chừng chỉ mất lần chạy đang dở.

### 8.6 Chạy bằng lệnh (CLI)

```bash
E=configs/experiments/rsna_pneumonia_dass.yaml
dass -c $E show-config                                   # xem cấu hình đã hợp nhất
dass -c $E run                                           # toàn bộ chuỗi prepare → report
dass -c $E run --from select --to report                 # một đoạn của chuỗi
dass -c $E train --model resnet50 --seeds 2026 2027 2028 # một backbone
dass -c $E evaluate --skip-generative                    # chỉ metric phân loại (không tính KID / FID)
dass -c $E -c configs/experiments/smoke.yaml run         # chạy thử
```

### 8.7 Ablation (nếu cần sau này)

- **Luôn kèm `--tag <tên>`**, để kết quả nằm trong thư mục riêng `results_<run_tag>_<tên>/`. Ví dụ:
  `--set selection.gamma=0 --tag nodiv`.
- Có thể dùng lại E_d của lần chạy chính bằng `--set encoder.e_d_from_run=base`. Khi đó ablation chỉ khác đúng tham số
  đang xét.

### 8.8 Không được làm

| Không làm | Lý do |
|---|---|
| `dass gan --fresh-start` với GAN đang dùng | **xoá** trạng thái GAN trên Drive |
| Xoá thư mục trên Drive | kết quả cũ không tái tạo lại được; muốn tách lần chạy thì dùng `--tag` hoặc `run_tag` mới |
| `select --force` mà giữ `run_tag` | các dự đoán đã train sẽ không còn khớp với lựa chọn mới |
| Sửa tay `notebooks/colab_pipeline.ipynb` để thêm logic | logic nằm trong `src/`; notebook được sinh từ `scripts/build_colab_notebook.py` |

---

## 9. Kiểm tra sau mỗi bước

Sau mỗi bước, bấm **Ctrl+S** để lưu output vào file `.ipynb` rồi đối chiếu với bảng sau:

| Bước | RSNA: phải thấy | ISIC: phải thấy | Nếu khác |
|---|---|---|---|
| `prepare` | `Thiểu số = positive (1059) \| đa số = negative (7522) \| cần chọn 6463 \| pool 9695 ảnh \| 1 kênh`; `Split trùng khớp split tham chiếu real_val_split.json` | `Thiểu số = malignant (148) \| đa số = benign (618) \| cần chọn 470 \| pool 705 ảnh \| 3 kênh` | sai cấu hình / `PROFILE` |
| `gan` | dùng `checkpoints_rsna` (KID 0,0145 tại 800 kimg), không train lại | dùng `checkpoints_v5` (KID 0,0201 tại 1.800 kimg), không train lại | sai `gan_tag` |
| `sample` | `Khôi phục 9695 ảnh từ …pool_positive_from800kimg_n9695_c1.zip` | `DONE 705 …` hoặc khôi phục từ zip | |
| `select` | `M6_dass: chọn 6463 / 9695 ảnh sinh` | `M6_dass: chọn 470 / 705 ảnh sinh` | |
| `select` (chẩn đoán) | AUC probe trên val: kỳ vọng E_d ≥ E_v. Tương quan M_v–M_d càng gần 1 thì E_d càng ít thêm thông tin. Jaccard giữa M1 và các biến thể khác ≈ 0,5 | như bên trái | ghi lại, nêu trong bài |
| `train` | dòng đầu `Giao thức train … \| augment = False … \| class weight: không`; mỗi lần chạy in `test ROC-AUC = …` | như bên trái | sai phiên bản code / cấu hình |
| `train` (ghép dữ liệu) | `M0_real_only_p1.5: {'train': {'negative': 7522, 'positive': 1059}, ...}` và `M0b_real_oversample_p1.5: {'train': {'negative': 7522, 'positive': 7522}, ...}` | `M0b_real_oversample_p1.5: {'train': {'benign': 618, 'malignant': 618}, ...}` | `selection.oversample_variant` đang tắt |
| `evaluate` | `Phân loại: 144 lần chạy, 48 nhóm` khi đủ 6 backbone; thêm các bảng nhóm con (tư thế AP / PA) | `Phân loại: 144 lần chạy, 48 nhóm` | thiếu: còn backbone chưa xong |

---

## 10. Kết quả: lưu ở đâu và đọc thế nào

### 10.1 Vị trí trên Google Drive

Kết quả nằm trong thư mục của từng bộ dữ liệu. Không bao giờ xoá; chạy lại không ghi đè.

| | RSNA (`…/RSNA Pneumonia/Result_Pneumonia/`) | ISIC (`…/ISBI2016_ISIC_Part3/`) |
|---|---|---|
| GAN | `checkpoints_rsna/stylegan2ada/` | `checkpoints_v5/stylegan2ada/` |
| Split, pool, lựa chọn, E_d, trọng số classifier | `checkpoints_rsna_v2/` | `checkpoints_v11/` |
| Dự đoán, số liệu thô, **bảng**, hình | `results_rsna_v2/` | `results_v11/` |
| Kết quả giao thức cũ (chuyển ra, không dùng trong bảng) | `results_rsna_v2/predictions_superseded/`, `checkpoints_rsna_v2/classifiers_superseded/` | — |
| Kết quả cấu hình trước (giữ nguyên, báo cáo riêng) | `results_rsna_v1/` (chỉ `prepare`) | `results_v10/` (k = 1,5, có augmentation, M0 có class weight); `results_v9/` (k = 2, không augmentation, M0 không class weight); `results_v7/` (notebook v5, k = 4); `v8` không dùng |
| Chạy thử | `checkpoints_smoke/`, `results_smoke/` | như bên trái |

Kết quả Brain Tumor cũ (`…/BrainTumor_GAN/`) vẫn nằm trên Drive, không xoá, nhưng không còn thuộc dự án.

Trong `results_<run_tag>/` có:
- `tables/`: bảng cho bài báo (`.csv` đã định dạng, `.json` số thô, `.tex` booktabs);
- `metrics/`: số liệu thô;
- `predictions/`: file `.npz`;
- các thư mục hình;
- `run_manifest.json`.

Chi tiết từng cột: [RESULTS_FORMAT.md](RESULTS_FORMAT.md).

### 10.2 Đọc kết quả

| Bảng / chỉ số | Đọc thế nào |
|---|---|
| `classification` | Hàng đã dùng tên hiển thị và chia nhóm (cột `Group`: Real Data Baselines / Generative Augmentation). So DASS (Ours) với Imbalanced Baseline (H1), ROS (H2), Unfiltered GAN (H3) và các Filter (ablation) **trong cùng một backbone**. Với dữ liệu mất cân bằng, ưu tiên ROC-AUC, PR-AUC, sensitivity, balanced accuracy, G-mean, MCC; không dựa vào accuracy |
| `significance` | Cột `vs` cho biết đối chứng (Imbalanced Baseline, ROS hoặc Unfiltered GAN). ΔAUC > 0 và khoảng tin cậy 95 % không chứa 0 thì phương pháp hơn đối chứng có ý nghĩa thống kê (lưu ý đa so sánh, mục 5.2). Hàng **DASS vs ROS** là bằng chứng mạnh nhất cho giá trị của ảnh sinh |
| Tính nhất quán | đếm số backbone (trên 6) mà M6 > M0, ở mỗi bộ dữ liệu; một kết quả tốt ở một backbone chưa đủ để kết luận |
| KID | thấp hơn là gần ảnh thật hơn; đọc tương đối so với hàng `real val vs real train` và `all candidates` |
| AUC thật-vs-sinh (`shortcut_check`) | gần 0,5 là khó phân biệt (tốt); gần 1 nghĩa là ảnh sinh dễ nhận ra, có nguy cơ shortcut |
| AUC probe (dòng val) | E_d > E_v nghĩa là không gian bệnh tách lớp tốt hơn, đúng mục đích thiết kế E_d |
| Jaccard | so với mốc ngẫu nhiên 0,5 (k = 1,5); thấp nghĩa là các tiêu chí chọn ra tập ảnh khác nhau |

---

## 11. Quy tắc báo cáo

**Nên:**
- Báo cáo **đủ 6 backbone × 8 biến thể × 2 bộ dữ liệu**, kể cả những cấu hình mà M6 không thắng.
- Mean ± std qua 3 seed, kèm ΔAUC, khoảng tin cậy 95 % và p của M6 so với M0, M0b và M1. Ghi rõ số seed và số lần
  bootstrap.
- KID là chỉ số chính về chất lượng ảnh sinh; FID kèm cảnh báo về độ chệch.
- Nêu AUC thật-vs-sinh và kết quả fingerprint như một hạn chế, cùng các mục ở [phần 7](#7-hạn-chế-đã-biết).
- Ghi phiên bản code (commit hash), `dass_version` và cấu hình (`run_manifest.json`).

**Không nên:**
- Chọn seed, backbone hay bộ dữ liệu "đẹp" sau khi đã xem kết quả.
- Dò ngưỡng, k, γ hay bất kỳ tham số nào trên test, hoặc chạy lại test.
- Trộn kết quả cũ (`results_v7`, `results_v9`, `results_v10`) với kết quả mới: cấu hình khác nhau.

---

## 12. Bảng tham số

| Nhóm | Tham số | Giá trị | File |
|---|---|---|---|
| GAN | cfg / batch / R1 γ / mirror / ADA target | paper256 / 16 / 1.0 / có / 0.6 | `configs/_base_/generator.yaml` |
| GAN | max kimg / snapshot / min kimg / patience / Δ tối thiểu | 3000 / 100 / 400 / 5 / 2 % | |
| GAN | KID khi train: số ảnh / tập con / seed | 1000 / 50 / 123 | |
| Pool | k = `pool_mult` (chung mọi bộ dữ liệu) / ψ / seed | 1.5 / 1.0 / 777 | `configs/_base_/selection.yaml` |
| DASS | K / λ_v / λ_d / α / β / γ | 5 / 1 / 1 / 1 / 1 / 0.5 | |
| Biến thể | M0b oversampling (`oversample_variant`) / M7 (`both_classes_variant`) | bật / tắt | |
| E_d | backbone / seed / epoch | DenseNet121 / 4242 / 5 + 20 | |
| Classifier | backbone | EfficientNetV2B0, ResNet50, DenseNet121, ConvNeXtTiny, ViT-B16, SwinT | `configs/_base_/classifier.yaml` |
| Classifier | ảnh / batch / dropout (CNN) | 224 / 16 / 0.3 | |
| Classifier | epoch / lr / weight decay / early stop | 5 + 30 / 1e-3, 1e-5 / 1e-4 / 8 | |
| Classifier | seed / chọn epoch | 2026, 2027, 2028 / `val_macro_recall` | |
| Classifier | augmentation (`augment`) / class weight cho M0 (`baseline_class_weight`) | tắt / tắt (mọi bộ dữ liệu) | |
| Đánh giá | ngưỡng / bootstrap / KID (tập con, kích thước tối đa) / cặp SSIM | 0.5 / 2000 / 50, 1000 / 200 | `configs/_base_/evaluation.yaml` |
| Đánh giá | đối chứng chính / cặp bổ sung (`comparisons`) | M0 / M6 vs M0b, M6 vs M1 | |
| Đánh giá | tên hiển thị / nhóm (`method_labels`, `method_groups`) | xem đầu tài liệu | |
| Chung | seed toàn cục / tất định | 2026 / false | `configs/_base_/runtime.yaml` |
| Dữ liệu | nguồn, tiền xử lý, cách chia, số kênh, thư mục Drive, `gan_tag`, `run_tag` | theo bộ dữ liệu | `configs/datasets/*.yaml` |
