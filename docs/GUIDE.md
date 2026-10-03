# Hướng dẫn tổng quan: bài toán, phương pháp, thiết kế thực nghiệm và cách chạy

Tài liệu dành cho người mới tiếp cận dự án, đồng thời là nguồn tham chiếu khi viết phần *Phương pháp* và *Thực
nghiệm* của bài báo. Mọi công thức, con số và tham số dưới đây khớp với code (`src/dass/`) và cấu hình (`configs/`)
của phiên bản **1.5.0** (k = 1,5; classifier có augmentation; baseline M0 có class weight, như ISIC v7; ISIC `v10`,
Brain Tumor `bt_v5`). Khi đổi code hoặc cấu hình, cập nhật file này.

**Các cấu hình đã chạy** (kết quả đều giữ trên Drive, không trộn với nhau):

| Run tag (ISIC / Brain) | k | Augmentation classifier | Class weight M0 | Ghi chú |
|---|---|---|---|---|
| `v7` / — | 4 | có | có | notebook v5, chỉ EfficientNetV2B0, 2 seed, không có ROS |
| — / `bt_v3` | 3 | có | có | Brain: EfficientNetV2B0, ResNet50, ConvNeXtTiny |
| `v9` / `bt_v4` | 2 | **không** | **không** | ISIC: EfficientNetV2B0, ResNet50 |
| **`v10` / `bt_v5`** | **1,5** | **có** | **có** | **cấu hình hiện tại** |

**Lỗi đã sửa ở 1.5.1:** trước đó 5 phép augmentation dùng chung một seed nên bị tương quan với nhau (xem CHANGELOG). Các lần chạy có augmentation trước 1.5.1 (`v7`, `bt_v3`) và mọi E_d cũ dùng augmentation tương quan này; `v10` / `bt_v5` dùng augmentation đã sửa, nên không giống hệt `v7` ở điểm này.

Cấu hình được đổi nhiều lần sau khi đã xem kết quả test. Khi viết bài báo phải nêu rõ điều này, và nên báo cáo kết
quả của mọi cấu hình đã chạy (ví dụ trong phụ lục) thay vì chỉ chọn cấu hình có kết quả đẹp nhất.

**Tên phương pháp:** trong code và file kết quả dùng **mã nội bộ** (M0, M0b, M1–M6); trong bảng bài báo dùng **tên
hiển thị** (`evaluation.method_labels`):

| Nhóm | Mã | Tên trong bài báo |
|---|---|---|
| Real Data Baselines | M0 (`M0_real_only`) | **Imbalanced Baseline (class-weighted)** |
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
| RQ6 | Kết quả có nhất quán giữa các backbone (CNN, Transformer) và giữa hai bộ dữ liệu không? | 6 backbone × 2 bộ dữ liệu | H5: chiều của H1–H3 giữ ở đa số cấu hình |
| RQ7 | Ảnh được chọn khác pool như thế nào (chất lượng, khả năng tạo shortcut)? | KID, AUC thật-vs-sinh, Jaccard | (mô tả) |

**H1, H2, H3** được kiểm định thống kê trực tiếp bằng paired bootstrap ΔAUC: mọi biến thể so với M0, cộng hai cặp
M6 vs M0b và M6 vs M1 (`evaluation.comparisons`). Các so sánh thành phần (RQ5) được đọc từ bảng mean ± std và nên
trình bày là **phân tích thăm dò** ([mục 5.2](#52-phân-tích-thống-kê)).

---

## 2. Dữ liệu

### 2.1 Hai bộ dữ liệu

| | ISIC 2016 Part 3 | Brain Tumor MRI |
|---|---|---|
| Loại ảnh | dermoscopy, RGB | MRI, xám (lưu dạng 3 kênh bằng nhau, xem 2.2) |
| Lớp đa số / thiểu số | benign (0) / **malignant (1)** | negative (0) / **positive (1)** |
| Nguồn | thư mục ảnh `.jpg` + CSV nhãn; có tập test chính thức | thư mục `Negative/`, `Positive/`; không có tập test riêng |
| Số ảnh gốc | train 900 (727 / 173), test 379 (304 / 75) | 2.000 / 400 |
| Tỉ lệ mất cân bằng | ≈ 4,2 : 1 | 5 : 1 |

### 2.2 Tiền xử lý (chạy một lần, áp dụng như nhau cho mọi tập)

| Bước | ISIC 2016 | Brain Tumor |
|---|---|---|
| Đọc ảnh | RGB | chuyển về luminance bằng `PIL.convert("L")`, lưu thành **3 kênh bằng nhau** (`force_grayscale`) |
| Cắt viền | cắt viền đen: pixel có trung bình kênh > 15 được coi là nội dung; chỉ cắt khi vùng nội dung chiếm **< 50 %** một cạnh | không cắt (ảnh đã sát viền, tỉ lệ nội dung trung vị ≈ 0,90) |
| Đưa về 256 × 256 | kéo giãn (bilinear) | **đệm nền đen thành ảnh vuông** (giữ tỉ lệ giải phẫu), rồi resize bilinear |
| Lưu | PNG | PNG |

- **Mọi ảnh lưu PNG** (nén không mất dữ liệu), để ảnh thật và ảnh sinh không khác nhau về kiểu nén.
- **Lý do dùng `force_grayscale` cho Brain Tumor:** 129 / 2.000 ảnh negative có nhiễu màu JPEG (độ lệch giữa các kênh
  ≤ 3,1 mức xám), trong khi 0 / 400 ảnh positive có nhiễu này. "Có chút màu" vì thế gắn với nhãn negative và trở thành
  manh mối giả (shortcut). Sau khi chuyển về luminance, 2.271 / 2.400 ảnh giữ nguyên giá trị. Ảnh sinh đi qua **đúng
  cùng phép chuyển** này.
- Khi đưa vào classifier, ảnh được resize về 224 × 224 rồi chuẩn hoá theo yêu cầu của từng backbone.

### 2.3 Cách chia

| | ISIC 2016 | Brain Tumor |
|---|---|---|
| Kiểu chia | `holdout_val`: tách val 15 % từ train (phân tầng theo lớp) | `stratified`: test 15 %, sau đó val 17,6 % phần còn lại, tức **70 / 15 / 15** |
| Train (đa số + thiểu số) | 618 + 148 | 1.401 + 281 |
| Val | 109 + 25 | 299 + 59 |
| Test | 304 + 75 (chính thức) | 300 + 60 |

- Split **tất định** (seed toàn cục 2026), lưu thành JSON. Chạy lại với cách chia khác mà vẫn giữ `run_tag` thì lệnh
  **dừng**, vì kết quả cũ gắn với split cũ.
- Split **tái lập đúng** split của notebook gốc (ISIC v5, Brain Tumor v1), vì GAN được dùng lại đã train trên phần
  train của split đó. Brain Tumor có kiểm tra tự động với `checkpoints_bt/data/real_split.json` (`split.expected`).
  ISIC được bảo đảm bằng test regression `tests/regression/test_split_reproduction.py`.
- **Val và test luôn 100 % ảnh thật.**

### 2.4 Ngân sách sinh và ý nghĩa của k

| | ISIC 2016 | Brain Tumor |
|---|---|---|
| n = n_maj − n_min | 618 − 148 = **470** | 1.401 − 281 = **1.120** |
| k (`pool_mult`) | 1,5 | 1,5 |
| N = ⌈k · n⌉ (pool) | **705** | **1.680** |
| Tập train sau khi thêm ảnh sinh | 618 : (148 + 470) = **1 : 1** | 1.401 : (281 + 1.120) = **1 : 1** |

- **n quyết định tỉ lệ lớp cuối cùng** (luôn 1 : 1). **k chỉ quyết định độ chọn lọc:** DASS giữ 1/k pool.
  - k = 1: pool đúng bằng n, nên mọi biến thể M1–M6 chọn **cùng một tập** (không có gì để chọn).
  - k = 1,5: mỗi biến thể giữ 2/3 pool (loại 1/3). k lớn hơn thì chọn lọc gắt hơn, nhưng tập được chọn lệch xa phân
    phối ảnh thật hơn.
- **k = 1,5** (phiên bản 1.5.0), dùng chung cho cả hai bộ dữ liệu. Lịch sử: 4 (ISIC `v7`) → 3 (`bt_v3`) → 2 (`v9` /
  `bt_v4`) → 1,5. Lý do giảm dần: chọn càng gắt thì tập DASS càng lệch phân phối ảnh thật (Brain k = 3: KID 0,100 so
  với 0,029 của toàn pool; độ đa dạng 0,168 so với 0,241). CosSIF (FAGT) cũng chỉ loại 15–25 % ảnh sinh (k ≈ 1,2–1,3).
  **Phải nêu trong bài báo** rằng k được đổi sau các lần chạy trước, kèm lý do. Không chạy ablation theo k.
- **Mốc tham chiếu khi đọc Jaccard:** hai tập con ngẫu nhiên độc lập, mỗi tập chiếm 1/k pool, có Jaccard kỳ vọng
  1/(2k − 1) = **0,5** khi k = 1,5.

### 2.5 Bộ dữ liệu thứ ba: RSNA Pneumonia (X-quang ngực)

Đề xuất thay cho Brain Tumor, vì có nguồn gốc rõ ràng, hai lớp cùng nguồn chụp và đủ ảnh lớp thiểu số để train GAN.

| | RSNA Pneumonia Detection Challenge 2018 |
|---|---|
| Nguồn | RSNA + NIH ChestX-ray8; nhãn do bác sĩ X-quang gán (Shih et al., *Radiology: AI* 2019) |
| Ảnh | DICOM xám 1024 × 1024, **mỗi bệnh nhân một ảnh** |
| Nhãn | positive = `Target = 1` (có đám mờ phổi); negative = *Normal* + *No Lung Opacity / Not Normal* |
| Quy mô | tập con phân tầng **6.000 ảnh** (`subset_size`, seed 2026), giữ tỉ lệ khoảng 22 % positive |
| Số kênh | **1** (PNG xám); GAN mới sinh thẳng ảnh 1 kênh; nhân bản 1 -> 3 kênh chỉ trên bộ nhớ |
| Chia | test của cuộc thi không có nhãn -> `stratified` 70 / 15 / 15 |
| Augmentation | `upright`: lật ngang, xoay ±10° (không lật dọc, không xoay 180°) |
| GAN | train mới (`gan_tag: rsna`), cùng cấu hình và cùng cách dừng sớm theo KID |

- Bước `prepare` tự đọc dữ liệu (`data.source.type: dicom_csv`): tìm `.dcm` (tự giải nén nếu cần), tìm CSV nhãn, chuyển
  sang PNG và ghi `checkpoints_rsna_v1/data/dicom_metadata.csv` (tư thế chụp, giới tính, tuổi).
- **Shortcut cần kiểm tra:** bệnh nhân nặng thường chụp tư thế AP. Log của `prepare` in tỉ lệ AP / PA theo lớp; nên báo
  cáo trong bài.

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
- **Hiện trạng:** cả hai bộ dữ liệu **dùng lại GAN đã train**. Brain Tumor dùng `checkpoints_bt`: KID tốt nhất 0,02985
  tại 1.200 kimg, dừng sớm ở 1.700 kimg. ISIC dùng `checkpoints_v5`. Khi đó lệnh `gan` chỉ báo cáo (đường KID, ảnh
  mẫu) và không train lại.

### 3.3 Candidate pool

- Sinh N = ⌈k · n⌉ ảnh lớp thiểu số từ `best.pkl` với `truncation ψ = 1` (không cắt bớt độ đa dạng). Seed cố định
  777, nên pool tái lập được. Pool được nén zip lên Drive.
- Ảnh sinh lưu đúng số kênh của bộ dữ liệu. Với Brain Tumor, ảnh sinh đi qua đúng phép chuyển luminance như ảnh thật
  (log in ra `force_gray=1`).
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
| M0 | Imbalanced Baseline (class-weighted) | không dùng ảnh sinh; dữ liệu thật mất cân bằng (5 : 1 / 4,2 : 1), **cân bằng trong loss bằng class weight** | – | **baseline** |
| M0b | Random Oversampling (ROS) | không dùng ảnh sinh: **nhân bản ảnh thật** lớp thiểu số lên 1 : 1 | – | **baseline oversampling**: cùng số ảnh, cùng số bước train, cùng cách cân bằng với M1–M6 |
| M1 | Unfiltered GAN (Random Selection) | ngẫu nhiên (seed 2026) | – | tác dụng của ảnh sinh khi không chọn lọc |
| M2 | Visual-only Filter (M_v) | M_v | – | chỉ không gian thị giác |
| M3 | Disease-only Filter (M_d) | M_d | – | chỉ không gian bệnh |
| M5 | Diversity-only Filter (S_div) | 0 (k-center greedy, γ = 1) | ✓ | chỉ đa dạng, không có điểm lề |
| M4 | Dual-Margin Filter (M_v + M_d) | α·M̃_v + β·M̃_d | – | kết hợp hai không gian, chưa có đa dạng |
| **M6** | **DASS (Ours)** | **α·M̃_v + β·M̃_d** | **γ = 0,5** | **DASS đầy đủ (phương pháp đề xuất)** |

**Cách M0b nhân bản ảnh** (`data.variants.oversample_indices`, seed 2026): cần thêm n bản sao từ n_min ảnh thật, nên
mỗi ảnh được lặp q = ⌊n / n_min⌋ lần, và r = n mod n_min ảnh (chọn ngẫu nhiên, không lặp) được lặp thêm một lần. Số
lần xuất hiện của các ảnh vì thế chênh nhau tối đa 1. Ví dụ Brain Tumor: 1.120 = 281 × 3 + 277, nên mỗi ảnh positive
thật xuất hiện 4 hoặc 5 lần trong tập train, mỗi lần với một phép augmentation khác nhau. M0b không chọn gì từ pool
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

| | Số ảnh train (Brain Tumor) | Bước mỗi epoch | Cân bằng bằng |
|---|---|---|---|
| M0 (Imbalanced Baseline, class-weighted) | 1.401 : 281 | 106 | **class weight** w_c = N_train / (2 · n_c) |
| M0b (ROS) | 1.401 : (281 + 1.120 bản sao) | 176 | dữ liệu (ảnh thật nhắc lại) |
| M1–M6 | 1.401 : (281 + 1.120 ảnh sinh) | 176 | dữ liệu (ảnh sinh) |

- Như ISIC `v7`: mọi biến thể dùng **cùng augmentation** (`classifier.augment: true`), M0 cân bằng bằng **class
  weight** (`classifier.baseline_class_weight: true`). Mọi biến thể vì thế có tỉ lệ lớp hiệu dụng 1 : 1, và ngưỡng 0,5
  so sánh được giữa các biến thể.
- M0b và M1–M6 giống hệt nhau về số ảnh, số bước train và cách cân bằng; chỉ khác **nội dung** ảnh thêm vào. Vì vậy
  **M6 vs M0b** là phép so sánh chặt chẽ nhất.
- Cấu hình không can thiệp dữ liệu (`v9` / `bt_v4`) chạy lại được bằng `--set classifier.augment=false
  --set classifier.baseline_class_weight=false` kèm `--tag` riêng.

**Backbone (pretrain ImageNet):**

| Backbone | Họ | Nguồn | Đầu ra |
|---|---|---|---|
| EfficientNetV2B0, ResNet50, DenseNet121, ConvNeXtTiny | CNN | `keras.applications` | pooling trung bình → Dropout 0,3 → Dense(1, sigmoid) |
| ViT-B16, SwinT | Transformer | KerasHub (preset Hugging Face) | đầu phân loại của preset, 1 đầu ra sigmoid |

**Augmentation** (chỉ tập train, **giống hệt nhau cho mọi biến thể, kể cả M0 và M0b**): lật ngang và dọc, xoay
ngẫu nhiên tới ±180° (biên phản chiếu), zoom ±10 %, độ sáng ±10 %, tương phản ±10 %. Augmentation được áp dụng trực
tuyến (biến đổi ngẫu nhiên mỗi batch), nên không làm tăng số ảnh và không đổi tỉ lệ lớp; tác dụng của ảnh sinh được
đo **trên nền** augmentation. Ngoài ra:
- **ADA của StyleGAN2-ADA** là augmentation cho discriminator khi train GAN (GAN được dùng lại, đã train xong);
- **E_d** (encoder bệnh của DASS) train với augmentation và class weight.

Tiền xử lý dữ liệu (cắt viền đen ở ISIC; đệm vuông và ép xám ở Brain Tumor) là làm sạch dữ liệu, áp dụng như nhau cho
mọi ảnh, không phải augmentation.

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
| **Biến được kiểm soát** | cùng split, cùng GAN, cùng pool, cùng n, cùng siêu tham số train, **cùng augmentation cho mọi biến thể**, cùng seed cho mọi phương pháp (thiết kế **ghép cặp**), cùng ngưỡng 0,5, cùng tập val / test |

### 4.2 Ma trận thực nghiệm

| | Mỗi bộ dữ liệu | Hai bộ dữ liệu |
|---|---|---|
| Lần train classifier | 8 biến thể × 6 backbone × 3 seed = **144** | **288** |
| Lần train E_d | 1 | 2 |
| GAN | dùng lại (0 lần train) | 0 |

### 4.3 Siêu tham số được cố định trước

- Tham số DASS (K = 5, λ = 1, α = β = 1, γ = 0,5) lấy theo công thức của notebook ISIC v5. Lịch train E_d và mọi siêu
  tham số classifier được **dùng chung cho cả hai bộ dữ liệu và mọi backbone**.
- k = 1,5 (xem [mục 2.4](#24-ngân-sách-sinh-và-ý-nghĩa-của-k)); augmentation và class weight cho M0 như ISIC `v7`.
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

Độ phân giải của sensitivity trên test: ISIC có 75 ca dương, nên mỗi ca tương ứng ≈ 1,3 điểm %; Brain Tumor có 60 ca
dương, mỗi ca ≈ 1,7 điểm %. Chênh lệch nhỏ hơn mức này chỉ là một ca bệnh.

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
- **Cách đọc và lưu ý:**
  - ΔAUC trong bảng `significance` là AUC của **ensemble**, nên có thể khác hiệu của các AUC trung bình trong bảng
    `classification`.
  - Bootstrap đo **độ bất định do mẫu test**, với điều kiện mô hình đã train cố định. Độ bất định do train được phản
    ánh qua std giữa các seed. Báo cáo cả hai.
  - Với 2.000 lần lấy mẫu, p nhỏ nhất khác 0 là 0,001. p = 0 nên ghi là **p < 0,001**.
  - **Đa so sánh:** mỗi backbone có 7 phép so sánh với M0 và 2 cặp bổ sung, tức 9 × 6 = 54 phép mỗi bộ dữ liệu. Code
    **không hiệu chỉnh** đa so sánh. Khuyến nghị coi **M6 vs M0, M6 vs M0b, M6 vs M1** (H1–H3) là các kiểm định xác
    nhận, và dùng hiệu chỉnh Holm–Bonferroni cho 3 phép này trong từng backbone. Các so sánh còn lại là thăm dò.
  - So sánh M6 với M2–M5 (RQ5) **chưa có kiểm định chính thức** (chỉ đọc từ mean ± std). Có thể thêm bằng cách bổ
    sung cặp vào `evaluation.comparisons`, không cần train lại.

### 5.3 Chất lượng ảnh sinh (Inception-v3 của StyleGAN2-ADA, 2.048 chiều)

So với **ảnh thật lớp thiểu số của tập train** (ISIC 148 ảnh, Brain Tumor 281 ảnh), tính cho từng tập ảnh được chọn:

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

---

## 6. Tính hợp lệ: rò rỉ, shortcut và các mối đe doạ

### 6.1 Biện pháp chống rò rỉ

| Nguyên tắc | Cách bảo đảm |
|---|---|
| Không rò rỉ test | GAN, DASS và E_d chỉ dùng ảnh train; test chỉ được đọc ở bước train, sau khi đã chọn epoch |
| Val chỉ để chọn epoch | val không dùng để chọn ngưỡng, chọn k hay chọn phương pháp; val và test 100 % ảnh thật (assert trước khi train) |
| Không dò ngưỡng | ngưỡng cố định 0,5 |
| Test chỉ dự đoán một lần | dự đoán sau khi nạp checkpoint tốt nhất; không có vòng lặp nào dùng kết quả test |
| GAN dùng lại hợp lệ | split phải trùng split mà GAN đã train (Brain Tumor: kiểm tra tự động; ISIC: test regression) |
| Không có shortcut định dạng | mọi ảnh lưu PNG; Brain Tumor: 3 kênh bằng nhau cho cả ảnh thật và ảnh sinh |
| So sánh công bằng | cùng pool, cùng n, cùng seed, cùng quy trình train, cùng augmentation; M0b có cùng số ảnh, số bước train và cách cân bằng với M1–M6 |

### 6.2 Các mối đe doạ đến tính hợp lệ

| Loại | Mối đe doạ | Cách xử lý hoặc đo |
|---|---|---|
| Nội tại | Ảnh sinh **chỉ** nằm ở lớp thiểu số, nên đặc điểm "trông như ảnh GAN" tương quan với nhãn trong tập train | Test toàn ảnh thật nên điều này không thổi phồng kết quả test, nhưng có thể làm giảm lợi ích. Đo bằng AUC thật-vs-sinh và fingerprint; M7 (tuỳ chọn) là đối chứng |
| Nội tại | M0 có ít bước train hơn (106 so với 176 bước mỗi epoch) và cân bằng bằng loss thay vì bằng dữ liệu, nên M6 vs M0 lẫn tác dụng của số bước / cách cân bằng | **M0b (ROS)** có cùng số ảnh, số bước và cách cân bằng với M6; M6 vs M0b tách riêng tác dụng của nội dung ảnh sinh |
| Nội tại | Cấu hình (k, augmentation, class weight) được đổi sau khi xem kết quả test | Báo cáo mọi cấu hình đã chạy (`v7`, `v9`, `v10`; `bt_v3`, `bt_v4`, `bt_v5`); kết quả `v9` (không augmentation) cho thấy lợi thế của ảnh sinh phụ thuộc vào augmentation |
| Nội tại | M1 chỉ có **một** lần rút ngẫu nhiên (seed 2026); M0b cũng chỉ một lần chọn ảnh nhân bản thêm | Độ biến thiên do việc rút ngẫu nhiên không nằm trong std; nêu rõ khi so M6 với M1 / M0b |
| Nội tại | Backbone DenseNet121 có **cùng kiến trúc** với E_d (khác seed, khác lần train) | Diễn giải riêng hàng DenseNet121; so sánh với 5 backbone còn lại |
| Nội tại | KID dùng trong early stopping của GAN so với chính ảnh train mà GAN đã thấy | Kiểm tra ảnh sinh cạnh ảnh thật gần nhất (học thuộc) |
| Nội tại | GPU không tất định | Báo cáo từ file `.npz` đã lưu; 3 seed |
| Cấu trúc | Metric tại ngưỡng 0,5 phụ thuộc tỉ lệ lớp lúc train: M0 (mất cân bằng) có xác suất lệch về lớp đa số, nên ở ngưỡng 0,5 sensitivity thấp là do thiết kế, không phải do mô hình kém hơn về khả năng xếp hạng | Chỉ số chính là ROC-AUC (không phụ thuộc ngưỡng); báo cáo thêm balanced accuracy, G-mean, MCC |
| Cấu trúc | Tập thật lớp thiểu số nhỏ (148 / 281) | KID là chỉ số chính, kèm std; FID chỉ tham khảo |
| Thống kê | Chỉ 3 seed; test nhỏ (75 / 60 ca dương); 54 so sánh mỗi bộ dữ liệu | [Mục 5.2](#52-phân-tích-thống-kê): báo cáo cả std lẫn CI; H1–H3 là kiểm định chính (Holm trong từng backbone) |
| Ngoại tại | Hai bộ dữ liệu, chỉ bài toán nhị phân, một họ GAN, ảnh 256 px | Nêu trong phần hạn chế |
| Ngoại tại | Brain Tumor **không chia theo bệnh nhân** (tên file không có mã bệnh nhân), nên các lát cắt của cùng một người có thể nằm ở cả train và test | Số tuyệt đối có thể lạc quan, nhưng ảnh hưởng như nhau tới mọi phương pháp. Nếu có mã bệnh nhân, đặt `split.group_regex` |

---

## 7. Hạn chế đã biết

Các điểm sau nên nêu trong bài báo:

- **Dấu vân tay tần số của GAN:** detector của Frank et al. phân biệt ảnh thật / ảnh sinh với AUC ≈ 1. Test toàn ảnh
  thật nên điều này không làm kết quả lạc quan giả tạo, nhưng có thể làm giảm lợi ích của ảnh sinh.
- **Nhiễu màu JPEG ở Brain Tumor** đã được loại bằng `force_grayscale` cho cả ảnh thật lẫn ảnh sinh. Tuy nhiên GAN được
  train **trước** khi làm sạch. Ảnh positive (lớp được sinh) vốn không có nhiễu này, nên ảnh hưởng không đáng kể.
- **Không chia theo bệnh nhân** ở Brain Tumor (xem 6.2).
- **Chỉ hỗ trợ bài toán nhị phân.** Bài toán đa lớp cần mở rộng DASS theo kiểu one-vs-rest.
- **Siêu tham số chung**, không dò riêng cho từng backbone; Transformer có thể chưa đạt mức tối ưu.
- **Kiểm định chính thức** có cho mọi biến thể so với M0 và cho M6 vs M0b, M6 vs M1. Các so sánh thành phần (M6 với
  M2–M5) chưa có kiểm định, và code chưa tự hiệu chỉnh đa so sánh.

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
EXPERIMENT = "configs/experiments/brain_tumor_dass.yaml"   # hoặc "configs/experiments/isic2016_dass.yaml"
PROFILE = ""                                                # "configs/experiments/smoke.yaml" = chạy thử
EXTRA = ""                                                  # ghi đè, ví dụ "--set selection.gamma=0 --tag nodiv"
```

| | Smoke (`PROFILE = "configs/experiments/smoke.yaml"`) | Chạy thật (`PROFILE = ""`) |
|---|---|---|
| Mục đích | kiểm tra mọi bước chạy thông, **không dùng để báo cáo** | kết quả cho bài báo |
| Pool | k = 1,2 | k = 1,5 |
| E_d | 1 + 1 epoch | 5 + 20 epoch |
| Classifier | EfficientNetV2B0, 1 seed, 1 + 1 epoch | 6 backbone, 3 seed, 5 + 30 epoch |
| Thư mục | `checkpoints_smoke/`, `results_smoke/` | `checkpoints_<run_tag>/`, `results_<run_tag>/` |

**Chuyển từ smoke sang chạy thật:** không cần tải lại notebook hay khởi động lại kernel.
1. Sửa `PROFILE = ""`, chạy lại ô chọn thực nghiệm. Kiểm tra cấu hình in ra có `"run_tag": "bt_v5"` (hoặc `"v10"` với
   ISIC), `"pool_mult": 1.5`, `"augment": true`, `"baseline_class_weight": true`.
2. **Bắt buộc** chạy lại ô tiện ích, vì `RESULTS` được tính từ cấu hình. Nếu không chạy lại, `show()` / `table()` vẫn
   đọc thư mục của smoke.
3. Chạy tiếp từ `prepare` trở xuống.

### 8.4 Thứ tự chạy và thời gian

1. **Brain Tumor:** `prepare` → `gan` → `sample` → `fingerprint` → `select` → 6 ô `train` → `evaluate` + `report`.
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
E=configs/experiments/brain_tumor_dass.yaml
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

| Bước | Brain Tumor: phải thấy | ISIC: phải thấy | Nếu khác |
|---|---|---|---|
| `prepare` | `Thiểu số = positive (281) \| đa số = negative (1401) \| cần chọn 1120 \| pool 1680 ảnh \| 3 kênh`; `Split trùng khớp split tham chiếu real_split.json` | `Thiểu số = malignant (148) \| đa số = benign (618) \| cần chọn 470 \| pool 705 ảnh \| 3 kênh` | sai cấu hình / `PROFILE` |
| `gan` | KID tốt nhất 0,02985 tại 1.200 kimg, không train lại | dùng `checkpoints_v5` (KID 0,0201 tại 1.800 kimg), không train lại | sai `gan_tag` |
| `sample` | `DONE 1680 … force_gray=1` | `DONE 705 …` | |
| `select` | `M6_dass: chọn 1120 / 1680 ảnh sinh` | `M6_dass: chọn 470 / 705 ảnh sinh` | |
| `select` (chẩn đoán) | AUC probe trên val: kỳ vọng E_d ≥ E_v. Tương quan M_v–M_d càng gần 1 thì E_d càng ít thêm thông tin. Jaccard giữa M1 và các biến thể khác ≈ 0,5 | như bên trái | ghi lại, nêu trong bài |
| `train` | mỗi lần chạy in `test ROC-AUC = …`; `best fine-tune epoch` không phải lúc nào cũng bằng 1 | như bên trái | nếu luôn bằng 1: lr / val có vấn đề |
| `train` (ghép dữ liệu) | `M0_real_only_p2: {'train': {'negative': 1401, 'positive': 281}, ...}` và `M0b_real_oversample_p2: {'train': {'negative': 1401, 'positive': 1401}, ...}` | `M0b_real_oversample_p2: {'train': {'benign': 618, 'malignant': 618}, ...}` | `selection.oversample_variant` đang tắt |
| `evaluate` | `Phân loại: 144 lần chạy, 48 nhóm` khi đủ 6 backbone | như bên trái | thiếu: còn backbone chưa xong |

---

## 10. Kết quả: lưu ở đâu và đọc thế nào

### 10.1 Vị trí trên Google Drive

Kết quả nằm trong thư mục của từng bộ dữ liệu. Không bao giờ xoá; chạy lại không ghi đè.

| | Brain Tumor (`…/BrainTumor_GAN/`) | ISIC (`…/ISBI2016_ISIC_Part3/`) |
|---|---|---|
| GAN | `checkpoints_bt/stylegan2ada/` | `checkpoints_v5/stylegan2ada/` |
| Split, pool, lựa chọn, E_d, trọng số classifier | `checkpoints_bt_v5/` | `checkpoints_v10/` |
| Dự đoán, số liệu thô, **bảng**, hình | `results_bt_v5/` | `results_v10/` |
| Kết quả cấu hình trước (giữ nguyên, báo cáo riêng) | `results_bt_v4/` (k = 2, không augmentation, M0 không class weight); `results_bt_v3/` (k = 3); `results_bt/` (notebook v1, công thức cũ) | `results_v9/` (k = 2, không augmentation, M0 không class weight); `results_v7/` (notebook v5, k = 4); `v8` không dùng |
| Chạy thử | `checkpoints_smoke/`, `results_smoke/` | như bên trái |

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
| `classification` | Hàng đã dùng tên hiển thị và chia nhóm (cột `Group`: Real Data Baselines / Generative Augmentation). So DASS (Ours) với Imbalanced Baseline (class-weighted) (H1), ROS (H2), Unfiltered GAN (H3) và các Filter (ablation) **trong cùng một backbone**. Với dữ liệu mất cân bằng, ưu tiên ROC-AUC, PR-AUC, sensitivity, balanced accuracy, G-mean, MCC; không dựa vào accuracy |
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
- Trộn kết quả cũ (`results_v7`, `results_bt`) với kết quả mới: cấu hình khác nhau.

---

## 12. Bảng tham số

| Nhóm | Tham số | Giá trị | File |
|---|---|---|---|
| GAN | cfg / batch / R1 γ / mirror / ADA target | paper256 / 16 / 1.0 / có / 0.6 | `configs/_base_/generator.yaml` |
| GAN | max kimg / snapshot / min kimg / patience / Δ tối thiểu | 3000 / 100 / 400 / 5 / 2 % | |
| GAN | KID khi train: số ảnh / tập con / seed | 1000 / 50 / 123 | |
| Pool | k = `pool_mult` (chung hai bộ dữ liệu) / ψ / seed | 1.5 / 1.0 / 777 | `configs/_base_/selection.yaml` |
| DASS | K / λ_v / λ_d / α / β / γ | 5 / 1 / 1 / 1 / 1 / 0.5 | |
| Biến thể | M0b oversampling (`oversample_variant`) / M7 (`both_classes_variant`) | bật / tắt | |
| E_d | backbone / seed / epoch | DenseNet121 / 4242 / 5 + 20 | |
| Classifier | backbone | EfficientNetV2B0, ResNet50, DenseNet121, ConvNeXtTiny, ViT-B16, SwinT | `configs/_base_/classifier.yaml` |
| Classifier | ảnh / batch / dropout (CNN) | 224 / 16 / 0.3 | |
| Classifier | epoch / lr / weight decay / early stop | 5 + 30 / 1e-3, 1e-5 / 1e-4 / 8 | |
| Classifier | seed / chọn epoch | 2026, 2027, 2028 / `val_macro_recall` | |
| Classifier | augmentation (`augment`) / class weight cho M0 (`baseline_class_weight`) | bật / bật | |
| Đánh giá | ngưỡng / bootstrap / KID (tập con, kích thước tối đa) / cặp SSIM | 0.5 / 2000 / 50, 1000 / 200 | `configs/_base_/evaluation.yaml` |
| Đánh giá | đối chứng chính / cặp bổ sung (`comparisons`) | M0 / M6 vs M0b, M6 vs M1 | |
| Đánh giá | tên hiển thị / nhóm (`method_labels`, `method_groups`) | xem đầu tài liệu | |
| Chung | seed toàn cục / tất định | 2026 / false | `configs/_base_/runtime.yaml` |
| Dữ liệu | nguồn, tiền xử lý, cách chia, số kênh, thư mục Drive, `gan_tag`, `run_tag` | theo bộ dữ liệu | `configs/datasets/*.yaml` |
