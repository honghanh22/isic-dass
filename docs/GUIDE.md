# Hướng dẫn tổng quan: bài toán, phương pháp, thiết kế thực nghiệm và cách chạy

Tài liệu dành cho người mới tiếp cận dự án, đồng thời là nguồn tham chiếu khi viết phần *Phương pháp* và *Thực
nghiệm* của bài báo (bản viết theo văn phong bài báo, **luôn được cập nhật trước**: [METHOD.md](METHOD.md)). Phiên bản
**2.0.0**. Ba bộ dữ liệu: **ISIC 2016**, **RSNA Pneumonia**, **PCam** (âm : dương = 12). Brain Tumor đã bị loại khỏi
dự án (1.10.0). Khi đổi code hoặc cấu hình, cập nhật file này.

**Thiết kế thí nghiệm 2.0.0** (người dùng quyết định, 2026-10-18):

| Bảng | Mã | Phương pháp | Lớp thiểu số | Lớp đa số | Khi train |
|---|---|---|---|---|---|
| Nhóm 1 `comparison` | B0 | Imbalanced Baseline | ảnh thật | ảnh thật | — |
| | B1 | Cost-sensitive (class-weighted loss) | ảnh thật | ảnh thật | class weight |
| | B2 | Random Oversampling | thật + g bản sao | ảnh thật | — |
| | B3 | Traditional Augmentation | ảnh thật | ảnh thật | augmentation |
| | B4 (= M1) | Balanced GAN: Random | thật + g ảnh sinh ngẫu nhiên | ảnh thật | — |
| | B5 (= M6) | Balanced GAN: DASS (Ours) | thật + g ảnh sinh chọn bằng DASS | ảnh thật | — |
| Nhóm 2 `ablation` | M1–M6 | Random / M_v / M_d / Diversity / M_v + M_d / DASS | thật + g ảnh sinh (cùng pool ⌈2g⌉) | ảnh thật | — |

g = n_đa số − n_thiểu số (cân bằng 1 : 1); k = 2. Ảnh sinh chỉ ở lớp thiểu số, nên **shortcut probe là bắt buộc**.
Họ giả thuyết chính: **H1** M6 vs M1, **H2** M6 vs B2, **H3** M6 vs B0. Classifier: ResNet50, ConvNeXt-Tiny, Swin-T
(cùng head), 3 seed; augmentation chỉ ở B3, class weight chỉ ở B1.

**Lần chạy:** ISIC `v14`, RSNA `rsna_v5`, PCam `pcam_v2` (server thêm `_srv`; PCam dùng lại E_d của `pcam_v1_srv`,
chính E_d đã chọn snapshot GAN). Cấu hình (k, thiết kế, tiêu chí chọn epoch, luật chọn checkpoint PCam) đã được đổi
nhiều lần sau khi xem kết quả của các lần chạy trước; bài báo phải nêu rõ và báo cáo mọi cấu hình đã chạy.

Tài liệu liên quan: [ARCHITECTURE.md](ARCHITECTURE.md) (cấu trúc code), [WORKFLOW.md](WORKFLOW.md) (làm việc hằng
ngày), [RESULTS_FORMAT.md](RESULTS_FORMAT.md) (ý nghĩa từng cột kết quả).

**Phần I — Nghiên cứu:** [1. Bài toán](#1-bài-toán) · [2. Dữ liệu](#2-dữ-liệu) · [3. Phương pháp](#3-phương-pháp) ·
[4. Thiết kế thực nghiệm](#4-thiết-kế-thực-nghiệm) · [5. Đánh giá và thống kê](#5-đánh-giá-và-phân-tích-thống-kê) ·
[6. Tính hợp lệ](#6-tính-hợp-lệ-rò-rỉ-shortcut-và-các-mối-đe-doạ) · [7. Hạn chế](#7-hạn-chế-đã-biết)

**Phần II — Thực hành:** [8. Cách chạy](#8-cách-chạy) · [9. Kiểm tra sau mỗi bước](#9-kiểm-tra-sau-mỗi-bước) ·
[10. Kết quả](#10-kết-quả-lưu-ở-đâu-và-đọc-thế-nào) · [11. Quy tắc báo cáo](#11-quy-tắc-báo-cáo) ·
[12. Bảng tham số](#12-bảng-tham-số)

---

# Phần I — Nghiên cứu

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
Ngân sách        g = n_maj − n_min   ảnh sinh lớp thiểu số (cân bằng 1 : 1); ROS (B2) dùng g bản sao ảnh thật.
Pool ứng viên    P = { G(z_j, min) : j = 1..N },  N = ⌈k · g⌉,  k = pool_mult = 2.
Bài toán chọn    Tìm S ⊂ P, |S| = g, sao cho classifier f train trên D ∪ S
                 đạt hiệu năng cao nhất trên tập test thật T.
```

Không thể tối ưu trực tiếp hiệu năng trên T, vì T không được phép dùng trong lúc xây dựng mô hình. DASS thay mục tiêu
đó bằng một **tiêu chí đại diện** chỉ tính từ D. Theo tiêu chí này, ảnh sinh tốt là ảnh:

1. **gần ảnh thật cùng lớp**,
2. **xa ảnh thật của lớp kia** (ảnh sinh lớp thiểu số nằm rõ về phía lớp bệnh, ảnh sinh lớp đa số rõ về phía lớp
   bình thường),
3. **không trùng lặp** với những ảnh đã chọn.

Hai tiêu chí đầu được đo trong **hai không gian đặc trưng**: không gian thị giác chung (E_v, pretrain ImageNet) và
không gian nhận biết bệnh (E_d, học có giám sát trên ảnh thật).

### 1.3 Câu hỏi nghiên cứu và giả thuyết

| | Câu hỏi | So sánh | Giả thuyết |
|---|---|---|---|
| RQ1 | Chọn ảnh sinh bằng DASS có tốt hơn chọn ngẫu nhiên, khi mọi thứ khác giống hệt? | M6 vs M1 | **H1 (chính)** |
| RQ2 | Ảnh sinh chọn bằng DASS có tốt hơn nhân bản ảnh thật tới cùng mức cân bằng? | M6 vs B2 | **H2 (chính)** |
| RQ3 | Balanced GAN + DASS có tốt hơn không xử lý mất cân bằng? | M6 vs B0 | **H3 (chính)** |
| RQ4 | So với class weight và augmentation truyền thống? | M6 vs B1, B3 | (khám phá) |
| RQ5 | Mỗi thành phần DASS đóng góp gì? | M2, M3 vs M5; M5 vs M6; M4 vs M6 | (khám phá) |
| RQ6 | Ảnh sinh chỉ ở lớp bệnh có tạo đường tắt không? | shortcut probe | (mô tả, bắt buộc) |

H1–H3 khai báo trước khi train (2.0.0, `evaluation.primary_comparisons`), Holm riêng trong họ chính.

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

| | ISIC 2016 | RSNA Pneumonia | PCam |
|---|---|---|---|
| g = n_đa số − n_thiểu số (ảnh thêm vào lớp thiểu số) | 470 | 6.463 | 33.022 |
| Pool ⌈k · g⌉ (k = 2) | 940 | 12.926 | 66.044 |
| Mỗi lớp trong B2 / Balanced GAN | 618 | 7.522 | 36.024 |
| Tỉ lệ ảnh sinh trong lớp thiểu số (Balanced GAN) | 76 % | 86 % | 92 % |

- k chỉ quyết định **độ chọn lọc**: mỗi biến thể giữ 1/k = 50 % pool. k đã đổi 4 → 3 → 2 → 1,5 → 2 (2.0.0); phải nêu
  trong bài. Ghi nhận cũ: chọn càng gắt thì tập chọn càng lệch phân phối ảnh thật.
- Mốc khi đọc Jaccard: hai tập con ngẫu nhiên độc lập, mỗi tập 1/k pool, có Jaccard kỳ vọng 1/(2k − 1) = **1/3** với k = 2.

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
  sang PNG và ghi `checkpoints_<run_tag>/data/dicom_metadata.csv` (tư thế chụp, giới tính, tuổi). Đổi thiết lập nguồn
  trong cùng runtime (ví dụ `subset_size`) thì ảnh cục bộ được đồng bộ lại; kết quả của run_tag cũ được bảo vệ bởi
  split đã lưu trên Drive (khác split -> báo lỗi, phải đổi run_tag).
- **Nhiễu nhãn (nêu trong bài):** khoảng 11 % ảnh âm từng được một bác sĩ khoanh "Lung Opacity (… Prob)" và khoảng 12 %
  ảnh dương từng được một bác sĩ ghi "No Lung Opacity / Not Normal"; dùng nhãn cuối cùng sau hội chẩn.
- **Vị trí trên Drive:** dữ liệu gốc ở `ColabData/RSNA Pneumonia` (`source.train_images`, đường dẫn tuyệt đối); kết
  quả ở `ColabData/RSNA Pneumonia/Result_Pneumonia` (`paths.drive_root`). Khi tìm dữ liệu, các thư mục kết quả
  (`checkpoints_*`, `results_*`) được bỏ qua ở mọi cấp; nếu thư mục dữ liệu rỗng mà có đúng một bản `RSNA Pneumonia (1)`
  chứa dữ liệu thì tự đọc từ bản đó.
- **Shortcut cần kiểm tra:** bệnh nhân nặng thường chụp tư thế AP. Log của `prepare` in tỉ lệ AP / PA theo lớp; nên báo
  cáo trong bài. Trên toàn bộ dữ liệu, AP chiếm 57,6 % ảnh dương nhưng chỉ 23,0 % ảnh âm, tức đoán theo tư thế đã
  đạt AUC khoảng 0,67 ((0,576 + 0,770) / 2).
- **Xem nhãn theo ảnh:** `dass -c configs/experiments/rsna_pneumonia_dass.yaml label-stats` in, cho từng tên nhãn trong
  JSON MD.ai, số chú thích, số ảnh, % trên tổng số ảnh, số ảnh theo lớp cuối cùng, số bệnh nhân; `--focus Exclude Flag`
  in thêm các nhãn đi kèm; `--samples 4` vẽ 4 ảnh ngẫu nhiên mỗi nhãn (khung đỏ = vùng đám mờ, dưới ảnh: lớp · tư thế
  chụp) vào `results_<run_tag>/label_samples.png`; cột "bị loại" = số ảnh bị `exclude_labels` loại. Không đổi dữ liệu
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
                        [3] sample ──► pool ảnh sinh lớp thiểu số + pool lớp đa số (⌈k·s⌉ ảnh mỗi lớp)
                              ▼
                 ( [3b] fingerprint — tuỳ chọn: đo dấu vết tần số của ảnh sinh )
                              ▼
                        [4] select ──► E_v, E_d → điểm lề + đa dạng → mỗi biến thể M1…M6 chọn s ảnh MỖI lớp
                              ▼
                        [5] train ──► backbone × 7 phương pháp × 3 seed (val chọn epoch, test dự đoán 1 lần)
                                       tập train M1–M6 = ảnh thật + ROS + s ảnh sinh mỗi lớp
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

- Sinh ảnh từ `best.pkl` với `truncation ψ = 1` (không cắt bớt độ đa dạng), seed cố định nên pool tái lập được. Pool
  được nén zip lên Drive.
  - Chỉ lớp thiểu số: seed 777, ⌈k · g⌉ ảnh (k = 2).
- Ảnh sinh lưu đúng số kênh của bộ dữ liệu: ISIC, PCam 3 kênh; RSNA 1 kênh (pool có hậu tố `_c1`).
- **Mọi biến thể chọn từ cùng pool**, nên khác biệt giữa các biến thể chỉ đến từ tiêu chí chọn.

### 3.4 Hai không gian đặc trưng

| | E_v (thị giác) | E_d (nhận biết bệnh) |
|---|---|---|
| Kiến trúc | EfficientNet-B0 (pooling trung bình, 1.280 chiều) | DenseNet-121 (pooling trung bình, 1.024 chiều) |
| Trọng số | ImageNet, **đóng băng** | ImageNet, rồi **train có giám sát** |
| Dữ liệu train | — | ảnh **thật** của tập train, nhãn thật, có class weight; val chỉ dùng để chọn epoch |
| Lịch train | — | 5 epoch head (lr 1e-3), rồi 20 epoch fine-tune (lr 1e-5); seed 4242 |
| Vai trò | ảnh có "trông giống" lớp thiểu số về mặt thị giác không | ảnh có mang đặc trưng phân biệt bệnh không |

- Mọi vector nhúng được chuẩn hoá L2, nên tích vô hướng chính là cosine.
- E_d là một mô hình **riêng** (seed riêng), cố ý khác classifier B0. Nếu chọn ảnh bằng chính B0 rồi so với B0, việc chọn
  chỉ củng cố ranh giới của B0.
- E_d **không thấy test, không thấy ảnh sinh**. E_d có dùng val để chọn epoch, giống như mọi classifier.

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
lặp đến khi |S| = s:
    với mọi x ∈ P \ S:   score(x) = b(x) + γ · S̃_div(x | S)        S̃_div: min-max trên P \ S, tính lại MỖI vòng
    S ← S ∪ { argmax score }                                        γ = 0,5
```

- γ cân bằng giữa **chất lượng** (điểm nền) và **độ phủ** (đa dạng). Khi γ = 0, thuật toán thành top-n theo b(x)
  (chính là M4).
- S_div được chuẩn hoá lại mỗi vòng vì giá trị của nó giảm dần khi S lớn lên. Nếu không chuẩn hoá lại, ảnh hưởng của
  đa dạng sẽ yếu dần.
- Độ phức tạp O(n · N) phép tích vô hướng. Mỗi vòng chỉ cập nhật khoảng cách tới ảnh vừa được chọn.

### 3.7 Các phương pháp so sánh

Xem bảng ở đầu tài liệu. M1–M6 chọn **đúng g ảnh sinh lớp thiểu số từ cùng một pool**; lớp đa số chỉ có ảnh thật.

| ID | Tiêu chí chọn | Đa dạng | Cô lập yếu tố gì |
|---|---|---|---|
| M1 | ngẫu nhiên (seed 2026) | – | không chọn lọc — đối chứng của **H1** |
| M2 | M_v | – | chỉ không gian thị giác |
| M3 | M_d | – | chỉ không gian bệnh |
| M4 | 0 (k-center greedy, γ = 1) | ✓ | chỉ đa dạng / coverage |
| M5 | α·M̃_v + β·M̃_d | – | hai không gian, chưa có đa dạng |
| **M6** | **α·M̃_v + β·M̃_d** | **γ = 0,5** | **DASS đầy đủ** |

**ROS (B2)** (`data.variants.oversample_indices`, seed 2026): mỗi ảnh lớp thiểu số được lặp ⌊g / n_min⌋ lần, và
g mod n_min ảnh (chọn ngẫu nhiên, không lặp) được lặp thêm một lần.

### 3.8 Classifier downstream

**Tập train:** xem bảng ở đầu tài liệu. Val và test giữ nguyên, 100 % ảnh thật.

| | ISIC | RSNA | PCam |
|---|---|---|---|
| B0, B1, B3 (đa số : thiểu số) | 618 : 148 | 7.522 : 1.059 | 36.024 : 3.002 |
| B2, M1–M6 | 618 : 618 | 7.522 : 7.522 | 36.024 : 36.024 |

- **Augmentation chỉ ở B3, class weight chỉ ở B1** (`Variant.augment`, `Variant.class_weight`, ghi trong `.npz`).
- B0 / B3 mất cân bằng nên ở ngưỡng 0,5 sensitivity thấp; so sánh chính dựa vào AUC.
- M1–M6 giống hệt nhau về ảnh thật, số ảnh và số bước train; chỉ khác **ảnh sinh được chọn**, nên **M6 vs M1 (H1)**
  là phép so sánh chặt chẽ nhất cho đóng góp của DASS. **M6 vs B2 (H2)** so ảnh sinh đã chọn với ảnh thật nhân bản ở
  cùng mức cân bằng.
- `.npz` còn ghi tiêu chí chọn epoch (`monitor`); trộn hai giá
  trị khác nhau trong cùng thư mục dự đoán -> báo lỗi. Ngoài ra `.npz` lưu dự đoán trên ảnh **thật** của tập train
  (`y_train`, `p_train`) cho bảng `auc_by_split`.
- Mỗi `.npz` ghi giao thức train (augment, class_weight). Gặp `.npz` của giao thức khác, `train` dừng báo lỗi;
  `train --archive-mismatched` chuyển nó (và trọng số) sang `predictions_superseded/<giao thức>/` (không xoá) rồi train
  lại, nên thư mục `predictions/` luôn chỉ có một giao thức.

**Backbone (pretrain ImageNet):**

| Backbone | Họ | Nguồn | Đầu ra |
|---|---|---|---|
| EfficientNetV2B0, ResNet50, DenseNet121, ConvNeXtTiny | CNN | `keras.applications` | pooling trung bình → Dropout 0,3 → Dense(1, sigmoid) |
| ViT-B16, SwinT | Transformer | KerasHub (`classifier.vit_preset`, `classifier.swin_preset` = `swin_tiny_patch4_window7_224`) | đầu phân loại của preset, 1 đầu ra sigmoid |

**Augmentation** của classifier: **tắt** ở cấu hình hiện tại (cả hai bộ dữ liệu). Khi bật (`classifier.augment: true`):
chỉ tập train, giống hệt nhau cho mọi biến thể; profile `rotation_invariant` = lật
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
- Theo dõi **`val_auc`** (ROC-AUC trên val; từ 1.12.0, người dùng quyết định). Lưu checkpoint tốt nhất; dừng sớm sau
  8 epoch không cải thiện. Áp dụng cho cả classifier lẫn E_d. Lý do: khớp chỉ số chính (AUC); không phụ thuộc ngưỡng
  nên không thiên vị giữa phương pháp mất cân bằng và cân bằng; ít nhiễu hơn với val nhỏ (ISIC 25 ca dương). Tiêu chí
  này được chọn sau khi đã xem kết quả test của lần chạy trước (dùng `val_macro_recall`) -> phải nêu trong bài. Tiêu
  chí được ghi vào `.npz` (`monitor`); trộn hai tiêu chí trong cùng thư mục dự đoán -> báo lỗi.
- Nạp lại checkpoint tốt nhất, rồi **dự đoán test đúng một lần**. Xác suất val và test được lưu vào
  `predictions/<model>__<biến thể>__s<seed>.npz`. Mọi bảng phân loại đều tính từ các file này.

---

## 4. Thiết kế thực nghiệm

### 4.1 Biến số

| Loại | Biến số |
|---|---|
| **Biến độc lập** | phương pháp: B0–B3, M1–M6 |
| **Yếu tố khối** | backbone (ResNet50, ConvNeXt-Tiny, Swin-T), bộ dữ liệu (3) |
| **Lặp lại** | seed classifier: 2026, 2027, 2028 |
| **Biến phụ thuộc** | metric phân loại trên test; shortcut probe; KID và chỉ số chẩn đoán của tập ảnh được chọn |
| **Biến được kiểm soát** | cùng split, GAN, pool, siêu tham số train, seed (thiết kế **ghép cặp**), ngưỡng 0,5, val / test |

### 4.2 Ma trận thực nghiệm

| | Mỗi bộ dữ liệu | Ba bộ dữ liệu |
|---|---|---|
| Lần train classifier | 10 phương pháp × 3 backbone × 3 seed = **90** | **270** |
| Lần train E_d | 1 (PCam: dùng lại E_d đã chọn GAN) | 2 |
| GAN | ISIC `v5`, RSNA `rsna` dùng lại; PCam `pcam` train mới (128 px) | 1 |

### 4.3 Siêu tham số được cố định trước

- Tham số DASS (K = 5, λ = 1, α = β = 1, γ = 0,5) lấy theo công thức của notebook ISIC v5. Lịch train E_d và mọi siêu
  tham số classifier được **dùng chung cho mọi bộ dữ liệu và mọi backbone**.
- k = 2 (2.0.0, xem [mục 2.4](#24-ngân-sách-sinh-và-ý-nghĩa-của-k)); augmentation chỉ ở B3, class weight chỉ ở B1. Trọng
  số DASS (α : β : γ = 0,4 : 0,4 : 0,2 sau chuẩn hoá) được giữ sau ablation chỉ trên val (1.20.0: không bộ trọng số nào
  hơn quá 0,01).
- **Lưu ý:** k, augmentation, class weight, tiêu chí chọn epoch và thiết kế tập train đã được đổi qua nhiều phiên bản
  **sau khi xem kết quả test** của các lần chạy trước. Đây không còn là thiết lập được chốt trước; phải báo cáo minh bạch mọi cấu hình đã chạy.
- **Không có tham số nào được dò trên test.** Val chỉ dùng để chọn epoch.
- Không dò siêu tham số riêng cho từng backbone. Cách này công bằng giữa các phương pháp, nhưng có thể chưa tối ưu cho
  từng backbone (xem [mục 7](#7-hạn-chế-đã-biết)).

### 4.4 Seed và khả năng tái lập

| Seed | Dùng cho |
|---|---|
| 2026 (`seed` toàn cục) | chia dữ liệu, tập ngẫu nhiên của M1 (lớp thiểu số; lớp đa số dùng 2027), ảnh nhân bản ROS, bootstrap, cross-validation của probe / shortcut, tập con KID và cặp SSIM khi đánh giá |
| 777 (`generator.gen_seed`) | sinh pool lớp thiểu số (778 = gen_seed + 1 cho pool lớp đa số) |
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
  - **mọi phương pháp so với B0** (`evaluation.baseline_method`);
  - cộng các cặp trong `evaluation.comparisons`: **M6 vs M1** (H1) và **M6 vs B2** (H2); H3 (M6 vs B0) đã nằm trong
    phép so với baseline.

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
  - **Đa so sánh (từ 1.14.0):** giả thuyết chính khai báo trước trong `evaluation.primary_comparisons` (H1 M6 vs
    M1, H2 M6 vs B2, H3 M6 vs B0) là **họ chính**; mọi so sánh khác là **họ khám phá**. Cột `family` / **Family**; **`p (Holm)`** hiệu chỉnh Holm–Bonferroni **riêng trong từng họ** của từng
    backbone. Kết luận chính dựa vào `p (Holm)` của họ chính; kết quả khám phá chỉ để gợi ý. Trước 1.14.0 mọi so sánh
    chung một họ nên quá bảo thủ. Nêu trong bài: hiệu chỉnh theo từng backbone,
    tách họ chính / khám phá; họ chính H1–H3 được khai báo trước khi train (2.0.0).
  - Bootstrap dùng xác suất **trung bình qua seed** -> chỉ đo dao động do tập test; dao động do train chỉ có ở cột
    ΔAUC theo seed (3 seed: yếu).
  - **AUC train / val / test** (bảng `auc_by_split`): AUC trên ảnh **thật** của tập train (không gồm ảnh nhân bản /
    ảnh sinh), val, test và hiệu train − test — đo mức overfit của từng phương pháp.
  - So sánh M6 với M2–M5 (RQ4) **chưa có kiểm định chính thức** (chỉ đọc từ mean ± std). Có thể thêm bằng cách bổ
    sung cặp vào `evaluation.comparisons`, không cần train lại.

### 5.3 Chất lượng ảnh sinh (Inception-v3 của StyleGAN2-ADA, 2.048 chiều)

Tính **theo từng lớp có ảnh sinh** (cột `class`; cả hai lớp), mỗi lớp so với **ảnh thật cùng lớp của
tập train** (lớp thiểu số: ISIC 148, RSNA 1.059 ảnh; lớp đa số: ISIC 618, RSNA 7.522 ảnh), cho từng tập ảnh được
chọn. AUC thật-vs-sinh chỉ có cho lớp thiểu số (tính bằng E_v ở bước `select`); đường tắt ở lớp đa số đo bằng
`scripts/shortcut_probe.py`.

| Chỉ số | Định nghĩa | Tốt khi |
|---|---|---|
| **KID** (chỉ số chính) | MMD² không chệch với kernel đa thức k(x, y) = (xᵀy / 2048 + 1)³; 50 tập con, mỗi tập m = min(n_thật, n_sinh, 1.000) ảnh; báo cáo mean ± std | thấp |
| **Học thuộc**: `nn_dist_ratio`, `near_copy_rate` | khoảng cách cosine (Inception-v3) từ mỗi ảnh tới ảnh train lớp thiểu số gần nhất. `nn_dist_ratio` = trung vị của tập ảnh / trung vị của ảnh **val** thật (ảnh thật chưa thấy); `near_copy_rate` = tỉ lệ ảnh gần ảnh train hơn phân vị 5 % của val. Hàng `real val vs real train` là mốc (= 1 và ≈ 0,05 theo định nghĩa). Val ở đây chỉ dùng để **báo cáo**, không chọn gì | ≈ mốc; ratio < 1 rõ rệt / near-copy cao -> ảnh sinh bám sát ảnh train. Val lớp thiểu số nhỏ (ISIC) -> phân vị 5 % thô |
| FID (tham khảo) | khoảng cách Fréchet giữa hai phân phối Gauss | thấp; **chệch mạnh khi số ảnh < 2.048** nên chỉ tham khảo |
| Đa dạng | 1 − cosine trung bình giữa các cặp ảnh trong tập | cao |
| SSIM nội bộ | SSIM trung bình của 200 cặp ảnh ngẫu nhiên (cửa sổ Gauss σ = 1,5) | thấp (ít trùng lặp) |
| AUC thật-vs-sinh | hồi quy logistic (chuẩn hoá, C = 0,1, cân bằng lớp), CV phân tầng 5 fold, trên E_v: ảnh thật lớp thiểu số vs tập được chọn | gần 0,5 (khó phân biệt) |

Có hai hàng tham chiếu:
- **`real val vs real train`:** mức nền của metric giữa hai tập ảnh **thật**. Val nhỏ (25 / 59 ảnh), nên m nhỏ và std
  của hàng này lớn.
- **`all candidates`:** đúng phần pool đã dùng để chọn (⌈k·s⌉ ảnh đầu), chưa lọc. So một tập được chọn với hàng này để thấy việc chọn lọc làm KID tăng
  hay giảm.

### 5.4 Chẩn đoán đi kèm

| Chẩn đoán | Đo gì | Bước |
|---|---|---|
| **Disease margin của tập chọn** (`selection_quality`) | M_d (E_d) của từng ảnh GAN được chọn, tổng hợp theo tập: trung bình, trung vị, tỉ lệ ảnh có M_d > 0 (gần ảnh bệnh thật hơn ảnh lớp đa số thật); kèm M_v | `evaluate` |
| **Đa dạng của tập chọn** (`selection_quality`, `generation_quality`) | 1 − cosine trung bình mọi cặp và khoảng cách tới ảnh gần nhất trong tập, trong E_v (đúng không gian của S_div); bản Inception-v3 ở `generation_quality` | `evaluate` |
| **Coverage / density** (`selection_quality`, `generation_quality`) | Naeem và cộng sự (ICML 2020), k = 5: coverage = tỉ lệ ảnh bệnh thật (train) có ít nhất một ảnh GAN trong bán kính k láng giềng gần nhất của nó (tập chọn phủ phân phối ảnh bệnh thật đến đâu); density ≈ 1 khi ảnh GAN phân bố như ảnh thật, > 1 khi dồn vào vùng dày. Trong E_d, E_v và Inception-v3. Chỉ so giữa các phương pháp CÙNG số ảnh (M1–M6) | `evaluate` |
| AUC probe | hai lớp bệnh tách nhau đến đâu trong E_v và E_d (logistic, CV 5 fold). Dòng **val** gần khách quan (E_d không học trên val, chỉ dùng val để chọn epoch); dòng train thì lạc quan | `select` |
| Kiểm tra shortcut | AUC tách ảnh thật / ảnh sinh trong E_v cho từng biến thể; hàng tham chiếu là AUC tách hai lớp bệnh thật | `select` |
| Jaccard giữa các biến thể | các tiêu chí có thật sự chọn ra các tập khác nhau không (mốc ngẫu nhiên 1/3 khi k = 2) | `select` |
| Hình ảnh | phân tán M_v–M_d, ảnh được chọn và bị loại, ảnh sinh cạnh ảnh thật gần nhất (kiểm tra GAN có "học thuộc" không) | `select` |
| Dấu vân tay tần số (Frank et al., ICML 2020) | ảnh xám → DCT 2 chiều → log → chuẩn hoá → CNN nông và hồi quy; ≤ 400 ảnh mỗi phía, 15 epoch. AUC ≈ 1 nghĩa là ảnh sinh mang dấu vết GAN. **Chỉ đo, không sửa ảnh** | `fingerprint` (tuỳ chọn) |

### 5.5 Phân tích theo nhóm con (RSNA: tư thế chụp AP / PA)

Bật bằng `evaluation.subgroup_columns` (RSNA: `[ViewPosition]`, lấy từ `dicom_metadata.csv`; ISIC không có). **Đăng ký trước** (2026-10-04, trước khi có kết quả test RSNA): báo cáo dù kết quả thế nào. Chạy trong
`evaluate`, không cần GPU, không đổi cách train.

| Bảng | Đo gì | Cách đọc |
|---|---|---|
| `classification_by_subgroup` | AUC, PR-AUC (mean ± std qua seed) trong từng nhóm AP, PA | Trong một nhóm, mọi ảnh cùng tư thế nên tư thế không giúp xếp hạng: đo khả năng nhận ra bệnh tách khỏi shortcut |
| `significance_by_subgroup` | ΔAUC vs B0 và các cặp bổ sung **trong từng nhóm**, p Holm trong mỗi model × nhóm | DASS hơn đối chứng ở **cả hai** nhóm -> lợi thế đến từ bệnh; chỉ hơn ở AUC toàn bộ -> nghi shortcut |
| `subgroup_reference` | AUC khi dự đoán nhãn **chỉ bằng tư thế**, tỉ lệ AP trong lớp dương / âm của test | Mức "ăn điểm nhờ shortcut" (RSNA: khoảng 0,67) |
| `subgroup_share` | Tỉ lệ ảnh "trông giống AP" trong ảnh dương thật, toàn pool và ảnh mỗi phương pháp chọn (probe logistic trên E_v của ảnh thật train, không class weight; ảnh thật dùng dự đoán ngoài fold) | Pool cao hơn ảnh dương thật -> GAN khuếch đại tư thế; ảnh DASS chọn cao hơn pool -> cách chọn khuếch đại |

Tên ảnh test lưu trong `.npz` (`test_files`, từ 1.9.0). `.npz` cũ (train trước 1.9.0)
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
| So sánh công bằng | cùng pool, cùng s, cùng ảnh nhân bản ROS, cùng seed, cùng quy trình train, không augmentation / class weight cho mọi biến thể; M1–M6 có cùng số ảnh, số bước train và thành phần (thật / nhân bản / sinh) ở mỗi lớp |

### 6.2 Các mối đe doạ đến tính hợp lệ

| Loại | Mối đe doạ | Cách xử lý hoặc đo |
|---|---|---|
| Nội tại | Ảnh sinh mang dấu vết GAN (AUC thật-vs-sinh ≈ 0,99). Nếu ảnh sinh **chỉ** nằm ở lớp thiểu số, dấu vết này là manh mối của nhãn | Thiết kế tập train đặt cùng số ảnh sinh (1/3) vào mỗi lớp nên dấu vết GAN không còn tương quan với nhãn. Còn lại: dấu vết GAN của hai lớp có thể khác nhau; `scripts/shortcut_probe.py` (chỉ val) chấm ảnh sinh lớp đa số giữ riêng bằng classifier đã train để đo trực tiếp |
| Nội tại | **Shortcut tư thế chụp (RSNA):** AP gắn với bệnh nặng (dương ≈ 58 % AP, âm ≈ 23 % AP) | Đo, không sửa dữ liệu: AUC trong từng nhóm AP / PA, mốc chỉ dùng tư thế, tỉ lệ "trông giống AP" của ảnh sinh (mục 5.5) |
| Nội tại | **Đường tắt nguồn ảnh:** ảnh sinh chỉ ở lớp thiểu số (76–92 %), classifier có thể học "trông giống GAN → bệnh" | Shortcut probe cho mọi phương pháp (bắt buộc), báo cáo trong bài |
| Nội tại | Cấu hình (k, augmentation, class weight, tiêu chí chọn epoch, **thiết kế tập train**) được đổi sau khi xem kết quả test; Brain Tumor bị loại sau khi thấy AUC chạm trần | Báo cáo mọi cấu hình đã chạy và nêu lý do loại Brain Tumor |
| Nội tại | M1 chỉ có **một** lần rút ngẫu nhiên (seed 2026 / 2027); ROS cũng chỉ một lần chọn ảnh nhân bản thêm | Độ biến thiên do việc rút ngẫu nhiên không nằm trong std; nêu rõ khi so M6 với M1 (H1) |
| Nội tại | Backbone DenseNet121 có **cùng kiến trúc** với E_d (khác seed, khác lần train) | Diễn giải riêng hàng DenseNet121; so sánh với 5 backbone còn lại |
| Nội tại | KID dùng trong early stopping của GAN so với chính ảnh train mà GAN đã thấy (thưởng cả việc học thuộc); KID của snapshot tốt nhất vì thế hơi lạc quan | `nn_dist_ratio` / `near_copy_rate` trong `generation_quality` + hình ảnh sinh cạnh ảnh thật gần nhất; so với hàng tham chiếu "real val vs real train" |
| Nội tại | KID dao động lớn khi ít ảnh thật (ISIC 148 ảnh): ngưỡng cải thiện 2 % có thể nằm trong nhiễu, snapshot "tốt nhất" có thể do may mắn | Patience 5 + `min_kimg` 400; xem `kid_history.png` — trên đoạn phẳng các snapshot tương đương, không diễn giải số kimg tốt nhất |
| Cấu trúc | Inception-v3 train trên ImageNet, xa miền ảnh y tế (nhất là X-quang): KID thấp ≠ giống về dấu hiệu bệnh | Nêu trong phần hạn chế; thước đo cuối cùng là kết quả classifier (M1 vs B0, M6 vs M1) |
| Nội tại | GPU không tất định | Báo cáo từ file `.npz` đã lưu; 3 seed |
| Cấu trúc | Metric tại ngưỡng 0,5 phụ thuộc tỉ lệ lớp lúc train: B0, B3 (mất cân bằng) có xác suất lệch về lớp đa số, nên ở ngưỡng 0,5 sensitivity thấp là do thiết kế, không phải do mô hình kém hơn về khả năng xếp hạng | Chỉ số chính là ROC-AUC (không phụ thuộc ngưỡng); báo cáo thêm balanced accuracy, G-mean, MCC |
| Cấu trúc | Tập thật lớp thiểu số nhỏ ở ISIC (148 ảnh) | KID là chỉ số chính, kèm std; FID chỉ tham khảo. Mỗi tập con KID dùng TOÀN BỘ 148 ảnh thật -> `kid_std` chỉ phản ánh dao động của ảnh sinh, đánh giá thấp độ bất định thật |
| Thiết kế | Trọng số thật của đa dạng trong S_DASS: nền α·M̃_v + β·M̃_d ∈ [0, 2], γ·S̃_div ∈ [0, 0,5] -> đa dạng ≈ 1/5 biên độ (không phải 50 %); S̃_div min-max lại mỗi vòng nên ở vòng cuối khác biệt rất nhỏ bị kéo giãn | Nêu rõ trong bài |
| Thiết kế | M_d tính với ảnh train mà E_d đã học (probe E_d: train ≈ 0,9, val ≈ 0,78) -> tham chiếu bị học thuộc | Tuỳ chọn `encoder.e_d_crossfit_folds` (K ≥ 2, 1.14.0): mỗi E_d so với ảnh nó không học |
| Thống kê | Chỉ 3 seed; test ISIC nhỏ (75 ca dương; RSNA 226); nhiều so sánh khám phá mỗi bộ dữ liệu; trên val (1.17) hiệu ứng H1 chỉ khoảng +0,004 AUC, nhỏ so với CI test (RSNA ≈ ±0,011) | [Mục 5.2](#52-phân-tích-thống-kê): báo cáo cả std lẫn CI; p Holm trong từng backbone |
| Ngoại tại | Hai bộ dữ liệu, chỉ bài toán nhị phân, một họ GAN, ảnh 256 px | Nêu trong phần hạn chế |
| Ngoại tại | RSNA: nhãn "Lung Opacity" là nhận định trên ảnh (không phải chẩn đoán viêm phổi xác nhận lâm sàng), có bất đồng giữa bác sĩ khoảng 11–12 % | Dùng nhãn cuối cùng sau hội chẩn; nêu trong phần hạn chế |

---

## 7. Hạn chế đã biết

Các điểm sau nên nêu trong bài báo:

- **Chọn epoch:** chỉ các epoch của giai đoạn fine-tune được xét (mô hình chỉ train head không là ứng viên); val AUC
  lúc train dùng `tf.keras.metrics.AUC` (xấp xỉ 200 ngưỡng); val nhỏ (ISIC 25 ca dương) nên val AUC của epoch được
  chọn lạc quan hơn test (ISIC `v14_srv`, EfficientNetV2B0: val ≈ 0,83, test ≈ 0,78). ResNet50 ISIC chọn epoch 1 ở
  20 / 21 lần chạy. Đường val qua từng epoch lưu trong `.npz` (`hist_*`) từ 1.21.1.
- **Dấu vân tay tần số của GAN:** detector của Frank et al. phân biệt ảnh thật / ảnh sinh với AUC ≈ 1. Test toàn ảnh
  thật nên điều này không làm kết quả lạc quan giả tạo, nhưng có thể làm giảm lợi ích của ảnh sinh.
- **Shortcut tư thế chụp ở RSNA** (AP gắn với bệnh nặng): không sửa dữ liệu, chỉ đo bằng phân tích nhóm con (mục 5.5).
- **Classifier không augmentation** (cả hai bộ dữ liệu): AUC tuyệt đối có thể thấp hơn các công trình dùng
  augmentation; kết luận của bài là về **chênh lệch giữa các phương pháp** trong cùng giao thức.
- **Chỉ hỗ trợ bài toán nhị phân.** Bài toán đa lớp cần mở rộng DASS theo kiểu one-vs-rest.
- **Siêu tham số chung**, không dò riêng cho từng backbone; Transformer có thể chưa đạt mức tối ưu.
- **Kiểm định chính thức** có cho H1 (M6 vs M1), H2 (M6 vs B2), H3 (M6 vs B0) và mọi phương pháp so với B0, kèm p hiệu chỉnh Holm theo
  họ trong từng backbone. Các so sánh thành phần (M6 với M2–M5) chưa có kiểm định riêng (thêm được qua
  `evaluation.comparisons`).
- **Không tách "nội dung ảnh sinh" với "thêm dữ liệu"** (không có baseline chỉ ROS).

---

# Phần II — Thực hành

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
| Tiện ích | định nghĩa `show()`, `table()`, `results_dir()` | không lỗi |
| `prepare` | dữ liệu, split | [mục 9](#9-kiểm-tra-sau-mỗi-bước) |
| `gan-setup` | clone + vá StyleGAN2-ADA, biên dịch plugin CUDA | chỉ cần chạy lại khi là **server mới** |
| `gan` | dùng lại GAN, vẽ đường KID | |
| `sample` | sinh pool | `DONE <N> …` |
| `fingerprint` | (tuỳ chọn) dấu vân tay tần số | bảng AUC |
| `select` | E_v, E_d, chọn ảnh mỗi lớp, chẩn đoán | `M6_dass: chọn s / ⌈k·s⌉ ảnh sinh` (hai dòng: lớp thiểu số, lớp đa số) |
| 6 ô `train` | mỗi ô một backbone (7 phương pháp × 3 seed) | mỗi lần chạy in `test ROC-AUC = …` |
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
| Pool | k = 1,2 | k = 2 |
| E_d | 1 + 1 epoch | 5 + 20 epoch |
| Classifier | 1 backbone, 1 seed, 1 + 1 epoch | ResNet50, ConvNeXt-Tiny, Swin-T, 3 seed, 5 + 30 epoch |
| Thư mục | `checkpoints_smoke/`, `results_smoke/` | `checkpoints_<run_tag>/`, `results_<run_tag>/` |

**Chuyển từ smoke sang chạy thật:** không cần tải lại notebook hay khởi động lại kernel.
1. Sửa `PROFILE = ""`, chạy lại ô chọn thực nghiệm. Kiểm tra cấu hình in ra: RSNA `"run_tag": "rsna_v5"`,
   ISIC `"run_tag": "v14"`; cả hai `"pool_mult": 2.0`, `"monitor": "val_auc"`.
2. Ô tiện ích đọc lại thư mục kết quả mỗi lần gọi (`results_dir()`), nên không bắt buộc chạy lại.
3. Chạy tiếp từ `prepare` trở xuống.

### 8.4 Thứ tự chạy và thời gian

1. **RSNA:** `prepare` → `gan` → `sample` → `select` → các ô `train` → `evaluate` + `report` (`fingerprint` tuỳ chọn).
   GAN `rsna` đã train xong (KID 0,0145 tại 800 kimg); `prepare` mất khoảng 30 phút trên server mới (chuyển DICOM).
   `sample` dùng lại pool lớp thiểu số và sinh thêm pool lớp đa số (5.642 ảnh); `select` train E_d mới (Colab) hoặc
   dùng lại E_d qua `encoder.e_d_from_run` (server: `rsna_v3_srv`).
2. **ISIC:** đổi `EXPERIMENT = "configs/experiments/isic2016_dass.yaml"`, chạy lại ô chọn thực nghiệm và ô tiện ích,
   rồi lặp lại đúng chuỗi trên.

Về thời gian:
- Bước `train` chiếm gần hết thời gian: 21 lần train cho mỗi backbone (tập train M1–M6 của RSNA có 22.566 ảnh), và
  Transformer chậm hơn CNN.
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
dass -c $E run                                           # toàn bộ chuỗi prepare → report (train: mỗi model × seed
                                                         # một tiến trình riêng, lần lượt)
dass -c $E run --from select --to report                 # một đoạn của chuỗi
dass -c $E train --model resnet50 --seeds 2026           # một backbone, MỘT seed (chạy lại với 2027, 2028)
dass -c $E evaluate --skip-generative                    # chỉ metric phân loại (không tính KID / FID)
dass -c $E -c configs/experiments/smoke.yaml run         # chạy thử
dass -c configs/server/rsna_pneumonia.yaml --tag srv run # server phòng lab: rsna_v5_srv (3 CNN, E_d của rsna_v3_srv)
dass -c configs/colab/rsna_pneumonia.yaml preflight --fix   # Colab: kiểm tra GAN / split trước khi chạy; --fix chép
                                                             # bản nằm trong thư mục trùng tên do Drive tạo về đúng chỗ
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
| `prepare` | `Thiểu số = positive (1059) \| đa số = negative (7522) \| cần chọn 6463 \| pool 9695 ảnh \| 1 kênh` (kích thước pool lớp thiểu số đã sinh; dòng tiếp theo in phần dùng để chọn); `Split trùng khớp split tham chiếu real_val_split.json` | `Thiểu số = malignant (148) \| đa số = benign (618) \| cần chọn 470 \| pool 705 ảnh \| 3 kênh` | sai cấu hình / `PROFILE` |
| `gan` | dùng `checkpoints_rsna` (KID 0,0145 tại 800 kimg), không train lại | dùng `checkpoints_v5` (KID 0,0201 tại 1.800 kimg), không train lại | sai `gan_tag` |
| `sample` | pool `positive` 12926 ảnh | pool `malignant` 940 ảnh | sai `pool_mult` |
| `select` | `M6_dass: chọn 6463 / 12926 ảnh sinh` | `M6_dass: chọn 470 / 940 ảnh sinh` | |
| `select` (chẩn đoán) | AUC probe trên val: kỳ vọng E_d ≥ E_v. Tương quan M_v–M_d càng gần 1 thì E_d càng ít thêm thông tin. Jaccard giữa M1 và các biến thể khác ≈ 1/3 | như bên trái | ghi lại, nêu trong bài |
| `train` | dòng đầu `Giao thức train … \| augmentation (…): ['B3_augment'] \| class weight: ['B1_class_weight']`; mỗi lần chạy in `test ROC-AUC = …` | như bên trái | sai phiên bản code / cấu hình |
| `train` (ghép dữ liệu) | `B0_imbalanced_k2: {'train': {'negative': 7522, 'positive': 1059}, ...}` và `M6_dass_k2: {'train': {'negative': 7522, 'positive': 7522}, ...}` | `M6_dass_k2: {'train': {'benign': 618, 'malignant': 618}, ...}` | sai thiết kế |
| `evaluate` | `Phân loại: 90 lần chạy, 30 nhóm` khi đủ 3 backbone; thêm các bảng nhóm con (tư thế AP / PA) | như bên trái (không có bảng nhóm con) | thiếu: còn backbone chưa xong |

---

## 10. Kết quả: lưu ở đâu và đọc thế nào

### 10.1 Vị trí trên Google Drive

Kết quả nằm trong thư mục của từng bộ dữ liệu. Không bao giờ xoá; chạy lại không ghi đè.

| | RSNA (`…/RSNA Pneumonia/Result_Pneumonia/`) | ISIC (`…/ISBI2016_ISIC_Part3/`) |
|---|---|---|
| GAN | `checkpoints_rsna/stylegan2ada/` | `checkpoints_v5/stylegan2ada/` |
| Split, pool, lựa chọn, E_d, trọng số classifier | `checkpoints_rsna_v5/` (split tham chiếu: `checkpoints_rsna_v2/data/real_val_split.json`; lựa chọn lớp đa số: `data/selections_majority.json`) | `checkpoints_v14/` |
| Dự đoán, số liệu thô, **bảng**, hình | `results_rsna_v5/` | `results_v14/` |
| Server phòng lab | `…/RSNA/results_rsna_v5_srv/` | `…/ISIC2016/results_v14_srv/` |
| Kết quả giao thức cũ (chuyển ra, không dùng trong bảng) | thư mục `*_superseded/` của lần chạy | thư mục `predictions_archived_*/` của lần chạy |
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
| `classification` | Hàng đã dùng tên hiển thị và chia nhóm (cột `Group`: Real Data Baseline / ROS + Class-balanced GAN). So DASS (Ours) với Random (H1), Imbalanced Baseline (H2) và các tiêu chí còn lại (ablation) **trong cùng một backbone**. Với dữ liệu mất cân bằng, ưu tiên ROC-AUC, PR-AUC, sensitivity, balanced accuracy, G-mean, MCC; không dựa vào accuracy |
| `significance` | Cột `vs` cho biết đối chứng (Imbalanced Baseline hoặc Random), cột `Family` cho biết họ chính / khám phá. ΔAUC > 0 và khoảng tin cậy 95 % không chứa 0 thì phương pháp hơn đối chứng có ý nghĩa thống kê; kết luận chính dựa vào `p (Holm)` của họ chính (mục 5.2). Hàng **DASS vs Random (H1)** là bằng chứng trực tiếp cho giá trị của cách chọn |
| `auc_by_split` | AUC train (ảnh thật) / val / test và hiệu train − test: hiệu lớn -> phương pháp overfit (ảnh nhân bản / ảnh sinh bị học thuộc) |
| Tính nhất quán | đếm số backbone (trên 6) mà M6 > M1, ở mỗi bộ dữ liệu; một kết quả tốt ở một backbone chưa đủ để kết luận |
| KID | thấp hơn là gần ảnh thật hơn; đọc tương đối so với hàng `real val vs real train` và `all candidates` |
| AUC thật-vs-sinh (`shortcut_check`) | gần 0,5 là khó phân biệt (tốt); gần 1 nghĩa là ảnh sinh dễ nhận ra, có nguy cơ shortcut |
| AUC probe (dòng val) | E_d > E_v nghĩa là không gian bệnh tách lớp tốt hơn, đúng mục đích thiết kế E_d |
| Jaccard | so với mốc ngẫu nhiên 1/3 (k = 2); thấp nghĩa là các tiêu chí chọn ra tập ảnh khác nhau |

---

## 11. Quy tắc báo cáo

**Nên:**
- Báo cáo **đủ mọi backbone × 7 phương pháp × 2 bộ dữ liệu**, kể cả những cấu hình mà M6 không thắng.
- Mean ± std qua 3 seed, kèm ΔAUC, khoảng tin cậy 95 % và p (Holm theo họ) của M6 so với M1 (H1), B2 (H2) và B0 (H3). Ghi rõ số seed và số lần
  bootstrap.
- KID là chỉ số chính về chất lượng ảnh sinh; FID kèm cảnh báo về độ chệch.
- Nêu AUC thật-vs-sinh và kết quả fingerprint như một hạn chế, cùng các mục ở [phần 7](#7-hạn-chế-đã-biết).
- Ghi phiên bản code (commit hash), `dass_version` và cấu hình (`run_manifest.json`).

**Không nên:**
- Chọn seed, backbone hay bộ dữ liệu "đẹp" sau khi đã xem kết quả.
- Dò ngưỡng, k, γ hay bất kỳ tham số nào trên test, hoặc chạy lại test.
- Trộn kết quả của các lần chạy khác giao thức (khác thiết kế tập train, tiêu chí chọn epoch, augmentation) trong
  cùng một bảng.

---

## 12. Bảng tham số

| Nhóm | Tham số | Giá trị | File |
|---|---|---|---|
| GAN | cfg / batch / R1 γ / mirror / ADA target | paper256 / 16 / 1.0 / có / 0.6 | `configs/_base_/generator.yaml` |
| GAN | max kimg / snapshot / min kimg / patience / Δ tối thiểu | 3000 / 100 / 400 / 5 / 2 % | |
| GAN | KID khi train: số ảnh / tập con / seed | 1000 / 50 / 123 | |
| Pool | k = `pool_mult` (chung mọi bộ dữ liệu) / ψ / seed | 2 / 1.0 / 777 | `configs/_base_/selection.yaml` |
| DASS | K / λ_v / λ_d / α / β / γ | 5 / 1 / 1 / 1 / 1 / 0.5 | |
| E_d | backbone / seed / epoch / dùng lại (`e_d_from_run`) | DenseNet121 / 4242 / 5 + 20 / "" (PCam server: `pcam_v1_srv`) | |
| Classifier | backbone | ResNet50, ConvNeXtTiny, SwinT (cùng head) | `configs/_base_/classifier.yaml` |
| Classifier | ảnh / batch / dropout (CNN) | 224 / 16 / 0.3 | |
| Classifier | epoch / lr / weight decay / early stop | 5 + 30 / 1e-3, 1e-5 / 1e-4 / 8 | |
| Classifier | seed / chọn epoch | 2026, 2027, 2028 / `val_auc` | |
| Classifier | augmentation / class weight | chỉ B3 / chỉ B1 | |
| Đánh giá | ngưỡng / bootstrap / KID (tập con, kích thước tối đa) / cặp SSIM | 0.5 / 2000 / 50, 1000 / 200 | `configs/_base_/evaluation.yaml` |
| Đánh giá | đối chứng / họ chính (`primary_comparisons`) | B0 / M6 vs M1, M6 vs B2, M6 vs B0 | |
| Đánh giá | tên hiển thị / hai bảng (`method_labels`, `result_tables`) | xem đầu tài liệu | |
| Chung | seed toàn cục / tất định | 2026 / false | `configs/_base_/runtime.yaml` |
| Dữ liệu | nguồn, tiền xử lý, cách chia, số kênh, thư mục Drive, `gan_tag`, `run_tag` | theo bộ dữ liệu | `configs/datasets/*.yaml` |
