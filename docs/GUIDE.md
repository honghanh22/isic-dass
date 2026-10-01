# Hướng dẫn chi tiết: bài toán, phương pháp, các bước và cách chạy

Tài liệu tổng quan cho người mới tiếp cận dự án. Chi tiết kỹ thuật: [ARCHITECTURE.md](ARCHITECTURE.md);
làm việc hằng ngày: [WORKFLOW.md](WORKFLOW.md); ý nghĩa từng cột kết quả: [RESULTS_FORMAT.md](RESULTS_FORMAT.md).

---

## 1. Bài toán

**Phân loại ảnh y tế nhị phân khi dữ liệu mất cân bằng lớp.** Lớp bệnh (lớp quan tâm) thường có rất ít ảnh so với lớp
bình thường, nên classifier dễ thiên về lớp đa số: độ chính xác tổng thể cao nhưng **bỏ sót ca bệnh** (sensitivity thấp).

| Bộ dữ liệu | Ảnh | Lớp đa số | Lớp thiểu số | Tỉ lệ |
|---|---|---|---|---|
| ISIC 2016 Part 3 (dermoscopy) | RGB | benign | malignant | ≈ 4,2 : 1 |
| Brain Tumor MRI | xám | negative (không u) | positive (có u) | 5 : 1 |

**Hướng giải quyết:** bù lớp thiểu số bằng **ảnh tổng hợp** do GAN sinh ra. Nhưng không phải ảnh sinh nào cũng có ích:
ảnh mờ, ảnh trông giống lớp đa số, hoặc ảnh lặp lại nhau có thể làm classifier tệ hơn. Vì vậy cần **chọn lọc**.

**Đóng góp chính — DASS:** từ một pool ảnh sinh dư thừa, chọn ra đúng số ảnh cần để cân bằng lớp, ưu tiên ảnh
1. **giống lớp thiểu số thật**,
2. **khác lớp đa số thật** (nằm rõ về phía lớp bệnh),
3. **đa dạng** (không trùng lặp nhau),

trong **hai không gian đặc trưng**: thị giác chung (E_v, ImageNet) và nhận biết bệnh (E_d, học có giám sát trên ảnh thật).

---

## 2. Dữ liệu và cách chia

| | ISIC 2016 | Brain Tumor |
|---|---|---|
| Nguồn | 2 thư mục ảnh `.jpg` + 2 file CSV nhãn | 1 thư mục, chia sẵn `Negative/`, `Positive/` |
| Số ảnh gốc | train 900 (727 / 173), test 379 có sẵn | 2.000 / 400, không có test riêng |
| Tiền xử lý | cắt viền đen (khi nội dung < 50 % cạnh) → resize 256 × 256 | đệm thành ảnh vuông (giữ tỉ lệ giải phẫu) → resize 256 × 256 |
| Số kênh | 3 (RGB) | 3 kênh **bằng nhau** (`force_grayscale`) — xem mục 7 |
| Cách chia | val = 15 % tập train (phân tầng theo lớp) | test 15 % → val 17,6 % phần còn lại → **70 / 15 / 15** |
| Train / val / test | 618 + 148 / 109 + 25 / 379 | 1401 + 281 / 299 + 59 / 300 + 60 |
| Cần sinh (`n_select`) | 618 − 148 = **470** ảnh malignant | 1401 − 281 = **1120** ảnh positive |

- Mọi ảnh lưu **PNG** (không nén mất dữ liệu) để ảnh thật và ảnh sinh không khác nhau về độ nén.
- Split **cố định** (seed 2026), lưu thành JSON, và được kiểm tra trùng khớp với split mà GAN đã dùng để train.
- **Val và test luôn 100 % ảnh thật.** GAN, E_d và DASS không bao giờ nhìn thấy val / test.

---

## 3. Toàn cảnh pipeline

```
 ảnh gốc ─► [1] prepare ─► ảnh 256×256 + split train/val/test
                               │ (chỉ phần train)
                               ▼
                         [2] gan ─► StyleGAN2-ADA có điều kiện (snapshot KID tốt nhất)
                               ▼
                         [3] sample ─► pool ảnh sinh lớp thiểu số (pool_mult × n_select)
                               ▼
              ( [3b] fingerprint — tuỳ chọn: đo dấu vết tần số của ảnh sinh )
                               ▼
                         [4] select ─► E_v, E_d → điểm → chọn n_select ảnh cho M0…M6
                               ▼
                         [5] train ─► 6 backbone × 7 biến thể × 3 seed (val chọn epoch, test dự đoán 1 lần)
                               ▼
                         [6] evaluate ─► metric phân loại, KID/FID, bootstrap ΔAUC
                               ▼
                         [7] report ─► bảng cho bài báo (.csv / .json / .tex)
```

Mỗi bước là một lệnh `dass …`, chạy độc lập, lưu kết quả lên Google Drive và **chạy lại được** (phần đã xong được bỏ qua).

---

## 4. Chi tiết từng bước

### Bước 1 — `prepare`: chuẩn bị dữ liệu
- Copy ảnh từ Drive về máy Colab, xếp theo lớp; tự nhận diện / áp dụng số kênh.
- Tiền xử lý (mục 2), chia train / val / test, tách sẵn thư mục test (Brain Tumor).
- Ghi `dataset_card.json`: số ảnh mỗi tập / lớp, số kênh, tham số chia, mã băm SHA-1 của split.
- Kiểm tra: split phải trùng `real_split.json` của lần train GAN (Brain Tumor) — nếu không, dừng (tránh rò rỉ).

### Bước 2 — `gan`: StyleGAN2-ADA có điều kiện
- **Mô hình:** StyleGAN2-ADA bản chính thức của NVlabs (PyTorch), cấu hình `paper256`, có điều kiện theo nhãn lớp.
  Mã gốc viết cho PyTorch 1.x; pipeline tự clone và **vá** 9 chỗ để chạy trên PyTorch 2.x / Python 3.12.
- **Dữ liệu train:** toàn bộ ảnh **train của cả hai lớp** (lớp đa số dạy các đặc trưng chung như giải phẫu, độ tương
  phản; nhãn điều khiển phần khác biệt).
- **ADA** (Adaptive Discriminator Augmentation, target 0,6): tự tăng / giảm augmentation để chống overfit khi ít ảnh.
- **Early stopping theo KID của lớp thiểu số:** mỗi 100 kimg lưu một snapshot, sinh 1.000 ảnh lớp thiểu số, tính KID
  với ảnh thật lớp thiểu số (Inception-v3). Sau 400 kimg, nếu KID không giảm ít nhất 2 % trong 5 snapshot liên tiếp →
  dừng. **Dùng snapshot có KID thấp nhất** (`best.pkl`), không phải snapshot cuối.
- **Hiện trạng:** dùng lại GAN đã train — Brain Tumor: best KID 0,02985 tại 1200 kimg (dừng sớm ở 1700 kimg);
  ISIC: `checkpoints_v5`. Lệnh `gan` chỉ báo cáo (đường KID, ảnh mẫu), không train lại.

### Bước 3 — `sample`: sinh candidate pool
- Sinh `k × n_select` ảnh lớp thiểu số, với **k = `pool_mult` = 4 cho cả hai bộ dữ liệu** (ISIC 1.880, Brain Tumor
  4.480 ảnh) → DASS giữ 25 %. k được chọn **trước** khi xem kết quả, theo CosSIF (2.000 ứng viên cho 554 ảnh ≈ 3,6);
  độ nhạy theo k ∈ {2, 3, 8} là ablation riêng (mục 6).
- `truncation ψ = 1` (không cắt bớt đa dạng), seed cố định → pool tái lập được, nén zip lên Drive.
- Brain Tumor: ảnh sinh đi qua đúng phép chuyển xám như ảnh thật (`force_gray=1`).

### Bước 3b — `fingerprint` (tuỳ chọn): dấu vết tần số
- Bộ phát hiện của Frank et al. (ICML 2020): biến đổi DCT → log → chuẩn hoá → CNN nông / hồi quy, phân biệt ảnh thật
  và ảnh sinh. AUC ≈ 1 nghĩa là ảnh sinh mang "dấu vân tay" GAN — chỉ đo để báo cáo, không sửa ảnh.

### Bước 4 — `select`: DASS
**Hai bộ mã hoá** (mọi vector chuẩn hoá L2):
- **E_v** — EfficientNet-B0 pretrain ImageNet, đóng băng: đặc trưng thị giác chung.
- **E_d** — DenseNet-121 **train riêng** trên nhãn thật của tập train (5 epoch head + 20 epoch fine-tune, class weight):
  đặc trưng nhận biết bệnh. Cố ý **khác** baseline M0 để việc chọn ảnh không củng cố ranh giới của chính M0.

**Điểm lề** cho mỗi ảnh ứng viên `x`, trong mỗi không gian (v hoặc d):

```
S⁺(x) = trung bình cosine với K = 5 ảnh THẬT lớp thiểu số gần nhất
S⁻(x) = trung bình cosine với K = 5 ảnh THẬT lớp đa số gần nhất
M(x)  = S⁺(x) − λ · S⁻(x)                (λ = 1)
```

Dùng trung bình top-K thay vì max để không phụ thuộc vào một ảnh ngoại lai.

**Điểm DASS** (chọn tham lam, mỗi vòng chọn ảnh có điểm cao nhất):

```
S_div(x)  = min_{y đã chọn} [1 − cos(z_x, z_y)]          (khác biệt với tập đã chọn, không gian E_v)
S_DASS(x) = α·M̃_v(x) + β·M̃_d(x) + γ·S̃_div(x)            (α = β = 1, γ = 0,5)
```

`~` = chuẩn hoá min–max về [0, 1] trên toàn pool (S_div chuẩn hoá lại mỗi vòng), để các thành phần cùng thang đo.

**Các biến thể so sánh** — mọi biến thể chọn **đúng `n_select` ảnh từ cùng một pool**, nên khác biệt chỉ đến từ tiêu chí:

| ID | Cách chọn ảnh sinh | Vai trò |
|---|---|---|
| M0 | không dùng ảnh sinh (chỉ ảnh thật + class weight) | **baseline** |
| M1 | ngẫu nhiên | ảnh sinh không chọn lọc |
| M2 | theo M_v (thị giác) | ablation: chỉ E_v |
| M3 | theo M_d (bệnh học) | ablation: chỉ E_d |
| M4 | theo α·M̃_v + β·M̃_d | ablation: không có đa dạng |
| M5 | chỉ đa dạng (k-center greedy) | ablation: chỉ đa dạng |
| **M6** | **DASS đầy đủ** | **phương pháp đề xuất** |

**Chẩn đoán đi kèm:** AUC probe (hai lớp bệnh tách nhau đến đâu trong E_v / E_d, đo trên val); **kiểm tra shortcut**
(AUC tách ảnh thật / ảnh sinh trong E_v); độ trùng lặp Jaccard giữa các biến thể; hình ảnh được chọn / bị loại; ảnh sinh
cạnh ảnh thật gần nhất (kiểm tra GAN có "học thuộc" không).

### Bước 5 — `train`: classifier
- **Tập train mỗi biến thể:** lớp đa số giữ nguyên ảnh thật; lớp thiểu số = ảnh thật + ảnh sinh đã chọn. Val giữ nguyên.
- **6 backbone pretrain ImageNet:** EfficientNetV2B0, ResNet50, DenseNet121, ConvNeXtTiny (CNN); ViT-B16, SwinT
  (Transformer, KerasHub). Ảnh 224 × 224, batch 16, dropout 0,3.
- **Augmentation** (chỉ tập train): lật ngang / dọc, xoay, zoom ±10 %, độ sáng ±10 %, tương phản ±10 %.
- **Train 2 giai đoạn:** (1) đóng băng backbone, train lớp đầu ra 5 epoch (AdamW, lr 1e-3); (2) mở toàn bộ, fine-tune
  tối đa 30 epoch (lr 1e-5, weight decay 1e-4).
- **Chọn epoch** theo `val_macro_recall` = (sensitivity + specificity) / 2, early stopping sau 8 epoch không cải thiện;
  nạp lại checkpoint tốt nhất rồi **dự đoán test một lần duy nhất**.
- **3 seed** (2026, 2027, 2028) → báo cáo mean ± std. Tổng mỗi bộ dữ liệu: 6 × 7 × 3 = **126 lần train**.

### Bước 6 — `evaluate`: đánh giá
- **Phân loại** (trên test, **ngưỡng cố định 0,5**): ROC-AUC, PR-AUC, F1, Sensitivity, Specificity, Balanced accuracy,
  G-mean, MCC (+ macro-F1, precision, accuracy).
- **Kiểm định:** paired bootstrap (2.000 lần, phân tầng theo lớp) cho ΔAUC của từng phương pháp so với M0, dùng xác suất
  trung bình qua các seed → khoảng tin cậy 95 % và p-value.
- **Chất lượng ảnh sinh** (Inception-v3): KID (chỉ số chính, mean ± std qua 50 tập con), FID (tham khảo), độ đa dạng,
  SSIM nội bộ, AUC thật-vs-sinh. Hai hàng tham chiếu: "real val vs real train" (mức nền giữa hai tập ảnh thật) và
  "all candidates" (pool chưa lọc).

### Bước 7 — `report`: bảng cho bài báo
`tables/dataset`, `tables/classification`, `tables/significance`, `tables/generation_quality` — mỗi bảng ở dạng `.csv`
(đã định dạng mean ± std), `.json` (số thô) và `.tex` (LaTeX booktabs, in đậm giá trị tốt nhất).

---

## 5. Nguyên tắc chống rò rỉ và shortcut

| Nguyên tắc | Cách đảm bảo |
|---|---|
| Không rò rỉ val / test | Val / test 100 % ảnh thật; GAN, E_d, DASS chỉ dùng phần train; có kiểm tra tự động |
| Không dò ngưỡng | Ngưỡng cố định 0,5; val chỉ dùng chọn epoch |
| Test dùng một lần | Dự đoán test sau khi đã nạp checkpoint tốt nhất |
| GAN dùng lại hợp lệ | Split phải trùng split GAN đã train (`split.expected`), không thì dừng |
| Không shortcut định dạng | Mọi ảnh PNG; Brain Tumor: 3 kênh bằng nhau cho cả ảnh thật và ảnh sinh |
| So sánh công bằng | Mọi biến thể cùng pool, cùng `n_select`, cùng seed, cùng quy trình train |
| Kết quả tin cậy | 3 seed + paired bootstrap; toàn bộ cấu hình và phiên bản thư viện lưu trong `run_manifest.json` |

---

## 6. Cách chạy

### Môi trường
Google Colab có GPU (A100 / L4 khuyến nghị). Hai cách mở notebook `notebooks/colab_pipeline.ipynb`:
- **VS Code + extension Colab:** mở notebook → *Select Kernel* → *Colab* → *New Colab Server* → *GPU* → *A100* →
  *Python 3 (ipykernel)*. (Thư mục dự án phải ở chế độ *Trust*.)
- **Trình duyệt:** https://colab.research.google.com/github/honghanh22/isic-dass/blob/main/notebooks/colab_pipeline.ipynb
  → *Runtime* → *Change runtime type* → *A100 GPU*.

Notebook **tự tải code mới nhất từ GitHub**, không cần upload tay.

### Chạy từng ô (Shift + Enter), từ trên xuống

| Ô | Việc | Ghi chú |
|---|---|---|
| Mount Drive | đăng nhập Google → Allow | `Mounted at /content/drive` |
| Lấy mã nguồn | clone / pull + cài đặt | in `Phiên bản code: …` |
| Chọn thực nghiệm | `EXPERIMENT`, `PROFILE`, `EXTRA` | xem bên dưới |
| Tiện ích | hàm `show()`, `table()` | |
| `prepare` → `gan-setup` → `gan` → `sample` → (`fingerprint`) → `select` → 6 ô `train` → `evaluate` + `report` | pipeline | mỗi ô in log; lỗi → Ctrl+S và báo |

**Chọn thực nghiệm:**
```python
EXPERIMENT = "configs/experiments/brain_tumor_dass.yaml"   # hoặc isic2016_dass.yaml
PROFILE = "configs/experiments/smoke.yaml"                 # chạy thử nhanh; "" = chạy thật
EXTRA = ""                                                  # ablation, ví dụ "--set selection.gamma=0 --tag nodiv"
```

- **Chạy thử (smoke):** 1 epoch, 1 seed, pool nhỏ, ghi vào thư mục `*_smoke` — để kiểm tra mọi bước chạy thông.
- **Chạy thật:** `PROFILE = ""`. Bước `train` mất nhiều giờ; có thể dừng giữa các ô và chạy tiếp hôm sau.
- **Ablation:** luôn kèm `--tag <tên>` để kết quả nằm ở thư mục riêng.
- **Độ nhạy theo k** (chạy sau khi lần chạy chính đã qua bước `select`): cùng một file cấu hình cho cả hai bộ dữ liệu
  — chỉ k thay đổi; GAN, split và **E_d dùng lại của lần chạy chính**; một backbone (EfficientNetV2B0) × 3 seed:
  ```python
  EXTRA = "-c configs/experiments/ablation/k3.yaml --tag k3"     # tương tự k2, k8
  ```
  rồi chạy `!dass {CFG} run --from prepare --to report` → kết quả ở `results_bt_v3_k3` / `results_v7_k3`.

### Hoặc bằng lệnh (CLI)
```bash
dass -c configs/experiments/brain_tumor_dass.yaml run                 # toàn bộ chuỗi
dass -c configs/experiments/brain_tumor_dass.yaml select              # một bước
dass -c configs/experiments/brain_tumor_dass.yaml train --model resnet50 --seeds 2026 2027 2028
dass -c configs/experiments/brain_tumor_dass.yaml --set selection.gamma=0 --tag nodiv run --from select
```

### Kết quả lưu ở đâu
Google Drive, trong thư mục của từng bộ dữ liệu (không bao giờ bị xoá; chạy lại không ghi đè):

| | Brain Tumor (`…/BrainTumor_GAN/`) | ISIC (`…/ISBI2016_ISIC_Part3/`) |
|---|---|---|
| GAN | `checkpoints_bt/stylegan2ada/` | `checkpoints_v5/stylegan2ada/` |
| Dữ liệu, pool, lựa chọn, E_d | `checkpoints_bt_v3/` | `checkpoints_v7/` |
| Dự đoán, số liệu, **bảng**, hình | `results_bt_v3/` (`tables/`) | `results_v7/` (`tables/`) |
| Chạy thử | `checkpoints_smoke/`, `results_smoke/` | như bên trái |

---

## 7. Đọc kết quả

| Bảng / chỉ số | Đọc thế nào |
|---|---|
| `classification` | So M6 với M0 (baseline) và M1–M5 (ablation) trong **cùng một backbone**. Quan trọng với dữ liệu mất cân bằng: **sensitivity**, **balanced accuracy**, **G-mean**, **MCC**, **PR-AUC** — không chỉ accuracy |
| `significance` | ΔAUC > 0 và khoảng tin cậy 95 % không chứa 0 (p < 0,05) → cải thiện có ý nghĩa thống kê so với M0 |
| KID | thấp hơn = ảnh sinh gần ảnh thật hơn; so với hàng "real val vs real train" (mức nền) |
| AUC thật-vs-sinh (`shortcut_check`) | gần 0,5 = khó phân biệt (tốt); gần 1 = ảnh sinh dễ nhận ra → nguy cơ shortcut |
| AUC probe (`probe_auc`, dòng val) | E_d cao hơn E_v → không gian bệnh học tách lớp tốt hơn, đúng mục đích của E_d |
| Jaccard | thấp giữa các biến thể → các tiêu chí chọn thực sự khác nhau |

---

## 8. Hạn chế đã biết (nên nêu trong bài báo)

- **Dấu vân tay tần số của GAN:** detector của Frank et al. phân biệt ảnh thật / sinh với AUC ≈ 1. Test toàn ảnh thật
  nên không làm kết quả lạc quan giả tạo, nhưng có thể giảm lợi ích của ảnh sinh.
- **Nhiễu màu JPEG ở Brain Tumor:** 129 / 2.000 ảnh negative (0 / 400 positive) có lệch kênh ≤ 3,1 / 255 — đã loại bằng
  `force_grayscale` cho cả ảnh thật lẫn ảnh sinh. GAN được train trước khi làm sạch (ảnh positive dùng để sinh vốn
  không có nhiễu, nên ảnh hưởng không đáng kể).
- **Không chia theo bệnh nhân** ở Brain Tumor (tên file không có mã bệnh nhân); nếu có, đặt `split.group_regex`.
- **Chỉ bài toán nhị phân**; đa lớp cần DASS one-vs-rest.
- Lớp thiểu số nhỏ (148 / 281 ảnh train) → FID chệch, chỉ dùng KID làm chỉ số chính.

---

## 9. Tham số chính

| Nhóm | Tham số | Giá trị | File |
|---|---|---|---|
| GAN | cfg / batch / γ (R1) / ADA target | paper256 / 16 / 1.0 / 0.6 | `configs/_base_/generator.yaml` |
| GAN | max / min kimg, patience, Δ tối thiểu | 3000 / 400, 5 snapshot, 2 % | |
| Pool | k = `pool_mult` (chung 2 bộ dữ liệu), ψ | 4, 1.0 | `configs/_base_/selection.yaml` |
| DASS | K, λ_v, λ_d, α, β, γ | 5, 1, 1, 1, 1, 0.5 | |
| E_d | backbone, seed, epoch | DenseNet121, 4242, 5 + 20 | |
| Ablation k | k ∈ {2, 3, 8}, E_d dùng lại, 1 backbone | `--tag k<k>` | `configs/experiments/ablation/` |
| Classifier | backbone | 6 (CNN + Transformer) | `configs/_base_/classifier.yaml` |
| Classifier | ảnh, batch, epoch, lr, early stop | 224, 16, 5 + 30, 1e-3 / 1e-5, 8 | |
| Classifier | seed, chọn epoch | 2026 / 2027 / 2028, `val_macro_recall` | |
| Đánh giá | ngưỡng, bootstrap, KID | 0.5, 2.000, 50 tập con | `configs/_base_/evaluation.yaml` |
| Dữ liệu | nguồn, tiền xử lý, chia, số kênh | theo bộ dữ liệu | `configs/datasets/*.yaml` |
