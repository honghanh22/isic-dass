# Phương pháp

Tài liệu này mô tả **đầy đủ phương pháp của bài báo**, viết lại từ code (`dass` 2.0.0) theo thứ tự của một mục
*Methods*: bài toán, dữ liệu, mô hình sinh, chọn ảnh sinh (DASS), thiết kế tập huấn luyện, huấn luyện classifier, đánh
giá và kiểm định, phân tích chẩn đoán. Mọi con số và công thức đều khớp với code và cấu hình mặc định
(`configs/_base_/*.yaml`, `configs/datasets/*.yaml`); mỗi mục có đường dẫn tới file thực hiện. Khi code và tài liệu
khác nhau, **code là nguồn sự thật**.

Tài liệu liên quan: [GUIDE.md](GUIDE.md) (hướng dẫn cho người mới, cách chạy), [ARCHITECTURE.md](ARCHITECTURE.md)
(kiến trúc phần mềm), [RESULTS_FORMAT.md](RESULTS_FORMAT.md) (định dạng kết quả).

---

## 0. Tóm tắt

Bài toán là **phân loại ảnh y tế nhị phân khi lớp bệnh ít ảnh** (ISIC 2016: 4,2 : 1; RSNA Pneumonia: 7,1 : 1; PCam:
12 : 1). Bài so sánh hai nhóm:

1. **Nhóm 1 — các cách xử lý mất cân bằng** (bảng `comparison`): B0 không xử lý, B1 class-weighted loss, B2 random
   oversampling, B3 augmentation truyền thống, B4 Balanced GAN chọn ngẫu nhiên, B5 Balanced GAN chọn bằng **DASS**.
2. **Nhóm 2 — ablation DASS** (bảng `ablation`): M1–M6, cùng pool, cùng thành phần dữ liệu, chỉ khác tiêu chí chọn
   (M1 = B4, M6 = B5).

```
 Ảnh thật train ──► (1) StyleGAN2-ADA có điều kiện ──► pool ⌈k·g⌉ ảnh sinh lớp bệnh
                          (chọn checkpoint theo lớp bệnh)               │
 Ảnh thật train ──► (2) E_v (ImageNet, đóng băng), E_d (học nhãn bệnh) │
                                                                       ▼
                   (3) DASS: chấm từng ảnh sinh = lề hình thức + lề bệnh + đa dạng -> chọn g ảnh (M6)
                                                                       │
                   (4) Balanced GAN: ảnh thật + g ảnh sinh lớp bệnh (cân bằng 1 : 1); B0–B3 dùng ảnh thật
                                                                       │
                   (5) Train classifier (3 backbone × 3 seed), chọn epoch trên val thật,
                       dự đoán test thật MỘT lần, kiểm định ΔAUC (bootstrap ghép cặp + Holm), shortcut probe
```

**DASS** (Dual-space Adaptive Synthetic Selection): mỗi ảnh sinh được chấm bằng một **lề tương phản** ("giống ảnh thật
cùng lớp hơn giống ảnh thật lớp kia bao nhiêu", $K$ láng giềng gần nhất) trong **hai** không gian đặc trưng: hình thức
chung (E_v) và đặc trưng bệnh (E_d), kèm một thành phần **đa dạng** tối ưu bằng chọn tham lam.

## 1. Bài toán và ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $y \in \{0, 1\}$ | nhãn; **lớp 1 = lớp bệnh = lớp thiểu số** (ISIC: `malignant`, RSNA: `positive`) |
| $R_1, R_0$ | ảnh **thật** lớp 1 / lớp 0 của tập train; $n_1 = \lvert R_1\rvert < n_0 = \lvert R_0\rvert$ |
| $\rho = n_0 / n_1$ | tỉ lệ mất cân bằng của tập train |
| $G$ | bộ sinh có điều kiện; $G(z, c)$ là ảnh sinh của lớp $c$ |
| $P_1$ | pool ảnh sinh ứng viên của lớp 1 |
| $g$ | số ảnh sinh được chọn (chỉ lớp 1): $g = n_0 - n_1$ (cân bằng 1 : 1) |
| $k$ | hệ số pool: $\lvert P_1\rvert = \lceil k\,g\rceil$, $k$ = 2 |
| $E_v, E_d$ | bộ mã hoá hình thức / bộ mã hoá bệnh; $z_v(x), z_d(x)$ là vector nhúng đã chuẩn hoá L2 |
| $f$ | classifier nhị phân; $f(x) \in [0, 1]$ là xác suất thuộc lớp 1 |

Mục tiêu là $f$ có ROC-AUC cao trên **tập test chỉ gồm ảnh thật**. Câu hỏi nghiên cứu: (i) Balanced GAN + DASS có tốt hơn các cách
xử lý mất cân bằng thông thường không; (ii) chọn ảnh sinh bằng DASS có tốt hơn chọn ngẫu nhiên không.

---

## 2. Dữ liệu

### 2.1 Ba bộ dữ liệu

| | ISIC 2016 (Part 3) | RSNA Pneumonia (2018) | PCam v1 |
|---|---|---|---|
| Loại ảnh | dermoscopy, RGB | X-quang ngực, DICOM, xám 1 kênh | mô bệnh học H&E, RGB, 96 × 96 |
| Lớp 0 / lớp 1 | `benign` / `malignant` | `negative` / `positive` | `negative` / `positive` (u ở ô 32 × 32 giữa ảnh) |
| Nhãn | CSV của ISBI 2016 | MD.ai đã thống nhất (Shih et al., 2019); loại 106 ảnh "Exclude" | nhãn PCam |
| Đơn vị lấy mẫu | ảnh | một ảnh mỗi bệnh nhân (29.894 → 12.249 ảnh) | ảnh (patch), chia theo WSI |
| Tập test | test chính thức | tách từ dữ liệu | tách từ dữ liệu |
| Ghi chú | – | tư thế `ViewPosition` (AP / PA) | file HDF5 đã tiền xử lý (mục 2.2) |

### 2.2 Tiền xử lý ([data/transforms.py](../src/dass/data/transforms.py), [data/sources/](../src/dass/data/sources/))

**PCam** được tiền xử lý MỘT lần bằng [scripts/preprocess_pcam.py](../scripts/preprocess_pcam.py) thành
`pcam_imb12.h5` (+ `pcam_imb12_split.csv`); pipeline chỉ đọc file này (nguồn `hdf5`). Các bước: gộp train / valid /
test gốc; bỏ ảnh trùng hệt (PCam lấy mẫu lặp lớp dương để đạt 50 / 50); bỏ ảnh có > 50 % pixel nền trắng (30 % ảnh âm,
5 % ảnh dương — đường tắt "trắng = không u"); chia 399 WSI 70 / 15 / 15 (không WSI nào ở hai phần); lấy mẫu phân tầng
theo WSI để âm : dương = 12 ở cả ba phần (seed 2026). Ảnh 96 → 128 (bilinear).

1. **Đọc ảnh, giữ đúng số kênh gốc.** ISIC lưu PNG RGB. RSNA chuyển DICOM sang PNG 8 bit, 1 kênh: ảnh `MONOCHROME1`
   được đảo theo giá trị lớn nhất **có thể** (theo số bit), không theo từng ảnh; ảnh nhiều hơn 8 bit được co giãn
   min-max về [0, 255].
2. **Cắt viền tối (chỉ ISIC):** nếu vùng có nội dung (cường độ trung bình > 15) chiếm dưới 50 % một cạnh, cắt theo
   khung bao của vùng đó. Mục đích là bỏ viền đen của ống kính dermoscopy.
3. **Resize** về 256 × 256 (ISIC, RSNA) hoặc 128 × 128 (PCam) (bilinear, kéo giãn).
4. Ảnh xám **chỉ được nhân bản 1 → 3 kênh trong bộ nhớ**, ngay trước các mạng pretrain ImageNet (E_v, E_d, classifier,
   Inception). Không bước nào biến đổi ảnh theo từng kênh.

### 2.3 Chia dữ liệu ([data/splits/](../src/dass/data/splits/))

Chia ngẫu nhiên **phân tầng theo lớp**, seed 2026:

- **ISIC** (`holdout_val`): giữ tập test chính thức; tách 15 % tập train chính thức làm val.
- **RSNA** (`stratified`): tách 15 % làm test, rồi 17,6 % phần còn lại làm val (≈ 70 / 15 / 15). Vì mỗi bệnh nhân chỉ
  có một ảnh, chia theo ảnh cũng là chia theo bệnh nhân.
- **PCam** (`file`): dùng đúng phần chia theo WSI trong file HDF5 (CSV đi kèm).

| Tập | ISIC (0 / 1) | RSNA (0 / 1) | PCam (0 / 1) |
|---|---|---|---|
| train | 618 / 148 ($\rho$ = 4,18) | 7.522 / 1.059 ($\rho$ = 7,10) | 36.024 / 3.002 ($\rho$ = 12) |
| val | 109 / 25 | 1.606 / 226 | 7.728 / 644 |
| test | 304 / 75 | 1.610 / 226 | 7.728 / 644 |

Tỉ lệ tư thế AP ở RSNA (toàn bộ dữ liệu): lớp âm 23,0 %, lớp dương 57,6 %. Tư thế gắn với nhãn, nên nó là một đường tắt
tiềm năng (mục 7.4).

**Nguyên tắc dùng dữ liệu:**

- **Val và test 100 % ảnh thật.** Ảnh sinh và ảnh nhân bản chỉ có trong tập train (có kiểm tra tự động).
- **Val chỉ dùng để chọn epoch** (của E_d và của classifier) và làm mốc cho vài chỉ số chẩn đoán. Không dùng để chọn
  ngưỡng hay chọn phương pháp.
- **Test chỉ được dự đoán một lần** cho mỗi lần chạy, sau khi đã nạp trọng số tốt nhất.
- GAN, E_d và các tập ảnh mốc của DASS **chỉ thấy phần train**. Split được lưu (`real_val_split.json`) và băm SHA-1;
  mọi bước dùng lại GAN kiểm tra split trùng với split GAN đã được train, để chặn rò rỉ val / test vào GAN.

---

## 3. Mô hình sinh ảnh có điều kiện

### 3.1 Huấn luyện ([models/generator/trainer.py](../src/dass/models/generator/trainer.py), [pipeline/stages/gan.py](../src/dass/pipeline/stages/gan.py))

**StyleGAN2-ADA** bản chính thức của NVlabs (PyTorch), **có điều kiện theo nhãn lớp** (`--cond=1`). GAN được train trên
**toàn bộ ảnh thật của tập train, cả hai lớp, kèm nhãn**, ở 256 × 256 (PCam: 128 × 128), đúng số kênh gốc (RSNA: GAN
1 kênh). Chỉ ảnh sinh lớp 1 được dùng trong tập train; train chung hai lớp để lớp 1 học được từ đặc trưng mô / ảnh
chung của cả tập.

| Tham số | Giá trị |
|---|---|
| Cấu hình | `paper256`, batch 16, R1 $\gamma$ = 1 |
| Tăng cường thích nghi (ADA) | `--aug=ada`, mục tiêu $r_t$ = 0,6 |
| Lật ngang (`mirror`) | ISIC, PCam: có; RSNA: **không** (X-quang ngực không đối xứng trái / phải) |
| Độ dài tối đa | 3.000 kimg; lưu snapshot mỗi 100 kimg |

### 3.2 Chọn checkpoint

**ISIC, RSNA — KID-Inception lớp thiểu số.** 
Mỗi snapshot được đánh giá bằng **KID** (Kernel Inception Distance; Bińkowski et al., 2018) giữa:

- 1.000 ảnh sinh **lớp thiểu số** ($\psi$ = 1, seed 123), và
- **ảnh thật lớp thiểu số của tập train** (không bao giờ dùng val).

Đặc trưng là vector 2.048 chiều của Inception-v3 (bản trong StyleGAN2-ADA). KID là MMD² không chệch với kernel đa thức
$k(a, b) = (a^\top b / d + 1)^3$, lấy trung bình trên 50 tập con, mỗi tập $\min(n_\text{thật}, n_\text{sinh}, 1000)$ ảnh.

**Quy tắc dừng sớm:** một snapshot chỉ được tính là "tốt hơn" nếu KID giảm ít nhất 2 % so với mức tốt nhất hiện có.
Sau 400 kimg, nếu 5 snapshot liên tiếp không tốt hơn thì dừng. Snapshot có KID thấp nhất được dùng (`best.pkl`).

*Lý do chọn KID thay vì FID:* KID không chệch theo số mẫu, phù hợp khi lớp thiểu số chỉ có vài trăm ảnh (ISIC: 148).
Với 148 ảnh, ma trận hiệp phương sai 2.048 chiều của FID có hạng tối đa 147.

**PCam — FID trên đặc trưng E_d, lớp 1, làm mượt EMA** (theo Xue et al., 2021): mỗi 100 kimg sinh 3.000 ảnh mỗi lớp,
FID giữa đặc trưng backbone (chưa chuẩn hoá L2) của **chính E_d** (DenseNet121, mục 4.2) cho ảnh sinh và cho **toàn
bộ ảnh train** cùng lớp; điểm chọn = FID lớp 1, làm mượt $\tilde F_t = 0{,}5\,F_t + 0{,}5\,\tilde F_{t-1}$; cùng
luật dừng sớm như trên. FID lớp 0 được chấm và lưu để báo cáo. Ban đầu điểm chọn là trung bình hai lớp; người dùng
đổi sang lớp 1 khi GAN ở ~1.400 kimg (FID lớp 1 tăng trong khi lớp 0 giảm, trung bình che mất điều đó), quyết định
dựa trên FID của tập train, trước mọi kết quả classifier; mọi snapshot được chấm lại theo luật mới
([scripts/rescore_gan_snapshots.py](../scripts/rescore_gan_snapshots.py)). Dùng chung E_d cho chọn checkpoint và cho
$M_d$ là một giới hạn (mục 10).

GAN đang dùng: ISIC `v5`, RSNA `rsna` (800 kimg), PCam `pcam`.

### 3.3 Sinh pool ứng viên ([pipeline/pool.py](../src/dass/pipeline/pool.py))

Ảnh sinh lớp 1 được tạo từ $z \sim \mathcal{N}(0, I)$ với nhãn one-hot, **không truncation** ($\psi$ = 1), nhiễu cố định,
seed 777: pool $\lceil k\,g\rceil$ ảnh với $k$ = 2 (ISIC 940, RSNA 12.926, PCam 66.044).

*Lý do $\psi$ = 1:* truncation làm ảnh "điển hình" hơn nhưng giảm đa dạng; việc lọc ảnh để dành cho DASS.

---

## 4. Hai không gian đặc trưng ([models/encoders/__init__.py](../src/dass/models/encoders/__init__.py))

Mọi vector nhúng được **chuẩn hoá L2**, nên độ giống giữa hai ảnh là cosine:
$\mathrm{sim}_e(a, b) = z_e(a)^\top z_e(b)$.

### 4.1 E_v: không gian hình thức

**EfficientNet-B0 pretrain ImageNet, đóng băng**, bỏ đầu phân loại, lấy trung bình toàn ảnh (global average pooling),
nên mỗi ảnh là một vector 1.280 chiều. Đầu vào 224 × 224.

*Vai trò:* một "ý kiến thứ hai" **không học trên nhãn bệnh**, nên không bị ảnh hưởng nếu E_d overfit tập train nhỏ.
Nó đo độ giống về hình thức chung (màu, kết cấu, hình dạng, kiểu chụp). E_v cũng là không gian để đo đa dạng.

### 4.2 E_d: không gian bệnh

**DenseNet121 pretrain ImageNet, fine-tune trên nhãn thật của tập train**, rồi dùng phần backbone (global average
pooling, 1.024 chiều) làm bộ trích đặc trưng. E_d được train **riêng**, không dùng lại classifier B0 (chọn ảnh bằng
chính ranh giới của B0 rồi so với B0 sẽ chỉ củng cố ranh giới đó).

| Thiết lập | Giá trị |
|---|---|
| Dữ liệu | ảnh thật train (chọn epoch trên val thật); không thấy test hay ảnh sinh |
| Kiến trúc đầu | giống classifier: Dropout 0,3 → Dense(1, sigmoid) |
| Huấn luyện | 2 giai đoạn như mục 6: 5 epoch đầu (backbone đóng băng), tối đa 20 epoch fine-tune |
| Chọn epoch | ROC-AUC trên val, dừng sớm sau 8 epoch không cải thiện |
| Augmentation | **có**, profile theo dữ liệu (mục 6.2) |
| Class weight | **có**: $w_c = N / (2 n_c)$ |
| Seed | 4242 |

Augmentation và class weight của E_d là **thành phần của phương pháp chọn ảnh**, độc lập với thiết lập của classifier
(classifier không dùng cả hai, mục 6).

Ở server, E_d được dùng lại từ một lần chạy trước (`v12_srv`, `rsna_v3_srv`), cùng công thức. Trên Colab, E_d được
train ngay trong lần chạy, cùng công thức.

---

## 5. DASS: chọn ảnh sinh

Code: [selection/scoring.py](../src/dass/selection/scoring.py), [selection/strategies.py](../src/dass/selection/strategies.py), [pipeline/stages/select.py](../src/dass/pipeline/stages/select.py).

### 5.1 Độ giống với ảnh thật của một lớp

Với ảnh sinh $x$ và tập ảnh thật $R$, trong không gian $e \in \{v, d\}$:

$$
S_e(x; R) = \frac{1}{K} \sum_{r \in \mathcal{N}_K(x;\, R)} \mathrm{sim}_e(x, r)
$$

trong đó $\mathcal{N}_K(x; R)$ là $K$ ảnh trong $R$ giống $x$ nhất, $K$ = 5. Đây là **trung bình độ giống với 5 ảnh
thật gần nhất**, không phải khoảng cách tới tâm lớp.

*Lý do dùng láng giềng gần nhất thay cho tâm lớp:* ảnh y tế trong một lớp có nhiều dạng (u hắc tố nhiều hình thái;
đám mờ phổi khác nhau về vị trí, mức độ, tư thế chụp). Tâm lớp có thể nằm ở vùng không có ảnh thật nào và ưu tiên ảnh
"trung bình". Láng giềng gần nhất giữ được cấu trúc cục bộ của phân phối. Đây cũng là cách các chỉ số precision /
recall và coverage / density ước lượng phân phối ảnh thật (Kynkäänniemi et al., 2019; Naeem et al., 2020).

### 5.2 Lề tương phản

Với ảnh sinh $x$ của lớp mục tiêu $c$ (lớp còn lại là $\bar c$):

$$
M_e(x) = S_e^{+}(x) - \lambda_e\, S_e^{-}(x), \qquad S_e^{+}(x) = S_e(x; R_c), \quad S_e^{-}(x) = S_e(x; R_{\bar c})
$$

với $\lambda_v = \lambda_d = 1$. Lề **dương và lớn** nghĩa là ảnh giống ảnh thật lớp của nó **hơn** giống ảnh thật lớp
kia. Lề âm nghĩa là ảnh trông giống lớp kia (nghi sai nhãn). Tính trong hai không gian, được **lề hình thức** $M_v$ và
**lề bệnh** $M_d$.

*Lý do có $S^{-}$:* GAN có điều kiện có thể sinh ảnh mang nhãn "bệnh" nhưng trông bình thường. Chỉ đo độ giống lớp của
mình (như khoảng cách tới tâm lớp) không bắt được trường hợp ảnh giống **cả hai** lớp. Trừ đi $S^{-}$ ưu tiên ảnh mang
đặc trưng phân biệt lớp.

### 5.3 Đa dạng

Với tập $S$ các ảnh đã chọn, độ đa dạng mà $x$ đóng góp đo trong E_v:

$$
S_\text{div}(x \mid S) = \min_{y \in S} \bigl[\,1 - \mathrm{sim}_v(x, y)\,\bigr]
$$

tức khoảng cách cosine tới ảnh đã chọn gần nhất. Giá trị cao nghĩa là $x$ khác mọi ảnh đã chọn.

### 5.4 Điểm DASS và thuật toán chọn tham lam

Mỗi lề được chuẩn hoá min-max **trên pool** $P$ của lớp đó:

$$
\tilde u(x) = \frac{u(x) - \min_{P} u}{\max_{P} u - \min_{P} u}
$$

(nếu $u$ không đổi trên $P$ thì $\tilde u \equiv 0{,}5$). **Điểm nền** là
$B(x) = \alpha\,\tilde M_v(x) + \beta\,\tilde M_d(x)$ với $\alpha = \beta = 1$, nên $B \in [0, 2]$. Điểm DASS khi đã
chọn tập $S$:

$$
S_\text{DASS}(x \mid S) = \alpha\,\tilde M_v(x) + \beta\,\tilde M_d(x) + \gamma\,\tilde S_\text{div}(x \mid S), \qquad \gamma = 0{,}5
$$

$\tilde S_\text{div}$ được chuẩn hoá min-max lại **ở mỗi vòng**, trên các ảnh còn lại, vì giá trị của nó giảm dần khi
$S$ lớn lên. Với $\gamma$ = 0,5, đa dạng chiếm khoảng 1/5 biên độ điểm (nền trong [0, 2], đa dạng trong [0, 0,5]).

**Thuật toán 1. Chọn tham lam DASS** (cho một lớp $c$)

```
Vào:  pool P (⌈k·s⌉ ảnh sinh lớp c), vector nhúng z_v, z_d; ảnh thật R_c, R_c̄; số ảnh cần chọn s
1. Với mỗi x ∈ P: tính M_v(x), M_d(x) (mục 5.1–5.2)
2. B(x) ← α·minmax_P(M_v)(x) + β·minmax_P(M_d)(x)
3. x₁ ← argmax_{x∈P} B(x);  S ← {x₁}
4. Lặp đến khi |S| = s:
       với mỗi x ∈ P \ S:  d(x) ← min_{y∈S} [1 − z_v(x)ᵀ z_v(y)]
       x* ← argmax_{x∈P\S} [ B(x) + γ·minmax_{P\S}(d)(x) ]
       S ← S ∪ {x*}
Ra:   S (s ảnh)
```

Độ phức tạp: $O(\lvert P\rvert \cdot s)$ phép nhân vector, vì khoảng cách tới tập đã chọn được cập nhật dần
(chỉ cần so với ảnh vừa thêm).

*Lý do cộng điểm thay vì lọc nối tiếp:* lọc nối tiếp (ví dụ Xue et al., 2021: lọc entropy rồi lọc khoảng cách tới tâm
lớp) phụ thuộc thứ tự các bộ lọc và mỗi bước chỉ dùng thông tin "trên hay dưới ngưỡng". Cộng điểm dùng toàn bộ thang
điểm và cho phép các tiêu chí bù trừ nhau. Đổi lại, nó không bảo đảm mức tối thiểu cho từng tiêu chí (mục 10).

*Lý do chọn tham lam cho đa dạng:* đây là chiến lược liên quan / đa dạng kiểu MMR (Carbonell & Goldstein, 1998). Với
$B \equiv 0$, nó trở thành k-center greedy (Sener & Savarese, 2018), chính là biến thể M5.

### 5.5 Kích thước pool ($k$)

$k$ = 2: mỗi biến thể giữ 1/2 pool. $k$ đã đổi qua các lần chạy (4 → 3 → 2 → 1,5 → 2), có lần sau khi đã xem kết quả
test; phải nêu trong bài và báo cáo mọi cấu hình đã chạy. Ghi nhận cũ: chọn càng gắt thì tập chọn càng lệch khỏi phân
phối ảnh thật (với $k$ = 3, KID tập chọn 0,100 so với 0,029 của pool). $k$ và mọi tham số DASS dùng chung cho mọi bộ
dữ liệu.

### 5.6 Các biến thể (nhóm 2, ablation)

Mọi biến thể chọn **đúng $g$ ảnh từ cùng pool $P_1$**; chỉ khác tiêu chí:

| Mã | Tên trong bảng ablation | Tiêu chí chọn |
|---|---|---|
| M1 | Random (= B4) | ngẫu nhiên (seed 2026) |
| M2 | Visual margin ($M_v$) | $g$ ảnh có $M_v$ cao nhất |
| M3 | Disease margin ($M_d$) | $g$ ảnh có $M_d$ cao nhất |
| M4 | Diversity/Coverage | chỉ đa dạng: Thuật toán 1 với $B \equiv 0$, $\gamma = 1$ (k-center greedy) |
| M5 | $M_v + M_d$ | $g$ ảnh có $B(x)$ cao nhất |
| M6 | $M_v + M_d$ + Diversity/Coverage (DASS, = B5) | Thuật toán 1 đầy đủ |

Các ablation trả lời: một hay hai không gian (M2, M3 so với M5), thêm đa dạng (M5 so với M6), thêm điểm lề (M4 so với
M6), và giá trị của việc chọn so với ngẫu nhiên (M6 so với M1).

---

## 6. Thiết kế tập huấn luyện và huấn luyện classifier

### 6.1 Các phương pháp và tập huấn luyện ([pipeline/stages/train.py](../src/dass/pipeline/stages/train.py), [data/variants.py](../src/dass/data/variants.py))

| Mã | Phương pháp | Lớp 1 | Lớp 0 | Khi train |
|---|---|---|---|---|
| B0 | Imbalanced | $R_1$ | $R_0$ | — |
| B1 | Cost-sensitive | $R_1$ | $R_0$ | class-weighted BCE, $w_c = N / (2 n_c)$ |
| B2 | Random oversampling | $R_1$ + $g$ bản sao | $R_0$ | — |
| B3 | Traditional augmentation | $R_1$ | $R_0$ | augmentation (bảng dưới), dữ liệu vẫn mất cân bằng |
| B4 = M1, B5 = M6, M2–M5 | Balanced GAN | $R_1$ + $g$ ảnh sinh đã chọn | $R_0$ | — |

**ROS (B2):** mỗi ảnh lớp 1 được nhân bản $\lfloor g/n_1\rfloor$ lần, cộng $g \bmod n_1$ ảnh (chọn ngẫu nhiên, seed
2026, không lặp) nhân thêm một lần — số lần xuất hiện chênh nhau tối đa 1.

| | ISIC | RSNA | PCam |
|---|---|---|---|
| $g$ (ảnh thêm vào lớp 1) | 470 | 6.463 | 33.022 |
| Pool $\lceil 2g\rceil$ | 940 | 12.926 | 66.044 |
| Mỗi lớp trong B2 / Balanced GAN | 618 | 7.522 | 36.024 |
| Tỉ lệ ảnh sinh trong lớp 1 (Balanced GAN) | 76 % | 86 % | 92 % |

**Rủi ro đường tắt nguồn ảnh:** ảnh sinh chỉ có ở lớp 1, nên "trông giống ảnh GAN" mang thông tin về nhãn; trong không
gian E_v ảnh sinh tách khỏi ảnh thật với AUC ≈ 0,99. Shortcut probe (mục 8.3) là bắt buộc và được báo cáo cho mọi
phương pháp. Các biến thể M1–M6 dùng cùng ảnh thật và cùng pool, nên khác biệt giữa chúng chỉ đến từ ảnh sinh được chọn.

### 6.2 Classifier ([models/classifiers/__init__.py](../src/dass/models/classifiers/__init__.py), [engine/trainer.py](../src/dass/engine/trainer.py))

**Backbone** (pretrain ImageNet): **ResNet50, ConvNeXt-Tiny** (`keras.applications`) và **Swin-T** (backbone KerasHub,
preset `swin_tiny_patch4_window7_224`, kèm bộ chuyển ảnh resize + chuẩn hoá của preset). DenseNet121 không dùng làm
classifier vì trùng kiến trúc với E_d. Cả ba dùng **cùng đầu phân loại**.

**Đầu phân loại:** global average pooling → Dropout(0,3) → Dense(1, sigmoid). BatchNorm của backbone luôn ở chế độ
suy luận (kể cả khi fine-tune). Hàm mất mát: binary cross-entropy.

**Huấn luyện hai giai đoạn:**

| Giai đoạn | Lớp được train | Tối ưu | Số epoch |
|---|---|---|---|
| 1. Head | chỉ đầu phân loại (backbone đóng băng) | AdamW, lr $10^{-3}$ | 5 |
| 2. Fine-tune | toàn bộ | AdamW, lr $10^{-5}$, weight decay $10^{-4}$ | tối đa 30; dừng sớm sau 8 epoch không cải thiện |

- **Chọn epoch:** checkpoint có **ROC-AUC trên val cao nhất** trong giai đoạn 2. Giai đoạn 1 không được chọn.
- Ảnh vào 224 × 224, batch 16.
- **Augmentation chỉ ở B3, class weight chỉ ở B1**; mọi phương pháp khác train trên đúng ảnh đã ghép. E_d luôn có
  augmentation + class weight (thành phần của DASS).
- **Seed:** 2026, 2027, 2028. Mỗi tổ hợp backbone × phương pháp × seed là một lần chạy: 3 × 10 × 3 = **90 lần chạy mỗi
  bộ dữ liệu** (B0–B3, M1–M6).
- **Dự đoán:** sau khi nạp checkpoint tốt nhất, dự đoán val, test (một lần) và ảnh **thật** của tập train (để đo khoảng
  cách train–test). Mọi dự đoán lưu trong `.npz` kèm giao thức train và phiên bản code.

Profile augmentation (B3 và E_d; [data/loaders.py](../src/dass/data/loaders.py)), áp dụng trực tuyến trên tập train:

| Profile | Dùng cho | Lật | Xoay | Zoom | Độ sáng / tương phản |
|---|---|---|---|---|---|
| `rotation_invariant` | ISIC, PCam (không có hướng) | ngang + dọc | ±180° | ±10 % | ±10 % |
| `upright` | RSNA (ảnh có hướng giải phẫu) | ngang | ±10° | ±10 % | ±10 % |

---

## 7. Đánh giá

### 7.1 Chỉ số ([evaluation/classification.py](../src/dass/evaluation/classification.py))

Trên tập test (ảnh thật), lớp dương là lớp 1:

- **Chỉ số chính: ROC-AUC** (không phụ thuộc ngưỡng, nên công bằng giữa phương pháp mất cân bằng và cân bằng).
- Không phụ thuộc ngưỡng: PR-AUC (average precision).
- Ở **ngưỡng cố định 0,5** (không dò trên val hay test): sensitivity, specificity, balanced accuracy, G-mean
  ($\sqrt{\text{sens}\cdot\text{spec}}$), MCC, F1, macro-F1, precision, accuracy.

Mỗi chỉ số được báo cáo dạng mean ± độ lệch chuẩn qua 3 seed. Các chỉ số theo ngưỡng phụ thuộc tỉ lệ lớp lúc train:
B0 và B3 có xác suất lệch về lớp đa số nên sensitivity thấp ở ngưỡng 0,5. Vì vậy so sánh chính dựa trên AUC.

### 7.2 Câu hỏi nghiên cứu và giả thuyết

| | Câu hỏi | So sánh | Loại |
|---|---|---|---|
| RQ1 | Chọn ảnh sinh bằng DASS có tốt hơn chọn ngẫu nhiên, khi mọi thứ khác giống hệt? | M6 vs M1 | **H1**, chính |
| RQ2 | Ảnh sinh chọn bằng DASS có tốt hơn nhân bản ảnh thật tới cùng mức cân bằng? | M6 vs B2 | **H2**, chính |
| RQ3 | Balanced GAN + DASS có tốt hơn không xử lý mất cân bằng? | M6 vs B0 | **H3**, chính |
| RQ4 | So với class weight và augmentation truyền thống? | M6 vs B1, B3 | khám phá |
| RQ5 | Mỗi thành phần DASS đóng góp gì? | M2–M5 vs M1, M6 | khám phá |
| RQ6 | Ảnh sinh chỉ ở lớp bệnh có tạo đường tắt không? | shortcut probe (mục 8.3) | mô tả, bắt buộc |

H1–H3 được **khai báo trước khi train** (`evaluation.primary_comparisons`).

### 7.3 Kiểm định thống kê ([evaluation/aggregate.py](../src/dass/evaluation/aggregate.py), [evaluation/statistics.py](../src/dass/evaluation/statistics.py))

Với mỗi backbone và mỗi cặp (phương pháp $a$, đối chứng $b$), có hai cách đo ΔAUC:

1. **Paired bootstrap trên ensemble seed (kiểm định chính).** Xác suất của mỗi phương pháp được lấy trung bình qua các
   seed chung: $\bar p_a = \tfrac{1}{3}\sum_s p_{a,s}$. Lấy lại mẫu tập test **phân tầng theo lớp** (ảnh dương và ảnh
   âm được lấy lại có hoàn lại, riêng từng nhóm), $B$ = 2.000 lần, cùng chỉ số cho $a$ và $b$ (ghép cặp):

   $$
   \Delta^{*}_b = \mathrm{AUC}(y^{*}_b, \bar p^{*}_{a,b}) - \mathrm{AUC}(y^{*}_b, \bar p^{*}_{b,b})
   $$

   KTC 95 % là phân vị 2,5 % và 97,5 % của $\Delta^{*}$. p-value hai phía:
   $p = 2\min\{\Pr(\Delta^{*} \le 0), \Pr(\Delta^{*} \ge 0)\}$. Cách này đo dao động **do mẫu test**.
2. **ΔAUC theo từng seed:** $\mathrm{AUC}(p_{a,s}) - \mathrm{AUC}(p_{b,s})$ ghép cặp theo seed, báo cáo mean ± độ lệch
   chuẩn. Cách này đo dao động **giữa các lần train**.

**Hiệu chỉnh Holm–Bonferroni** trong từng backbone, **riêng cho hai họ**:

- họ **chính**: H1, H2, H3 (3 phép thử);
- họ **khám phá**: mọi phương pháp khác so với B0.

### 7.4 Phân tích nhóm con: tư thế chụp của RSNA ([evaluation/subgroups.py](../src/dass/evaluation/subgroups.py))

Khai báo trước (2026-10-04, trước khi có kết quả test RSNA), báo cáo dù kết quả thế nào. Tư thế AP gắn với bệnh nặng
(Oakden-Rayner et al., 2020; Janizek et al., 2020), nên classifier có thể dùng tư thế làm đường tắt.

1. **AUC trong từng nhóm AP / PA** của tập test: trong một nhóm mọi ảnh cùng tư thế, nên tư thế không còn giúp xếp hạng.
   AUC trong nhóm đo khả năng nhận ra bệnh thật sự. ΔAUC so với B0 trong từng nhóm, Holm theo backbone × nhóm.
2. **Mốc "chỉ dùng tư thế":** AUC khi dự đoán nhãn chỉ bằng việc ảnh là AP.
3. **Tỉ lệ ảnh sinh "trông giống AP":** probe logistic trên E_v, học trên ảnh thật train (kiểm tra bằng 5-fold), áp
   lên pool và từng tập được chọn, để xem việc chọn ảnh có khuếch đại tư thế gắn với bệnh không.

---

## 8. Phân tích chẩn đoán (không dùng để chọn ảnh hay train)

Các chỉ số này giải thích **vì sao** một phương pháp tốt hay kém; chúng không tham gia bất kỳ quyết định nào của pipeline.

### 8.1 Chất lượng ảnh sinh, theo từng lớp ([pipeline/stages/evaluate.py](../src/dass/pipeline/stages/evaluate.py), [evaluation/generative.py](../src/dass/evaluation/generative.py))

Trong không gian Inception-v3 (độc lập với E_v, E_d và các classifier), so với **ảnh thật cùng lớp của tập train**. Tính
cho: hàng mốc (ảnh thật val so với ảnh thật train), pool (đúng phần dùng để chọn), và từng tập được chọn.

| Chỉ số | Định nghĩa | Đọc |
|---|---|---|
| **KID** (chính) | MMD² không chệch, kernel đa thức bậc 3; mean ± std qua 50 tập con | thấp = gần phân phối ảnh thật |
| FID (tham khảo) | khoảng cách Fréchet giữa hai Gauss | chệch khi ít mẫu |
| Đa dạng | $1 -$ cosine trung bình giữa mọi cặp ảnh | cao = đa dạng |
| SSIM nội bộ | SSIM trung bình của 200 cặp ngẫu nhiên, trên đúng số kênh gốc | thấp = đa dạng |
| AUC thật / sinh | probe logistic 5-fold trên E_v tách ảnh thật và ảnh sinh lớp 1 | gần 1 = ảnh sinh dễ nhận ra (nguy cơ đường tắt) |
| NN ratio | trung vị khoảng cách tới ảnh train gần nhất của ảnh sinh ÷ cùng đại lượng của ảnh thật val | < 1 rõ rệt = bám sát ảnh train; > 1 = xa phân phối train |
| Near-copy | tỉ lệ ảnh sinh gần ảnh train hơn phân vị 5 % của ảnh thật val | cao hơn nhiều so với 5 % = nhiều ảnh gần như bản sao |
| **Coverage / density** | Naeem et al. (ICML 2020), $k$ = 5 (dưới đây) | coverage cao = phủ rộng phân phối ảnh thật |

Với ảnh thật $\{r_i\}_{i=1}^{N}$, ảnh sinh $\{f_j\}_{j=1}^{M}$ và $\mathrm{NND}_k(r_i)$ là khoảng cách từ $r_i$ tới láng
giềng thật thứ $k$ của nó:

$$
\text{coverage} = \frac{1}{N}\sum_{i} \mathbb{1}\Bigl[\exists j:\ \lVert r_i - f_j\rVert \le \mathrm{NND}_k(r_i)\Bigr], \qquad
\text{density} = \frac{1}{kM}\sum_{j}\sum_{i} \mathbb{1}\Bigl[\lVert f_j - r_i\rVert \le \mathrm{NND}_k(r_i)\Bigr]
$$

Coverage và density phụ thuộc số ảnh, nên chỉ so sánh giữa các tập có cùng kích thước (M1–M6).

### 8.2 Chất lượng tập ảnh được chọn, lớp 1 ([evaluation/selection_quality.py](../src/dass/evaluation/selection_quality.py))

Từ vector nhúng của bước chọn ảnh: thống kê $M_d$, $M_v$ (trung bình, trung vị, tỉ lệ $M_d > 0$); đa dạng và khoảng
cách tới ảnh gần nhất trong cùng tập (E_v); coverage / density so với ảnh thật lớp 1 trong E_d và E_v.

Lưu ý khi đọc:

- E_d và E_v là chính các không gian dùng để chọn, nên các chỉ số ở đây **thiên vị** cho phương pháp chọn theo không
  gian đó. Bằng chứng độc lập là các chỉ số trong Inception (mục 8.1).
- Dấu của $M$ không diễn giải trực tiếp được khi hai tập ảnh mốc lệch kích thước (mục 10, ý 2).

### 8.3 Các chẩn đoán khác

| Chẩn đoán | Nội dung |
|---|---|
| AUC theo tập (`auc_by_split`) | AUC trên ảnh thật train (không gồm bản sao, ảnh sinh), val và test: khoảng cách train − test cho thấy overfit |
| Probe AUC | probe logistic 5-fold tách hai lớp ảnh thật trong E_v và E_d (trên train và val): mỗi không gian mang bao nhiêu thông tin về bệnh |
| Jaccard | mức trùng ảnh giữa các tập chọn của các biến thể |
| Fingerprint tần số (tuỳ chọn) | detector DCT của Frank et al. (ICML 2020) tách ảnh thật và ảnh sinh |
| Lịch sử train | AUC / loss từng epoch của cả hai giai đoạn, lưu trong `.npz` |
| **Shortcut probe** (bắt buộc; [scripts/shortcut_probe.py](../scripts/shortcut_probe.py)) | classifier đã train (mọi phương pháp, trung bình qua seed) chấm ảnh **sinh lớp 0** chưa từng thấy (pool riêng, seed 779) so với ảnh **thật lớp 0** của val: tỉ lệ bị gọi là bệnh, chênh lệch xác suất, AUC sinh-vs-thật. B0–B3 (chưa thấy ảnh sinh) là mốc |

---

## 9. Tái lập

- **Seed:** chia dữ liệu, M1, ROS, bootstrap: 2026; classifier: 2026–2028; E_d: 4242; pool: 777; KID / FID khi chọn
  snapshot: 123; tiền xử lý PCam: 2026. Phép toán GPU không tất định (`deterministic: false`).
- **Mỗi lần chạy** (`run_tag`: ISIC `v14`, RSNA `rsna_v5`, PCam `pcam_v2`; server thêm `_srv`) ghi `run_manifest.json`
  gồm commit, phiên bản thư viện và cấu hình của từng stage.
- **Mỗi file dự đoán `.npz`** ghi giao thức train (augmentation, class weight, tiêu chí chọn epoch) và phiên bản code.
- **Split** được lưu và băm; GAN dùng lại phải khớp split của nó.
- Lệnh: `dass -c configs/server/<bộ dữ liệu>.yaml --tag srv run`.

---

## 10. Giới hạn đã biết của phương pháp

1. **Đường tắt nguồn ảnh:** ảnh sinh chỉ ở lớp bệnh (76–92 % lớp 1), nên classifier có thể học "trông giống ảnh GAN →
   bệnh". Thiết kế tương tự trước đây đo được 62 % (ISIC) và 99 % (RSNA) ảnh sinh lớp đa số bị gọi là bệnh. Shortcut
   probe được báo cáo cho mọi phương pháp.
2. **Lề là đại lượng tương đối.** Ảnh xa mọi ảnh thật có $S^{+}$ và $S^{-}$ cùng thấp, nên $M \approx 0$; DASS loại
   ảnh "giống lớp kia", không loại trực tiếp ảnh "không giống ảnh thật".
3. **Lệch do kích thước tập mốc:** tìm 5 láng giềng trong lớp lớn hơn dễ gặp ảnh giống hơn, nên dấu của $M$ không diễn
   giải được; chỉ thứ hạng trong pool có nghĩa.
4. **Chuẩn hoá min-max** cho $\tilde M_v$ và $\tilde M_d$ trọng số ngang nhau, bất kể lề nào mang nhiều thông tin hơn.
5. **Thành phần đa dạng ưu tiên ngoại lai** (tính chất của k-center).
6. **E_d học trên chính các ảnh làm mốc**, nên $S_d$ lạc quan với ảnh giống ảnh train. Ở PCam, **cùng E_d** vừa chọn
   checkpoint GAN vừa tính $M_d$.
7. **GAN chỉ học từ tập train**, nên ảnh sinh không mang thông tin mới ngoài tập train.
8. **Các thay đổi sau khi xem kết quả của các lần chạy trước:** $k$, tiêu chí chọn epoch, thiết kế thí nghiệm (2.0.0),
   luật chọn checkpoint của PCam. Mọi cấu hình đã chạy phải được báo cáo.
9. **Mất cân bằng của PCam là nhân tạo** (lấy mẫu 12 : 1).
10. **Giai đoạn chỉ train đầu phân loại không được chọn làm epoch tốt nhất.** Với backbone dễ overfit (ResNet50), epoch
    fine-tune đầu tiên thường đã là tốt nhất.

---

## 11. Bảng tham số

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Dữ liệu | kích thước lưu / vào mạng | 256 (PCam 128) / 224 |
| GAN | StyleGAN2-ADA có điều kiện | `paper256`, batch 16, R1 $\gamma$ 1, ADA target 0,6 |
| | lật ngang | ISIC, PCam có; RSNA không |
| | chọn checkpoint | ISIC, RSNA: KID-Inception lớp 1; PCam: FID-E_d lớp 1, EMA 0,5 |
| | dừng sớm | snapshot 100 kimg, tối thiểu 400 kimg, kiên nhẫn 5, giảm tối thiểu 2 %, tối đa 3.000 kimg |
| | sinh pool | $\psi$ = 1, seed 777, $\lceil 2g\rceil$ ảnh lớp 1 |
| DASS | $K$ láng giềng | 5 |
| | $\lambda_v, \lambda_d$ | 1, 1 |
| | $\alpha, \beta, \gamma$ | 1, 1, 0,5 (≡ 0,4 / 0,4 / 0,2) |
| | $k$ (pool / số chọn) | 2 |
| E_v | mạng | EfficientNet-B0 ImageNet, đóng băng, 1.280 chiều |
| E_d | mạng | DenseNet121, fine-tune trên ảnh thật, 1.024 chiều; 5 + tối đa 20 epoch, augmentation, class weight, seed 4242 |
| Classifier | backbone | ResNet50, ConvNeXt-Tiny, Swin-T |
| | đầu | GAP → Dropout 0,3 → Dense sigmoid |
| | giai đoạn 1 / 2 | 5 epoch AdamW lr $10^{-3}$ / tối đa 30 epoch AdamW lr $10^{-5}$, wd $10^{-4}$, kiên nhẫn 8 |
| | chọn epoch | val ROC-AUC |
| | augmentation / class weight | chỉ B3 / chỉ B1 |
| | batch / seed | 16 / 2026, 2027, 2028 |
| Đánh giá | ngưỡng | 0,5 (cố định) |
| | bootstrap | 2.000 lần, phân tầng theo lớp, trên ensemble seed |
| | hiệu chỉnh | Holm trong từng backbone, riêng họ chính (H1–H3) / họ khám phá |

## Tài liệu tham khảo được nhắc

- Bińkowski et al. (2018). *Demystifying MMD GANs.* ICLR. (KID)
- Buda, Maki & Mazurowski (2018). *A systematic study of the class imbalance problem in CNNs.* Neural Networks. (ROS)
- Carbonell & Goldstein (1998). *The use of MMR, diversity-based reranking.* SIGIR.
- Frank et al. (2020). *Leveraging frequency analysis for deep fake image recognition.* ICML.
- Janizek, Erion, DeGrave & Lee (2020). *An adversarial approach for the robust classification of pneumonia from chest
  radiographs.* ACM CHIL.
- Karras et al. (2020). *Training generative adversarial networks with limited data.* NeurIPS. (StyleGAN2-ADA)
- Kynkäänniemi et al. (2019). *Improved precision and recall metric for assessing generative models.* NeurIPS.
- Naeem et al. (2020). *Reliable fidelity and diversity metrics for generative models.* ICML. (coverage / density)
- Oakden-Rayner et al. (2020). *Hidden stratification causes clinically meaningful failures in machine learning for
  medical imaging.* ACM CHIL.
- Sener & Savarese (2018). *Active learning for convolutional neural networks: a core-set approach.* ICLR.
- Shih et al. (2019). *Augmenting the NIH chest radiograph dataset with expert annotations of possible pneumonia.*
  Radiology: AI.
- Xue et al. (2021). *Selective synthetic augmentation with HistoGAN for improved histopathology image
  classification.* Medical Image Analysis.
