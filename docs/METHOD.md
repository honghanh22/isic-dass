# Phương pháp

Tài liệu này mô tả **đầy đủ phương pháp của bài báo**, viết lại từ code (`dass` 1.22.1) theo thứ tự của một mục
*Methods*: bài toán, dữ liệu, mô hình sinh, chọn ảnh sinh (DASS), thiết kế tập huấn luyện, huấn luyện classifier, đánh
giá và kiểm định, phân tích chẩn đoán. Mọi con số và công thức đều khớp với code và cấu hình mặc định
(`configs/_base_/*.yaml`, `configs/datasets/*.yaml`); mỗi mục có đường dẫn tới file thực hiện. Khi code và tài liệu
khác nhau, **code là nguồn sự thật**.

Tài liệu liên quan: [GUIDE.md](GUIDE.md) (hướng dẫn cho người mới, cách chạy), [ARCHITECTURE.md](ARCHITECTURE.md)
(kiến trúc phần mềm), [RESULTS_FORMAT.md](RESULTS_FORMAT.md) (định dạng kết quả).

---

## 0. Tóm tắt

Bài toán là **phân loại ảnh y tế nhị phân khi lớp bệnh ít ảnh** (ISIC 2016: tỉ lệ 4,2 : 1; RSNA Pneumonia: 7,1 : 1).
Quy trình gồm năm bước:

```
 Ảnh thật train ──► (1) StyleGAN2-ADA có điều kiện ──► pool ảnh sinh mỗi lớp
                          (chọn checkpoint theo KID)            │
                                                                ▼
 Ảnh thật train ──► (2) Hai bộ mã hoá: E_v (ImageNet, đóng băng), E_d (học nhãn bệnh)
                                                                │
                                                                ▼
                   (3) DASS: chấm điểm từng ảnh sinh = lề hình thức + lề bệnh + đa dạng
                       -> chọn s ảnh mỗi lớp (M6); các biến thể M1–M5 dùng một phần tiêu chí
                                                                │
                                                                ▼
                   (4) Tập train (thiết kế B) = ảnh thật + ROS lớp bệnh lên 1 : 1 + s ảnh sinh MỖI lớp
                                                                │
                                                                ▼
                   (5) Train classifier (3 backbone × 3 seed), chọn epoch trên val thật,
                       dự đoán test thật MỘT lần, kiểm định ΔAUC (bootstrap ghép cặp + Holm)
```

Đóng góp về phương pháp gồm hai phần:

1. **DASS** (Dual-space Adaptive Synthetic Selection): mỗi ảnh sinh được chấm bằng một **lề tương phản**, tức "giống
   ảnh thật cùng lớp hơn giống ảnh thật lớp kia bao nhiêu", đo bằng $K$ láng giềng gần nhất trong **hai** không gian
   đặc trưng: hình thức chung (E_v) và đặc trưng bệnh (E_d). Kèm theo là một thành phần **đa dạng** được tối ưu bằng
   cách chọn tham lam.
2. **Thiết kế tập huấn luyện cân bằng nguồn ảnh (thiết kế B):** sau khi cân bằng lớp bằng ROS, mỗi lớp nhận **cùng
   số** ảnh sinh. Ảnh sinh chiếm cùng tỉ lệ 1/3 ở cả hai lớp, nên "ảnh thật hay ảnh sinh" không mang thông tin về nhãn.

Giao thức đánh giá cô lập đúng đóng góp của **cách chọn ảnh**: mọi biến thể M1–M6 dùng cùng ảnh thật, cùng ảnh nhân
bản, cùng pool ảnh sinh và cùng số ảnh; chỉ khác tiêu chí chọn.

---

## 1. Bài toán và ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $y \in \{0, 1\}$ | nhãn; **lớp 1 = lớp bệnh = lớp thiểu số** (ISIC: `malignant`, RSNA: `positive`) |
| $R_1, R_0$ | ảnh **thật** lớp 1 / lớp 0 của tập train; $n_1 = \lvert R_1\rvert < n_0 = \lvert R_0\rvert$ |
| $\rho = n_0 / n_1$ | tỉ lệ mất cân bằng của tập train |
| $G$ | bộ sinh có điều kiện; $G(z, c)$ là ảnh sinh của lớp $c$ |
| $P_c$ | pool ảnh sinh ứng viên của lớp $c$ |
| $s$ | số ảnh sinh được chọn **mỗi lớp** |
| $k$ | hệ số pool: $\lvert P_c\rvert = \lceil k\,s\rceil$ |
| $E_v, E_d$ | bộ mã hoá hình thức / bộ mã hoá bệnh; $z_v(x), z_d(x)$ là vector nhúng đã chuẩn hoá L2 |
| $f$ | classifier nhị phân; $f(x) \in [0, 1]$ là xác suất thuộc lớp 1 |

Mục tiêu là $f$ có ROC-AUC cao trên **tập test chỉ gồm ảnh thật**. Câu hỏi nghiên cứu chính: khi thêm ảnh sinh vào tập
train, **chọn ảnh sinh nào** thì tốt hơn chọn ngẫu nhiên.

---

## 2. Dữ liệu

### 2.1 Hai bộ dữ liệu

| | ISIC 2016 (Part 3) | RSNA Pneumonia (2018) |
|---|---|---|
| Loại ảnh | dermoscopy, RGB | X-quang ngực, DICOM, xám 1 kênh |
| Lớp 0 / lớp 1 | `benign` / `malignant` | `negative` / `positive` (có đám mờ phổi) |
| Nhãn | CSV của ISBI 2016 | bản MD.ai đã thống nhất (Shih et al., *Radiology: AI* 2019): dương = có chú thích "Lung Opacity"; âm = "Normal" hoặc "No Lung Opacity / Not Normal"; loại 106 ảnh mang nhãn "Exclude" |
| Đơn vị lấy mẫu | ảnh | **một ảnh mỗi bệnh nhân** (mapping NIH): 29.894 → 12.249 ảnh |
| Tập test | tập test chính thức (riêng) | tách từ dữ liệu (test của cuộc thi không có nhãn) |
| Thuộc tính phụ | – | tư thế chụp `ViewPosition` (AP / PA) từ DICOM |

### 2.2 Tiền xử lý ([data/transforms.py](../src/dass/data/transforms.py), [data/sources/](../src/dass/data/sources/))

1. **Đọc ảnh, giữ đúng số kênh gốc.** ISIC lưu PNG RGB. RSNA chuyển DICOM sang PNG 8 bit, 1 kênh: ảnh `MONOCHROME1`
   được đảo theo giá trị lớn nhất **có thể** (theo số bit), không theo từng ảnh; ảnh nhiều hơn 8 bit được co giãn
   min-max về [0, 255].
2. **Cắt viền tối (chỉ ISIC):** nếu vùng có nội dung (cường độ trung bình > 15) chiếm dưới 50 % một cạnh, cắt theo
   khung bao của vùng đó. Mục đích là bỏ viền đen của ống kính dermoscopy.
3. **Resize** về 256 × 256 (bilinear, kéo giãn; ảnh RSNA vốn vuông 1024 × 1024).
4. Ảnh xám **chỉ được nhân bản 1 → 3 kênh trong bộ nhớ**, ngay trước các mạng pretrain ImageNet (E_v, E_d, classifier,
   Inception). Không bước nào biến đổi ảnh theo từng kênh.

### 2.3 Chia dữ liệu ([data/splits/](../src/dass/data/splits/))

Chia ngẫu nhiên **phân tầng theo lớp**, seed 2026:

- **ISIC** (`holdout_val`): giữ tập test chính thức; tách 15 % tập train chính thức làm val.
- **RSNA** (`stratified`): tách 15 % làm test, rồi 17,6 % phần còn lại làm val (≈ 70 / 15 / 15). Vì mỗi bệnh nhân chỉ
  có một ảnh, chia theo ảnh cũng là chia theo bệnh nhân.

| Tập | ISIC (0 / 1) | RSNA (0 / 1) |
|---|---|---|
| train | 618 / 148 ($\rho$ = 4,18) | 7.522 / 1.059 ($\rho$ = 7,10) |
| val | 109 / 25 | 1.606 / 226 |
| test | 304 / 75 | 1.610 / 226 |

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
**toàn bộ ảnh thật của tập train, cả hai lớp, kèm nhãn**, ở độ phân giải 256 × 256, đúng số kênh gốc (RSNA: GAN 1 kênh).

| Tham số | Giá trị |
|---|---|
| Cấu hình | `paper256`, batch 16, R1 $\gamma$ = 1 |
| Tăng cường thích nghi (ADA) | `--aug=ada`, mục tiêu $r_t$ = 0,6 |
| Lật ngang (`mirror`) | ISIC: có; RSNA: **không** (X-quang ngực không đối xứng trái / phải) |
| Độ dài tối đa | 3.000 kimg; lưu snapshot mỗi 100 kimg |

### 3.2 Chọn checkpoint bằng KID

Mỗi snapshot được đánh giá bằng **KID** (Kernel Inception Distance; Bińkowski et al., 2018) giữa:

- 1.000 ảnh sinh **lớp thiểu số** ($\psi$ = 1, seed 123), và
- **ảnh thật lớp thiểu số của tập train** (không bao giờ dùng val).

Đặc trưng là vector 2.048 chiều của Inception-v3 (bản trong StyleGAN2-ADA). KID là MMD² không chệch với kernel đa thức
$k(a, b) = (a^\top b / d + 1)^3$, lấy trung bình trên 50 tập con, mỗi tập $\min(n_\text{thật}, n_\text{sinh}, 1000)$ ảnh.

**Quy tắc dừng sớm:** một snapshot chỉ được tính là "tốt hơn" nếu KID giảm ít nhất 2 % so với mức tốt nhất hiện có.
Sau 400 kimg, nếu 5 snapshot liên tiếp không tốt hơn thì dừng. Snapshot có KID thấp nhất được dùng (`best.pkl`).

*Lý do chọn KID thay vì FID:* KID không chệch theo số mẫu, phù hợp khi lớp thiểu số chỉ có vài trăm ảnh (ISIC: 148).
Với 148 ảnh, ma trận hiệp phương sai 2.048 chiều của FID có hạng tối đa 147.

GAN đang dùng: ISIC `v5` (train bằng notebook gốc theo cùng quy trình, cùng split); RSNA `rsna` (train bằng package
này; snapshot tốt nhất ở 800 kimg).

### 3.3 Sinh pool ứng viên ([pipeline/pool.py](../src/dass/pipeline/pool.py))

Ảnh sinh được tạo từ $z \sim \mathcal{N}(0, I)$ với nhãn one-hot của lớp, **không truncation** ($\psi$ = 1) và nhiễu
cố định (`noise_mode="const"`):

| Pool | Seed | Kích thước sinh | Phần dùng để chọn (thiết kế B) |
|---|---|---|---|
| Lớp 1 (bệnh) | 777 | ISIC 705, RSNA 9.695 | $\lceil k s\rceil$ ảnh **đầu tiên**: ISIC 464, RSNA 5.642 |
| Lớp 0 | 778 | $\lceil k s\rceil$: ISIC 464, RSNA 5.642 | toàn bộ |

Pool lớp 1 được sinh theo kích thước của thiết kế A ($\lceil k\,(n_0 - n_1)\rceil$) để dùng chung cho cả hai thiết kế.
Thiết kế B chỉ lấy phần đầu, nên giữ đúng tỉ lệ $k$ giữa pool và số ảnh chọn.

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
pooling, 1.024 chiều) làm bộ trích đặc trưng. E_d được train **riêng**, không dùng lại classifier M0 (chọn ảnh bằng
chính ranh giới của M0 rồi so với M0 sẽ chỉ củng cố ranh giới đó).

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

Ở server, lần chạy thiết kế B dùng lại E_d đã train ở lần chạy thiết kế A (`v12_srv`, `rsna_v3_srv`; cùng công thức),
để hai thiết kế chỉ khác nhau ở tập train. Trên Colab, E_d được train ngay trong lần chạy, cùng công thức.

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

### 5.5 Lớp đa số: lề đảo chiều

Ở thiết kế B, ảnh sinh lớp 0 cũng được chọn bằng **cùng tiêu chí** của từng biến thể, với lề **đảo chiều**:
$S^{+}$ tính với $R_0$ và $S^{-}$ tính với $R_1$, tức chọn ảnh "giống bình thường hơn giống bệnh". Pool lớp 0 riêng
(seed 778); M1 dùng seed 2027.

*Lý do:* xử lý **đối xứng** hai lớp. Nếu chỉ lọc lớp bệnh, ảnh sinh lớp bệnh sẽ là ảnh "đã lọc" còn ảnh sinh lớp bình
thường là ảnh "ngẫu nhiên". Hai nhóm này khác nhau về thống kê ảnh (ví dụ SSIM, độ đa dạng), và khác biệt đó gắn với
nhãn, nên có thể thành tín hiệu phụ cho classifier.

### 5.6 Kích thước pool ($k$)

$k$ = 1,5: DASS giữ 2/3 pool. Giá trị $k$ đã giảm dần qua các lần chạy ISIC (4 → 3 → 2 → 1,5). Lý do được ghi nhận: chọn
càng gắt thì tập chọn càng lệch khỏi phân phối ảnh thật (với $k$ = 3, KID của tập chọn là 0,100, so với 0,029 của
pool). Thay đổi này được quyết định **sau khi đã xem kết quả test** của các lần chạy trước, nên phải nêu trong bài và
báo cáo mọi cấu hình đã chạy. $k$ và mọi tham số DASS dùng chung cho hai bộ dữ liệu.

### 5.7 Các biến thể so sánh

Mọi biến thể chọn **đúng $s$ ảnh từ cùng pool $P$** của mỗi lớp; chỉ khác tiêu chí:

| Mã | Tên trong bảng | Tiêu chí chọn |
|---|---|---|
| M0 | Imbalanced Baseline | không ảnh sinh, không ROS: chỉ ảnh thật (mất cân bằng) |
| M1 | ROS + Balanced GAN: Random | ngẫu nhiên (seed 2026; lớp 0: 2027) |
| M2 | … Visual margin ($M_v$) | $s$ ảnh có $M_v$ cao nhất |
| M3 | … Disease margin ($M_d$) | $s$ ảnh có $M_d$ cao nhất |
| M4 | … Dual margin ($M_v + M_d$) | $s$ ảnh có $B(x)$ cao nhất |
| M5 | … Diversity ($S_\text{div}$) | chỉ đa dạng: Thuật toán 1 với $B \equiv 0$, $\gamma = 1$ |
| M6 | … DASS (Ours) | Thuật toán 1 đầy đủ |

Các ablation trả lời: một hay hai không gian (M2, M3 so với M4), thêm đa dạng (M4 so với M6), thêm điểm lề (M5 so với
M6), và giá trị của việc chọn so với ngẫu nhiên (M6 so với M1).

---

## 6. Thiết kế tập huấn luyện (thiết kế B) và huấn luyện classifier

### 6.1 Tập huấn luyện ([pipeline/stages/train.py](../src/dass/pipeline/stages/train.py), [data/variants.py](../src/dass/data/variants.py))

**Bước 1. Cân bằng lớp bằng ảnh thật (ROS).** Cần thêm $n_0 - n_1$ bản sao ảnh thật lớp 1. Mỗi ảnh lớp 1 được nhân
bản $q = \lfloor (n_0 - n_1)/n_1 \rfloor$ lần, cộng thêm $r = (n_0 - n_1) \bmod n_1$ ảnh chọn ngẫu nhiên (seed 2026,
không lặp) được nhân thêm một lần. Số lần xuất hiện của các ảnh chênh nhau tối đa 1: đây là random oversampling tới
đúng 1 : 1 nhưng ít biến thiên hơn lấy mẫu có hoàn lại.

**Bước 2. Thêm cùng số ảnh sinh vào mỗi lớp.** $s = \mathrm{round}(q_s \cdot n_0)$ với $q_s$ = 0,5
(`balanced_synth_ratio`), chọn theo mục 5.

$$
\mathcal{T}_{j} \;=\; \underbrace{R_0 \cup R_1}_{\text{ảnh thật}} \;\cup\; \underbrace{\mathrm{ROS}(R_1)}_{n_0 - n_1 \text{ bản sao}} \;\cup\; \underbrace{G^{(j)}_1 \cup G^{(j)}_0}_{s \text{ ảnh sinh mỗi lớp}}, \qquad j = 1, \dots, 6
$$

| | ISIC | RSNA |
|---|---|---|
| Ảnh thật lớp 0 / lớp 1 | 618 / 148 | 7.522 / 1.059 |
| Bản sao ROS (lớp 1) | 470 (mỗi ảnh xuất hiện 4–5 lần) | 6.463 (mỗi ảnh xuất hiện 7–8 lần) |
| $s$ ảnh sinh mỗi lớp | 309 | 3.761 |
| Pool mỗi lớp $\lceil 1{,}5\, s\rceil$ | 464 | 5.642 |
| **Mỗi lớp trong tập train M1–M6** | **927** | **11.283** |
| Thành phần lớp 1: thật / nhân bản / sinh | 16 % / 51 % / 33 % | 9 % / 57 % / 33 % |
| Thành phần lớp 0: thật / sinh | 67 % / 33 % | 67 % / 33 % |

M0 dùng đúng ảnh thật (618 / 148; 7.522 / 1.059), không thêm gì.

*Lý do của thiết kế B:* nếu chỉ thêm ảnh sinh vào lớp bệnh (thiết kế A), "ảnh sinh" trở thành dấu hiệu của nhãn: trong
không gian E_v, ảnh sinh tách khỏi ảnh thật với AUC khoảng 0,99, dễ hơn nhiều so với việc tách hai lớp bệnh. Classifier
khi đó có thể học "ảnh GAN → bệnh". Ở thiết kế B, ảnh sinh chiếm đúng 1/3 mỗi lớp, nên nguồn ảnh **độc lập** với nhãn.
Mọi biến thể M1–M6 dùng **cùng** ảnh thật và **cùng** bản sao ROS, nên khác biệt giữa chúng chỉ đến từ ảnh sinh được
chọn.

Thiết kế A (ảnh sinh chỉ ở lớp 1, bù tới 1 : 1, kèm baseline ROS M0b) được giữ để tái lập các lần chạy cũ
(`configs/experiments/design_a.yaml`) và báo cáo riêng.

### 6.2 Classifier ([models/classifiers/__init__.py](../src/dass/models/classifiers/__init__.py), [engine/trainer.py](../src/dass/engine/trainer.py))

**Backbone** (pretrain ImageNet, Keras): **EfficientNetV2B0, ResNet50, ConvNeXtTiny**. DenseNet121 không dùng làm
classifier vì trùng kiến trúc với E_d. Mỗi backbone dùng đúng hàm chuẩn hoá đầu vào của nó.

**Đầu phân loại:** global average pooling → Dropout(0,3) → Dense(1, sigmoid). BatchNorm của backbone luôn ở chế độ
suy luận (kể cả khi fine-tune). Hàm mất mát: binary cross-entropy.

**Huấn luyện hai giai đoạn:**

| Giai đoạn | Lớp được train | Tối ưu | Số epoch |
|---|---|---|---|
| 1. Head | chỉ đầu phân loại (backbone đóng băng) | AdamW, lr $10^{-3}$ | 5 |
| 2. Fine-tune | toàn bộ | AdamW, lr $10^{-5}$, weight decay $10^{-4}$ | tối đa 30; dừng sớm sau 8 epoch không cải thiện |

- **Chọn epoch:** checkpoint có **ROC-AUC trên val cao nhất** trong giai đoạn 2. Giai đoạn 1 không được chọn.
- Ảnh vào 224 × 224, batch 16.
- **Không augmentation, không class weight** ở mọi biến thể và mọi bộ dữ liệu. M0 là baseline mất cân bằng thật; việc
  cân bằng chỉ đến từ dữ liệu (ROS + ảnh sinh) trong M1–M6. Augmentation chỉ dùng cho E_d (bảng dưới).
- **Seed:** 2026, 2027, 2028 (khởi tạo đầu phân loại, thứ tự xáo trộn, dropout). Mỗi tổ hợp backbone × biến thể × seed
  là một lần chạy: 3 × 7 × 3 = **63 lần chạy mỗi bộ dữ liệu**.
- **Dự đoán:** sau khi nạp checkpoint tốt nhất, dự đoán val, test (một lần) và ảnh **thật** của tập train (để đo khoảng
  cách train–test). Mọi dự đoán lưu trong `.npz` kèm giao thức train và phiên bản code.

Profile augmentation của E_d ([data/loaders.py](../src/dass/data/loaders.py)), áp dụng trực tuyến trên tập train:

| Profile | Dùng cho | Lật | Xoay | Zoom | Độ sáng / tương phản |
|---|---|---|---|---|---|
| `rotation_invariant` | ISIC (tổn thương da không có hướng) | ngang + dọc | ±180° | ±10 % | ±10 % |
| `upright` | RSNA (ảnh có hướng giải phẫu) | ngang | ±10° | ±10 % | ±10 % |

---

## 7. Đánh giá

### 7.1 Chỉ số ([evaluation/classification.py](../src/dass/evaluation/classification.py))

Trên tập test (ảnh thật), lớp dương là lớp 1:

- **Chỉ số chính: ROC-AUC** (không phụ thuộc ngưỡng, nên công bằng giữa M0 mất cân bằng và các biến thể cân bằng).
- Không phụ thuộc ngưỡng: PR-AUC (average precision).
- Ở **ngưỡng cố định 0,5** (không dò trên val hay test): sensitivity, specificity, balanced accuracy, G-mean
  ($\sqrt{\text{sens}\cdot\text{spec}}$), MCC, F1, macro-F1, precision, accuracy.

Mỗi chỉ số được báo cáo dạng mean ± độ lệch chuẩn qua 3 seed. Các chỉ số theo ngưỡng phụ thuộc tỉ lệ lớp lúc train: M0
có xác suất lệch về lớp đa số nên sensitivity thấp ở ngưỡng 0,5. Vì vậy so sánh chính dựa trên AUC.

### 7.2 Câu hỏi nghiên cứu và giả thuyết

| | Câu hỏi | So sánh | Loại |
|---|---|---|---|
| RQ1 | Chọn ảnh sinh bằng DASS có tốt hơn chọn ngẫu nhiên, khi mọi thứ khác giống hệt? | M6 vs M1 | **H1: AUC(M6) > AUC(M1)**, chính |
| RQ2 | Toàn bộ quy trình có tốt hơn không làm gì? | M6 vs M0 | **H2: AUC(M6) > AUC(M0)**, chính |
| RQ3 | Ảnh sinh cân bằng hai lớp **không chọn lọc** có giúp không? | M1 vs M0 | khám phá |
| RQ4 | Mỗi thành phần đóng góp gì? | M2, M3, M4, M5 vs M0 và giữa các biến thể | khám phá |
| RQ5 | Kết quả có nhất quán giữa backbone và bộ dữ liệu? | 3 backbone × 2 bộ dữ liệu | tiêu chí: chiều của H1, H2 giữ ở đa số cấu hình |
| RQ6 | Tập được chọn khác pool thế nào? | chỉ số chẩn đoán (mục 8) | mô tả |

H1 và H2 được **khai báo trước khi train** thiết kế B (`evaluation.primary_comparisons`).

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

- họ **chính**: H1, H2 (2 phép thử);
- họ **khám phá**: M1, …, M5 so với M0 (5 phép thử).

### 7.4 Phân tích nhóm con: tư thế chụp của RSNA ([evaluation/subgroups.py](../src/dass/evaluation/subgroups.py))

Khai báo trước (2026-10-04, trước khi có kết quả test RSNA), báo cáo dù kết quả thế nào. Tư thế AP gắn với bệnh nặng
(Oakden-Rayner et al., 2020; Janizek et al., 2020), nên classifier có thể dùng tư thế làm đường tắt.

1. **AUC trong từng nhóm AP / PA** của tập test: trong một nhóm mọi ảnh cùng tư thế, nên tư thế không còn giúp xếp hạng.
   AUC trong nhóm đo khả năng nhận ra bệnh thật sự. ΔAUC so với M0 trong từng nhóm, Holm theo backbone × nhóm.
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

---

## 9. Tái lập

- **Seed:** chia dữ liệu, M1, ROS, bootstrap: 2026 (M1 lớp 0: 2027); classifier: 2026–2028; E_d: 4242; pool: 777 (lớp 1),
  778 (lớp 0); KID khi train GAN: 123. Phép toán GPU không tất định (`deterministic: false`), nên chạy lại có thể chênh
  rất nhỏ.
- **Mỗi lần chạy** (`run_tag`: ISIC `v13` / `v13_srv`, RSNA `rsna_v4` / `rsna_v4_srv`) ghi `run_manifest.json` gồm
  commit, phiên bản thư viện và cấu hình của từng stage.
- **Mỗi file dự đoán `.npz`** ghi giao thức train (augmentation, class weight, tiêu chí chọn epoch, thiết kế, lọc lớp đa
  số) và phiên bản code. Thư mục dự đoán trộn hai giao thức sẽ bị từ chối khi đánh giá.
- **Split** được lưu và băm; GAN dùng lại phải khớp split của nó.
- Lệnh: `dass -c configs/server/<bộ dữ liệu>.yaml run` (server) hoặc notebook `notebooks/colab_pipeline.ipynb` (Colab).

---

## 10. Giới hạn đã biết của phương pháp

Các điểm sau suy ra từ thiết kế và đã được kiểm tra trên dữ liệu khi có thể. Nên nêu trong phần thảo luận / hạn chế
của bài.

1. **Lề là đại lượng tương đối.** Ảnh xa mọi ảnh thật (ảnh lỗi) có $S^{+}$ và $S^{-}$ cùng thấp, nên $M \approx 0$ và
   không bị đẩy xuống cuối bảng xếp hạng. DASS loại ảnh "giống lớp kia", không loại trực tiếp ảnh "không giống ảnh
   thật". Kiểm tra độ thật chỉ có ở cấp checkpoint (KID).
2. **Lệch do kích thước tập mốc.** Tìm 5 láng giềng trong lớp có nhiều ảnh hơn ($n_0 \gg n_1$) dễ gặp ảnh giống hơn, nên
   $S^{-}$ của ảnh sinh lớp 1 bị đẩy lên có hệ thống. Thứ hạng trong pool vẫn dùng được, nhưng dấu của $M$ không
   diễn giải được (ở RSNA, chỉ 22 % pool có $M_d > 0$).
3. **Chuẩn hoá min-max** cho $\tilde M_v$ và $\tilde M_d$ trọng số ngang nhau, bất kể lề nào mang nhiều thông tin hơn
   (RSNA, probe trên val: E_v 0,70, E_d 0,78).
4. **Thành phần đa dạng ưu tiên ngoại lai** (tính chất của k-center).
5. **E_d là một mô hình đơn lẻ** và học trên chính các ảnh làm mốc $R_0, R_1$, nên $S_d$ lạc quan với ảnh giống ảnh
   train. Tuỳ chọn cross-fitting (`encoder.e_d_crossfit_folds`) có sẵn nhưng không dùng trong lần chạy chính.
6. **Checkpoint GAN được chọn theo KID lớp 1**, trong khi thiết kế B dùng ảnh sinh của cả hai lớp. KID của lớp 0 được
   báo cáo nhưng không dùng để chọn.
7. **Độ nhạy của H1 bị giới hạn bởi $k$.** M1 lấy ngẫu nhiên $s$ trong $1{,}5\,s$ ảnh, nên trung bình 2/3 số ảnh của
   tập DASS cũng có trong M1. Vì ảnh sinh chiếm 1/3 mỗi lớp, tập train của M6 và M1 chỉ khác nhau khoảng 11 % số ảnh
   mỗi lớp.
8. **Thiết kế B không có baseline chỉ ROS**, nên không tách được đóng góp của **nội dung** ảnh sinh với việc **có thêm
   dữ liệu train**. Câu hỏi này được trả lời ở thiết kế A (M6 vs M0b).
9. **Các thay đổi sau khi xem kết quả test:** $k$ (4 → 1,5), tiêu chí chọn epoch (`val_macro_recall` → `val_auc`),
   tắt augmentation và class weight của classifier. Mọi cấu hình đã chạy được giữ nguyên và phải được báo cáo. Việc
   lọc cả lớp 0 chưa được kiểm chứng trên val trước lần chạy chính `rsna_v4`.
10. **Giai đoạn chỉ train đầu phân loại không được chọn làm epoch tốt nhất.** Với backbone dễ overfit (ResNet50), epoch
    fine-tune đầu tiên thường đã là tốt nhất.

---

## 11. Bảng tham số

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Dữ liệu | kích thước lưu / kích thước vào mạng | 256 / 224 |
| | split | ISIC: val 15 % (giữ test chính thức); RSNA: test 15 %, val 17,6 % phần còn lại |
| GAN | StyleGAN2-ADA có điều kiện | `paper256`, batch 16, R1 $\gamma$ 1, ADA target 0,6 |
| | lật ngang | ISIC có, RSNA không |
| | dừng sớm theo KID lớp 1 | snapshot 100 kimg, tối thiểu 400 kimg, kiên nhẫn 5, giảm tối thiểu 2 %, tối đa 3.000 kimg |
| | sinh pool | $\psi$ = 1, seed 777 / 778 |
| DASS | $K$ láng giềng | 5 |
| | $\lambda_v, \lambda_d$ | 1, 1 |
| | $\alpha, \beta, \gamma$ | 1, 1, 0,5 |
| | $k$ (pool / số chọn) | 1,5 |
| Thiết kế B | $q_s$ (ảnh sinh mỗi lớp / ảnh thật lớp 0) | 0,5 |
| | lọc lớp 0 | có, lề đảo chiều |
| E_v | mạng | EfficientNet-B0 ImageNet, đóng băng, 1.280 chiều |
| E_d | mạng | DenseNet121, fine-tune trên ảnh thật, 1.024 chiều |
| | huấn luyện | 5 + tối đa 20 epoch, augmentation, class weight, seed 4242 |
| Classifier | backbone | EfficientNetV2B0, ResNet50, ConvNeXtTiny |
| | đầu | GAP → Dropout 0,3 → Dense sigmoid |
| | giai đoạn 1 | 5 epoch, AdamW lr $10^{-3}$ |
| | giai đoạn 2 | tối đa 30 epoch, AdamW lr $10^{-5}$, weight decay $10^{-4}$, dừng sớm kiên nhẫn 8 |
| | chọn epoch | val ROC-AUC |
| | augmentation / class weight | không / không |
| | batch / seed | 16 / 2026, 2027, 2028 |
| Đánh giá | ngưỡng | 0,5 (cố định) |
| | bootstrap | 2.000 lần, phân tầng theo lớp, trên ensemble seed |
| | hiệu chỉnh | Holm trong từng backbone, riêng họ chính / họ khám phá |
| | KID / coverage | 50 tập con × tối đa 1.000 ảnh / $k$ = 5 |

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
