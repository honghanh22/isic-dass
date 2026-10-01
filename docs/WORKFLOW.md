# Hướng dẫn làm việc: chạy, đồng bộ code, thêm bộ dữ liệu, tổ chức code

## 1. Ba nơi và vai trò

```
┌──────── Máy bạn (VS Code) ────────┐   Upload to Colab   ┌────── Máy Colab (GPU) ──────┐
│ src/dass/   ← CODE (nguồn sự thật) │ ──────────────────▶ │ /content/src, /content/...  │
│ configs/    ← THAM SỐ              │                     │ chạy lệnh `dass`             │
│ notebooks/colab_pipeline.ipynb     │ ── chạy cell ─────▶ │                              │
│   (chỉ bấm chạy + xem kết quả)     │ ◀── output cell ─── │                              │
└────────────────────────────────────┘                     └──────────────┬───────────────┘
                                                                          │ đọc / ghi
                                                           ┌──────────────▼───────────────┐
                                                           │ Google Drive (lưu lâu dài)    │
                                                           │ dữ liệu gốc, GAN, kết quả     │
                                                           └──────────────────────────────┘
```

- Code chỉ sửa ở `src/dass/` trên máy bạn; máy Colab giữ bản sao, mất khi server bị xoá.
- Dữ liệu, checkpoint, kết quả nằm trên Drive.
- `notebooks/colab_pipeline.ipynb` chỉ gọi lệnh `dass`. Sửa notebook = sửa `scripts/build_colab_notebook.py` rồi chạy lại.

Cấu trúc code và cách mở rộng: [ARCHITECTURE.md](ARCHITECTURE.md). Định dạng kết quả: [RESULTS_FORMAT.md](RESULTS_FORMAT.md).

## 2. Cấu hình

```
configs/_base_/        siêu tham số dùng chung — MỘT công thức cho mọi bộ dữ liệu
configs/datasets/      chỉ phần phụ thuộc dữ liệu: paths, nguồn, số kênh, tiền xử lý, cách chia
configs/experiments/   thực nghiệm = _base_ + dataset (+ ghi đè); smoke.yaml ghép được với mọi thực nghiệm
```

| Muốn | Cách |
|---|---|
| Đổi một tham số cho một lần chạy | `--set selection.gamma=0.25` |
| Tách thư mục kết quả cho ablation | `--tag gamma025` (thêm hậu tố vào `run_tag`) |
| Chạy thử nhanh | thêm `-c configs/experiments/smoke.yaml` |
| Thực nghiệm cố định trong bài | file mới trong `configs/experiments/` với `_base_` trỏ tới thực nghiệm gốc |

## 3. Chạy lần đầu trên một server Colab mới

1. Mở `notebooks/colab_pipeline.ipynb` trong VS Code → **Select Kernel → Colab → New Colab Server** → chọn GPU.
2. Explorer: chọn `src`, `configs`, `pyproject.toml`, `README.md` → chuột phải → **Upload to Colab**.
3. Chạy cell mount Drive, cell lấy mã nguồn (tự `pip install -e .`), cell chọn `EXPERIMENT`.
4. Chạy các stage theo thứ tự, hoặc một lệnh: `!dass {CFG} run`.

## 4. Vòng làm việc hằng ngày

1. Chạy một cell → **Ctrl+S** (output được lưu vào file `.ipynb` trên máy bạn).
2. Nhắn Claude Code: "đọc output cell X", "cell Y lỗi", "thêm backbone Z", …
3. Claude sửa `src/` / `configs/`, chạy `pytest`.
4. Upload lại thư mục vừa sửa → chạy lại đúng cell đó. Không cần cài lại (cài `-e`), trừ khi `pyproject.toml` đổi.

Mọi stage chạy lại được: GAN, pool, lựa chọn, dự đoán `.npz` đã có được khôi phục / bỏ qua.
`select` không ghi đè `selections.json` đã có (dự đoán gắn với nó) — muốn chọn lại: `select --force` và `--tag` mới.

## 5. Thêm một bộ dữ liệu (nhị phân)

1. Đặt dữ liệu lên Drive theo một trong hai dạng: thư mục theo lớp (`<root>/<lớp>/*`) hoặc thư mục ảnh phẳng + CSV
   nhãn (cột 1 image_id, cột 2 tên lớp hoặc 0/1).
2. Copy `configs/datasets/_template.yaml` → `configs/datasets/<tên>.yaml`, sửa nguồn, lớp, tiền xử lý, cách chia:

   | Cách chia | Cấu hình | Khi nào |
   |---|---|---|
   | `holdout_val` | `val: 0.15` | có tập test riêng |
   | `stratified` | `test: 0.15, val: 0.176` (+ `group_regex`) | một nguồn ảnh, tự tách test (theo bệnh nhân nếu có mã) |
   | `file` | `file: my_split.csv` (`image_id,split`) | cách chia cố định cho trước |

3. Tạo `configs/experiments/<tên>_dass.yaml` (copy `brain_tumor_dass.yaml`, đổi dòng dataset cuối).
4. `dass -c configs/experiments/<tên>_dass.yaml run`. Bộ dữ liệu mới cần GAN mới (`gan_tag` mới).

`data.channels: auto` tự nhận diện ảnh xám / màu; ảnh xám được giữ 1 kênh xuyên suốt.
Đổi cách chia / seed mà giữ `run_tag` → lệnh báo lỗi (kết quả cũ gắn với split cũ). Pipeline chỉ hỗ trợ 2 lớp.

## 6. Khi có một notebook / ý tưởng mới

| Trường hợp | Cách tổ chức |
|---|---|
| Cùng pipeline, khác dữ liệu / cách chia | chỉ thêm YAML (mục 5) |
| Ý tưởng mới trên cùng pipeline | thêm vào đúng tầng (`selection/`, `models/classifiers/`, …) + tham số + test |
| Dự án khác hẳn | repo riêng, dùng lại cấu trúc này |

Notebook cũ: đặt vào `notebooks/archive/`, nhờ Claude chuyển thành module (như đã làm với ISIC v5 và Brain Tumor v1).

## 7. Đồng bộ code giữa máy bạn và Colab

| Cách | Làm thế nào | Ưu / nhược |
|---|---|---|
| **Upload to Colab** | chuột phải `src` → Upload to Colab | không cần cài gì; làm tay mỗi lần sửa / mỗi server mới |
| **Google Drive for Desktop** | dự án trong thư mục Drive → Colab chạy từ `/content/drive/MyDrive/dass` | tự đồng bộ; Claude đọc được kết quả trên Drive |
| **Git + GitHub** | `git push` trên máy → `git pull` trên Colab | có lịch sử phiên bản; khuyến nghị lâu dài |

## 8. Đặt tên lần chạy

- `paths.gan_tag`: phiên bản GAN — chỉ đổi khi train GAN mới.
- `paths.run_tag` (+ `--tag`): phiên bản thí nghiệm (split, lựa chọn, classifier).
- Không bao giờ xoá artefact trên Drive; so sánh các lần chạy bằng các `results_<run_tag>/`.

## 9. Lỗi thường gặp

| Lỗi | Nguyên nhân / xử lý |
|---|---|
| `No module named 'google.colab'` | kernel là Python trên máy bạn → Select Kernel → Colab |
| `dass: command not found` / `No module named 'dass'` | chưa chạy cell cài đặt, hoặc upload `src/dass` thay vì `src` → chạy lại cell đó |
| `Lỗi cấu hình: …` | khoá sai / giá trị không hợp lệ — thông báo chỉ rõ khoá nào |
| `Cách chia hiện tại khác split đã lưu` | đổi `data.*` / `seed` mà giữ `run_tag` → dùng `--tag` hoặc `run_tag` mới |
| `Split khác split tham chiếu` | split không trùng split GAN đã train → giữ cấu hình chia, hoặc train GAN mới và bỏ `split.expected` |
| `ChannelMismatchError` khi `sample` | GAN 3 kênh sinh ảnh có màu cho bộ dữ liệu ảnh xám → train GAN 1 kênh (`gan_tag` mới, `dass gan`) |
| `Chưa có GAN đã train` / `Chưa có candidate pool` / `Chưa có selections.json` | chạy `gan` / `sample` / `select` trước |
| Hết bộ nhớ GPU | giảm `classifier.batch_size` / `generator.batch`, hoặc GPU lớn hơn |
