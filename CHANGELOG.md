# Changelog

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
