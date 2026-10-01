"""DATA LOADER — mọi khác biệt giữa các bộ dữ liệu nằm ở đây và chỉ ở đây:

1. `sources/`   đọc nguồn (CSV + thư mục phẳng | thư mục theo lớp)
2. `transforms` tiền xử lý (cắt viền, đệm vuông, resize) — `image_io` giữ đúng số kênh gốc
3. `splits/`    chiến lược chia (holdout_val | stratified [theo nhóm] | file)

`loaders` (tf.data) import TensorFlow — chỉ dùng trong stage cần TF.
"""
