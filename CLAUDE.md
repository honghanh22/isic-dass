# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này.

## Tổng quan

Package Python `isic_dass` (thư mục `src/`) cho pipeline nghiên cứu: StyleGAN2-ADA có điều kiện + chọn ảnh sinh bằng DASS để xử lý mất cân bằng lớp trên ISIC 2016 (benign / malignant). Xem [README.md](README.md) để biết luồng và artefact.

Chú thích, docstring và log viết bằng **tiếng Việt**; tên hàm/biến bằng tiếng Anh. Giữ quy ước này.

`notebooks/archive/` là notebook gốc v5 (có output, khoảng 10 MB). Đây là tài liệu tham khảo: **không sửa** và không dùng làm nguồn sự thật. Nguồn sự thật là `src/`.

## Lệnh

```bash
pip install -e ".[dev]"     # cục bộ (không cần GPU)
pytest                      # ~10 s; test chỉ dùng numpy/sklearn/PIL
ruff check src tests
isic-dass --config configs/default.yaml show-config
isic-dass --config configs/smoke.yaml <stage>        # chạy thử nhanh trên Colab, ghi vào *_smoke
```

Các stage thật (`gan-*`, `generate`, `frequency`, `select`, `train`) cần Colab có GPU cùng Google Drive đã mount, nên **không chạy được trên máy Windows cục bộ**. Kiểm tra cục bộ gồm `pytest`, `ruff` và `python -m compileall src`.

## Kiến trúc

- `config.py`: `Config` là các dataclass lồng nhau, nạp từ YAML và ghi đè bằng `--set a.b=v`; khoá không tồn tại thì báo lỗi. `Layout` là **nơi duy nhất** định nghĩa đường dẫn. Thêm artefact mới thì thêm thuộc tính vào `Layout`, không ghép đường dẫn rải rác trong code.
- `pipeline.py`: mỗi `stage_*` là một bước độc lập. Bước đó đọc artefact từ Drive, ghi artefact lên Drive, và có thể chạy lại. `Context.create()` chuẩn bị dữ liệu cục bộ (idempotent), nạp split và tính `ClassBudget` (lớp thiểu số, `n_select`, `pool_size`). `resolve_candidate_pool()` khôi phục pool gốc rồi áp các bước harmonize/mitigation theo config, mỗi bước có cache zip riêng.
- `cli.py` gọi stage. Mỗi stage chạy trong **một tiến trình riêng** để PyTorch (GAN) và TensorFlow (classifier) không tranh VRAM.
- Quy tắc import:
  - `classify/*` và `selection/encoders.py` import TensorFlow ở đầu module.
  - `gan/metrics.py` và `gan/train.py` chỉ import torch bên trong hàm.
  - Mọi module khác chỉ dùng numpy/sklearn/PIL để test chạy được mà không cần GPU.
  - `pipeline.py` import module nặng **bên trong** từng stage.
  - Đừng thêm import TF/torch ở cấp module vào những file đang không có.
- StyleGAN2-ADA không được vendor vào repo. `gan/setup.py` clone repo NVlabs rồi áp `PATCHES`. Mỗi bản vá phải idempotent (kiểm tra `new in text` trước) và báo lỗi nếu không tìm thấy `old`. Muốn thêm bản vá thì thêm một `Patch(...)`, kèm `reason`.
- `train.py` và bước sinh ảnh của StyleGAN chạy trong tiến trình con (`gan/_generate_worker.py`).

## Nguyên tắc phương pháp (không được phá vỡ)

- **Val và test luôn 100 % ảnh thật.** GAN, `E_d` và DASS không bao giờ thấy val hoặc test. `assert_clean_eval_sets` kiểm tra điều này trước khi train.
- Val **chỉ** dùng để chọn epoch, theo `classifier.monitor`. Metric test tính ở **ngưỡng cố định 0,5**; không dò ngưỡng trên val hay test.
- Test chỉ được dự đoán một lần, sau khi đã nạp lại checkpoint tốt nhất.
- Mọi ảnh (thật, sinh, test) lưu **PNG**.
- `E_d` phải là model riêng, khác baseline M0 (kiến trúc `encoder.e_d_model` và seed riêng).
- Generator dùng về sau là **snapshot có KID tốt nhất** (`best.pkl`).
- Mọi biến thể M1–M6 chọn đúng `n_select` ảnh từ **cùng một pool**. Baseline `M0_real_only` dùng class weight.

## Tương thích artefact trên Drive

Các tên sau được giữ nguyên để dùng lại kết quả cũ: `gan_state.json`, `best.pkl`, `latest.pkl`, `real_val_split.json`, `selections.json` (dạng `{method: [tên file]}`), `pool_<lớp>_<tag>.zip`, `Ed_<model>_s<seed>.weights.h5`, `<model>__<variant>__s<seed>.npz` (có trường `k` = `pool_mult`, `lam`).

Khi đổi định dạng những file này, cập nhật cả code đọc lẫn code ghi, và ghi vào [CHANGELOG.md](CHANGELOG.md). Không bao giờ xoá artefact trên Drive; muốn tách lần chạy mới thì đổi `paths.run_tag` (hoặc `paths.gan_tag`).

## Mở rộng thường gặp

- **Thêm classifier:** thêm builder vào `MODEL_BUILDERS` trong `classify/models.py`. CLI tự nhận tên mới.
- **Thêm biến thể chọn ảnh:** thêm vào `METHODS` và `select_all_methods` trong `selection/dass.py`, cập nhật test trong `tests/test_selection.py`.
- **Thêm tham số:** thêm field vào dataclass tương ứng trong `config.py` **và** vào `configs/default.yaml`. Test `test_default_yaml_matches_dataclass_defaults` sẽ báo lỗi nếu giá trị ở hai nơi khác nhau.
- **Thêm stage:** viết `stage_xxx(cfg)` trong `pipeline.py`, thêm subparser trong `cli.py`, thêm một cell vào `notebooks/colab_pipeline.ipynb`.
