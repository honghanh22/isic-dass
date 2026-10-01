# isic-dass

Pipeline nghiên cứu xử lý **mất cân bằng lớp** trên **ISIC 2016 (Part 3, benign / malignant)**:

1. Train **StyleGAN2-ADA có điều kiện** (bản chính thức NVlabs) trên toàn bộ ảnh train thật, early stopping theo **KID** của lớp thiểu số.
2. Sinh một **candidate pool** ảnh malignant.
3. Chọn ảnh sinh bằng **DASS** — điểm lề trong hai không gian đặc trưng `E_v` (EfficientNet-B0 ImageNet) và `E_d` (DenseNet121 fine-tune trên ảnh thật) cộng thành phần đa dạng.
4. So sánh 7 biến thể **M0–M6** bằng nhiều classifier × nhiều seed, kiểm định **paired bootstrap ΔAUC** so với baseline.

Kèm các bước kiểm soát shortcut: kiểm tra thật-vs-sinh, fingerprint miền tần số (Frank et al., ICML 2020) và spectral mitigation tuỳ chọn (Dong et al., CVPR 2022).

## Cấu trúc

```
.
├── configs/
│   ├── default.yaml          # cấu hình chuẩn (khớp notebook v5/v7)
│   └── smoke.yaml            # chạy thử nhanh, ghi vào thư mục *_smoke
├── notebooks/
│   ├── colab_pipeline.ipynb  # notebook chạy pipeline trên Colab (chỉ gọi CLI)
│   ├── colab_gpu_check.ipynb # kiểm tra GPU của runtime
│   └── archive/              # notebook gốc v5 (tham khảo, không bảo trì)
├── src/isic_dass/
│   ├── config.py             # Config (dataclass) + Layout (mọi đường dẫn)
│   ├── pipeline.py           # các stage; Context dùng chung
│   ├── cli.py                # `isic-dass <stage>`
│   ├── data/                 # ingest, preprocess, split, variants
│   ├── gan/                  # setup (vá repo), dataset, metrics (KID), train, generate
│   ├── frequency/            # spectrum, fingerprint, harmonize, mitigation
│   ├── selection/            # encoders (E_v, E_d), scoring, dass, diagnostics
│   ├── classify/             # tf.data, models, MacroRecall, train 2 giai đoạn
│   └── evaluation/           # metrics, stats (bootstrap), aggregate, quality
└── tests/                    # pytest, chạy được không cần GPU / TensorFlow
```

## Chạy trên Google Colab

Mở `notebooks/colab_pipeline.ipynb` trên Colab (runtime GPU), điền `REPO_URL` (hoặc đường dẫn dự án trên Drive) rồi chạy lần lượt. Tương đương với:

```bash
pip install -e . ninja click keras-hub          # torch / tensorflow đã có sẵn trên Colab
isic-dass --config configs/default.yaml prepare
isic-dass --config configs/default.yaml gan-setup
isic-dass --config configs/default.yaml gan-train       # Colab ngắt -> chạy lại, tự resume
isic-dass --config configs/default.yaml gan-report
isic-dass --config configs/default.yaml generate
isic-dass --config configs/default.yaml frequency       # tuỳ chọn
isic-dass --config configs/default.yaml select
isic-dass --config configs/default.yaml train --model ResNet50 --seeds 2026 2027 2028
isic-dass --config configs/default.yaml aggregate
isic-dass --config configs/default.yaml quality
```

Ghi đè cấu hình không cần sửa file: `--set classifier.monitor=val_auc --set selection.gamma=0.25`.

Dữ liệu gốc đặt trên Drive tại `paths.drive_root`:

```
ISBI2016_ISIC_Part3/
├── ISBI2016_ISIC_Part3_Training_Data/*.jpg
├── ISBI2016_ISIC_Part3_Training_GroundTruth.csv
├── ISBI2016_ISIC_Part3_Test_Data/*.jpg
└── ISBI2016_ISIC_Part3_Test_GroundTruth.csv
```

## Artefact

| Artefact | Vị trí (dưới `drive_root`) | Tạo bởi |
|---|---|---|
| Split train/val cố định | `checkpoints_<run>/data/real_val_split.json` | `prepare` |
| GAN (`best.pkl`, `latest.pkl`, `gan_state.json`) | `checkpoints_<gan>/stylegan2ada/` | `gan-train` |
| Candidate pool (zip) | `checkpoints_<run>/data/pool_<lớp>_from<kimg>kimg_n<N>.zip` | `generate` |
| Lựa chọn M0–M6, embeddings | `checkpoints_<run>/data/selections.json`, `embeddings.npz` | `select` |
| Trọng số E_d / classifier | `checkpoints_<run>/classifiers/` | `select`, `train` |
| Dự đoán từng lần chạy | `results_<run>/predictions/<model>__<variant>__s<seed>.npz` | `train` |
| Bảng kết quả, hình | `results_<run>/*.csv`, `results_<run>/**/*.png` | mọi stage |

`<gan>` = `paths.gan_tag`, `<run>` = `paths.run_tag`. Tên file giữ tương thích với notebook v5/v7 nên artefact cũ trên Drive được dùng lại.

## Phát triển

```bash
pip install -e ".[dev]"
pytest          # ~10 s, không cần GPU
ruff check src tests
```

Các nguyên tắc phương pháp (val/test 100 % ảnh thật, ngưỡng cố định 0,5, ...) và hướng dẫn mở rộng: xem [CLAUDE.md](CLAUDE.md). Thay đổi so với notebook gốc: [CHANGELOG.md](CHANGELOG.md).

## Tham khảo

- Karras et al., *Training Generative Adversarial Networks with Limited Data* (StyleGAN2-ADA), NeurIPS 2020 — [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition*, ICML 2020 — [RUB-SysSec/GANDCTAnalysis](https://github.com/RUB-SysSec/GANDCTAnalysis)
- Dong et al., *Think Twice Before Detecting GAN-generated Fake Images from their Spectral Domain Imprints*, CVPR 2022
- CosSIF — lọc ảnh sinh theo cosine similarity cho dữ liệu y tế mất cân bằng
