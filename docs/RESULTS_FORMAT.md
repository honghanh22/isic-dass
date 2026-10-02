# Định dạng kết quả

Mọi file nằm trên Drive dưới `<drive_root>/results_<run_tag>/`.

## `metrics/` — số liệu thô (`.csv` + `.json`, cùng nội dung)

| File | Một dòng là | Cột chính |
|---|---|---|
| `classification_runs` | một lần chạy (model × phương pháp × seed) | `model, method, lam, k, seed, best_epoch, n_test, val_roc_auc, roc_auc, pr_auc, f1, macro_f1, sensitivity, specificity, balanced_accuracy, g_mean, mcc, precision, accuracy` |
| `classification_summary` | (model, phương pháp) | `<metric>_mean`, `<metric>_std` (độ lệch chuẩn mẫu qua seed), `n_seeds` |
| `significance_vs_baseline` | (model, phương pháp, đối chứng): mọi phương pháp vs M0, cộng các cặp trong `evaluation.comparisons` (M6 vs M0b, M6 vs M1) | `vs` (đối chứng), `delta_auc, ci95_low, ci95_high, p_value, n_seeds` (paired bootstrap trên xác suất trung bình qua seed chung) |
| `generation_quality` | một tập ảnh | `set` (`reference` / `pool` / `selected`), `method, n, n_real, kid, kid_std, fid, diversity, ssim, auc_real_vs_synth` |
| `probe_auc` | không gian đặc trưng × tập | `space, subset, auc_probe` (chỉ dòng `val` là khách quan) |
| `shortcut_check` | phương pháp | `auc_5fold` = AUC tách ảnh thật / sinh lớp thiểu số (E_v); dòng `reference` = hai lớp bệnh thật |
| `selection_jaccard` | phương pháp | Jaccard giữa tập ảnh được chọn của các phương pháp |
| `gan_kid_history` | snapshot GAN | `cum_kimg, kid, improved` |
| `frequency_fingerprint` | (tuỳ chọn) | `auc_frank_cnn, auc_frank_regression, chenh_lech_kenh_that, chenh_lech_kenh_sinh` |

Quy ước:
- Metric phân loại tính trên **test**, ngưỡng cố định `evaluation.threshold` (0.5); lớp dương = chỉ số 1.
- KID / FID trên đặc trưng Inception-v3 (2048 chiều), so với ảnh **thật lớp thiểu số của tập train**.
  KID là chỉ số chính (mean ± std qua `kid_subsets` tập con). FID chỉ để tham khảo: chệch khi số ảnh < 2048.
- Hàng `reference / real val vs real train` là mức nền của KID / FID giữa hai tập ảnh thật — giá trị của ảnh sinh
  nên được đọc tương đối so với hàng này.

## `tables/` — bảng cho bài báo

Mỗi bảng: `.csv` (đã định dạng `mean ± std`), `.json` (số thô), `.tex` (booktabs; in đậm giá trị tốt nhất mỗi cột
trong từng model / nhóm; KID, FID, SSIM, AUC thật-vs-sinh: thấp hơn là tốt hơn).

| Bảng | Nguồn |
|---|---|
| `dataset` | `checkpoints_<run_tag>/data/dataset_card.json` |
| `classification` | `metrics/classification_summary` |
| `significance` | `metrics/significance_vs_baseline` |
| `generation_quality` | `metrics/generation_quality` |

## Khác

- `run_manifest.json`: config đã hợp nhất, phiên bản thư viện, thời điểm + lệnh của từng stage.
- `checkpoints_<run_tag>/data/dataset_card.json`: số kênh, số ảnh mỗi tập / lớp, tham số chia, SHA-1 của split.
- `predictions/<model>__<variant>__s<seed>.npz`: `y_val, p_val, y_test, p_test` + metadata — nguồn của mọi bảng phân loại.
