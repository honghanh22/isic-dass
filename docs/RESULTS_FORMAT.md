# Định dạng kết quả

Mọi file nằm trên Drive dưới `<drive_root>/results_<run_tag>/`.

## `metrics/` — số liệu thô (`.csv` + `.json`, cùng nội dung)

| File | Một dòng là | Cột chính |
|---|---|---|
| `classification_runs` | một lần chạy (model × phương pháp × seed) | `model, method, lam, k, seed, best_epoch, n_test, train_roc_auc, val_roc_auc, roc_auc, pr_auc, f1, macro_f1, sensitivity, specificity, balanced_accuracy, g_mean, mcc, precision, accuracy` |
| `classification_summary` | (model, phương pháp) | `<metric>_mean`, `<metric>_std` (độ lệch chuẩn mẫu qua seed), `n_seeds` |
| `significance_vs_baseline` | (model, phương pháp, đối chứng): mọi phương pháp vs B0, cộng các cặp trong `evaluation.comparisons` (M6 vs M1, M6 vs B2; cặp trùng phép so với B0 chỉ tính một lần) | `vs` (đối chứng), `family` (`primary` = cặp trong `evaluation.primary_comparisons`, còn lại `exploratory`), `delta_auc, ci95_low, ci95_high, p_value, n_seeds` (paired bootstrap trên xác suất trung bình qua seed chung — dao động do mẫu test); `p_holm` (Holm trong mỗi model × họ `family`); `delta_auc_seed_mean, delta_auc_seed_std` (ΔAUC ghép cặp theo seed — dao động giữa các lần train) |
| `classification_by_subgroup_runs` | (lần chạy, nhóm con) — chỉ khi `evaluation.subgroup_columns` | `attribute, group` (ví dụ `ViewPosition`, `AP`), `n, n_pos` + metric |
| `classification_by_subgroup` | (model, phương pháp, nhóm con) | `roc_auc_*`, `pr_auc_*` (mean / std qua seed), `n, n_pos, n_seeds` |
| `significance_by_subgroup` | (model, phương pháp, đối chứng, nhóm con) | như `significance_vs_baseline`, `p_holm` trong mỗi model × nhóm |
| `subgroup_reference` | (thuộc tính, giá trị) trên test | `share_in_positive, share_in_negative, auc_attribute_only` (AUC khi CHỈ dùng thuộc tính) |
| `subgroup_share` | tập ảnh (thật / pool / ảnh mỗi phương pháp chọn) | `true_share` (theo metadata), `predicted_share` (probe logistic trên E_v), `probe_auc` |
| `generation_quality` | một tập ảnh của một lớp | `class` (lớp có ảnh sinh: cả hai lớp, mỗi lớp so với ảnh thật CÙNG lớp của train; file trước 1.21.1 không có cột này = lớp thiểu số), `set` (`reference` / `pool` = đúng phần pool đã dùng để chọn / `selected`), `method, n, n_real, kid, kid_std, fid, diversity, ssim, auc_real_vs_synth, nn_dist_median, nn_dist_ratio, near_copy_rate`; từ 1.22.0 thêm `coverage, density` (Naeem et al., 2020, k = `evaluation.coverage_k`, so với ảnh thật train cùng lớp, đặc trưng Inception) |
| `selection_quality` | tập ảnh sinh lớp thiểu số (mốc ảnh bệnh thật / pool / mỗi phương pháp) | `set, method, n`; **disease margin** `md_mean, md_median, md_std, md_pos_share` (tỉ lệ ảnh có M_d > 0) và `mv_mean, mv_median` (E_d / E_v, từ `embeddings.npz`); **đa dạng** `diversity_ev` (1 − cosine trung bình mọi cặp) và `nn_within_ev` (khoảng cách tới ảnh gần nhất trong tập) trong E_v; **coverage / density** `coverage_d, density_d` (E_d) và `coverage_v, density_v` (E_v) so với ảnh bệnh thật của train. Không cần GPU (tính cả khi `--skip-generative`). Coverage / density phụ thuộc số ảnh: chỉ so giữa các phương pháp cùng số ảnh |
| `probe_auc` | không gian đặc trưng × tập | `space, subset, auc_probe` (dòng `val` sát thực tế hơn; vẫn hơi lạc quan vì E_d dùng val để chọn epoch) |
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

Bảng dùng **tên hiển thị** của phương pháp (`evaluation.method_labels`, ví dụ "ROS + Balanced GAN: DASS (Ours)")
theo đúng thứ tự khai báo. Hai bảng của bài: `table_comparison`, `table_ablation` (+ `_auc_by_split`), theo `evaluation.result_tables`. Công thức
`$...$` trong tên được giữ trong `.tex` và bỏ ký hiệu LaTeX trong `.csv` (`$S_{\text{div}}$` -> `S_div`). File `.json`
và `metrics/` giữ **mã nội bộ** (`M6_dass`, …).

| Bảng | Nguồn |
|---|---|
| `dataset` | `checkpoints_<run_tag>/data/dataset_card.json` |
| `classification` | `metrics/classification_summary` |
| `significance` | `metrics/significance_vs_baseline` (cột `Family`: Primary / Exploratory, họ chính xếp trước) |
| `auc_by_split` | `metrics/classification_runs`: ROC-AUC train (ảnh **thật** của tập train, không gồm ảnh nhân bản / ảnh sinh) / val / test, mean ± std qua seed, cột Train − Test |
| `classification_by_subgroup`, `significance_by_subgroup`, `subgroup_reference`, `subgroup_share` | `metrics/` cùng tên |
| `generation_quality` | `metrics/generation_quality` (cột Coverage / Density từ 1.22.0) |
| `selection_quality` | `metrics/selection_quality`: M_d, M_v, đa dạng E_v, coverage / density (E_d, E_v) của từng tập được chọn |

Chú thích bảng (`caption` trong `.tex`) viết bằng tiếng Anh, cùng ngôn ngữ với tên phương pháp.

## Khác

- `run_manifest.json`: cấp trên cùng = lần ghi gần nhất (config, phiên bản, commit); `stages.<stage>` lưu riêng thời
  điểm, lệnh, `dass_version`, `code_version` (commit git) và config của **từng** stage (ví dụ `train:ResNet50`).
- `checkpoints_<run_tag>/data/selections.json`: tên ảnh sinh lớp thiểu số mỗi phương pháp M1–M6 chọn.
- `checkpoints_<run_tag>/data/dataset_card.json`: số kênh, số ảnh mỗi tập / lớp, tham số chia, SHA-1 của split.
- `predictions/<model>__<variant>__s<seed>.npz`: `y_val, p_val, y_test, p_test` (từ 1.18.0 thêm `y_train, p_train` trên ảnh thật
  của tập train) + metadata (`augment`, `class_weight`, `monitor`;
  từ 1.21.1 thêm `hist_<metric>` = metric từng epoch: giai đoạn fine-tune (`hist_val_auc`, `hist_val_loss`,
  `hist_auc`, `hist_loss`, …) và giai đoạn head (`hist_head_…`); từ 1.9.0 thêm `test_files` = tên ảnh test `lớp/tệp` đúng thứ tự dự đoán, `code_version`, `dass_version`) — nguồn
  của mọi bảng phân loại. `.npz` cũ không có `test_files`: phân tích nhóm con dựng lại từ split (thứ tự dự đoán =
  lớp theo cấu hình, tên file tăng dần), chỉ dùng khi nhãn khớp đúng thứ tự.
- `predictions_superseded/<giao thức>/`: kết quả của giao thức train cũ do `train --archive-mismatched` chuyển ra
  (không xoá); `evaluate` không đọc thư mục này.
