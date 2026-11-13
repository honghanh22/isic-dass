# DASS-RC: chọn ảnh sinh theo margin mục tiêu và độ phủ ảnh thật

Thiết kế theo thuyết minh đề tài ngày 13/11/2026. Code: [selection/rc.py](../src/dass/selection/rc.py),
[pipeline/stages/rc.py](../src/dass/pipeline/stages/rc.py). Config: [configs/_base_/rc.yaml](../configs/_base_/rc.yaml),
[configs/server/isic2016_rc.yaml](../configs/server/isic2016_rc.yaml).

## 1. Phương pháp so sánh (8 cấu hình)

| Mã | Cấu hình | Bảng |
|---|---|---|
| `B0_imbalanced` | Chỉ ảnh thật, giữ mất cân bằng, xáo trộn thường, không class weight | chính |
| `B2_ros` | Lặp ảnh thật lớp thiểu số đến 1 : 1 | chính |
| `A0_random` | Ngẫu nhiên trong ℰ_c (qua NC + gate) | chính, ablation |
| `A1_tmu` | Top-K theo TMU | ablation (RQ1) |
| `A2_rdc` | Greedy chỉ RDC (α = 0) | ablation |
| `A3_tmu_poolcov` | TMU + coverage với mốc là chính ℰ_c (ảnh sinh) | ablation (RQ2) |
| `A4_maxmargin_rdc` | Rank margin + RDC | ablation (RQ3) |
| `A6_dass_rc` | TMU + RDC (DASS-RC) | chính, ablation |

## 2. Quy trình cho mỗi lớp c

K_c = N_c (số ảnh thật train), pool ⌈3·K_c⌉ ảnh sinh cho MỌI lớp, latent seed `gen_seed + 7919·chỉ số lớp`.

1. **NC** (Inception-v3 của StyleGAN2-ADA, độc lập với bộ chọn): d_i = khoảng cách từ ảnh thật train i tới ảnh train
   gần nhất (bỏ chính nó), τ = Q_0.01({d_i}). Ảnh sinh trùng pixel với train -> loại. Ảnh sinh có khoảng cách tới
   train (cả hai lớp) < τ -> nghi vấn, xuất cặp ảnh vào `results_<run>/rc_nc_review/`, chỉ loại khi người dùng ghi
   vào `rc_nc_confirmed.json`. Trùng trong pool (< τ hoặc trùng pixel): giữ ID nhỏ hơn. τ = 1 % là lựa chọn vận
   hành, không phải tỷ lệ loại nhầm.
2. **Gate**: margin E_d m_c ≥ 0 (sigmoid: m_1 = z, m_0 = −z; z = logit từ đặc trưng E_d chưa chuẩn hoá).
3. **TMU**: m̄ = (m − a_c) / max(b_c, ε), a_c, b_c theo ảnh thật train lớp c; T = exp(−(m̄ − μ)² / 2σ²), μ = phân vị
   của m̄ trên ảnh thật có m > 0.
4. **RDC**: embedding E_d chuẩn hoá L2; κ = exp(−d² / 2h²), h = η · median khoảng cách (≤ 10.000 cặp ảnh thật cùng
   lớp, seed cố định); C(U) = (1/N_c) Σ_x max_{q∈U} κ(x, q).
5. **Greedy**: F(U) = (α/K)·Σ u(q) + (1 − α)·C(U), lazy greedy chính xác (bằng greedy đầy đủ, có test), hoà -> chỉ số
   nhỏ hơn; log phần tăng utility / coverage mỗi lượt.
6. **Thiếu ứng viên** (|ℰ_c| < K_c): sinh thêm tới `max_pool_mult` = 5K_c; vẫn thiếu thì không nới gate / NC, mọi
   cách chọn dùng chung ngân sách khả thi, ghi số thiếu trong `meta` của file lựa chọn.

## 3. Dò siêu tham số (`rc-search`, chỉ VAL)

Tuần tự, khoá từng giai đoạn, ConvNeXt-Tiny + seed 2026, tiêu chí AP trên val (hoà -> giá trị đứng trước trong lưới):
η ∈ {0,5; 1; 2} bằng RDC-only -> (μ ∈ {P25, P50, P75}) × (σ ∈ {0,5; 1; 2}) bằng TMU-only -> α ∈ {0,25; 0,5; 0,75}
bằng DASS-RC. Không dự đoán test; kết quả ở `results_<run>/rc_search_predictions/`, tham số khoá ở
`checkpoints_<run>/data/rc_params.json`. Đây là tìm kiếm tiết kiệm, không bảo đảm tổ hợp tối ưu; cấu hình được chọn
trên đúng backbone + seed của kết quả đầu tiên nên kết quả đó có thể hơi lạc quan.

## 4. Huấn luyện và đánh giá

- `classifier.fixed_steps: true`: mọi cấu hình cùng số update, epoch = ⌈ảnh thật train / batch⌉ bước, dữ liệu lặp.
- Sampler lớp × nguồn (6 cấu hình có GAN): batch 16 = 4 thật âm + 4 GAN âm + 4 thật dương + 4 GAN dương; mỗi nhóm
  xáo lại sau mỗi lượt, lặp khi cạn. BCE, không class weight, không augmentation.
- Chọn epoch theo `val_pr_auc`. Chỉ số chính: AP (sklearn `average_precision_score`). Ngưỡng: tối đa MCC trên val của
  từng lần chạy (hoà -> sensitivity cao hơn -> ngưỡng lớn hơn), khoá trước test.
- Paired bootstrap ΔAP (2.000 lần, phân tầng theo lớp, lấy mẫu theo ẢNH — chưa theo bệnh nhân / WSI), Holm riêng cho
  RQ1–RQ3.

## 5. Lệnh

```bash
dass -c configs/server/isic2016_rc.yaml rc-prepare      # GPU: pool, NC, E_d -> rc_cache.npz
# xác minh rc_nc_review/ -> ghi rc_nc_confirmed.json (nếu có ảnh nghi vấn)
dass -c configs/server/isic2016_rc.yaml rc-search       # GPU: 15 lần train ConvNeXt-T trên val
dass -c configs/server/isic2016_rc.yaml rc-train --model ConvNeXtTiny --seeds 2026
dass -c configs/server/isic2016_rc.yaml evaluate --skip-generative
```
