"""Sinh notebooks/colab_pipeline.ipynb — notebook chỉ GỌI CLI `dass`, không chứa logic.

    python scripts/build_colab_notebook.py                   # sinh lại toàn bộ (XOÁ output đang có)
    python scripts/build_colab_notebook.py --markdown-only   # chỉ cập nhật các ô markdown, GIỮ code + output
Sửa notebook bằng cách sửa file này rồi chạy lại (không sửa tay file .ipynb). Đóng notebook trong VS Code trước khi
chạy, nếu không VS Code có thể ghi đè lại khi lưu.
"""

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pipeline.ipynb"
cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# DASS — StyleGAN2-ADA + chọn ảnh sinh cho ảnh y tế mất cân bằng (chạy trên Colab)

Notebook này **không chứa logic**: mọi code nằm trong package `dass` (`src/`), mỗi cell gọi một lệnh CLI.
Mỗi lệnh là một tiến trình riêng (PyTorch và TensorFlow không tranh VRAM) và **chạy lại được**: artefact lưu
trên Drive, phần đã xong được khôi phục / bỏ qua.

`prepare → gan → sample → [fingerprint] → select → train → evaluate → report` — hoặc cả chuỗi: `dass … run`.

Runtime: **GPU** (A100 / L4 khuyến nghị; T4 chạy được nhưng train GAN chậm).
""")

md("## 0. Kết nối Drive, lấy mã nguồn, cài đặt")
code("""
from google.colab import drive
drive.mount("/content/drive")
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
code("""
import os

# Nguồn mã — ưu tiên theo thứ tự:
# (a) GitHub (mặc định): mỗi lần chạy cell này tự clone / cập nhật bản mới nhất -> KHÔNG cần upload tay,
#     kể cả khi Colab cấp máy mới.
# (b) REPO_URL = "": dùng bản đã "Upload to Colab" từ VS Code (src/, configs/, pyproject.toml ở /content).
# (c) Không có cả hai: thư mục dự án trên Google Drive (PROJECT_ON_DRIVE).
REPO_URL = "https://github.com/honghanh22/isic-dass.git"
PROJECT_ON_DRIVE = "/content/drive/MyDrive/dass"

if REPO_URL:                                                         # (a)
    PROJECT_DIR = "/content/dass-repo"
    if os.path.isdir(f"{PROJECT_DIR}/.git"):
        !git -C {PROJECT_DIR} pull -q --ff-only
    else:
        !git clone -q {REPO_URL} {PROJECT_DIR}
    !git -C {PROJECT_DIR} log -1 --format="Phiên bản code: %h — %s (%cr)"
elif os.path.exists("/content/pyproject.toml"):                     # (b)
    PROJECT_DIR = "/content"
    if os.path.isdir("/content/dass"):        # lỡ upload src/dass thay vì src/ -> đưa về đúng chỗ
        !mkdir -p /content/src/dass && rsync -a /content/dass/ /content/src/dass/ && rm -rf /content/dass
else:                                                                # (c)
    PROJECT_DIR = "/content/dass-repo"
    !rsync -a --delete --exclude .git "{PROJECT_ON_DRIVE}/" {PROJECT_DIR}/

print("PROJECT_DIR =", PROJECT_DIR)
%cd {PROJECT_DIR}
# torch / tensorflow có sẵn trên Colab -> chỉ cài package + phụ thuộc nhẹ
!pip install -q -e . ninja click keras-hub
""")

md("""
## 1. Chọn thực nghiệm

`EXPERIMENT`: `configs/experiments/rsna_pneumonia_dass.yaml` (RSNA: `prepare` tự đọc DICOM; GAN `rsna` đã train) hoặc
`configs/experiments/isic2016_dass.yaml`.
`PROFILE = "configs/experiments/smoke.yaml"` để chạy thử nhanh (ghi vào thư mục `*_smoke`). Ghi đè thêm:
`EXTRA = "--set selection.gamma=0.25 --tag gamma025"`.
""")
code("""
EXPERIMENT = "configs/experiments/rsna_pneumonia_dass.yaml"
PROFILE = ""          # "configs/experiments/smoke.yaml" để chạy thử
EXTRA = ""            # ví dụ "--set selection.gamma=0.25 --tag gamma025"
CFG = f"-c {EXPERIMENT}" + (f" -c {PROFILE}" if PROFILE else "") + (f" {EXTRA}" if EXTRA else "")
!dass {CFG} show-config | head -n 40
""")
code("""
# Tiện ích xem hình / bảng mà các stage đã lưu lên Drive. Thư mục kết quả đọc lại từ cấu hình MỖI LẦN gọi
# -> đổi run_tag (git pull, sửa EXPERIMENT) không cần chạy lại ô này.
import glob, json, shlex, subprocess
import pandas as pd
from IPython.display import Image, display

def results_dir():
    out = subprocess.run(["dass", *shlex.split(CFG), "show-config"], capture_output=True, text=True, check=True)
    paths = json.loads(out.stdout)["paths"]
    return f"{paths['drive_root']}/results_{paths['run_tag']}"

def show(pattern):
    found = sorted(glob.glob(f"{results_dir()}/{pattern}"))
    for p in found:
        print(p)
        display(Image(p))
    if not found:
        print("chưa có", pattern)

def table(name, folder="tables"):
    p = f"{results_dir()}/{folder}/{name}.csv"
    display(pd.read_csv(p)) if os.path.exists(p) else print("chưa có", p)
""")

md("## 2. Dữ liệu: đọc nguồn, nhận diện số kênh, tiền xử lý, chia train / val / test")
code("""
!dass {CFG} prepare
show("class_distribution.png")
""")
code("""
# (Chỉ RSNA — nhãn JSON MD.ai) số ảnh theo từng tên nhãn + 4 ảnh mẫu mỗi nhãn (khung đỏ = vùng đám mờ)
!dass {CFG} label-stats --focus Exclude Flag --samples 4
show("label_samples.png")
""")

md("""
## 3. StyleGAN2-ADA có điều kiện

`gan-setup` clone repo NVlabs, vá cho PyTorch 2.x và biên dịch plugin CUDA. `gan` train với early stopping theo KID
của lớp thiểu số — **Colab ngắt thì chạy lại đúng cell này** để resume. GAN đã train xong (ISIC: `checkpoints_v5`,
RSNA: `checkpoints_rsna`) được dùng lại ngay. `--fresh-start` XOÁ GAN cũ.
""")
code("""
!dass {CFG} gan-setup
""")
code("""
!dass {CFG} gan
show("gan/*.png")
""")

md("""
## 4. Candidate pool

Ảnh sinh lưu đúng số kênh của bộ dữ liệu. Dữ liệu ảnh xám + GAN 3 kênh: chỉ gộp về 1 kênh khi 3 kênh giống hệt
nhau — nếu không, lệnh dừng và yêu cầu train lại GAN 1 kênh.
""")
code("""
!dass {CFG} sample
""")

md("## 5. (Tuỳ chọn) Fingerprint miền tần số — detector của Frank et al.")
code("""
!dass {CFG} fingerprint
table("frequency_fingerprint", "metrics")
show("frequency_analysis/*.png")
""")

md(r"""
## 6. Chọn ảnh sinh: E_v / E_d, chấm điểm, kiểm tra shortcut

Mỗi phương pháp Generative Augmentation chọn đúng n ảnh từ cùng một pool. Log và tên file dùng **mã nội bộ**; bảng
cho bài báo (`report`) dùng **tên hiển thị** (`evaluation.method_labels`):

| Mã nội bộ | Tên trong bài báo |
|---|---|
| `M1_random` | Unfiltered GAN (Random Selection) |
| `M2_visual` | Visual-only Filter ($M_v$) |
| `M3_disease` | Disease-only Filter ($M_d$) |
| `M5_diversity` | Diversity-only Filter ($S_{\text{div}}$) |
| `M4_visual_disease` | Dual-Margin Filter ($M_v + M_d$) |
| `M6_dass` | **DASS (Ours)** |
""")
code("""
!dass {CFG} select
table("probe_auc", "metrics")
table("shortcut_check", "metrics")
show("selection_figures/scatter_Mv_Md.png")
show("selection_figures/grid_M6_dass_*.png")
""")

md("""
## 7. Train classifier — 6 backbone × 8 phương pháp × 3 seed

Mỗi cell một backbone, train cả 8 phương pháp với 3 seed (`classifier.seeds`); chạy lại được (lần chạy đã có dự đoán
trên Drive sẽ bỏ qua). Trong một bộ dữ liệu, mọi phương pháp dùng cùng augmentation (RSNA: không augmentation);
baseline cân bằng bằng class weight.

- **Real Data Baselines:** Imbalanced Baseline (class-weighted) (`M0_real_only`), Random Oversampling (ROS)
  (`M0b_real_oversample`)
- **Generative Augmentation (StyleGAN2-ADA):** 6 phương pháp ở mục 6, từ Unfiltered GAN đến **DASS (Ours)**

CNN: `EfficientNetV2B0`, `ResNet50`, `DenseNet121`, `ConvNeXtTiny`; Transformer (KerasHub): `ViT-B16`, `SwinT`.
Transformer nặng hơn CNN nhiều — nên dùng GPU L4 / A100.

Báo lỗi "được train với thiết lập khác" (dự đoán cũ theo giao thức khác): thêm `--archive-mismatched` để chuyển kết
quả cũ sang `predictions_superseded/` (không xoá) rồi train lại.

Thử nhanh các backbone mới trước khi chạy thật (1 epoch, 1 seed):
`!dass {CFG} -c configs/experiments/smoke.yaml run --from train --to train --models ConvNeXtTiny ViT-B16 SwinT`
""")
for model in ["EfficientNetV2B0", "ResNet50", "DenseNet121", "ConvNeXtTiny", "ViT-B16", "SwinT"]:
    code(f"!dass {{CFG}} train --model {model}")

md("""
## 8. Đánh giá và bảng cho bài báo (`tables/*.csv`, `*.json`, `*.tex`)

Bảng dùng tên hiển thị và cột `Group` (Real Data Baselines / Generative Augmentation (StyleGAN2-ADA)); kiểm định
gồm mọi phương pháp vs Imbalanced Baseline, cộng DASS (Ours) vs ROS và DASS (Ours) vs Unfiltered GAN.
""")
code("""
!dass {CFG} evaluate
!dass {CFG} report
table("dataset")
table("classification")
table("significance")
table("generation_quality")
# RSNA: phân tích theo tư thế chụp (bộ dữ liệu khác: "chưa có")
table("classification_by_subgroup")
table("significance_by_subgroup")
table("subgroup_reference")
table("subgroup_share")
""")

md("""
## Chạy cả chuỗi bằng một lệnh

```
!dass {CFG} run                         # prepare -> report, train mọi model trong classifier.models
!dass {CFG} run --from select --to report
```
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

def update_markdown_only(path: Path) -> int:
    """Thay nội dung các ô markdown của notebook đang có bằng bản trong file này; ô code và output giữ nguyên.

    Ghép theo thứ tự: ô markdown thứ i của notebook <- ô markdown thứ i ở đây (số ô markdown phải bằng nhau).
    """
    existing = json.loads(path.read_text(encoding="utf-8"))
    old = [c for c in existing["cells"] if c["cell_type"] == "markdown"]
    new = [c for c in cells if c["cell_type"] == "markdown"]
    if len(old) != len(new):
        raise SystemExit(f"Notebook có {len(old)} ô markdown, bản sinh có {len(new)} -> không ghép được. "
                         "Sinh lại toàn bộ (mất output) hoặc sửa tay.")
    changed = 0
    for o, n in zip(old, new):
        if o["source"] != n["source"]:
            o["source"] = n["source"]
            changed += 1
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


if __name__ == "__main__":
    if "--markdown-only" in sys.argv[1:]:
        print(f"Đã cập nhật {update_markdown_only(OUT)} ô markdown trong {OUT} (giữ nguyên code và output)")
    else:
        OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Đã ghi {OUT} ({len(cells)} cell)")
