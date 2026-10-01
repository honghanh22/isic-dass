# DASS — Synthetic Sample Selection for Imbalanced Medical Image Classification

Official implementation <!-- TODO: tên đầy đủ của DASS và tiêu đề bài báo -->. A class-conditional **StyleGAN2-ADA** generates candidates for the minority class;
**DASS** selects the subset that is close to real minority images, far from the majority class (in both an ImageNet
feature space and a supervised disease-aware space) and diverse; downstream classifiers are trained on real + selected
images and compared with six controls under a fixed, leakage-free protocol.

Benchmarks: **ISIC 2016** (dermoscopy, RGB, benign / malignant) and **Brain Tumor MRI** (grayscale,
negative / positive). Both run on the same code and the same formulas; only `configs/datasets/*.yaml` differs.

## Method

For a candidate `x`, with `S±(x)` the mean cosine similarity to its top-K nearest *real* minority / majority images:

```
M(x)      = S⁺(x) − λ · S⁻(x)                        (computed in E_v and in E_d)
S_div(x)  = min_{y ∈ selected} [1 − cos(z_x, z_y)]
S_DASS(x) = α · M̃_v(x) + β · M̃_d(x) + γ · S̃_div(x)    (~ : min-max normalised; S_div re-normalised every step)
```

`E_v`: frozen ImageNet EfficientNet-B0. `E_d`: a separate DenseNet-121 trained with cross-entropy on real training
labels only. Selection is greedy and picks exactly `n_select = |majority| − |minority|` images.

| ID | Selection criterion | Diversity |
|---|---|---|
| M0 | real images only (class-weighted) — baseline | – |
| M1 | random | – |
| M2 | M̃_v | – |
| M3 | M̃_d | – |
| M4 | α·M̃_v + β·M̃_d | – |
| M5 | 0 (k-center greedy) | γ = 1 |
| **M6** | **α·M̃_v + β·M̃_d (DASS)** | **γ** |

## Repository layout

```
configs/
  _base_/            shared hyper-parameters (generator, selection, classifier, evaluation, runtime)
  datasets/          dataset-specific settings only (source, channels, preprocessing, split, paths)
  experiments/       _base_ + dataset (+ overrides): one file per experiment in the paper; smoke.yaml profile
src/dass/
  data/              DATA LOADER    — sources (csv | folders), transforms, splits, channel-preserving image I/O
  models/            MODEL BACKBONE — generator (StyleGAN2-ADA), encoders (E_v, E_d), classifiers (registry)
  selection/         SELECTION      — margin scoring, DASS and control strategies
  evaluation/        METRICS        — classification, KID/FID (Inception-v3), bootstrap, CSV/JSON/LaTeX tables
  analysis/          diagnostics    — shortcut test, feature probes, frequency fingerprint (Frank et al.), figures
  engine/            classifier training loop
  pipeline/          context + stages: prepare, gan, sample, fingerprint, select, train, evaluate, report
  config/            schema, loader (_base_ inheritance, --set, --tag), artefact paths
notebooks/           colab_pipeline.ipynb (calls the CLI only); archive/ (original notebooks, reference)
scripts/             build_colab_notebook.py
tests/               unit/ and regression/ (split reproduction, artefact paths, channel preservation)
docs/                ARCHITECTURE.md, WORKFLOW.md, RESULTS_FORMAT.md (Vietnamese)
```

## Installation

Google Colab with a GPU (A100 / L4 recommended). PyTorch and TensorFlow are pre-installed on Colab:

```bash
pip install -e . ninja click keras-hub
```

Local development (no GPU needed): `pip install -e ".[dev]" && pytest && ruff check src tests`.

## Data

Place each dataset on Google Drive and point `configs/datasets/<name>.yaml` to it.

```
ISBI2016_ISIC_Part3/                               Brain_Tumor_Dataset/
├── ISBI2016_ISIC_Part3_Training_Data/*.jpg        ├── Negative/*.png|jpg
├── ISBI2016_ISIC_Part3_Training_GroundTruth.csv   └── Positive/*.png|jpg
├── ISBI2016_ISIC_Part3_Test_Data/*.jpg
└── ISBI2016_ISIC_Part3_Test_GroundTruth.csv
```

| | ISIC 2016 | Brain Tumor MRI |
|---|---|---|
| Source | flat folders + CSV labels | class sub-folders |
| Channels | 3 (auto-detected) | 3 (grayscale MRI stored as three identical channels, as the generator was trained) |
| Preprocessing | dark-border crop, resize | pad to square, resize |
| Split | official test set; 15 % of train → val | stratified 70 / 15 / 15 (optional patient grouping) |

## Reproducing the experiments

```bash
dass -c configs/experiments/isic2016_dass.yaml run
dass -c configs/experiments/brain_tumor_dass.yaml run
```

or stage by stage (every stage is resumable; finished artefacts are restored from Drive):

```bash
E=configs/experiments/brain_tumor_dass.yaml
dass -c $E prepare                     # ingest, detect channels, preprocess, split, dataset card
dass -c $E gan-setup                   # clone + patch NVlabs StyleGAN2-ADA for PyTorch 2.x
dass -c $E gan                         # train with KID early stopping (or reuse the trained generator)
dass -c $E sample                      # candidate pool
dass -c $E select                      # E_v / E_d, DASS + controls, shortcut test
dass -c $E train --model resnet50 --seeds 2026 2027 2028
dass -c $E evaluate                    # classification metrics, KID/FID, paired bootstrap
dass -c $E report                      # paper tables: tables/*.csv, *.json, *.tex
```

Quick check: `dass -c $E -c configs/experiments/smoke.yaml run`. Ablations without editing files:
`dass -c $E --set selection.gamma=0 --tag nodiv run --from select`.

## Outputs

`results_<run_tag>/tables/` (on Drive), each as `.csv` (formatted), `.json` (raw) and `.tex` (booktabs):

| Table | Content |
|---|---|
| `dataset` | images per split and class |
| `classification` | AUC, PR-AUC, F1, sensitivity, specificity, balanced accuracy, G-mean, MCC — mean ± std over seeds |
| `significance` | paired bootstrap ΔAUC vs. M0 with 95 % CI and p-value |
| `generation_quality` | KID (primary, mean ± std) and FID (reference) on Inception-v3, diversity, intra-set SSIM, real-vs-synthetic AUC |

Raw per-run numbers are in `results_<run_tag>/metrics/`; `run_manifest.json` stores the resolved configuration,
seeds and library versions. Column definitions: `docs/RESULTS_FORMAT.md`.

## Protocol

- Validation and test sets contain **real images only**; the generator, `E_d` and DASS never see them.
- Validation is used **only** for epoch selection (`val_macro_recall`); test metrics use a **fixed threshold of 0.5**;
  the test set is predicted once, after reloading the best checkpoint.
- Every image is stored as PNG in its **native channel count**. Grayscale images are replicated to three identical
  channels only in memory, immediately before ImageNet-pretrained networks / Inception-v3.
- The generator checkpoint used downstream is the snapshot with the **lowest minority-class KID**.
- Splits are deterministic (global seed) and validated against the split the reused generator was trained on.
- Results are reported as mean ± std over three classifier seeds with paired bootstrap tests against M0.

## Acknowledgements

[StyleGAN2-ADA (NVlabs)](https://github.com/NVlabs/stylegan2-ada-pytorch) — generator, used unmodified apart from
compatibility patches applied at runtime. [GANDCTAnalysis (Frank et al., ICML 2020)](https://github.com/RUB-SysSec/GANDCTAnalysis)
— frequency fingerprint detector (optional `fingerprint` stage).

## Citation

```bibtex
@inproceedings{dass2026,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```
