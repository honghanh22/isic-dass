# DASS — Synthetic Sample Selection for Imbalanced Medical Image Classification

Official implementation <!-- TODO: tên đầy đủ của DASS và tiêu đề bài báo -->. A class-conditional **StyleGAN2-ADA** generates candidates for the minority class;
**DASS** selects the subset that is close to real minority images, far from the majority class (in both an ImageNet
feature space and a supervised disease-aware space) and diverse; downstream classifiers are trained on real + selected
images and compared with seven controls (including a random-oversampling baseline) under a fixed, leakage-free
protocol.

Benchmarks: **ISIC 2016** (dermoscopy, RGB, benign / malignant) and **Brain Tumor MRI** (grayscale,
negative / positive). Both run on the same code and the same formulas; only `configs/datasets/*.yaml` differs.

**Contents:** [Method](#method) · [Project structure](#project-structure) · [Installation](#installation) ·
[Data](#data) · [Reproducing the experiments](#reproducing-the-experiments) · [Outputs](#outputs) ·
[Protocol](#protocol) · [Development](#development) · [Citation](#citation)

## Method

For a candidate `x`, with `S±(x)` the mean cosine similarity to its top-K nearest *real* minority / majority images:

```
M(x)      = S⁺(x) − λ · S⁻(x)                        (computed in E_v and in E_d)
S_div(x)  = min_{y ∈ selected} [1 − cos(z_x, z_y)]
S_DASS(x) = α · M̃_v(x) + β · M̃_d(x) + γ · S̃_div(x)    (~ : min-max normalised; S_div re-normalised every step)
```

`E_v`: frozen ImageNet EfficientNet-B0. `E_d`: a separate DenseNet-121 trained with cross-entropy on real training
labels only. Selection is greedy and picks exactly `n_select = |majority| − |minority|` images from a pool of
`k · n_select` candidates (`k = pool_mult = 1.5` for both datasets), so every balanced variant trains on a 1 : 1 class
ratio.

| Group | Method (paper name) | Internal ID | Training set of the minority class |
|---|---|---|---|
| Real Data Baselines | **Imbalanced Baseline (class-weighted)** | M0 | real images only, imbalanced; class weights in the loss |
| | **Random Oversampling (ROS)** | M0b | real images duplicated to 1 : 1 |
| Generative Augmentation (StyleGAN2-ADA) | **Unfiltered GAN (Random Selection)** | M1 | real + random synthetic |
| | **Visual-only Filter (M_v)** | M2 | real + top-n by M̃_v |
| | **Disease-only Filter (M_d)** | M3 | real + top-n by M̃_d |
| | **Diversity-only Filter (S_div)** | M5 | real + k-center greedy (γ = 1, no margin) |
| | **Dual-Margin Filter (M_v + M_d)** | M4 | real + top-n by α·M̃_v + β·M̃_d |
| | **DASS (Ours)** | M6 | real + greedy α·M̃_v + β·M̃_d + γ·S̃_div |

**Identical training protocol:** every variant uses the same online data augmentation, hyper-parameters and seeds; the
baseline compensates the imbalance with class weights. ROS and the GAN-based variants also share the training-set
size, the number of optimisation steps and the balancing mechanism (data, not loss weights), so **DASS vs ROS**
isolates the contribution of the synthetic image *content* and **DASS vs Unfiltered GAN** that of the selection. Paired bootstrap tests are reported
for every method vs the Imbalanced Baseline and for DASS vs ROS and DASS vs Unfiltered GAN. Internal IDs are kept in
all artefacts; paper names are applied when tables are written (`evaluation.method_labels`). Full method,
experimental design and statistics: [docs/GUIDE.md](docs/GUIDE.md) (Vietnamese).

## Project structure

### Top level

```
.
├── README.md                  this file (English, for reviewers / users)
├── CLAUDE.md                  invariants and conventions for code assistants (Vietnamese)
├── CHANGELOG.md               version history (Vietnamese)
├── pyproject.toml             package `dass` (src layout), CLI entry point `dass`, extras: gan / tf / dev
├── .gitignore
├── configs/                   all hyper-parameters and dataset settings (YAML, no code)
├── src/dass/                  the Python package — all logic lives here
├── notebooks/                 Colab notebook that only calls the CLI; archived original notebooks
├── scripts/                   helper scripts (notebook generator)
├── tests/                     unit + regression tests (no GPU / TF / torch needed)
├── docs/                      guide, architecture, workflow, results format (Vietnamese)
└── colab_gpu_check.ipynb      legacy GPU check (superseded by notebooks/colab_gpu_check.ipynb)
```

### `configs/` — configuration

```
configs/
├── _base_/                    shared hyper-parameters: ONE recipe for every dataset
│   ├── generator.yaml         StyleGAN2-ADA (paper256, ADA target 0.6), KID early stopping, pool seed
│   ├── selection.yaml         DASS (pool_mult k = 1.5, K, λ, α, β, γ), M0b / M7 switches, E_d (DenseNet121, seed 4242)
│   ├── classifier.yaml        6 backbones, seeds 2026–2028, two-stage training, val_macro_recall,
│   │                          online augmentation, class-weighted baseline
│   ├── evaluation.yaml        fixed threshold 0.5, baseline M0 + extra comparisons, paper names / groups, bootstrap
│   └── runtime.yaml           global seed, local paths, fingerprint settings
├── datasets/                  dataset-specific settings ONLY (paths, source, channels, preprocessing, split)
│   ├── _template.yaml         starting point for a new dataset
│   ├── isic2016.yaml          CSV source, dark-border crop, holdout_val split, run_tag v10
│   └── brain_tumor.yaml       folder source, force_grayscale, pad_square, stratified 70/15/15, run_tag bt_v5
└── experiments/               experiment = _base_ + dataset (+ overrides)
    ├── isic2016_dass.yaml
    ├── brain_tumor_dass.yaml
    └── smoke.yaml             quick-check profile, composable with any experiment (not for reporting)
```

Resolution order: `_base_/*` → `datasets/<name>.yaml` → `experiments/<name>.yaml` → extra `-c` profiles →
`--set key=value` → `--tag <suffix>` (appends to `run_tag` so ablations get their own output folders). The schema
(`src/dass/config/schema.py`) rejects unknown keys; its defaults are tested against `configs/_base_/`.

### `src/dass/` — the package

```
src/dass/
├── __init__.py / __main__.py  version; `python -m dass`
├── cli.py                     `dass` command: prepare, gan-setup, gan, sample, fingerprint, select, train,
│                              evaluate, report, run, show-config
├── utils.py                   logging, seeding, atomic JSON writes, vector helpers, figure saving, package versions
│
├── config/                    CONFIGURATION
│   ├── schema.py              dataclasses + validation (unknown keys / invalid values -> error)
│   ├── loader.py              YAML `_base_` inheritance, multiple -c files, --set, --tag
│   └── paths.py               `Layout`: the ONLY place where artefact paths (Drive / local) are defined
│
├── data/                      DATA LOADER — every dataset difference lives here and only here
│   ├── image_io.py            the ONLY image reader / writer; preserves native channel count
│   ├── transforms.py          dark-border crop, pad-to-square, resize, PNG export
│   ├── sources/               where images and labels come from
│   │   ├── base.py            `DatasetSource` interface
│   │   ├── csv_source.py      flat image folder + CSV labels (ISIC)
│   │   └── folder_source.py   one sub-folder per class (Brain Tumor)
│   ├── splits/                how images are split into train / val / test
│   │   ├── base.py            split I/O, canonical hash, reference-split check, class budget (n_select, pool size)
│   │   ├── stratified.py      holdout_val and stratified (optionally patient-grouped); reproduces the original notebooks
│   │   └── from_file.py       fixed split from a CSV
│   ├── variants.py            assembles train/val folders per variant (real + selected synthetic, or real copies for
│   │                          M0b); balanced oversampling indices; asserts val/test are real
│   ├── manifest.py            dataset card (counts per split / class, channels, split hash)
│   └── loaders.py             tf.data pipelines, optional augmentation (off for classifiers), 1 -> 3 channel replication
│
├── models/                    MODEL BACKBONES
│   ├── generator/             StyleGAN2-ADA (PyTorch, imported inside functions only)
│   │   ├── patches.py         clones NVlabs repo and applies idempotent compatibility patches (PyTorch 2.x / Py 3.12)
│   │   ├── export.py          packs the TRAIN split into a StyleGAN2-ADA dataset zip with class labels
│   │   ├── trainer.py         training with KID early stopping, resumable; best.pkl / latest.pkl / gan_state.json
│   │   ├── inception.py       StyleGAN2-ADA Inception-v3 features (KID / FID), generator loading, class-conditional sampling
│   │   ├── sampler.py         candidate-pool generation in a subprocess, zip cache on Drive
│   │   └── _sample_worker.py  subprocess entry point (torch); channel merge / force_grayscale
│   ├── encoders/__init__.py   E_v (frozen EfficientNet-B0) and E_d (DenseNet121 trained on real images)
│   └── classifiers/__init__.py  backbone registry: EfficientNetV2B0, ResNet50, DenseNet121, ConvNeXtTiny (Keras),
│                              ViT-B16, SwinT (KerasHub)
│
├── selection/                 SELECTION STRATEGY (numpy only)
│   ├── scoring.py             top-K cosine margins M_v, M_d
│   └── strategies.py          greedy DASS, controls M0–M6 (+ M0b name, optional M7), Jaccard overlap
│
├── engine/                    CLASSIFIER TRAINING (TensorFlow)
│   ├── metrics.py             macro recall (epoch-selection criterion)
│   └── trainer.py             two-stage training, best-epoch reload, single test prediction, .npz per run
│
├── evaluation/                METRICS & TABLES (numpy / sklearn only)
│   ├── classification.py      AUC, PR-AUC, sensitivity, specificity, BA, G-mean, MCC, F1 at a fixed threshold
│   ├── statistics.py          stratified paired bootstrap ΔAUC
│   ├── generative.py          KID (primary), FID (reference), diversity, intra-set SSIM
│   ├── aggregate.py           loads all predictions, mean ± std over seeds, comparison with the baseline
│   └── reporting.py           paper tables as .csv (formatted), .json (raw) and .tex (booktabs)
│
├── analysis/                  DIAGNOSTICS — measure and plot only, never modify training images
│   ├── shortcut.py            probe AUC (class separability), real-vs-synthetic AUC
│   ├── fingerprint.py         frequency fingerprint detector of Frank et al. (optional stage)
│   └── figures.py             class distribution, KID curve, samples, (M_v, M_d) scatter, selected / rejected grids
│
└── pipeline/                  ORCHESTRATION
    ├── context.py             shared stage context: ingest -> channels -> preprocess -> split -> budget; run manifest
    ├── pool.py                resolves / restores the candidate pool, maps selections.json to files
    └── stages/                one module per CLI stage
        ├── prepare.py  gan.py  sample.py  fingerprint.py  select.py  train.py  evaluate.py  report.py
        └── __init__.py        TensorFlow init (memory growth, seeds), metrics I/O
```

**Import rules** (keep the package importable and testable without a GPU):
`config/`, `data/` (except `loaders.py`), `selection/` and `evaluation/` are pure numpy / sklearn / PIL.
TensorFlow is used only in `data/loaders.py`, `models/encoders`, `engine/` and inside functions of
`models/classifiers` and `analysis/fingerprint.py`; PyTorch only inside functions of `models/generator/*` and in the
`_sample_worker.py` subprocess. Stages import heavy modules inside `run()`.

### Pipeline: stage → code → artefacts

| Stage (CLI) | Main modules | Reads | Writes (on Google Drive) |
|---|---|---|---|
| `prepare` | `pipeline/context.py`, `data/` | raw images + labels | `checkpoints_<run>/data/real_val_split.json`, `dataset_card.json`; `results_<run>/class_distribution.png` |
| `gan-setup` | `models/generator/patches.py` | NVlabs repository | patched clone in `/content` (local) |
| `gan` | `pipeline/stages/gan.py`, `models/generator/{export,trainer}.py` | train split | `checkpoints_<gan_tag>/stylegan2ada/{best,latest}.pkl`, `gan_state.json`; `results_<run>/gan/`, `metrics/gan_kid_history` |
| `sample` | `pipeline/pool.py`, `models/generator/sampler.py` | `best.pkl` | `checkpoints_<run>/data/pool_<class>_from<kimg>kimg_n<N>[_c1\|_gray].zip` |
| `fingerprint` (optional) | `analysis/fingerprint.py` | pool, real train images | `results_<run>/metrics/frequency_fingerprint`, `frequency_analysis/` |
| `select` | `models/encoders`, `selection/`, `analysis/shortcut.py` | pool, train / val images | `checkpoints_<run>/data/{selections.json, embeddings.npz}`, `classifiers/Ed_DenseNet121_s4242.weights.h5`; `metrics/{probe_auc, shortcut_check, selection_jaccard}`, `selection_figures/` |
| `train --model X` | `data/variants.py`, `engine/trainer.py`, `models/classifiers` | selections, pool, real train images (M0b) | `results_<run>/predictions/<model>__<variant>__s<seed>.npz`; `checkpoints_<run>/classifiers/*.weights.h5` |
| `evaluate` | `evaluation/` | predictions, pool | `results_<run>/metrics/{classification_runs, classification_summary, significance_vs_baseline, generation_quality}` |
| `report` | `evaluation/reporting.py` | metrics | `results_<run>/tables/*.{csv,json,tex}` |
| `run` | `cli.py` | — | every stage above, in order (`--from` / `--to` to run a range) |

Every stage is resumable: finished artefacts are restored from Drive or skipped.

### Artefact layout on Google Drive

```
<drive_root>/                                  e.g. .../ColabData/BrainTumor_GAN
├── checkpoints_<gan_tag>/stylegan2ada/        generator (shared by all runs of a dataset)
│   ├── best.pkl  latest.pkl  gan_state.json
│   └── logs_and_samples/
├── checkpoints_<run_tag>/
│   ├── data/                                  split, dataset card, candidate pool zip, selections, embeddings
│   └── classifiers/                           E_d and classifier weights
└── results_<run_tag>/
    ├── predictions/                           one .npz per (model, variant, seed) — source of every table
    ├── metrics/                               raw numbers (.csv + .json)
    ├── tables/                                paper tables (.csv / .json / .tex)
    ├── gan/  selection_figures/  frequency_analysis/
    ├── class_distribution.png
    └── run_manifest.json                      resolved config, library versions, stage commands
```

Current tags: ISIC `gan_tag v5`, `run_tag v10`; Brain Tumor `gan_tag bt`, `run_tag bt_v5`; quick checks `run_tag smoke`.
Earlier configurations are kept untouched and reported separately: ISIC `v7` (k = 4), ISIC `v9` / Brain Tumor `bt_v4`
(k = 2, no augmentation, unweighted baseline), Brain Tumor `bt_v3` (k = 3).
Artefacts are never deleted; a new configuration gets a new `run_tag` (or `--tag`).

### `notebooks/`, `scripts/`, `tests/`, `docs/`

```
notebooks/
├── colab_pipeline.ipynb       one cell per CLI stage; contains no logic (generated, do not edit by hand)
├── colab_gpu_check.ipynb      checks the GPU assigned by Colab
└── archive/                   original notebooks (ISIC v5, Brain Tumor v1) — reference only, not run
scripts/
└── build_colab_notebook.py    regenerates notebooks/colab_pipeline.ipynb
tests/
├── conftest.py
├── unit/                      config, image I/O, transforms / sources, splits / variants, selection,
│                              evaluation, generator helpers and analysis
└── regression/                split reproduction, artefact paths, channel preservation, force_grayscale,
                               no spectral mitigation, .gitignore
docs/
├── GUIDE.md                   problem, method, experimental design, statistics, validity, how to run
├── ARCHITECTURE.md            layers and extension points
├── WORKFLOW.md                day-to-day workflow (VS Code + Colab + GitHub)
└── RESULTS_FORMAT.md          every column of every output file
```

## Installation

Google Colab with a GPU (A100 / L4 recommended). PyTorch and TensorFlow are pre-installed on Colab:

```bash
pip install -e . ninja click keras-hub
```

Local development (no GPU needed): `pip install -e ".[dev]" && pytest && ruff check src tests scripts`.

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
| Channels | 3 (auto-detected) | 3 identical channels (`force_grayscale`: removes JPEG chroma noise present only in 129 negative images) |
| Preprocessing | dark-border crop, resize | pad to square, resize |
| Split | official test set; 15 % of train → val | stratified 70 / 15 / 15 (optional patient grouping) |

Adding a dataset: copy `configs/datasets/_template.yaml`, set the source, classes, preprocessing and split, then create
an experiment file next to `configs/experiments/brain_tumor_dass.yaml`. No code change is needed for folder / CSV
sources (see [docs/WORKFLOW.md](docs/WORKFLOW.md)).

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

On Colab, open `notebooks/colab_pipeline.ipynb`: it clones this repository, installs the package and runs one stage
per cell.

## Outputs

`results_<run_tag>/tables/` (on Drive), each as `.csv` (formatted), `.json` (raw) and `.tex` (booktabs):

| Table | Content |
|---|---|
| `dataset` | images per split and class |
| `classification` | AUC, PR-AUC, F1, sensitivity, specificity, balanced accuracy, G-mean, MCC — mean ± std over seeds |
| `significance` | paired bootstrap ΔAUC with 95 % CI and p-value: every method vs. Imbalanced Baseline, plus DASS vs. ROS and DASS vs. Unfiltered GAN (column `vs`) |
| `generation_quality` | KID (primary, mean ± std) and FID (reference) on Inception-v3, diversity, intra-set SSIM, real-vs-synthetic AUC |

Raw per-run numbers are in `results_<run_tag>/metrics/`; `run_manifest.json` stores the resolved configuration,
seeds and library versions. Column definitions: [docs/RESULTS_FORMAT.md](docs/RESULTS_FORMAT.md).

## Protocol

- Validation and test sets contain **real images only**; the generator and DASS never see them, and `E_d` uses the
  validation set only for epoch selection.
- Validation is used **only** for epoch selection (`val_macro_recall`); test metrics use a **fixed threshold of 0.5**;
  the test set is predicted once, after reloading the best checkpoint.
- Every image is stored as PNG in its **native channel count**. Grayscale images are replicated to three identical
  channels only in memory, immediately before ImageNet-pretrained networks / Inception-v3.
- The generator checkpoint used downstream is the snapshot with the **lowest minority-class KID**.
- Splits are deterministic (global seed) and validated against the split the reused generator was trained on.
- All classifiers share the same online augmentation; the real-data baseline uses class weights, every other variant
  is balanced 1 : 1 by data.
- Results are reported as mean ± std over three classifier seeds with paired bootstrap tests against the Imbalanced
  Baseline, and for DASS against ROS and Unfiltered GAN.

## Development

```bash
pip install -e ".[dev]"
pytest                                   # unit + regression, ~10 s, no GPU / TF / torch needed
ruff check src tests scripts
python -m compileall -q src              # syntax check of TF / torch modules
python scripts/build_colab_notebook.py   # regenerate the Colab notebook
```

Invariants protected by regression tests (channel preservation, split reproduction, artefact paths, removal of
spectral mitigation) are listed in [CLAUDE.md](CLAUDE.md). Code comments, logs and `docs/` are in Vietnamese;
identifiers and this README are in English.

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
