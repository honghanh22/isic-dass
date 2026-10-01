"""Các bước (stage) của pipeline. Mỗi stage đọc/ghi artefact trên Drive nên chạy độc lập và chạy lại được.

    prepare → gan-setup → gan-train → gan-report → generate → frequency (tuỳ chọn)
            → select → train (mỗi model × seed) → aggregate → quality

Nên chạy mỗi stage trong một tiến trình riêng (qua CLI) để PyTorch (GAN) và TensorFlow (classifier)
không tranh VRAM trong cùng một tiến trình.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Layout
from .data.ingest import copy_images_by_class, count_per_class
from .data.preprocess import preprocess_and_save
from .data.split import ClassBudget, Split, compute_budget, load_or_create_split, split_paths
from .utils import read_json, save_figure, set_seed, write_json_atomic

log = logging.getLogger(__name__)


# ============================================================================================ context
@dataclass
class Context:
    cfg: Config
    layout: Layout
    split: Split
    budget: ClassBudget

    @classmethod
    def create(cls, cfg: Config) -> Context:
        """Đảm bảo dữ liệu cục bộ đã sẵn sàng (idempotent), nạp split cố định và tính ngân sách ảnh sinh."""
        set_seed(cfg.seed)
        layout = Layout(cfg)
        layout.makedirs()
        prepare_local_data(cfg, layout)
        split = load_or_create_split(layout.split_json, layout.train_pp, cfg.data.class_names,
                                     cfg.data.val_fraction, cfg.seed)
        budget = compute_budget(split, cfg.selection.pool_mult)
        log.info("Thiểu số = %s (%d) | đa số = %s (%d) | cần chọn %d | pool %d ảnh",
                 budget.minority, budget.n_real[budget.minority], budget.majority,
                 budget.n_real[budget.majority], budget.n_select, budget.pool_size)
        return cls(cfg, layout, split, budget)

    @property
    def train_paths(self) -> dict[str, list[str]]:
        return split_paths(self.layout.train_pp, self.split, "train")

    @property
    def val_paths(self) -> dict[str, list[str]]:
        return split_paths(self.layout.train_pp, self.split, "val")

    @property
    def minority_idx(self) -> int:
        return self.cfg.data.class_to_idx[self.budget.minority]


def prepare_local_data(cfg: Config, layout: Layout) -> None:
    """Copy ảnh từ Drive về ổ cục bộ và tiền xử lý sang PNG. Bỏ qua ảnh đã có."""
    names = cfg.data.class_names
    copy_images_by_class(layout.drive_train_gt_csv, layout.drive_train_img_dir, layout.train_raw, names)
    copy_images_by_class(layout.drive_test_gt_csv, layout.drive_test_img_dir, layout.test_raw, names)
    preprocess_and_save(layout.train_raw, layout.train_pp, cfg.data.img_size, names)
    preprocess_and_save(layout.test_raw, layout.test_pp, cfg.data.img_size, names)


# ============================================================================================ stages
def stage_prepare(cfg: Config) -> Context:
    ctx = Context.create(cfg)
    names = cfg.data.class_names
    counts = {"Train": [len(ctx.split[c]["train"]) for c in names],
              "Validation": [len(ctx.split[c]["val"]) for c in names],
              "Test": list(count_per_class(ctx.layout.test_pp, names).values())}
    log.info("Phân bố lớp: %s", {k: dict(zip(names, v)) for k, v in counts.items()})
    if cfg.save_figures:
        import matplotlib.pyplot as plt

        x = np.arange(len(names))
        fig = plt.figure(figsize=(8, 5))
        for i, (name, vals) in enumerate(counts.items()):
            bars = plt.bar(x + (i - 1) * 0.25, vals, 0.25, label=name)
            for b in bars:
                plt.text(b.get_x() + b.get_width() / 2, b.get_height(), int(b.get_height()), ha="center", va="bottom")
        plt.xticks(x, names)
        plt.ylabel("Number of images")
        plt.title("Class Distribution of Train, Validation and Test Sets")
        plt.legend()
        save_figure(fig, ctx.layout.results_dir / "class_distribution.png")
    return ctx


def stage_gan_setup(cfg: Config, reset: bool = True, verify: bool = True, clear_cache: bool = False) -> None:
    from .gan.setup import clear_torch_extension_cache, ensure_stylegan_repo, verify_cuda_plugins

    layout = Layout(cfg)
    if clear_cache:
        clear_torch_extension_cache()
    ensure_stylegan_repo(layout.sg2_repo, reset=reset)
    if verify:
        verify_cuda_plugins(layout.sg2_repo)


def stage_gan_train(cfg: Config, fresh_start: bool = False, dry_run: bool = False) -> dict:
    import sys

    from .gan.dataset import build_stylegan_dataset_zip
    from .gan.metrics import inception_features, load_images_uint8
    from .gan.setup import ensure_stylegan_repo
    from .gan.train import StyleGanTrainer
    from .utils import run_command

    ctx = Context.create(cfg)
    layout, g = ctx.layout, cfg.gan
    ensure_stylegan_repo(layout.sg2_repo)
    build_stylegan_dataset_zip(layout.gan_dataset_zip, layout.train_pp, ctx.split, cfg.data.class_to_idx)
    if dry_run:   # kiểm tra dataset + cấu hình, không train
        run_command([sys.executable, "train.py", f"--outdir={layout.gan_runs / 'dryrun'}",
                     f"--data={layout.gan_dataset_zip}", "--gpus=1", "--cond=1", f"--mirror={int(g.mirror)}",
                     f"--cfg={g.cfg}", f"--batch={g.batch}", f"--gamma={g.gamma}", "--kimg=10",
                     "--metrics=none", "--dry-run"], cwd=layout.sg2_repo)
        return {}

    real_feats = inception_features(load_images_uint8(ctx.train_paths[ctx.budget.minority]))
    log.info("Inception features %s thật (train): %s", ctx.budget.minority, real_feats.shape)
    trainer = StyleGanTrainer(layout, g, cfg.seed, real_feats, ctx.minority_idx)
    return trainer.train(fresh_start=fresh_start)


def stage_gan_report(cfg: Config) -> None:
    import matplotlib.pyplot as plt

    from .gan.metrics import generate_class_images, load_generator
    from .gan.setup import ensure_stylegan_repo
    from .gan.train import load_state

    layout = Layout(cfg)
    layout.makedirs()
    ensure_stylegan_repo(layout.sg2_repo)
    state = load_state(layout.gan_state_json)
    log.info("Trạng thái GAN: %s | best KID = %s @ %s kimg", state["stop_reason"], state["best_kid"],
             state["best_cum_kimg"])
    hist = pd.DataFrame(state["history"])
    if len(hist):
        hist.to_csv(layout.gan_fig_dir.parent / "gan_kid_history.csv", index=False)
        fig = plt.figure(figsize=(8, 4))
        plt.plot(hist["cum_kimg"], hist["kid"] * 1e3, marker="o")
        plt.axvline(state["best_cum_kimg"], color="red", ls="--", label=f"best @ {state['best_cum_kimg']} kimg")
        plt.xlabel("kimg")
        plt.ylabel("KID thiểu số (×1e-3, thấp hơn = tốt hơn)")
        plt.title("StyleGAN2-ADA — KID theo snapshot")
        plt.legend()
        plt.grid(alpha=0.3)
        save_figure(fig, layout.gan_fig_dir / "kid_history.png")

    G = load_generator(layout.gan_best_pkl)
    classes = cfg.data.class_to_idx
    fig, axes = plt.subplots(len(classes), 8, figsize=(16, 2.2 * len(classes)))
    for row, (label, idx) in enumerate(classes.items()):
        imgs = generate_class_images(G, 8, idx, psi=cfg.gan.trunc_psi, seed=1)
        for col in range(8):
            axes[row, col].imshow(imgs[col])
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"sinh: {label}", loc="left", fontsize=11)
    fig.suptitle(f"Ảnh sinh từ snapshot tốt nhất ({state['best_cum_kimg']} kimg)")
    save_figure(fig, layout.gan_fig_dir / "samples_best.png")


# -------------------------------------------------------------------------------------- candidate pool
@dataclass
class CandidatePool:
    raw: list[str]
    harmonized: list[str] | None = None
    mitigated: list[str] | None = None

    @property
    def final(self) -> list[str]:
        return self.mitigated or self.harmonized or self.raw


def _pool_tag(ctx: Context) -> str:
    from .gan.generate import pool_tag
    from .gan.train import load_state

    state = load_state(ctx.layout.gan_state_json)
    if state["best_cum_kimg"] is None:
        raise RuntimeError("Chưa có GAN đã train (gan_state.json trống). Chạy stage gan-train trước.")
    return pool_tag(state["best_cum_kimg"], ctx.budget.pool_size)


def resolve_candidate_pool(ctx: Context, generate_if_missing: bool = True) -> CandidatePool:
    """Khôi phục (hoặc sinh) pool gốc rồi áp các bước chuẩn hoá / mitigation đã bật trong config (đều có cache)."""
    from .frequency.harmonize import harmonize_pool, median_short_side, resolve_harmonize_channels
    from .gan.generate import archive_pool, generate_images, restore_pool

    cfg, layout, minority = ctx.cfg, ctx.layout, ctx.budget.minority
    tag = _pool_tag(ctx)
    pool_name = f"pool_{minority}"

    raw_zip = layout.data_dir / f"{pool_name}_{tag}.zip"
    raw_dir = layout.candidates / pool_name
    raw = restore_pool(raw_zip, raw_dir)
    if raw is None:
        if not generate_if_missing:
            raise RuntimeError(f"Chưa có candidate pool {raw_zip}. Chạy stage generate trước.")
        from .gan.setup import ensure_stylegan_repo

        ensure_stylegan_repo(layout.sg2_repo)
        raw = generate_images(layout.sg2_repo, layout.gan_best_pkl, raw_dir, ctx.minority_idx,
                              ctx.budget.pool_size, cfg.gan.trunc_psi, cfg.gan.gen_seed)
        archive_pool(raw_dir, raw_zip)
    pool = CandidatePool(raw=raw)

    fq = cfg.frequency
    real_minority = ctx.train_paths[minority]
    to_gray = resolve_harmonize_channels(fq.harmonize_channels, real_minority)
    if to_gray or fq.match_resize_chain:
        ref_side = median_short_side(layout.train_raw / minority) if fq.match_resize_chain else None
        pool.harmonized = harmonize_pool(pool.raw, layout.data_dir / f"{pool_name}_harmonized_{tag}.zip",
                                         layout.candidates / f"{pool_name}_harmonized", cfg.data.img_size,
                                         to_gray, ref_side)

    if fq.apply_spectral_mitigation:
        from .frequency.mitigation import mitigate_pool

        source = pool.harmonized or pool.raw
        pool.mitigated, _ = mitigate_pool(source, real_minority,
                                          layout.data_dir / f"{pool_name}_mitigated_{fq.mitigation_mode}_{tag}.zip",
                                          layout.candidates / f"{pool_name}_mitigated", fq.mitigation_mode)
    log.info("Candidate pool: %d ảnh (%s) | cần chọn %d", len(pool.final), minority, ctx.budget.n_select)
    return pool


def stage_generate(cfg: Config) -> CandidatePool:
    return resolve_candidate_pool(Context.create(cfg), generate_if_missing=True)


def stage_frequency(cfg: Config) -> pd.DataFrame:
    """Bước 4b — đo fingerprint tần số qua các giai đoạn: gốc → sau chuẩn hoá → sau mitigation."""
    from .classify.data import configure_gpu
    from .frequency.fingerprint import ensure_frank_repo, fingerprint_report
    from .frequency.mitigation import fetch_reference_code, plot_before_after

    ctx = Context.create(cfg)
    configure_gpu()
    pool = resolve_candidate_pool(ctx, generate_if_missing=False)
    frank = ensure_frank_repo(ctx.layout.frank_repo)
    fq, minority, fig_dir = cfg.frequency, ctx.budget.minority, ctx.layout.freq_fig_dir
    fig_dir = fig_dir if cfg.save_figures else None

    stages = [("goc", pool.raw), ("sau_chuan_hoa", pool.harmonized), ("sau_mitigation", pool.mitigated)]
    rows = []
    for stage, paths in stages:
        if paths:
            rows += fingerprint_report(stage, {minority: paths}, ctx.train_paths, frank, fig_dir,
                                       fq.fingerprint_n_max, fq.fingerprint_epochs, cfg.seed)
    if pool.mitigated:
        fetch_reference_code(ctx.layout.dong_dir)
        if fig_dir is not None:
            plot_before_after(pool.harmonized or pool.raw, pool.mitigated, minority,
                              fig_dir / f"before_after_{minority}.png")
    df = pd.DataFrame(rows)
    df.to_csv(ctx.layout.results_dir / "frequency_fingerprint_auc.csv", index=False)
    print(df.round(4).to_string(index=False))
    return df


def stage_select(cfg: Config) -> dict[str, list[int]]:
    """Bước 5–7b — nhúng E_v / E_d, chấm điểm, chọn ảnh cho M0–M6 và chẩn đoán."""
    from .classify.data import configure_gpu
    from .data.variants import assemble_variant
    from .selection import diagnostics as diag
    from .selection.dass import jaccard_matrix, select_all_methods
    from .selection.encoders import disease_encoder, disease_encoder_ckpt_name, visual_encoder
    from .selection.scoring import compute_pool_scores

    ctx = Context.create(cfg)
    configure_gpu(cfg.classifier.mixed_precision)
    layout, sel, budget = ctx.layout, cfg.selection, ctx.budget
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    classes = cfg.data.class_names
    train_paths, val_paths = ctx.train_paths, ctx.val_paths

    ev = visual_encoder(cfg)
    z_v_real = {c: ev.embed(train_paths[c]) for c in classes}
    z_v_pool = ev.embed(pool)

    assemble_variant(layout.real_only, layout.train_pp, ctx.split, budget.minority, [])
    ed = disease_encoder(cfg, layout.real_only, layout.clf_dir / disease_encoder_ckpt_name(cfg), layout.clf_ckpt)
    z_d_real = {c: ed.embed(train_paths[c]) for c in classes}
    z_d_pool = ed.embed(pool)

    # AUC probe: chỉ dòng 'val' là khách quan (E_d đã học chính các ảnh train)
    c2i = cfg.data.class_to_idx
    probe = pd.DataFrame([
        {"khong_gian": "E_v (ImageNet)", "tap": "train (E_d đã thấy)", "auc_probe": diag.probe_auc(z_v_real, c2i, cfg.seed)},
        {"khong_gian": "E_d (có giám sát)", "tap": "train (E_d đã thấy)", "auc_probe": diag.probe_auc(z_d_real, c2i, cfg.seed)},
        {"khong_gian": "E_v (ImageNet)", "tap": "val (khách quan)",
         "auc_probe": diag.probe_auc({c: ev.embed(val_paths[c]) for c in classes}, c2i, cfg.seed)},
        {"khong_gian": "E_d (có giám sát)", "tap": "val (khách quan)",
         "auc_probe": diag.probe_auc({c: ed.embed(val_paths[c]) for c in classes}, c2i, cfg.seed)},
    ])
    probe.to_csv(layout.results_dir / "probe_auc_Ev_vs_Ed.csv", index=False)
    print(probe.round(4).to_string(index=False))

    scores = compute_pool_scores(z_v_pool, z_v_real, z_d_pool, z_d_real, budget.minority, budget.majority,
                                 sel.lambda_v, sel.lambda_d, sel.sim_topk)
    selections = select_all_methods(scores, z_v_pool, budget.n_select, sel.alpha, sel.beta, sel.gamma, cfg.seed)
    for m, idx in selections.items():
        log.info("%s: chọn %d / %d ảnh sinh", m, len(idx), len(pool))

    write_json_atomic(layout.selections_json, {m: [Path(pool[i]).name for i in idx] for m, idx in selections.items()})
    np.savez(layout.embeddings_npz, pool_names=np.array([Path(p).name for p in pool]), z_v_pool=z_v_pool,
             z_d_pool=z_d_pool, **{f"z_v_real__{c}": z_v_real[c] for c in classes},
             **{f"z_d_real__{c}": z_d_real[c] for c in classes}, **{f"score__{k}": v for k, v in scores.items()})

    J = jaccard_matrix(selections)
    J.to_csv(layout.results_dir / "selection_jaccard.csv")
    print("\n=== Jaccard giữa các biến thể ===\n" + J.round(2).to_string())

    shortcut = diag.shortcut_table(z_v_real, z_v_pool, selections, budget.minority, budget.majority, cfg.seed)
    shortcut.to_csv(layout.results_dir / "shortcut_check.csv", index=False)
    print(shortcut.round(4).to_string(index=False))

    if cfg.save_figures:
        fig_dir = layout.selection_fig_dir
        diag.plot_score_scatter(scores, selections, fig_dir / "scatter_Mv_Md.png")
        for m, idx in selections.items():
            if idx:
                diag.plot_selected_vs_removed(pool, scores, idx, m, sel.alpha, sel.beta, fig_dir / f"grid_{m}_random.png")
        for mode in ["extreme", "boundary"]:
            diag.plot_selected_vs_removed(pool, scores, selections["M6_dass"], "M6_dass", sel.alpha, sel.beta,
                                          fig_dir / f"grid_M6_dass_{mode}.png", mode=mode)
        diag.plot_nearest_real(pool, train_paths[budget.minority], z_d_pool, z_d_real[budget.minority],
                               selections["M6_dass"], "M6_dass", fig_dir / "nearest_real_M6_dass.png")
    return selections


def load_selections(ctx: Context, pool: list[str]) -> dict[str, list[str]]:
    """selections.json lưu tên file -> ánh xạ về đường dẫn trong pool hiện tại."""
    names = read_json(ctx.layout.selections_json)
    if names is None:
        raise RuntimeError(f"Chưa có {ctx.layout.selections_json}. Chạy stage select trước.")
    by_name = {Path(p).name: p for p in pool}
    missing = [n for files in names.values() for n in files if n not in by_name]
    if missing:
        raise RuntimeError(f"{len(missing)} ảnh trong selections.json không có trong pool hiện tại "
                           "(pool hoặc cấu hình frequency đã đổi?). Chạy lại stage select.")
    return {m: [by_name[n] for n in files] for m, files in names.items()}


def stage_train(cfg: Config, model_name: str, seeds: list[int] | None = None) -> None:
    """Bước 8 + 10a — ghép tập huấn luyện cho các biến thể rồi train `model_name` với các seed."""
    from .classify.data import configure_gpu
    from .classify.train import run_experiments
    from .data.variants import assert_clean_eval_sets, prepare_variants

    ctx = Context.create(cfg)
    configure_gpu(cfg.classifier.mixed_precision)
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)
    sel = cfg.selection
    variants = prepare_variants(ctx.layout.variants, ctx.layout.train_pp, ctx.split, ctx.budget.minority, selections,
                                sel.pool_mult, f"v{sel.lambda_v:g}_d{sel.lambda_d:g}", cfg.evaluation.baseline_method)
    assert_clean_eval_sets(variants, ctx.layout.test_pp, cfg.data.class_names)
    run_experiments(cfg, ctx.layout, model_name, variants, seeds or cfg.classifier.seeds)


def stage_aggregate(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .evaluation.aggregate import compare_to_baseline, load_all_runs, mean_std_table

    layout = Layout(cfg)
    layout.makedirs()
    runs, probs = load_all_runs(layout.pred_dir)
    log.info("Đã đọc %d lần chạy từ %s", len(runs), layout.pred_dir)
    if runs.empty:
        return runs, pd.DataFrame()
    runs.to_csv(layout.results_dir / "all_runs.csv", index=False)
    summary = mean_std_table(runs)
    summary.to_csv(layout.results_dir / "summary_mean_std.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Kết quả (mean ± std qua các seed) — ngưỡng 0.5 ===\n" + summary.to_string(index=False))

    baseline = cfg.evaluation.baseline_method
    cmp = compare_to_baseline(probs, baseline, cfg.evaluation.n_bootstrap, cfg.seed)
    if len(cmp):
        cmp.to_csv(layout.results_dir / f"paired_bootstrap_vs_{baseline}.csv", index=False)
        print(f"\n=== Paired bootstrap ΔAUC so với {baseline} ===\n" + cmp.round(4).to_string(index=False))
    return summary, cmp


def stage_quality(cfg: Config) -> pd.DataFrame:
    """Chất lượng tập ảnh sinh được chọn (KID/FID trên đặc trưng E_v, đa dạng, SSIM nội bộ, AUC thật-vs-sinh)."""
    from .evaluation.quality import compute_diversity, compute_fid, compute_ssim
    from .gan.metrics import kid_from_features

    ctx = Context.create(cfg)
    layout = ctx.layout
    if not layout.embeddings_npz.exists():
        raise RuntimeError(f"Chưa có {layout.embeddings_npz}. Chạy stage select trước.")
    emb = np.load(layout.embeddings_npz)
    index = {str(n): i for i, n in enumerate(emb["pool_names"])}
    real = emb[f"z_v_real__{ctx.budget.minority}"].astype(np.float64)
    pool = resolve_candidate_pool(ctx, generate_if_missing=False).final
    selections = load_selections(ctx, pool)

    rows = []
    for method, paths in selections.items():
        if not paths:
            continue
        fake = emb["z_v_pool"][[index[Path(p).name] for p in paths]].astype(np.float64)
        rows.append({"method": method, "k": cfg.selection.pool_mult, "n": len(paths),
                     "kid_effnet": kid_from_features(real, fake, num_subsets=cfg.gan.kid_subsets,
                                                     seed=cfg.gan.kid_seed),
                     "fid_effnet_tham_khao": compute_fid(real, fake),
                     "diversity": compute_diversity(fake),
                     "ssim_noi_bo": compute_ssim(paths, cfg.evaluation.ssim_pairs, cfg.seed)})
    df = pd.DataFrame(rows)
    shortcut_csv = layout.results_dir / "shortcut_check.csv"
    if shortcut_csv.exists():
        sc = pd.read_csv(shortcut_csv)[["method", "auc_5fold"]].rename(columns={"auc_5fold": "auc_that_vs_sinh"})
        df = df.merge(sc, on="method", how="left")
    df.to_csv(layout.results_dir / f"quality_p{cfg.selection.pool_mult:g}.csv", index=False)
    print(f"=== Chất lượng ảnh sinh — FID trên {len(real)} ảnh thật chỉ để tham khảo ===\n"
          + df.round(4).to_string(index=False))
    return df
