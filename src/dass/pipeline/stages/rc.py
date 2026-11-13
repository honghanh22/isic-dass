"""Stage DASS-RC (thuyết minh 13/11/2026).

    rc-prepare  (GPU) sinh pool ⌈3·K_c⌉ cho MỌI lớp, Inception-v3 -> NC, E_d -> đặc trưng + logit -> rc_cache.npz;
                xuất ảnh nghi vấn gần trùng để người dùng xác minh (rc_nc_review/)
    rc-select   (CPU) gate + TMU + RDC + lazy greedy cho 6 cách chọn -> rc_selections/<tên>.json
    rc-search   (GPU) dò tuần tự trên VAL: η (RDC-only) -> μ (TMU-only) -> (σ, α) (DASS-RC); khoá rc_params.json
    rc-train    (GPU) train Imbalanced, ROS và 6 cách chọn (sampler lớp × nguồn) với tham số đã khoá

K_c = N_c (số ảnh thật train lớp c). Mọi cách chọn dùng chung ℰ_c và cùng ngân sách khả thi.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Config
from ...data.variants import VariantSpec, assert_clean_eval_sets, oversample_indices, prepare_variants
from ...selection import IMBALANCED, OVERSAMPLE
from ...selection.rc import (
    RC_DASS,
    RC_RDC,
    RC_SELECTION_METHODS,
    RC_TMU,
    NCResult,
    RCParams,
    nc_check,
    nc_threshold,
    select_class,
)
from ...utils import read_json, write_json_atomic
from ..context import Context
from . import init_tensorflow, save_metrics

log = logging.getLogger(__name__)

RC_TRAIN_METHODS = (IMBALANCED, OVERSAMPLE, *RC_SELECTION_METHODS)


def class_budget(ctx: Context) -> dict[str, int]:
    """K_c = N_c."""
    return {c: len(v) for c, v in ctx.train_paths.items()}


def pool_size(ctx: Context, mult: float) -> dict[str, int]:
    return {c: math.ceil(mult * k) for c, k in class_budget(ctx).items()}


def _pixel_hash(img: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()


def _inception(paths: list[str], channels: int, chunk: int = 2000) -> tuple[np.ndarray, list[str]]:
    from ...data.image_io import load_images
    from ...models.generator.inception import inception_features

    feats, hashes = [], []
    for s in range(0, len(paths), chunk):
        imgs = load_images(paths[s:s + chunk], channels)
        hashes.extend(_pixel_hash(im) for im in imgs)
        feats.append(inception_features(imgs))
    return np.concatenate(feats), hashes


# ------------------------------------------------------------------------------------------------------- rc-prepare
def prepare(cfg: Config) -> None:
    from ...models.generator.patches import ensure_stylegan_repo
    from ..pool import resolve_pool
    from .select import load_disease_encoder

    ctx = Context.create(cfg, "rc-prepare")
    layout, rc, classes = ctx.layout, cfg.rc, cfg.data.class_names
    sizes = pool_size(ctx, rc.pool_mult)
    pools = {c: resolve_pool(ctx, c, sizes[c], cfg.generator.gen_seed + 7919 * ctx.class_idx(c), True,
                             label="_rc").final for c in classes}
    for c in classes:
        log.info("Pool DASS-RC lớp %s: %d ảnh (K_c = %d)", c, len(pools[c]), class_budget(ctx)[c])

    # ---- NC: Inception-v3, so với TOÀN BỘ train của GAN (cả hai lớp)
    ensure_stylegan_repo(layout.sg2_repo)
    train = ctx.train_paths
    train_all = [p for c in classes for p in train[c]]
    f_train, h_train = _inception(train_all, ctx.channels)
    tau, d_real = nc_threshold(f_train, rc.nc_quantile)
    log.info("τ_NC = Q_%.2f = %.4f (khoảng cách láng giềng gần nhất giữa %d ảnh thật train)", rc.nc_quantile, tau,
             len(train_all))
    nc: dict[str, NCResult] = {}
    for c in classes:
        f_pool, h_pool = _inception(pools[c], ctx.channels)
        nc[c] = nc_check(f_pool, f_train, tau, h_pool, h_train)
        r = nc[c]
        log.info("NC lớp %s: trùng pixel %d | nghi vấn %d | trùng trong pool %d", c, int(r.pixel_duplicate.sum()),
                 int(r.suspect.sum()), int((r.pool_duplicate_of >= 0).sum()))

    # ---- E_d: đặc trưng chưa chuẩn hoá + logit (ảnh thật train và pool)
    init_tensorflow(cfg)
    ed, _ = load_disease_encoder(cfg, ctx)
    arrays: dict[str, np.ndarray] = {"tau": np.array(tau), "d_real_nn": d_real}
    for c in classes:
        fr = ed.embed(train[c], normalize=False)
        fp = ed.embed(pools[c], normalize=False)
        r = nc[c]
        arrays.update({f"{c}__real_feat": fr, f"{c}__real_logit": ed.logits(fr),
                       f"{c}__pool_feat": fp, f"{c}__pool_logit": ed.logits(fp),
                       f"{c}__pool_names": np.array([Path(p).name for p in pools[c]]),
                       f"{c}__nc_dist": r.nn_dist, f"{c}__nc_index": r.nn_index,
                       f"{c}__nc_pixel": r.pixel_duplicate, f"{c}__nc_suspect": r.suspect,
                       f"{c}__nc_dup_of": r.pool_duplicate_of})
    arrays["train_names"] = np.array([f"{Path(p).parent.name}/{Path(p).name}" for p in train_all])
    np.savez(layout.rc_cache_npz, **arrays)
    log.info("Đã lưu %s", layout.rc_cache_npz)
    _write_nc_review(cfg, ctx, pools, train_all, nc)


def _write_nc_review(cfg: Config, ctx: Context, pools: dict[str, list[str]], train_all: list[str],
                     nc: dict[str, NCResult]) -> None:
    """Danh sách + hình cặp (ảnh sinh | ảnh train gần nhất) của ảnh nghi vấn để xác minh bằng mắt."""
    from PIL import Image

    out = ctx.layout.rc_nc_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, r in nc.items():
        for i in np.flatnonzero(r.suspect | r.pixel_duplicate):
            rows.append({"class": c, "pool_image": Path(pools[c][i]).name, "nearest_train": train_all[r.nn_index[i]],
                         "distance": float(r.nn_dist[i]), "tau": r.tau,
                         "status": "pixel_duplicate (loại)" if r.pixel_duplicate[i] else "suspect (cần xác minh)"})
            if not r.pixel_duplicate[i]:
                a = Image.open(pools[c][i]).convert("RGB")
                b = Image.open(train_all[r.nn_index[i]]).convert("RGB").resize(a.size)
                pair = Image.new("RGB", (a.width * 2 + 4, a.height), "white")
                pair.paste(a, (0, 0))
                pair.paste(b, (a.width + 4, 0))
                pair.save(out / f"{c}__{Path(pools[c][i]).stem}__vs__{Path(train_all[r.nn_index[i]]).stem}.png")
    df = pd.DataFrame(rows, columns=["class", "pool_image", "nearest_train", "distance", "tau", "status"])
    df.to_csv(out / "nc_suspects.csv", index=False)
    n_suspect = int(sum(r.suspect.sum() for r in nc.values()))
    if n_suspect:
        log.warning("%d ảnh nghi vấn gần trùng -> xem %s, rồi ghi tên ảnh ĐÃ xác minh là bản sao vào %s dạng "
                    '{"<lớp>": ["<tên ảnh>", ...]} (danh sách rỗng nếu không có ảnh nào là bản sao).',
                    n_suspect, out, ctx.layout.rc_nc_confirmed)
    elif not ctx.layout.rc_nc_confirmed.exists():
        write_json_atomic(ctx.layout.rc_nc_confirmed, {c: [] for c in nc})
        log.info("Không có ảnh nghi vấn -> ghi %s rỗng", ctx.layout.rc_nc_confirmed)


# -------------------------------------------------------------------------------------------------------- rc-select
def locked_params(cfg: Config, layout) -> RCParams:
    saved = read_json(layout.rc_params_json)
    if saved:
        return RCParams(**{k: saved[k] for k in ("eta", "mu_percentile", "sigma", "alpha", "delta")})
    rc = cfg.rc
    return RCParams(eta=rc.eta, mu_percentile=rc.mu_percentile, sigma=rc.sigma, alpha=rc.alpha, delta=rc.delta)


def params_tag(p: RCParams) -> str:
    return f"eta{p.eta:g}_mu{p.mu_percentile:g}_sig{p.sigma:g}_a{p.alpha:g}"


def select(cfg: Config, params: RCParams | None = None, methods: tuple[str, ...] = RC_SELECTION_METHODS,
           name: str = "final", ctx: Context | None = None) -> Path:
    """Chọn ảnh cho mỗi lớp; ghi rc_selections/<name>.json = {"selections": {cách: {lớp: [tên]}}, "meta": …}."""
    ctx = ctx or Context.create(cfg, "rc-select")
    layout, classes = ctx.layout, cfg.data.class_names
    params = params or locked_params(cfg, layout)
    if not layout.rc_cache_npz.exists():
        raise RuntimeError(f"Chưa có {layout.rc_cache_npz} -> chạy `dass rc-prepare` trước")
    confirmed = read_json(layout.rc_nc_confirmed)
    if confirmed is None:
        raise RuntimeError(f"Chưa xác minh ảnh nghi vấn gần trùng: tạo {layout.rc_nc_confirmed} (xem rc_nc_review/)")
    cache = np.load(layout.rc_cache_npz, allow_pickle=False)
    k = class_budget(ctx)
    sel: dict[str, dict[str, list[str]]] = {m: {} for m in methods}
    meta: dict = {"params": params.__dict__, "classes": {}}
    for c in classes:
        names = [str(n) for n in cache[f"{c}__pool_names"]]
        r = NCResult(tau=float(cache["tau"]), nn_dist=cache[f"{c}__nc_dist"], nn_index=cache[f"{c}__nc_index"],
                     pixel_duplicate=cache[f"{c}__nc_pixel"], suspect=cache[f"{c}__nc_suspect"],
                     pool_duplicate_of=cache[f"{c}__nc_dup_of"])
        idx = {n: i for i, n in enumerate(names)}
        keep = r.keep({idx[n] for n in confirmed.get(c, [])})
        res = select_class(pool_emb=cache[f"{c}__pool_feat"], pool_logit=cache[f"{c}__pool_logit"],
                           real_emb=cache[f"{c}__real_feat"], real_logit=cache[f"{c}__real_logit"],
                           cls=ctx.class_idx(c), nc_keep=keep, k_target=k[c], params=params, seed=cfg.seed,
                           methods=methods)
        for m in methods:
            sel[m][c] = [names[i] for i in res.selections[m]]
        meta["classes"][c] = {"K_c": k[c], "pool": len(names), "after_nc": int(keep.sum()),
                              "eligible": len(res.eligible), "budget": res.budget, "shortfall": res.shortfall,
                              "h": res.h, "mu": res.mu}
        log.info("Lớp %s: pool %d | sau NC %d | qua gate %d | ngân sách %d (thiếu %d) | h = %.4f | μ = %.3f", c,
                 len(names), int(keep.sum()), len(res.eligible), res.budget, res.shortfall, res.h, res.mu)
    out = layout.rc_selections_dir / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, {"selections": sel, "meta": meta})
    return out


# --------------------------------------------------------------------------------------------------------- training
def _specs(ctx: Context, selection_file: Path, methods: list[str], suffix: str = "") -> dict[str, VariantSpec]:
    from ..pool import resolve_pool

    cfg = ctx.cfg
    b = ctx.budget
    specs: dict[str, VariantSpec] = {}
    if IMBALANCED in methods:
        specs[IMBALANCED] = VariantSpec()
    if OVERSAMPLE in methods:
        real_min = ctx.train_paths[b.minority]
        n_extra = b.n_real[b.majority] - b.n_real[b.minority]
        specs[OVERSAMPLE] = VariantSpec(extra_real=[real_min[i] for i in oversample_indices(len(real_min), n_extra,
                                                                                             cfg.seed)])
    rc_methods = [m for m in methods if m in RC_SELECTION_METHODS]
    if rc_methods:
        saved = read_json(selection_file)["selections"]
        sizes = pool_size(ctx, cfg.rc.pool_mult)
        pools = {c: {Path(p).name: p for p in resolve_pool(ctx, c, sizes[c],
                                                            cfg.generator.gen_seed + 7919 * ctx.class_idx(c), False,
                                                            label="_rc").final}
                 for c in cfg.data.class_names}
        for m in rc_methods:
            specs[f"{m}{suffix}"] = VariantSpec(synth_by_class={c: [pools[c][n] for n in names]
                                                                for c, names in saved[m].items()},
                                                sampler="class_source")
    return specs


def _train(cfg: Config, ctx: Context, model: str, seeds: list[int], specs: dict[str, VariantSpec],
           pred_dir: Path | None = None, predict_test: bool = True) -> None:
    from ...engine.trainer import run_experiments

    if not cfg.classifier.fixed_steps:
        raise ValueError("DASS-RC cần classifier.fixed_steps: true (mọi phương pháp cùng số update)")
    variants = prepare_variants(ctx.layout.variants, ctx.layout.train_pp, ctx.split, ctx.budget.minority, specs,
                                cfg.rc.pool_mult)
    assert_clean_eval_sets(variants, ctx.layout.test_pp, cfg.data.class_names)
    run_experiments(cfg, ctx.layout, model, variants, seeds, ctx.channels, train_real=ctx.train_paths,
                    pred_dir=pred_dir, predict_test=predict_test)


def train(cfg: Config, model: str, seeds: list[int] | None = None, methods: list[str] | None = None) -> None:
    ctx = Context.create(cfg, f"rc-train:{model}")
    init_tensorflow(cfg)
    if not ctx.layout.rc_params_json.exists():
        raise RuntimeError(f"Chưa khoá tham số ({ctx.layout.rc_params_json}) -> chạy `dass rc-search` trước")
    methods = list(methods or RC_TRAIN_METHODS)
    sel_file = select(cfg, name="final", ctx=ctx)
    _train(cfg, ctx, model, seeds or cfg.classifier.seeds, _specs(ctx, sel_file, methods))


# -------------------------------------------------------------------------------------------------------- rc-search
def _val_ap(pred_dir: Path, model: str, tag: str, seed: int) -> float:
    from sklearn.metrics import average_precision_score

    with np.load(pred_dir / f"{model}__{tag}__s{seed}.npz", allow_pickle=True) as d:
        return float(average_precision_score(d["y_val"], d["p_val"]))


def search(cfg: Config) -> RCParams:
    """Dò tuần tự trên VAL (không dự đoán test), khoá từng giai đoạn; hoà -> giá trị đứng trước trong lưới."""
    from ...data.variants import variant_tag
    from ...models.classifiers import resolve_model_name

    ctx = Context.create(cfg, "rc-search")
    init_tensorflow(cfg)
    layout, rc = ctx.layout, cfg.rc
    if layout.rc_params_json.exists():
        log.warning("Đã khoá tham số: %s -> giữ nguyên", read_json(layout.rc_params_json))
        return locked_params(cfg, layout)
    model, seed = resolve_model_name(rc.search_model), rc.search_seed
    p = RCParams(eta=rc.eta, mu_percentile=rc.mu_percentile, sigma=rc.sigma, alpha=rc.alpha, delta=rc.delta)
    log_rows = []

    def run_phase(phase: str, method: str, candidates: list[RCParams]) -> RCParams:
        best, best_ap = None, -np.inf
        for cand in candidates:
            name = f"search__{phase}__{params_tag(cand)}"
            sel = select(cfg, cand, (method,), name, ctx=ctx)
            suffix = f"__{phase}_{params_tag(cand)}"
            _train(cfg, ctx, model, [seed], _specs(ctx, sel, [method], suffix), pred_dir=layout.rc_search_pred_dir,
                   predict_test=False)
            ap = _val_ap(layout.rc_search_pred_dir, model, variant_tag(f"{method}{suffix}", rc.pool_mult), seed)
            log.info("[%s] %s -> val AP = %.4f", phase, params_tag(cand), ap)
            log_rows.append({"phase": phase, "method": method, **cand.__dict__, "val_ap": ap})
            if ap > best_ap:
                best, best_ap = cand, ap
        log.info("[%s] khoá %s (val AP = %.4f)", phase, params_tag(best), best_ap)
        return best

    p = run_phase("eta", RC_RDC, [RCParams(**{**p.__dict__, "eta": e}) for e in rc.eta_grid])
    # σ không đổi THỨ TỰ của TMU-only (top-K theo |m̄ − μ|) -> dò μ bằng TMU-only, dò (σ, α) bằng DASS-RC
    p = run_phase("tmu", RC_TMU, [RCParams(**{**p.__dict__, "mu_percentile": m}) for m in rc.mu_grid])
    p = run_phase("alpha", RC_DASS, [RCParams(**{**p.__dict__, "sigma": sg, "alpha": a})
                                     for sg in rc.sigma_grid for a in rc.alpha_grid])
    save_metrics(layout, "rc_search", pd.DataFrame(log_rows))
    write_json_atomic(layout.rc_params_json, {**p.__dict__, "search_model": model, "search_seed": seed,
                                              "coverage": "probabilistic"})
    log.info("Đã khoá tham số DASS-RC: %s -> %s", params_tag(p), layout.rc_params_json)
    return p
