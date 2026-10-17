"""Train classifier 2 giai đoạn (head -> fine-tune) và lưu dự đoán val / test.

- Augmentation theo `classifier.augment` (mặc định tắt từ 1.11.0 -> train trên đúng ảnh đã lưu; bật thì như nhau cho
  mọi biến thể).
- Val chỉ dùng để chọn epoch (EarlyStopping + ModelCheckpoint theo `classifier.monitor`, mode = max).
- Test chỉ được dự đoán MỘT lần, sau khi đã nạp lại checkpoint tốt nhất.
- Mỗi (model, biến thể, seed) lưu một .npz trên Drive (ghi nguyên tử); file đã có -> bỏ qua (chạy lại được), nhưng
  nếu thiết lập train lưu trong file (augment, class_weight, monitor, synth_fraction) khác cấu hình hiện tại thì báo lỗi thay vì dùng lại
  (`--archive-mismatched`: chuyển kết quả cũ sang thư mục *_superseded/ rồi train lại).
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .. import __version__
from ..config import Config, Layout
from ..config.schema import ClassifierConfig
from ..data.loaders import build_dataset
from ..data.variants import Variant
from ..evaluation.aggregate import archive_superseded_run, protocol_mismatch
from ..models.classifiers import build_model, resolve_model_name
from ..utils import code_version, read_json, save_npz_atomic, write_json_atomic
from .metrics import training_metrics

log = logging.getLogger(__name__)


def compute_class_weight_from_dir(train_dir: str | Path, classes: dict[str, int]) -> dict[int, float]:
    n = {i: len(list((Path(train_dir) / c).iterdir())) for c, i in classes.items()}
    total = sum(n.values())
    return {i: total / (len(n) * cnt) for i, cnt in n.items()}


class ResumableBest(tf.keras.callbacks.Callback):
    """EarlyStopping + ModelCheckpoint(save_best_only) theo `monitor` (mode max, min_delta 0) — cùng luật với hai
    callback của Keras — nhưng lưu trạng thái (best, wait, lịch sử TÍCH LUỸ) ra JSON sau mỗi epoch để train tiếp được
    sau khi bị ngắt (dùng cùng `BackupAndRestore`, thứ khôi phục trọng số + optimizer + số epoch)."""

    def __init__(self, monitor: str, patience: int, ckpt_path: Path, state_path: Path):
        super().__init__()
        self.monitor, self.patience, self.ckpt_path, self.state_path = monitor, patience, ckpt_path, state_path
        st = read_json(state_path, default=None) or {"best": None, "wait": 0, "history": {}}
        self.best, self.wait, self.history = st["best"], st["wait"], st["history"]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))
        cur = float(logs[self.monitor])
        if self.best is None or cur > self.best:
            self.best, self.wait = cur, 0
            self.model.save_weights(str(self.ckpt_path))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        write_json_atomic(self.state_path, {"best": self.best, "wait": self.wait, "history": self.history})


def _fit_resumable(model, train_ds, val_ds, *, epochs: int, class_weight: dict | None, backup_dir: Path,
                   extra_callbacks: list | None = None) -> None:
    """model.fit có `BackupAndRestore`: bị ngắt (hết giờ GPU) -> gọi lại thì chạy tiếp từ epoch đã xong cuối cùng.
    Thứ tự xáo trộn dữ liệu sau khi train tiếp khác lần chạy liền một mạch (Keras không lưu trạng thái tf.data)."""
    callbacks = [tf.keras.callbacks.BackupAndRestore(str(backup_dir)), *(extra_callbacks or [])]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=callbacks,
              verbose=2)


def fit_two_stage(model, base, train_ds, val_ds, *, clf: ClassifierConfig, class_weight: dict | None,
                  ckpt_path: Path, head_epochs: int, ft_epochs: int, with_pr_auc: bool = True,
                  resume_dir: Path | None = None) -> tuple[int, dict]:
    """Giai đoạn 1: chỉ train head. Giai đoạn 2: fine-tune toàn bộ, giữ epoch tốt nhất theo `clf.monitor`.

    Trả về (best_epoch của giai đoạn 2, history). `history`: metric từng epoch của giai đoạn 2 (khoá của Keras: `loss`,
    `auc`, `val_loss`, `val_auc`, …) và của giai đoạn 1 với tiền tố `head_`. Model đã được nạp lại trọng số tốt nhất.
    `resume_dir` (tuỳ chọn): lưu trạng thái sau mỗi epoch để gọi lại thì train tiếp (dùng khi một lần train dài hơn
    giới hạn thời gian GPU, ví dụ E_d của PCam); cùng luật chọn epoch / dừng sớm, `resume_dir` bị xoá khi xong.
    """
    if resume_dir is not None:
        return _fit_two_stage_resumable(model, base, train_ds, val_ds, clf=clf, class_weight=class_weight,
                                        ckpt_path=ckpt_path, head_epochs=head_epochs, ft_epochs=ft_epochs,
                                        with_pr_auc=with_pr_auc, resume_dir=resume_dir)
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.AdamW(clf.head_lr), loss="binary_crossentropy",
                  metrics=training_metrics(with_pr_auc))
    head = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, class_weight=class_weight, verbose=2)

    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.AdamW(clf.ft_lr, weight_decay=clf.weight_decay),
                  loss="binary_crossentropy", metrics=training_metrics(with_pr_auc))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=clf.monitor, mode="max", patience=clf.early_stop_patience),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor=clf.monitor, mode="max",
                                           save_best_only=True, save_weights_only=True),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs, class_weight=class_weight,
                     callbacks=callbacks, verbose=2)
    model.load_weights(str(ckpt_path))
    best_epoch = int(np.argmax(hist.history[clf.monitor])) + 1
    log.info("best fine-tune epoch = %d, %s = %.4f, val AUC = %.4f", best_epoch, clf.monitor,
             max(hist.history[clf.monitor]), hist.history["val_auc"][best_epoch - 1])
    return best_epoch, {**hist.history, **{f"head_{k}": v for k, v in head.history.items()}}


def _fit_two_stage_resumable(model, base, train_ds, val_ds, *, clf: ClassifierConfig, class_weight: dict | None,
                             ckpt_path: Path, head_epochs: int, ft_epochs: int, with_pr_auc: bool,
                             resume_dir: Path) -> tuple[int, dict]:
    resume_dir.mkdir(parents=True, exist_ok=True)
    head_done = resume_dir / "head_done.weights.h5"
    head_state = resume_dir / "head_history.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.AdamW(clf.head_lr), loss="binary_crossentropy",
                  metrics=training_metrics(with_pr_auc))
    model.optimizer.build(model.trainable_variables)   # tạo biến optimizer trước -> BackupAndRestore nạp lại được
    if head_done.exists():
        model.load_weights(str(head_done))
        log.info("Train tiếp: giai đoạn head đã xong (%s)", head_done)
    else:
        rec = ResumableBest("val_auc", head_epochs + 1, resume_dir / "head_best.weights.h5", head_state)
        _fit_resumable(model, train_ds, val_ds, epochs=head_epochs, class_weight=class_weight,
                       backup_dir=resume_dir / "backup_head", extra_callbacks=[rec])
        model.save_weights(str(head_done))       # head luôn chạy đủ head_epochs như bản không resume

    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.AdamW(clf.ft_lr, weight_decay=clf.weight_decay),
                  loss="binary_crossentropy", metrics=training_metrics(with_pr_auc))
    model.optimizer.build(model.trainable_variables)
    best = ResumableBest(clf.monitor, clf.early_stop_patience, ckpt_path, resume_dir / "ft_state.json")
    ft_done = resume_dir / "ft_done"
    if not ft_done.exists():                     # fine-tune chưa xong (hết epoch hoặc dừng sớm) ở lần chạy trước
        _fit_resumable(model, train_ds, val_ds, epochs=ft_epochs, class_weight=class_weight,
                       backup_dir=resume_dir / "backup_ft", extra_callbacks=[best])
        ft_done.touch()
    model.load_weights(str(ckpt_path))
    hist = best.history
    best_epoch = int(np.argmax(hist[clf.monitor])) + 1
    log.info("best fine-tune epoch = %d, %s = %.4f, val AUC = %.4f", best_epoch, clf.monitor,
             max(hist[clf.monitor]), hist["val_auc"][best_epoch - 1])
    head_hist = (read_json(head_state, default=None) or {}).get("history", {})
    shutil.rmtree(resume_dir, ignore_errors=True)
    return best_epoch, {**hist, **{f"head_{k}": v for k, v in head_hist.items()}}


def predict_dir(model, preprocess, data_dir: str | Path, cfg: Config,
                channels: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(nhãn, xác suất, tên ảnh `lớp/tệp` theo đúng thứ tự dự đoán)."""
    clf = cfg.classifier
    ds = build_dataset(data_dir, cfg.data.class_names, clf.size, clf.batch_size, channels, preprocess, shuffle=False)
    y_true = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    files = [f"{Path(p).parent.name}/{Path(p).name}" for p in ds.file_paths]
    if len(files) != len(y_true):
        raise RuntimeError(f"Số tên ảnh ({len(files)}) khác số dự đoán ({len(y_true)}) ở {data_dir}")
    return y_true, y_prob, files


def train_classifier(cfg: Config, layout: Layout, model_name: str, variant: Variant, seed: int, channels: int):
    clf = cfg.classifier
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model, base, preprocess = build_model(model_name, clf)
    names = cfg.data.class_names
    train_dir, val_dir = variant.dir / "train", variant.dir / "val"
    train_ds = build_dataset(train_dir, names, clf.size, clf.batch_size, channels, preprocess, shuffle=True,
                             augment=clf.augment, seed=seed, augment_profile=cfg.data.augment_profile)
    val_ds = build_dataset(val_dir, names, clf.size, clf.batch_size, channels, preprocess, shuffle=False)
    class_weight = compute_class_weight_from_dir(train_dir, cfg.data.classes) if variant.class_weight else None
    ckpt = layout.clf_ckpt / f"{model_name}__{variant.tag}__s{seed}.weights.h5"
    best_epoch, history = fit_two_stage(model, base, train_ds, val_ds, clf=clf, class_weight=class_weight,
                                        ckpt_path=ckpt, head_epochs=clf.head_epochs, ft_epochs=clf.ft_epochs)
    if clf.save_weights_to_drive:
        shutil.copy2(ckpt, layout.clf_dir / ckpt.name)
    return model, preprocess, best_epoch, history


def history_arrays(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Lịch sử train -> mảng `hist_<khoá>` để lưu trong .npz (chỉ thêm thông tin, không thuộc giao thức train): xem
    đường val AUC / loss qua từng epoch, ví dụ phân biệt overfit với val phẳng khi epoch tốt nhất là 1."""
    return {f"hist_{k}": np.asarray(v, dtype=np.float64) for k, v in history.items()}


def prediction_path(layout: Layout, model_name: str, variant_tag: str, seed: int) -> Path:
    return layout.pred_dir / f"{model_name}__{variant_tag}__s{seed}.npz"


def predict_paths(model, preprocess, paths_by_class: dict[str, list[str]], cfg: Config,
                  channels: int) -> tuple[np.ndarray, np.ndarray]:
    """(nhãn, xác suất) trên danh sách ảnh theo lớp — cùng cách đọc / resize như `build_dataset` (bilinear)."""
    from ..data.loaders import paths_dataset

    labels = [cfg.data.classes[c] for c, ps in paths_by_class.items() for _ in ps]
    paths = [p for ps in paths_by_class.values() for p in ps]
    ds = paths_dataset(paths, cfg.classifier.size, channels, preprocess, cfg.classifier.batch_size)
    return np.asarray(labels, dtype=int), model.predict(ds, verbose=0).ravel()


def run_experiments(cfg: Config, layout: Layout, model_name: str, variants: dict[str, Variant], seeds: list[int],
                    channels: int, archive_mismatched: bool = False,
                    train_real: dict[str, list[str]] | None = None) -> None:
    """`archive_mismatched`: .npz đã có nhưng train theo giao thức khác cấu hình hiện tại -> chuyển nó (và trọng số
    trên Drive) sang `*_superseded/<giao thức>/` rồi train lại; mặc định báo lỗi.
    `train_real` ({lớp: ảnh THẬT của tập train}, không gồm ảnh nhân bản / ảnh sinh): có -> dự đoán thêm trên tập này
    (y_train / p_train trong .npz) để so AUC train với val / test."""
    model_name = resolve_model_name(model_name)
    code = code_version()
    log.info("Giao thức train %s | code %s (dass %s) | augment = %s (%s) | class weight: %s | seeds %s",
             model_name, code or "?", __version__, cfg.classifier.augment, cfg.data.augment_profile,
             [v.method for v in variants.values() if v.class_weight] or "không", list(seeds))
    for tag, variant in variants.items():
        for seed in seeds:
            out = prediction_path(layout, model_name, tag, seed)
            if out.exists():
                mismatch = protocol_mismatch(out, {"augment": cfg.classifier.augment,
                                                   "class_weight": variant.class_weight,
                                                   "monitor": cfg.classifier.monitor,
                                                   "synth_fraction": cfg.selection.synth_fraction,
                                                   "design": cfg.selection.design,
                                                   "filter_majority": cfg.selection.filter_majority})
                if not mismatch:
                    log.info("[bỏ qua, đã có] %s", out.name)
                    continue
                if not archive_mismatched:
                    raise RuntimeError(f"{out.name} được train với thiết lập khác ({mismatch}). Không dùng lại kết quả "
                                       "của giao thức khác: đổi --tag / run_tag cho cấu hình mới, hoặc chạy "
                                       "`train --archive-mismatched` để chuyển kết quả cũ sang predictions_superseded/ "
                                       "(không xoá) rồi train lại.")
                moved = archive_superseded_run(out, layout.clf_dir / f"{model_name}__{tag}__s{seed}.weights.h5",
                                               layout.superseded_pred_dir, layout.superseded_clf_dir)
                log.warning("Giao thức cũ (%s): đã chuyển %s -> %s, train lại", mismatch, out.name, moved)
            log.info("=== %s | %s | seed %d ===", model_name, tag, seed)
            model, preprocess, best_epoch, history = train_classifier(cfg, layout, model_name, variant, seed,
                                                                      channels)
            y_val, p_val, _ = predict_dir(model, preprocess, variant.dir / "val", cfg, channels)
            y_test, p_test, test_files = predict_dir(model, preprocess, layout.test_pp, cfg, channels)
            extra = {}
            if train_real:
                extra["y_train"], extra["p_train"] = predict_paths(model, preprocess, train_real, cfg, channels)
            save_npz_atomic(out, y_val=y_val, p_val=p_val, y_test=y_test, p_test=p_test, **extra,
                            test_files=np.array(test_files), model=model_name, method=variant.method,
                            k=cfg.selection.pool_mult, seed=seed, best_epoch=best_epoch, lam=variant.lam,
                            feature_space=variant.feature_space, augment=cfg.classifier.augment,
                            class_weight=variant.class_weight, monitor=cfg.classifier.monitor,
                            synth_fraction=cfg.selection.synth_fraction, design=cfg.selection.design,
                            filter_majority=cfg.selection.filter_majority,
                            code_version=code or "",
                            dass_version=__version__, **history_arrays(history))
            log.info("test ROC-AUC = %.4f | đã lưu %s", roc_auc_score(y_test, p_test), out.name)
            del model
            gc.collect()
            tf.keras.backend.clear_session()
