"""Train classifier 2 giai đoạn (head -> fine-tune) và lưu dự đoán val / test.

- Augmentation theo `classifier.augment` (mặc định tắt từ 1.11.0 -> train trên đúng ảnh đã lưu; bật thì như nhau cho
  mọi biến thể).
- Val chỉ dùng để chọn epoch (EarlyStopping + ModelCheckpoint theo `classifier.monitor`, mode = max).
- Test chỉ được dự đoán MỘT lần, sau khi đã nạp lại checkpoint tốt nhất.
- Mỗi (model, biến thể, seed) lưu một .npz trên Drive (ghi nguyên tử); file đã có -> bỏ qua (chạy lại được), nhưng
  nếu thiết lập train lưu trong file (augment, class_weight, monitor) khác cấu hình hiện tại thì báo lỗi thay vì dùng lại
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
from ..utils import code_version, save_npz_atomic
from .metrics import training_metrics

log = logging.getLogger(__name__)


def compute_class_weight_from_dir(train_dir: str | Path, classes: dict[str, int]) -> dict[int, float]:
    n = {i: len(list((Path(train_dir) / c).iterdir())) for c, i in classes.items()}
    total = sum(n.values())
    return {i: total / (len(n) * cnt) for i, cnt in n.items()}


def fit_two_stage(model, base, train_ds, val_ds, *, clf: ClassifierConfig, class_weight: dict | None,
                  ckpt_path: Path, head_epochs: int, ft_epochs: int, with_pr_auc: bool = True) -> tuple[int, dict]:
    """Giai đoạn 1: chỉ train head. Giai đoạn 2: fine-tune toàn bộ, giữ epoch tốt nhất theo `clf.monitor`.

    Trả về (best_epoch của giai đoạn 2, history). Model đã được nạp lại trọng số tốt nhất.
    """
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.AdamW(clf.head_lr), loss="binary_crossentropy",
                  metrics=training_metrics(with_pr_auc))
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, class_weight=class_weight, verbose=2)

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
    history = hist.history
    best_epoch = int(np.argmax(history[clf.monitor])) + 1
    log.info("best fine-tune epoch = %d, %s = %.4f, val AUC = %.4f", best_epoch, clf.monitor,
             max(history[clf.monitor]), history["val_auc"][best_epoch - 1])
    return best_epoch, history


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
    best_epoch, _ = fit_two_stage(model, base, train_ds, val_ds, clf=clf, class_weight=class_weight, ckpt_path=ckpt,
                                  head_epochs=clf.head_epochs, ft_epochs=clf.ft_epochs)
    if clf.save_weights_to_drive:
        shutil.copy2(ckpt, layout.clf_dir / ckpt.name)
    return model, preprocess, best_epoch


def prediction_path(layout: Layout, model_name: str, variant_tag: str, seed: int) -> Path:
    return layout.pred_dir / f"{model_name}__{variant_tag}__s{seed}.npz"


def run_experiments(cfg: Config, layout: Layout, model_name: str, variants: dict[str, Variant], seeds: list[int],
                    channels: int, archive_mismatched: bool = False) -> None:
    """`archive_mismatched`: .npz đã có nhưng train theo giao thức khác cấu hình hiện tại -> chuyển nó (và trọng số
    trên Drive) sang `*_superseded/<giao thức>/` rồi train lại; mặc định báo lỗi."""
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
                                                   "monitor": cfg.classifier.monitor})
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
            model, preprocess, best_epoch = train_classifier(cfg, layout, model_name, variant, seed, channels)
            y_val, p_val, _ = predict_dir(model, preprocess, variant.dir / "val", cfg, channels)
            y_test, p_test, test_files = predict_dir(model, preprocess, layout.test_pp, cfg, channels)
            save_npz_atomic(out, y_val=y_val, p_val=p_val, y_test=y_test, p_test=p_test,
                            test_files=np.array(test_files), model=model_name, method=variant.method,
                            k=cfg.selection.pool_mult, seed=seed, best_epoch=best_epoch, lam=variant.lam,
                            feature_space=variant.feature_space, augment=cfg.classifier.augment,
                            class_weight=variant.class_weight, monitor=cfg.classifier.monitor, code_version=code or "",
                            dass_version=__version__)
            log.info("test ROC-AUC = %.4f | đã lưu %s", roc_auc_score(y_test, p_test), out.name)
            del model
            gc.collect()
            tf.keras.backend.clear_session()
