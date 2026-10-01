"""Train classifier 2 giai đoạn (head -> fine-tune) và lưu dự đoán val / test.

- Val chỉ dùng để chọn epoch (EarlyStopping + ModelCheckpoint theo `classifier.monitor`, mode = max).
- Test chỉ được dự đoán MỘT lần, sau khi đã nạp lại checkpoint tốt nhất.
- Mỗi (model, biến thể, seed) lưu một .npz trên Drive; file đã có -> bỏ qua (chạy lại được).
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from ..config import Config, Layout
from ..config.schema import ClassifierConfig
from ..data.loaders import build_dataset
from ..data.variants import Variant
from ..models.classifiers import build_model, resolve_model_name
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


def predict_dir(model, preprocess, data_dir: str | Path, cfg: Config, channels: int) -> tuple[np.ndarray, np.ndarray]:
    clf = cfg.classifier
    ds = build_dataset(data_dir, cfg.data.class_names, clf.size, clf.batch_size, channels, preprocess, shuffle=False)
    y_true = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    return y_true, y_prob


def train_classifier(cfg: Config, layout: Layout, model_name: str, variant: Variant, seed: int, channels: int):
    clf = cfg.classifier
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model, base, preprocess = build_model(model_name, clf)
    names = cfg.data.class_names
    train_dir, val_dir = variant.dir / "train", variant.dir / "val"
    train_ds = build_dataset(train_dir, names, clf.size, clf.batch_size, channels, preprocess, shuffle=True,
                             augment=True, seed=seed)
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
                    channels: int) -> None:
    model_name = resolve_model_name(model_name)
    for tag, variant in variants.items():
        for seed in seeds:
            out = prediction_path(layout, model_name, tag, seed)
            if out.exists():
                log.info("[bỏ qua, đã có] %s", out.name)
                continue
            log.info("=== %s | %s | seed %d ===", model_name, tag, seed)
            model, preprocess, best_epoch = train_classifier(cfg, layout, model_name, variant, seed, channels)
            y_val, p_val = predict_dir(model, preprocess, variant.dir / "val", cfg, channels)
            y_test, p_test = predict_dir(model, preprocess, layout.test_pp, cfg, channels)
            np.savez(out, y_val=y_val, p_val=p_val, y_test=y_test, p_test=p_test,
                     model=model_name, method=variant.method, k=cfg.selection.pool_mult, seed=seed,
                     best_epoch=best_epoch, lam=variant.lam, feature_space=variant.feature_space)
            log.info("test ROC-AUC = %.4f | đã lưu %s", roc_auc_score(y_test, p_test), out.name)
            del model
            gc.collect()
            tf.keras.backend.clear_session()
