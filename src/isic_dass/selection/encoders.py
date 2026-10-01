"""Bước 5 / 5b — hai bộ mã hoá đặc trưng.

- E_v: EfficientNet-B0 ImageNet, đóng băng, bỏ đầu phân loại.
- E_d: model riêng (mặc định DenseNet121, seed riêng) train Cross-Entropy trên nhãn THẬT của tập train
  (không thấy val, test, hay ảnh sinh), rồi giữ backbone làm bộ trích đặc trưng.
  Không dùng lại baseline M0: nếu dùng M0 để chọn ảnh rồi so với M0, ảnh được chọn chỉ củng cố ranh giới của M0.

Mọi vector nhúng được chuẩn hoá L2.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..classify.data import build_dataset, paths_dataset
from ..classify.models import build_model
from ..classify.train import compute_class_weight_from_dir, fit_two_stage
from ..config import Config
from ..utils import l2_normalize

log = logging.getLogger(__name__)


class Encoder:
    def __init__(self, model, preprocess, image_size: int, batch_size: int):
        self.model = model
        self.preprocess = preprocess
        self.image_size = image_size
        self.batch_size = batch_size

    def embed(self, paths: list[str]) -> np.ndarray:
        ds = paths_dataset(paths, self.image_size, self.preprocess, self.batch_size)
        return l2_normalize(self.model.predict(ds, verbose=0))


def visual_encoder(cfg: Config) -> Encoder:
    size = cfg.classifier.size
    model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg",
                                                 input_shape=(size, size, 3))
    model.trainable = False
    return Encoder(model, tf.keras.applications.efficientnet.preprocess_input, size, cfg.encoder.embed_batch_size)


def disease_encoder(cfg: Config, real_only_dir: Path, ckpt: Path, local_ckpt_dir: Path) -> Encoder:
    """Nạp E_d từ `ckpt` nếu có; nếu chưa, train trên `real_only_dir/{train,val}` rồi lưu vào `ckpt`."""
    enc, clf = cfg.encoder, cfg.classifier
    model, base, preprocess = build_model(enc.e_d_model, clf)
    if ckpt.exists():
        model.load_weights(str(ckpt))
        log.info("Đã nạp E_d từ %s", ckpt.name)
    else:
        log.info("Chưa có %s -> huấn luyện E_d trên ảnh thật ...", ckpt.name)
        names = cfg.data.class_names
        tf.keras.utils.set_random_seed(enc.e_d_seed)
        train_ds = build_dataset(real_only_dir / "train", names, clf.size, clf.batch_size, preprocess,
                                 shuffle=True, augment=True, seed=enc.e_d_seed)
        val_ds = build_dataset(real_only_dir / "val", names, clf.size, clf.batch_size, preprocess, shuffle=False)
        class_weight = compute_class_weight_from_dir(real_only_dir / "train", cfg.data.class_to_idx)
        local_ckpt = local_ckpt_dir / ckpt.name
        fit_two_stage(model, base, train_ds, val_ds, clf=clf, class_weight=class_weight, ckpt_path=local_ckpt,
                      head_epochs=enc.e_d_head_epochs, ft_epochs=enc.e_d_ft_epochs, with_pr_auc=False)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_ckpt, ckpt)
        log.info("Đã lưu E_d -> %s", ckpt)

    feat = tf.keras.Model(base.input, base.output, name=f"E_d_{enc.e_d_model}")
    feat.trainable = False
    return Encoder(feat, preprocess, clf.size, enc.embed_batch_size)


def disease_encoder_ckpt_name(cfg: Config) -> str:
    return f"Ed_{cfg.encoder.e_d_model}_s{cfg.encoder.e_d_seed}.weights.h5"
