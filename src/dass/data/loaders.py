"""tf.data cho classifier và bộ mã hoá. Ảnh đọc ĐÚNG số kênh gốc; ảnh xám chỉ được nhân bản 1 -> 3 kênh
trên bộ nhớ, ngay trước bước preprocess của backbone pretrain ImageNet (không ghi ra đĩa, không đổi giá trị).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers


def configure_gpu(mixed_precision: bool = False) -> None:
    """BẮT BUỘC gọi trước mọi op TF: mặc định TF chiếm toàn bộ VRAM."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def to_backbone_input(x: tf.Tensor) -> tf.Tensor:
    """(…, 1) -> (…, 3) bằng nhân bản; (…, 3) giữ nguyên."""
    return tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x


def make_augmenter(seed: int) -> tf.keras.Sequential:
    """Augmentation hình học + độ sáng / tương phản — áp dụng như nhau cho mọi kênh, không tạo màu."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.5, fill_mode="reflect", seed=seed),
        layers.RandomZoom((-0.1, 0.1), fill_mode="reflect", seed=seed),
        layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
        layers.RandomContrast(0.1, seed=seed),
    ], name="augment")


def build_dataset(data_dir: str | Path, class_names: list[str], image_size: int, batch_size: int, channels: int,
                  preprocess_fn: Callable, shuffle: bool, augment: bool = False, seed: int = 0) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir), labels="inferred", label_mode="binary", class_names=list(class_names),
        color_mode="grayscale" if channels == 1 else "rgb", image_size=(image_size, image_size),
        batch_size=batch_size, shuffle=shuffle, seed=seed, verbose=False)
    if augment:
        aug = make_augmenter(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_fn(to_backbone_input(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def paths_dataset(paths: list[str], image_size: int, channels: int, preprocess_fn: Callable,
                  batch_size: int = 32) -> tf.data.Dataset:
    def _load(path):
        img = tf.io.decode_png(tf.io.read_file(path), channels=channels)
        img = tf.image.resize(img, [image_size, image_size])
        return preprocess_fn(to_backbone_input(img))

    return (tf.data.Dataset.from_tensor_slices(list(paths))
            .map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE))
