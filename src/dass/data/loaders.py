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


AUGMENT_GEOMETRY = {
    # profile: (chế độ lật, biên độ xoay theo phần của 2π)
    "rotation_invariant": ("horizontal_and_vertical", 0.5),     # ±180° — tổn thương da không có hướng
    "upright": ("horizontal", 10 / 360),                        # ±10° — ảnh có hướng giải phẫu (X-quang ngực)
}


def make_augmenter(seed: int, profile: str = "rotation_invariant") -> tf.keras.Sequential:
    """Augmentation hình học + độ sáng / tương phản — áp dụng như nhau cho mọi kênh, không tạo màu.

    Mỗi lớp một seed RIÊNG: các lớp ngẫu nhiên của Keras 3 rút số bằng RNG không trạng thái theo bộ đếm, nên cùng
    seed -> cùng dãy số -> lật / xoay / zoom / độ sáng / tương phản của một ảnh bị khoá vào CÙNG một số ngẫu nhiên
    (ví dụ ảnh bị lật thì luôn tối hơn và xoay về một phía).
    """
    flip, rotation = AUGMENT_GEOMETRY[profile]
    return tf.keras.Sequential([
        layers.RandomFlip(flip, seed=seed),
        layers.RandomRotation(rotation, fill_mode="reflect", seed=seed + 1),
        layers.RandomZoom((-0.1, 0.1), fill_mode="reflect", seed=seed + 2),
        layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed + 3),
        layers.RandomContrast(0.1, seed=seed + 4),
    ], name="augment")


def build_dataset(data_dir: str | Path, class_names: list[str], image_size: int, batch_size: int, channels: int,
                  preprocess_fn: Callable, shuffle: bool, augment: bool = False, seed: int = 0,
                  augment_profile: str = "rotation_invariant") -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir), labels="inferred", label_mode="binary", class_names=list(class_names),
        color_mode="grayscale" if channels == 1 else "rgb", image_size=(image_size, image_size),
        batch_size=batch_size, shuffle=shuffle, seed=seed, verbose=False)
    if augment:
        aug = make_augmenter(seed, augment_profile)
        # Tuần tự (không num_parallel_calls): trạng thái seed của các lớp ngẫu nhiên là biến dùng chung; gọi song song
        # làm thứ tự rút seed không xác định và hai batch có thể dùng cùng seed. Tiền xử lý phía sau vẫn song song.
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y))
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
