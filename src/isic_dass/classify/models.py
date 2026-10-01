"""Kiến trúc classifier. Mỗi builder trả về (model, backbone, preprocess_fn).

Thêm model mới: viết một hàm `build_xxx(image_size, dropout, cfg)` rồi đăng ký vào `MODEL_BUILDERS`.
TensorFlow chỉ được import khi gọi builder, để CLI liệt kê được tên model mà không cần TF.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Built = tuple[Any, Any, Callable]


def _head(base, image_size: int, dropout: float, preprocess: Callable) -> Built:
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)          # BatchNorm luôn ở chế độ inference khi fine-tune
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs), base, preprocess


def _keras_app(app_name: str, preprocess_module: str | None):
    def build(image_size: int, dropout: float, cfg) -> Built:
        import tensorflow as tf

        kwargs = dict(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3), pooling="avg")
        if app_name == "EfficientNetV2B0":
            kwargs["include_preprocessing"] = True   # model tự chuẩn hoá, input giữ [0, 255]
        base = getattr(tf.keras.applications, app_name)(**kwargs)
        if preprocess_module is None:              # model tự chuẩn hoá bên trong
            preprocess = lambda x: tf.cast(x, tf.float32)  # noqa: E731
        else:
            preprocess = getattr(tf.keras.applications, preprocess_module).preprocess_input
        return _head(base, image_size, dropout, preprocess)

    return build


def _keras_hub(preset_attr: str):
    """ViT / Swin từ KerasHub: preprocessor của preset tự resize + chuẩn hoá -> preprocess ở tf.data là identity."""
    def build(image_size: int, dropout: float, cfg) -> Built:
        import keras_hub
        import tensorflow as tf

        model = keras_hub.models.ImageClassifier.from_preset(getattr(cfg, preset_attr), num_classes=1,
                                                             activation="sigmoid")
        return model, model.backbone, (lambda x: tf.cast(x, tf.float32))

    return build


MODEL_BUILDERS: dict[str, Callable[..., Built]] = {
    "ResNet50": _keras_app("ResNet50", "resnet50"),
    "DenseNet121": _keras_app("DenseNet121", "densenet"),
    "EfficientNetV2B0": _keras_app("EfficientNetV2B0", None),
    # Các model dùng trong bài báo CosSIF
    "ConvNeXtTiny": _keras_app("ConvNeXtTiny", None),
    "ViT-B16": _keras_hub("vit_preset"),
    "SwinT": _keras_hub("swin_preset"),
}

MODEL_NAMES = tuple(MODEL_BUILDERS)


def build_model(name: str, clf_cfg) -> Built:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Model không hỗ trợ: {name}. Có: {', '.join(MODEL_NAMES)}")
    return MODEL_BUILDERS[name](clf_cfg.size, clf_cfg.dropout, clf_cfg)
