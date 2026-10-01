"""Backbone phân loại (Keras). Mỗi builder trả về (model, backbone, preprocess_fn); model nhận ảnh 3 kênh
(ảnh xám được `data.loaders` nhân bản 1 -> 3 kênh ngay trước preprocess).

Thêm model: viết builder rồi đăng ký vào `MODEL_BUILDERS`. TensorFlow chỉ được import khi gọi builder,
để CLI liệt kê được tên model mà không cần TF.
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
    "ConvNeXtTiny": _keras_app("ConvNeXtTiny", None),
    "ViT-B16": _keras_hub("vit_preset"),
    "SwinT": _keras_hub("swin_preset"),
}
MODEL_NAMES = tuple(MODEL_BUILDERS)


def resolve_model_name(name: str) -> str:
    """Không phân biệt hoa thường / gạch: 'resnet50', 'vit_b16' -> tên chuẩn."""
    norm = lambda s: s.lower().replace("-", "").replace("_", "")  # noqa: E731
    for canonical in MODEL_NAMES:
        if norm(canonical) == norm(name):
            return canonical
    raise KeyError(f"Model không hỗ trợ: {name}. Có: {', '.join(MODEL_NAMES)}")


def build_model(name: str, clf_cfg) -> Built:
    return MODEL_BUILDERS[resolve_model_name(name)](clf_cfg.size, clf_cfg.dropout, clf_cfg)
