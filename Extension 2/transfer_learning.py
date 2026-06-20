"""Transfer-learning models for sickle-cell binary classification.

Provides builders for EfficientNetB0, ResNet50 and MobileNetV2 with a shared
two-stage training routine (feature-extraction then fine-tuning).
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from project_paths import preferred_model_path


# ---------------------------------------------------------------------------
# Custom preprocessing layer (registered so .keras files deserialize)
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="sickle_cell")
class AppPreprocess(tf.keras.layers.Layer):
    """Apply the backbone-specific preprocessing inside the graph."""

    def __init__(self, app_name: str, **kwargs):
        super().__init__(**kwargs)
        self.app_name = app_name

    def call(self, inputs):
        if self.app_name == "mobilenet_v2":
            return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        if self.app_name == "resnet50":
            return tf.keras.applications.resnet50.preprocess_input(inputs)
        if self.app_name == "efficientnet":
            return tf.keras.applications.efficientnet.preprocess_input(inputs)
        raise ValueError(f"Unknown app_name: {self.app_name}")

    def get_config(self):
        config = super().get_config()
        config.update({"app_name": self.app_name})
        return config


# ---------------------------------------------------------------------------
# Shared head + compile helper
# ---------------------------------------------------------------------------
INPUT_SIZE = (64, 64)


def _add_head(base_model: tf.keras.Model, app_name: str) -> tf.keras.Model:
    """Wrap *base_model* with resizing, preprocessing and a binary head."""
    inp = layers.Input(shape=(64, 64, 3))
    x = layers.Resizing(224, 224)(inp)
    x = AppPreprocess(app_name)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


def _compile(model: tf.keras.Model, lr: float = 1e-3) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_efficientnet_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base.trainable = False
    model = _add_head(base, "efficientnet")
    _compile(model)
    return model, base


def build_resnet_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base.trainable = False
    model = _add_head(base, "resnet50")
    _compile(model)
    return model, base


def build_mobilenet_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base.trainable = False
    model = _add_head(base, "mobilenet_v2")
    _compile(model)
    return model, base


# ---------------------------------------------------------------------------
# Two-stage training routine
# ---------------------------------------------------------------------------
def train_transfer_model(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds,
    val_ds,
    feature_epochs: int = 10,
    finetune_epochs: int = 12,
    finetune_fraction: float = 0.20,
):
    """Train a transfer-learning model in two stages.

    Stage 1 – Feature extraction: backbone frozen, train the head.
    Stage 2 – Fine-tuning: unfreeze top *finetune_fraction* of backbone layers,
              keeping BatchNorm frozen, and train at a lower LR.

    Returns (history_stage1, history_stage2).
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7
        ),
    ]

    # Stage 1 – feature extraction
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=feature_epochs,
        callbacks=callbacks,
        verbose=0,
    )

    # Stage 2 – fine-tuning
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * (1.0 - finetune_fraction))
    base_model.trainable = True
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    # Keep BatchNorm layers frozen to avoid instability
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    _compile(model, lr=1e-5)

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=callbacks,
        verbose=0,
    )

    return h1, h2


# ---------------------------------------------------------------------------
# Standalone training (single split)
# ---------------------------------------------------------------------------
def main():
    from data_generators import create_train_val_generators

    train_gen, val_gen = create_train_val_generators()

    for name, builder, save_path in [
        ("EfficientNetB0", build_efficientnet_model, "efficientnetb0_transfer.keras"),
        ("ResNet50", build_resnet_model, "resnet50_transfer.keras"),
        ("MobileNetV2", build_mobilenet_model, "mobilenetv2_transfer.keras"),
    ]:
        print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
        model, base = builder()
        h1, h2 = train_transfer_model(model, base, train_gen, val_gen)
        final_save_path = preferred_model_path(save_path)
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(final_save_path)
        print(f"Saved {final_save_path}")


if __name__ == "__main__":
    main()
