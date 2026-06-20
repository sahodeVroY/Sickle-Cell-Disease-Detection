import argparse
from pathlib import Path
from typing import Optional, Iterable

import cv2
import numpy as np
import tensorflow as tf

from project_paths import existing_model_paths
from project_paths import preferred_model_path

@tf.keras.utils.register_keras_serializable(package="sickle_cell")
class AppPreprocess(tf.keras.layers.Layer):
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


def list_images(folder: Path) -> list[Path]:
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid]


def load_image_for_model(image_path: Path, target_size=(64, 64)) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32)
    return img_rgb, x


def iter_all_layers(layers: Iterable[tf.keras.layers.Layer]) -> Iterable[tf.keras.layers.Layer]:
    for layer in layers:
        yield layer
        if isinstance(layer, tf.keras.Model):
            for sub in iter_all_layers(layer.layers):
                yield sub


def find_layer_by_name(model: tf.keras.Model, layer_name: str) -> tf.keras.layers.Layer:
    for layer in iter_all_layers(model.layers):
        if layer.name == layer_name:
            return layer
    raise ValueError(f"Layer not found: {layer_name}")


def auto_find_last_conv(model: tf.keras.Model) -> tf.keras.layers.Layer:
    last = None
    for layer in iter_all_layers(model.layers):
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.SeparableConv2D,
                tf.keras.layers.DepthwiseConv2D,
            ),
        ):
            last = layer
    if last is None:
        raise RuntimeError("No Conv2D-like layer found for Grad-CAM.")
    return last


def find_base_app_model(model: tf.keras.Model) -> Optional[tf.keras.Model]:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(
            k in layer.name.lower() for k in ("efficientnet", "resnet", "mobilenet")
        ):
            return layer
    return None


def _model_call(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    try:
        return model(x, training=False)
    except Exception:
        return model({model.input_names[0]: x}, training=False)


def compute_gradcam(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    target_layer: Optional[tf.keras.layers.Layer],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(img_batch, np.ndarray):
        img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    _ = _model_call(model, img_batch)

    base_app_model = find_base_app_model(model)

    if base_app_model is None:
        inp = tf.keras.Input(shape=tuple(img_batch.shape[1:]), name="gradcam_input")
        cloned = tf.keras.models.clone_model(model, input_tensors=inp)
        cloned.set_weights(model.get_weights())

        if target_layer is None:
            target = auto_find_last_conv(cloned)
        else:
            target = cloned.get_layer(target_layer.name)

        grad_model = tf.keras.Model(cloned.inputs, [target.output, cloned.outputs[0]])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_batch, training=False)
            if preds.shape[-1] == 1:
                eps = 1e-7
                p = tf.clip_by_value(preds[:, 0], eps, 1.0 - eps)
                loss = tf.math.log(p / (1.0 - p))
            else:
                idx = tf.argmax(preds, axis=-1)
                loss = tf.gather_nd(preds, tf.stack([tf.range(tf.shape(preds)[0]), idx], axis=1))
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            raise RuntimeError("Gradients are None. Use --layer to choose a different conv layer.")
    else:
        resizing_layer = None
        preprocess_layer = None
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.layers.Resizing):
                resizing_layer = lyr
            if isinstance(lyr, AppPreprocess):
                preprocess_layer = lyr

        x = img_batch
        if resizing_layer is not None:
            x = resizing_layer(x)
        if preprocess_layer is not None:
            x = preprocess_layer(x)

        if target_layer is None:
            target_layer = auto_find_last_conv(base_app_model)

        conv_and_base = tf.keras.Model(
            base_app_model.inputs,
            [target_layer.output, base_app_model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, base_out = conv_and_base(x, training=False)

            y = base_out
            seen_base = False
            for lyr in model.layers:
                if lyr is base_app_model:
                    seen_base = True
                    continue
                if not seen_base:
                    continue
                y = lyr(y, training=False)
            preds = y

            if preds.shape[-1] == 1:
                eps = 1e-7
                p = tf.clip_by_value(preds[:, 0], eps, 1.0 - eps)
                loss = tf.math.log(p / (1.0 - p))
            else:
                idx = tf.argmax(preds, axis=-1)
                loss = tf.gather_nd(preds, tf.stack([tf.range(tf.shape(preds)[0]), idx], axis=1))

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            raise RuntimeError("Gradients are None. Use --layer to choose a different conv layer.")

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1, keepdims=True))
    cam = tf.image.resize(cam, (img_batch.shape[1], img_batch.shape[2]))
    cam = cam.numpy()
    
    # Use 98th percentile to saturate the hotspots (push them to true RED)
    # instead of letting a single outlier pixel compress the rest of the colors into yellow/green
    cam = cam - cam.min(axis=(1, 2, 3), keepdims=True)
    cam_max = np.percentile(cam, 98, axis=(1, 2, 3), keepdims=True)
    cam = np.clip(cam / (cam_max + 1e-8), 0.0, 1.0)
    
    return cam[..., 0], preds.numpy().reshape(-1)


def overlay_heatmap(img_rgb: np.ndarray, heat: np.ndarray, alpha=0.45) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    hm = cv2.resize((heat * 255).astype(np.uint8), (w, h))
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1 - alpha, hm_color, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to .keras model file (or 'all')")
    parser.add_argument("--image", default=None, help="Path to input image, or just a filename like 1.jpg")
    parser.add_argument("--out", default=None, help="Output image path (single-model mode only)")
    parser.add_argument("--layer", default=None, help="Optional conv layer name override")
    args = parser.parse_args()

    pos_dir = Path("sickle cell dataset") / "Positive" / "Labelled"
    imgs = list_images(pos_dir) if pos_dir.exists() else []
    if not imgs:
        raise SystemExit("Could not find any positive images under: sickle cell dataset/Positive/Labelled")

    if args.image is None:
        chosen = imgs[0]
    else:
        p = Path(args.image)
        if p.exists():
            chosen = p
        else:
            match = [x for x in imgs if x.name.lower() == p.name.lower()]
            if not match:
                raise SystemExit(f"Image not found: {args.image}")
            chosen = match[0]

    img_rgb, x = load_image_for_model(chosen, target_size=(64, 64))
    x_batch = np.expand_dims(x, 0)

    candidate_models = [
        "efficientnetb0_transfer.keras",
        "resnet50_transfer.keras",
        "mobilenetv2_transfer.keras",
        "sickle_cnn.keras",
    ]

    if args.model is None or str(args.model).lower() == "all":
        model_paths = existing_model_paths(candidate_models)
        if not model_paths:
            raise SystemExit("No .keras model files found in models/ or project root.")
    else:
        requested = Path(args.model)
        model_paths = [requested if requested.exists() else preferred_model_path(requested.name)]
        if not model_paths[0].exists():
            raise SystemExit(f"Model file not found: {model_paths[0]}")

    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        target_layer = find_layer_by_name(model, args.layer) if args.layer else None
        heat_batch, preds = compute_gradcam(model, x_batch, target_layer)
        overlay = overlay_heatmap(img_rgb, heat_batch[0], alpha=0.45)

        out_name = f"{chosen.stem}_{model_path.stem}_gradcam_tf.png"
        out_path = (
            Path(args.out)
            if (len(model_paths) == 1 and args.out is not None)
            else (chosen.parent / out_name)
        )
        cv2.imwrite(str(out_path), overlay)
        prob = float(preds[0]) if len(preds) else float("nan")
        print(f"{model_path.name}: prob={prob:.4f} saved={out_path}")


if __name__ == "__main__":
    main()
