"""Option A — Predict sickle cell from an image and attach clinical notes.

Loads one or more .keras models, predicts sickle probability for a given
image, and prints clinical notes along with the prediction.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from project_paths import existing_model_paths
from project_paths import preferred_model_path
from transfer_learning import AppPreprocess


def list_images(folder: Path) -> list[Path]:
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid]


def load_image(image_path: Path, target_size=(64, 64)):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    return img_rgb, resized.astype(np.float32)


def _clinical_notes(pred_class: int, prob_sickle: float) -> list[str]:
    if pred_class != 1:
        return [
            "Model predicts non-sickle morphology on this image. This is not a diagnosis.",
        ]
    return [
        f"Model predicts sickle morphology (prob={prob_sickle:.3f}). This is not a diagnosis.",
        "Sickle cell disease can be associated with cardiovascular/cerebrovascular complications "
        "such as stroke and pulmonary hypertension.",
        "In clinical practice, confirm with hemoglobin electrophoresis/HPLC and follow local "
        "guidelines for stroke screening and cardiopulmonary assessment.",
    ]


def main():
    parser = argparse.ArgumentParser(description="Option A: predict + clinical notes")
    parser.add_argument("--model", default=None, help=".keras model path (or 'all')")
    parser.add_argument("--image", default=None, help="Input image path")
    args = parser.parse_args()

    # Find image
    pos_dir = Path("sickle cell dataset") / "Positive" / "Labelled"
    imgs = list_images(pos_dir) if pos_dir.exists() else []

    if args.image:
        chosen = Path(args.image)
        if not chosen.exists():
            raise SystemExit(f"Image not found: {chosen}")
    elif imgs:
        chosen = imgs[0]
    else:
        raise SystemExit("No images found. Provide --image.")

    candidate_models = [
        "efficientnetb0_transfer.keras",
        "resnet50_transfer.keras",
        "mobilenetv2_transfer.keras",
        "sickle_cnn.keras",
    ]

    if args.model is None or str(args.model).lower() == "all":
        model_paths = existing_model_paths(candidate_models)
    else:
        requested = Path(args.model)
        model_paths = [requested if requested.exists() else preferred_model_path(requested.name)]

    if not model_paths:
        raise SystemExit("No model files found.")

    _, x = load_image(chosen, target_size=(64, 64))

    for model_path in model_paths:
        model = tf.keras.models.load_model(
            model_path, custom_objects={"AppPreprocess": AppPreprocess}
        )
        is_custom_cnn = model_path.name.lower() == "sickle_cnn.keras"
        x_input = np.expand_dims(x / 255.0 if is_custom_cnn else x, 0).astype(np.float32)

        prob = float(model.predict(x_input, verbose=0).reshape(-1)[0])
        pred_class = 1 if prob >= 0.5 else 0
        label = "sickle" if pred_class == 1 else "normal"

        print(f"\n{'=' * 50}")
        print(f"Model:      {model_path.name}")
        print(f"Image:      {chosen}")
        print(f"Prediction: {label} (prob={prob:.4f})")
        print("Clinical Notes:")
        for note in _clinical_notes(pred_class, prob):
            print(f"  • {note}")


if __name__ == "__main__":
    main()
