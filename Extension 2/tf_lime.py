"""LIME explainability for sickle-cell image classifiers.

Produces:
  1. <stem>_lime.png: original image, positive regions, negative regions,
     and a heatmap overlay.
  2. <stem>_lime_bar.png: bar chart of the most influential regions.

Usage:
    python tf_lime.py --model sickle_cnn.keras --image path/to/img.jpg
    python tf_lime.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from lime import lime_image
from project_paths import existing_model_paths
from project_paths import preferred_model_path
from scipy.ndimage import gaussian_filter, zoom

from transfer_learning import AppPreprocess

CLASS_NAMES = {0: "Normal", 1: "Sickle-cell"}


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


def _upscale(img, scale=4):
    """Upscale a small image so it looks clean in figures."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def _upscale_segments(segments: np.ndarray, scale=4) -> np.ndarray:
    """Upscale integer segment ids without changing their values."""
    return np.repeat(np.repeat(segments, scale, axis=0), scale, axis=1)


def _segment_center(mask: np.ndarray) -> tuple[int, int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    center_y, center_x = points.mean(axis=0)
    return int(center_x), int(center_y)


def _annotate_regions(ax, segs, regions, max_regions, text_color):
    for rank, (sp_id, weight) in enumerate(regions[:max_regions], start=1):
        center = _segment_center(segs == sp_id)
        if center is None:
            continue
        x, y = center
        ax.text(
            x,
            y,
            f"{rank}: {weight:+.2e}",
            color=text_color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )


def _blend_regions(base_img, segs, regions, color, max_weight):
    blended = base_img.copy().astype(np.float32)
    for sp_id, weight in regions:
        mask = segs == sp_id
        strength = 0.25 + 0.35 * (abs(weight) / (max_weight + 1e-8))
        blended[mask] = blended[mask] * (1 - strength) + color * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def _make_predict_fn(model, is_custom_cnn: bool):
    """Return a function mapping (N, H, W, 3) float images -> (N, 2) probs."""

    def predict_fn(images: np.ndarray) -> np.ndarray:
        x = images.astype(np.float32)
        if is_custom_cnn:
            x = x / 255.0
        prob = model.predict(x, verbose=0).reshape(-1)
        return np.column_stack([1.0 - prob, prob])

    return predict_fn


def _save_main_figure(img, explanation, pred_label, pred_prob, stem, num_features, out_dir):
    """Save the main four-panel LIME explanation figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
    segments = explanation.segments
    local_exp = dict(explanation.local_exp[pred_label])

    weight_map = np.zeros(segments.shape, dtype=np.float64)
    for sp_id, weight in local_exp.items():
        weight_map[segments == sp_id] = weight

    sorted_sps = sorted(local_exp.items(), key=lambda item: abs(item[1]), reverse=True)
    top_sps = sorted_sps[:num_features]
    pos_sps = [(sp_id, weight) for sp_id, weight in top_sps if weight > 0]
    neg_sps = [(sp_id, weight) for sp_id, weight in top_sps if weight < 0]

    img_u8 = img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)
    big = _upscale(img_u8, scale=4)
    big_segs = _upscale_segments(segments, scale=4)

    green = np.array([0, 220, 0], dtype=np.float32)
    red = np.array([220, 30, 30], dtype=np.float32)
    max_w = max(abs(weight) for _, weight in top_sps) if top_sps else 1.0

    pro_img = _blend_regions(big, big_segs, pos_sps, green, max_w)
    con_img = _blend_regions(big, big_segs, neg_sps, red, max_w)

    smooth = gaussian_filter(weight_map, sigma=1.0)
    vmax = np.percentile(np.abs(smooth), 99)
    normed = np.clip(smooth / vmax, -1.0, 1.0) if vmax > 0 else smooth.copy()
    normed_big = zoom(normed, 4, order=1)

    total_pos = float(sum(weight for _, weight in local_exp.items() if weight > 0))
    total_neg = float(sum(weight for _, weight in local_exp.items() if weight < 0))

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(big)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(pro_img)
    axes[1].set_title(
        f'Supports "{pred_name}"\n({len(pos_sps)} regions highlighted)',
        fontsize=13,
        fontweight="bold",
        color="#1a7a1a",
    )
    _annotate_regions(axes[1], big_segs, pos_sps, max_regions=5, text_color="#124d12")
    axes[1].axis("off")

    axes[2].imshow(con_img)
    axes[2].set_title(
        f'Opposes "{pred_name}"\n({len(neg_sps)} regions highlighted)',
        fontsize=13,
        fontweight="bold",
        color="#c62828",
    )
    _annotate_regions(axes[2], big_segs, neg_sps, max_regions=5, text_color="#7f1d1d")
    axes[2].axis("off")

    axes[3].imshow(big)
    im = axes[3].imshow(normed_big, cmap="RdBu_r", alpha=0.55, vmin=-1, vmax=1)
    axes[3].set_title("Importance Heatmap", fontsize=14, fontweight="bold")
    axes[3].axis("off")
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("opposes  |  supports", fontsize=10)

    plt.suptitle(
        f"LIME Explanation - {stem}\nPredicted: {pred_name} (confidence: {pred_prob:.1%})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.02,
        f"Top-region contribution summary: total positive = {total_pos:+.3e}, total negative = {total_neg:+.3e}. "
        "Labels show rank and LIME weight for the strongest regions.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = out_dir / f"{stem}_lime.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved main figure: {out_path}")


def _save_bar_chart(explanation, pred_label, pred_prob, stem, num_features, out_dir):
    """Save a bar chart of the most important LIME regions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))

    local_exp = explanation.local_exp[pred_label]
    sp_ids = np.array([item[0] for item in local_exp])
    weights = np.array([item[1] for item in local_exp])

    order = np.argsort(np.abs(weights))[::-1]
    top_n = min(num_features, len(order))
    top = order[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_weights = weights[top][::-1]
    bar_labels = [f"Region {sp_ids[i]}" for i in top][::-1]
    bar_colors = ["#2e7d32" if weight > 0 else "#c62828" for weight in bar_weights]
    y_pos = np.arange(top_n)

    ax.barh(y_pos, bar_weights, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=11)
    ax.set_xlabel("LIME Weight", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)

    legend_elements = [
        Patch(facecolor="#2e7d32", label=f'Supports "{pred_name}"'),
        Patch(facecolor="#c62828", label=f'Opposes "{pred_name}"'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9)

    ax.set_title(
        f'LIME Feature Importance - {stem}\nWhich image regions influenced "{pred_name}"?',
        fontsize=13,
        fontweight="bold",
    )

    max_abs = max(np.max(np.abs(bar_weights)), 1e-8) if top_n else 1e-8
    for index, value in enumerate(bar_weights):
        offset = 0.02 * max_abs * (1 if value >= 0 else -1)
        ax.text(offset + value, index, f"{value:.2e}", va="center", fontsize=8, color="#333")

    plt.tight_layout()
    out_path = out_dir / f"{stem}_lime_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar chart: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="LIME explanations for sickle-cell models")
    parser.add_argument("--model", default=None, help="Path to .keras model (or 'all')")
    parser.add_argument("--image", default=None, help="Input image path")
    parser.add_argument("--num-samples", type=int, default=1000, help="LIME perturbation samples")
    parser.add_argument("--num-features", type=int, default=10, help="Top superpixels to show")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    pos_dir = Path("sickle cell dataset") / "Positive" / "Labelled"
    imgs = list_images(pos_dir) if pos_dir.exists() else []

    if args.image:
        chosen = Path(args.image)
        if not chosen.exists():
            raise SystemExit(f"Image not found: {chosen}")
    else:
        if not imgs:
            raise SystemExit("No images found. Provide --image.")
        chosen = imgs[0]

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, x = load_image(chosen, target_size=(64, 64))
    explainer = lime_image.LimeImageExplainer()

    for model_path in model_paths:
        print(f"\n{'=' * 60}")
        print(f"  LIME - {model_path.name}")
        print(f"  Image : {chosen.name}")
        print(f"{'=' * 60}")

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"AppPreprocess": AppPreprocess},
        )
        is_custom_cnn = model_path.name.lower() == "sickle_cnn.keras"
        stem = model_path.stem
        predict_fn = _make_predict_fn(model, is_custom_cnn)

        probs = predict_fn(np.expand_dims(x, 0))[0]
        pred_label = int(np.argmax(probs))
        pred_prob = float(probs[pred_label])
        print(f"  P(Normal)={probs[0]:.4f}   P(Sickle)={probs[1]:.4f}")
        print(f"  Predicted: {CLASS_NAMES[pred_label]} ({pred_prob:.1%})")

        explanation = explainer.explain_instance(
            x.astype(np.double),
            predict_fn,
            labels=(pred_label,),
            hide_color=0,
            num_samples=args.num_samples,
        )

        _save_main_figure(
            x, explanation, pred_label, pred_prob, stem, args.num_features, out_dir
        )
        _save_bar_chart(
            explanation, pred_label, pred_prob, stem, args.num_features, out_dir
        )

        print(f"\n  Done - {stem}!")


if __name__ == "__main__":
    main()
