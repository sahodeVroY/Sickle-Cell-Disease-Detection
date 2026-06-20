"""SHAP explainability for sickle-cell image classifiers.

Produces the official SHAP image plot (pixel-level heatmap).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import shap

from project_paths import preferred_model_path
from transfer_learning import AppPreprocess


def list_images(folder: Path) -> list[Path]:
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid]


def load_and_preprocess(image_path: Path, target_size=(64, 64), normalize=True):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32)
    if normalize:
        x = x / 255.0
    return img_rgb, x


def _get_background(image_paths, n=100, target_size=(64, 64), normalize=True):
    chosen = image_paths[:n]
    bg = np.zeros((len(chosen), target_size[0], target_size[1], 3), dtype=np.float32)
    for i, p in enumerate(chosen):
        _, x = load_and_preprocess(p, target_size=target_size, normalize=normalize)
        bg[i] = x
    return bg


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="SHAP explanations for sickle-cell models")
    parser.add_argument("--model", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--bg-size", type=int, default=100, help="Background dataset size")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    pos_dir = Path("sickle cell dataset") / "Positive" / "Labelled"
    neg_dir = Path("sickle cell dataset") / "Negative" / "Clear"
    all_imgs: list[Path] = []
    if pos_dir.exists():
        all_imgs.extend(list_images(pos_dir))
    if neg_dir.exists():
        all_imgs.extend(list_images(neg_dir))
    if not all_imgs:
        raise SystemExit("No images found.")

    chosen = Path(args.image) if args.image else all_imgs[0]

    candidate_models = [preferred_model_path("sickle_cnn.keras")]
    model_paths = [Path(args.model)] if args.model else [p for p in candidate_models if p.exists()]
    if args.model and not model_paths[0].exists():
        model_paths = [preferred_model_path(model_paths[0].name)]
    if not model_paths:
        raise SystemExit("No model files found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle images to get a diverse background (mix of Pos/Neg)
    import random
    random.seed(42)
    random.shuffle(all_imgs)

    for model_path in model_paths:
        print(f"\n=== SHAP for {model_path.name} ===")
        model = tf.keras.models.load_model(model_path, custom_objects={"AppPreprocess": AppPreprocess})
        is_custom_cnn = model_path.name.lower() == "sickle_cnn.keras"
        stem = model_path.stem

        last_layer = model.layers[-1]
        headless_model = tf.keras.Model(inputs=model.inputs, outputs=last_layer.input)
        
        logit_model = tf.keras.Sequential([
            headless_model,
            tf.keras.layers.Dense(units=last_layer.units, activation=None)
        ])
        logit_model.layers[1].set_weights(last_layer.get_weights())

        bg = _get_background(all_imgs, n=args.bg_size, normalize=is_custom_cnn)
        explainer = shap.GradientExplainer(logit_model, bg)

        _, x_single = load_and_preprocess(chosen, normalize=is_custom_cnn)
        x_single_batch = np.expand_dims(x_single, 0)
        
        print(f"Computing SHAP values for {chosen.name}...")
        sv_single = explainer.shap_values(x_single_batch)
        
        if isinstance(sv_single, list):
            sv_single = sv_single[0]
            
        print(f"  SHAP range (raw): min={sv_single.min():.2e}, max={sv_single.max():.2e}")

        # ---- Clean Heatmap Visualization ----
        # 1. Sum across RGB channels to get a single saliency value per pixel
        sv_img = sv_single[0]  # (H, W, 3)
        saliency = sv_img.sum(axis=-1)  # (H, W)

        # 2. Gaussian smoothing to reduce per-pixel noise
        from scipy.ndimage import gaussian_filter
        saliency = gaussian_filter(saliency, sigma=1.5)

        # 3. Percentile-based normalization for strong contrast
        vmax = np.percentile(np.abs(saliency), 99)
        if vmax > 0:
            saliency = saliency / vmax
        saliency = np.clip(saliency, -1, 1)

        print(f"  Saliency range (after smoothing): min={saliency.min():.3f}, max={saliency.max():.3f}")

        # 4. Create a clean 3-panel figure: Original | Heatmap | Overlay
        img_display = x_single if x_single.max() <= 1.0 else x_single / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original image
        axes[0].imshow(img_display)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Panel 2: SHAP heatmap (red = positive attribution, blue = negative)
        im = axes[1].imshow(saliency, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_title("SHAP Attribution Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="SHAP value")

        # Panel 3: Overlay of heatmap on original image
        axes[2].imshow(img_display)
        axes[2].imshow(saliency, cmap="RdBu_r", alpha=0.5, vmin=-1, vmax=1)
        axes[2].set_title("SHAP Overlay")
        axes[2].axis("off")

        plt.suptitle(f"SHAP Explanation — {stem}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = out_dir / f"{stem}_shap_image_plot.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved SHAP heatmap: {out_path}")

        # ---- Superpixel-Level Aggregation for Waterfall & Bar Plots ----
        from skimage.segmentation import slic, mark_boundaries

        # Segment the image into superpixels
        n_segments = 20
        img_for_slic = (x_single * 255).astype(np.uint8) if is_custom_cnn else x_single.astype(np.uint8)
        segments = slic(img_for_slic, n_segments=n_segments, compactness=10, start_label=0)
        n_sp = segments.max() + 1

        # Aggregate SHAP values per superpixel
        sp_shap = np.zeros(n_sp, dtype=np.float32)
        for sp_id in range(n_sp):
            mask = segments == sp_id
            sp_shap[sp_id] = saliency[mask].sum()

        sp_names = [f"Region-{i}" for i in range(n_sp)]

        # Get the base value (model prediction on background mean)
        bg_mean = bg.mean(axis=0, keepdims=True)
        base_val = float(logit_model.predict(bg_mean, verbose=0).reshape(-1)[0])
        pred_val = float(logit_model.predict(x_single_batch, verbose=0).reshape(-1)[0])

        print(f"  Base logit: {base_val:.4f}, Predicted logit: {pred_val:.4f}")

        # ---- 5. Waterfall Plot ----
        try:
            explanation_obj = shap.Explanation(
                values=sp_shap,
                base_values=base_val,
                feature_names=sp_names,
            )
            fig_wf = plt.figure(figsize=(10, 8))
            shap.plots.waterfall(explanation_obj, show=False)
            plt.title(f"SHAP Waterfall — {stem}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            wf_path = out_dir / f"{stem}_shap_waterfall.png"
            plt.savefig(wf_path, dpi=150, bbox_inches="tight")
            plt.close(fig_wf)
            print(f"  ✓ Saved waterfall: {wf_path}")
        except Exception as e:
            print(f"  ✗ Waterfall failed: {e}")

        # ---- 6. Bar Plot (Feature Importance) ----
        try:
            # Sort by absolute value
            order = np.argsort(np.abs(sp_shap))[::-1]
            top_n = min(15, len(order))
            top_idx = order[:top_n]

            fig_bar, ax = plt.subplots(figsize=(10, 7))
            colors = ["#d73027" if v > 0 else "#4575b4" for v in sp_shap[top_idx]]
            y_pos = np.arange(top_n)
            ax.barh(y_pos, sp_shap[top_idx][::-1], color=colors[::-1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels([sp_names[i] for i in top_idx][::-1], fontsize=10)
            ax.set_xlabel("SHAP Value (impact on logit)", fontsize=12)
            ax.set_title(f"SHAP Feature Importance — {stem}", fontsize=13, fontweight="bold")
            ax.axvline(x=0, color="black", linewidth=0.8)

            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#d73027", label="Pushes → Positive (sickle)"),
                Patch(facecolor="#4575b4", label="Pushes → Negative (normal)")
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
            plt.tight_layout()

            bar_path = out_dir / f"{stem}_shap_bar.png"
            plt.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig_bar)
            print(f"  ✓ Saved bar plot: {bar_path}")
        except Exception as e:
            print(f"  ✗ Bar plot failed: {e}")

        # ---- 7. Superpixel Overlay (shows which region is which) ----
        try:
            marked = mark_boundaries(img_for_slic, segments, color=(1, 1, 0))
            fig_sp, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(marked)
            # Label each superpixel with its ID
            for sp_id in range(n_sp):
                ys, xs = np.where(segments == sp_id)
                cy, cx = ys.mean(), xs.mean()
                ax.text(cx, cy, str(sp_id), fontsize=7, color="yellow",
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
            ax.set_title(f"Superpixel Regions — {stem}", fontsize=13, fontweight="bold")
            ax.axis("off")
            plt.tight_layout()

            sp_path = out_dir / f"{stem}_superpixel_map.png"
            plt.savefig(sp_path, dpi=150, bbox_inches="tight")
            plt.close(fig_sp)
            print(f"  ✓ Saved superpixel map: {sp_path}")
        except Exception as e:
            print(f"  ✗ Superpixel map failed: {e}")

        print(f"\n  All SHAP outputs saved for {stem}!")


if __name__ == "__main__":
    main()
