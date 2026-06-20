"""Attach image file paths to the synthetic tabular CSV.

Reads synthetic_scd_heart_stroke.csv and assigns each row a random image
path from the sickle cell dataset, producing
synthetic_scd_heart_stroke_with_images.csv.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from project_paths import SYNTHETIC_DIR


def list_images(folder: Path) -> list[Path]:
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(SYNTHETIC_DIR / "synthetic_scd_heart_stroke.csv"))
    parser.add_argument("--out", default=str(SYNTHETIC_DIR / "synthetic_scd_heart_stroke_with_images.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Collect all images from dataset
    pos_dir = Path("sickle cell dataset") / "Positive" / "Labelled"
    neg_dir = Path("sickle cell dataset") / "Negative" / "Clear"
    pos_imgs = list_images(pos_dir) if pos_dir.exists() else []
    neg_imgs = list_images(neg_dir) if neg_dir.exists() else []
    all_imgs = pos_imgs + neg_imgs

    if not all_imgs:
        alt_pos = Path("dataset") / "sickle"
        alt_neg = Path("dataset") / "normal"
        all_imgs = list_images(alt_pos) + list_images(alt_neg) if alt_pos.exists() else []

    if not all_imgs:
        raise SystemExit("No images found to attach.")

    random.seed(args.seed)
    df["image_path"] = [str(random.choice(all_imgs)) for _ in range(len(df))]

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows with image paths -> {out_path}")


if __name__ == "__main__":
    main()
