from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --------------------
# Config
# --------------------
DATA_DIR = Path("dataset")  # dataset/sickle , dataset/normal
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42


def create_train_val_generators(
    data_dir: Path | None = None,
    img_size: tuple[int, int] | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    test_size: float = 0.20,
):
    """Build Keras ImageDataGenerators with augmentation (train) and plain rescaling (val).

    Parameters
    ----------
    data_dir : Path, optional
        Root folder containing ``normal/`` and ``sickle/`` sub-directories.
    img_size : tuple[int, int], optional
        ``(height, width)`` to resize every image to.
    batch_size : int, optional
        Mini-batch size for both generators.
    seed : int, optional
        Random seed for reproducibility.
    test_size : float, optional
        Fraction of the data to hold out for validation.

    Returns
    -------
    train_gen, val_gen : DirectoryIterator
        Keras generators ready for ``model.fit()``.
    """
    data_dir = data_dir or DATA_DIR
    img_size = img_size or IMG_SIZE
    batch_size = batch_size or BATCH_SIZE
    seed = seed or SEED

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    filepaths: list[str] = []
    labels: list[str] = []

    for cls in ["normal", "sickle"]:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            # Fall back to the alternative layout
            alt = Path("sickle cell dataset")
            if cls == "normal":
                cls_dir = alt / "Negative" / "Clear"
            else:
                cls_dir = alt / "Positive" / "Labelled"
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_ext:
                filepaths.append(str(p))
                labels.append(cls)

    df = pd.DataFrame({"filename": filepaths, "label": labels})
    if len(df) == 0:
        raise RuntimeError(f"No images found under {data_dir.resolve()}")

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="label",
        target_size=img_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="label",
        target_size=img_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
    )

    return train_gen, val_gen


if __name__ == "__main__":
    train_gen, val_gen = create_train_val_generators()
    print("Train samples:", train_gen.samples, " Val samples:", val_gen.samples)
    print("Class indices:", train_gen.class_indices)
