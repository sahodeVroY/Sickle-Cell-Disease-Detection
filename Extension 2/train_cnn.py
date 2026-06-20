from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_generators import create_train_val_generators
from project_paths import preferred_model_path


EPOCHS = 30
MODEL_PATH = "sickle_cnn.keras"


def build_model(input_shape=(64, 64, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def plot_history(history, output_path="training_curves.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(model, val_gen, output_path="confusion_matrix.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    val_gen.reset()
    y_true = val_gen.classes
    y_prob = model.predict(val_gen, steps=len(val_gen), verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype("int32")

    cm = confusion_matrix(y_true, y_pred)
    class_labels = [None] * len(val_gen.class_indices)
    for name, idx in val_gen.class_indices.items():
        class_labels[idx] = name

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45, ha="right")
    plt.yticks(tick_marks, class_labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return cm


def _iter_layers(layers_list):
    for layer in layers_list:
        yield layer
        if isinstance(layer, tf.keras.Model):
            for sub in _iter_layers(layer.layers):
                yield sub


def _auto_last_conv(model: tf.keras.Model):
    last = None
    for layer in _iter_layers(model.layers):
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
        raise RuntimeError("No convolutional layer found for Grad-CAM.")
    return last


def _load_image_for_cnn(image_path: Path, target_size=(64, 64)):
    import cv2
    import numpy as np

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    return img_rgb, x


def _overlay_heatmap(img_rgb, heatmap, alpha=0.45):
    import cv2
    import numpy as np

    h, w = img_rgb.shape[:2]
    hm = cv2.resize((heatmap * 255).astype(np.uint8), (w, h))
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base_bgr, 1 - alpha, hm_color, alpha, 0)


def _enhance_heatmap(heatmap, low_pct=5.0, high_pct=99.0, gamma=0.7):
    import numpy as np

    h = heatmap.astype(np.float32)
    lo = float(np.percentile(h, low_pct))
    hi = float(np.percentile(h, high_pct))
    h = (h - lo) / (hi - lo + 1e-8)
    h = np.clip(h, 0.0, 1.0)
    if gamma is not None:
        h = np.power(h, float(gamma))
    return h


def compute_gradcam_cnn(
    model: tf.keras.Model,
    img_batch,
    layer_name: str | None = None,
    class_idx: int | None = None,
    use_logits: bool = True,
):
    import numpy as np

    if isinstance(img_batch, np.ndarray):
        img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    inp = tf.keras.Input(shape=tuple(img_batch.shape[1:]), name="gradcam_input")
    cloned = tf.keras.models.clone_model(model, input_tensors=inp)
    cloned.set_weights(model.get_weights())

    if layer_name:
        target_layer = cloned.get_layer(layer_name)
    else:
        target_layer = _auto_last_conv(cloned)

    grad_model = tf.keras.Model(cloned.inputs, [target_layer.output, cloned.outputs[0]])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        if preds.shape[-1] == 1:
            prob = preds[:, 0]
            pred_class = tf.cast(prob >= 0.5, tf.int32)
            target_class = pred_class if class_idx is None else tf.cast(class_idx, tf.int32)
            if use_logits:
                eps = tf.constant(1e-7, dtype=prob.dtype)
                logit = tf.math.log((prob + eps) / (1.0 - prob + eps))
                loss = tf.where(target_class == 1, logit, -logit)
            else:
                loss = tf.where(target_class == 1, prob, 1.0 - prob)
        else:
            idx = tf.argmax(preds, axis=-1)
            loss = tf.gather_nd(preds, tf.stack([tf.range(tf.shape(preds)[0]), idx], axis=1))

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError("Gradients are None for chosen layer. Try --gradcam-layer with a different conv layer.")
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1, keepdims=True))
    cam = tf.image.resize(cam, (tf.shape(img_batch)[1], tf.shape(img_batch)[2]))
    cam = cam.numpy()
    cam = cam - cam.min(axis=(1, 2, 3), keepdims=True)
    cam = cam / (cam.max(axis=(1, 2, 3), keepdims=True) + 1e-8)
    return cam[..., 0], preds.numpy().reshape(-1)


def _option1_list_images(folder: Path) -> list[Path]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext]


def _option1_find_class_dirs() -> dict[str, Path]:
    dataset_dir = Path("dataset")
    if (dataset_dir / "normal").exists() and (dataset_dir / "sickle").exists():
        return {"normal": dataset_dir / "normal", "sickle": dataset_dir / "sickle"}

    alt_dir = Path("sickle cell dataset")
    normal_dir = alt_dir / "Negative" / "Clear"
    sickle_dir = alt_dir / "Positive" / "Labelled"
    if normal_dir.exists() and sickle_dir.exists():
        return {"normal": normal_dir, "sickle": sickle_dir}

    raise SystemExit("Could not find dataset folders (dataset/normal+sickle or sickle cell dataset/...).")


def _option1_load_paths_and_labels():
    import numpy as np

    class_dirs = _option1_find_class_dirs()
    normal = _option1_list_images(class_dirs["normal"])
    sickle = _option1_list_images(class_dirs["sickle"])
    x = np.array([str(p) for p in (normal + sickle)])
    y = np.array([0] * len(normal) + [1] * len(sickle), dtype=np.int32)
    if len(x) == 0:
        raise SystemExit("No images found.")
    return x, y


def _option1_make_augmenter() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
        ]
    )


def _option1_make_dataset(
    paths,
    labels,
    batch_size: int,
    shuffle: bool,
    seed: int,
    augment: tf.keras.Model | None,
):
    import numpy as np

    paths = np.asarray(paths)
    labels = np.asarray(labels).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (64, 64), method="bilinear")
        img = tf.cast(img, tf.float32)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment is not None:
        def _aug(img, label):
            return augment(img, training=True), label

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _option1_build_custom_cnn() -> tf.keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


def _option1_compile_binary(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def _option1_histories_to_frame(histories, fold: int):
    import numpy as np
    import pandas as pd

    rows = []
    epoch_offset = 0
    for h in histories:
        acc = h.history.get("accuracy", [])
        val_acc = h.history.get("val_accuracy", [])
        loss = h.history.get("loss", [])
        val_loss = h.history.get("val_loss", [])
        n = max(len(loss), len(val_loss), len(acc), len(val_acc))
        for i in range(n):
            rows.append(
                {
                    "fold": int(fold),
                    "epoch": int(epoch_offset + i + 1),
                    "loss": float(loss[i]) if i < len(loss) else float("nan"),
                    "val_loss": float(val_loss[i]) if i < len(val_loss) else float("nan"),
                    "accuracy": float(acc[i]) if i < len(acc) else float("nan"),
                    "val_accuracy": float(val_acc[i]) if i < len(val_acc) else float("nan"),
                }
            )
        epoch_offset += n

    return pd.DataFrame(rows)


def _option1_mean_curve(history_df, metric: str):
    import pandas as pd

    g = history_df.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
    g.columns = ["epoch", "mean", "std"]
    return g


def _option1_save_curves(history_df, out_path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = _option1_mean_curve(history_df, "accuracy")
    val_acc = _option1_mean_curve(history_df, "val_accuracy")
    loss = _option1_mean_curve(history_df, "loss")
    val_loss = _option1_mean_curve(history_df, "val_loss")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc["epoch"], acc["mean"], label="Train Acc")
    plt.fill_between(acc["epoch"], acc["mean"] - acc["std"], acc["mean"] + acc["std"], alpha=0.2)
    plt.plot(val_acc["epoch"], val_acc["mean"], label="Val Acc")
    plt.fill_between(
        val_acc["epoch"],
        val_acc["mean"] - val_acc["std"],
        val_acc["mean"] + val_acc["std"],
        alpha=0.2,
    )
    plt.title(f"{title} - Accuracy (mean±std over folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss["epoch"], loss["mean"], label="Train Loss")
    plt.fill_between(loss["epoch"], loss["mean"] - loss["std"], loss["mean"] + loss["std"], alpha=0.2)
    plt.plot(val_loss["epoch"], val_loss["mean"], label="Val Loss")
    plt.fill_between(
        val_loss["epoch"],
        val_loss["mean"] - val_loss["std"],
        val_loss["mean"] + val_loss["std"],
        alpha=0.2,
    )
    plt.title(f"{title} - Loss (mean±std over folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _option1_save_confusion(cm, out_path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.asarray(cm)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    labels = ["normal", "sickle"]
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _option1_summarize_roc(roc_per_fold):
    import numpy as np

    grid = np.linspace(0.0, 1.0, 101)
    tprs = []
    aucs = []
    for fpr, tpr, auc in roc_per_fold:
        tpr_i = np.interp(grid, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
        aucs.append(float(auc))
    tprs = np.stack(tprs, axis=0)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    return {
        "fpr": grid,
        "tpr_mean": mean_tpr,
        "tpr_std": tprs.std(axis=0),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def _option1_save_roc_comparison(items, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for name, s in items:
        fpr = s["fpr"]
        tpr = s["tpr_mean"]
        auc_m = s["auc_mean"]
        auc_s = s["auc_std"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_m:.3f}±{auc_s:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison (5-Fold CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _option1_train_and_eval_one_fold(
    model_key: str,
    use_augmentation: bool,
    train_paths,
    train_y,
    val_paths,
    val_y,
    batch_size: int,
    seed: int,
    cnn_epochs: int,
    feature_epochs: int,
    finetune_epochs: int,
):
    import time

    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    import transfer_learning as tl

    tf.keras.backend.clear_session()
    augmenter = _option1_make_augmenter() if use_augmentation else None
    train_ds = _option1_make_dataset(train_paths, train_y, batch_size=batch_size, shuffle=True, seed=seed, augment=augmenter)
    val_ds = _option1_make_dataset(val_paths, val_y, batch_size=batch_size, shuffle=False, seed=seed, augment=None)

    if model_key == "custom_cnn":
        model = _option1_build_custom_cnn()
        _option1_compile_binary(model, lr=1e-3)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        ]
        t0 = time.perf_counter()
        h = model.fit(train_ds, validation_data=val_ds, epochs=cnn_epochs, callbacks=callbacks, verbose=0)
        train_time_s = time.perf_counter() - t0
        histories = [h]
    else:
        if model_key == "efficientnetb0":
            model, base = tl.build_efficientnet_model()
        elif model_key == "resnet50":
            model, base = tl.build_resnet_model()
        elif model_key == "mobilenetv2":
            model, base = tl.build_mobilenet_model()
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        t0 = time.perf_counter()
        h1, h2 = tl.train_transfer_model(
            model,
            base_model=base,
            train_ds=train_ds,
            val_ds=val_ds,
            feature_epochs=feature_epochs,
            finetune_epochs=finetune_epochs,
        )
        train_time_s = time.perf_counter() - t0
        histories = [h1, h2]

    val_count = int(len(val_paths))
    _ = model.predict(val_ds.take(1), verbose=0)
    t1 = time.perf_counter()
    prob = model.predict(val_ds, verbose=0).reshape(-1)
    infer_time_s = time.perf_counter() - t1
    infer_time_ms_per_image = (infer_time_s / max(1, val_count)) * 1000.0

    y_true = np.asarray(val_y).astype(int)
    y_pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    history_df = _option1_histories_to_frame(histories, fold=seed)
    epochs_trained = int(history_df["epoch"].max()) if not history_df.empty else 0
    train_time_s_per_epoch = float(train_time_s / max(1, epochs_trained))

    return {
        "params": int(model.count_params()),
        "train_time_s": float(train_time_s),
        "train_time_s_per_epoch": float(train_time_s_per_epoch),
        "epochs_trained": int(epochs_trained),
        "infer_time_ms_per_image": float(infer_time_ms_per_image),
        "history_df": history_df,
        "cm": cm,
        "roc": (fpr, tpr, auc),
        "auc": float(auc),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "model": model,
    }


def _option1_main(argv):
    import argparse

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", default="custom_cnn,efficientnetb0")
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--feature-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--out-dir", default="option1_outputs")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y = _option1_load_paths_and_labels()

    model_keys = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    experiments = []
    for mk in model_keys:
        experiments.append((mk, False))
        experiments.append((mk, True))

    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))

    summary_rows = []
    fold_rows = []
    roc_summaries = []
    metrics_rows = []

    for model_key, use_aug in experiments:
        exp_name = f"{model_key}_{'aug' if use_aug else 'noaug'}"
        exp_dir = out_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Option1] Experiment start: {exp_name} (folds={args.folds})")

        cms = []
        history_frames = []
        rocs = []
        train_times = []
        train_time_per_epoch = []
        epochs_trained = []
        infer_times = []
        params = None

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
            print(f"[Option1]  Fold {fold_idx}/{args.folds} start: {exp_name}")
            train_paths, train_y = x[train_idx], y[train_idx]
            val_paths, val_y = x[val_idx], y[val_idx]

            r = _option1_train_and_eval_one_fold(
                model_key=model_key,
                use_augmentation=use_aug,
                train_paths=train_paths,
                train_y=train_y,
                val_paths=val_paths,
                val_y=val_y,
                batch_size=int(args.batch_size),
                seed=int(args.seed) + fold_idx,
                cnn_epochs=int(args.cnn_epochs),
                feature_epochs=int(args.feature_epochs),
                finetune_epochs=int(args.finetune_epochs),
            )
            print(
                "[Option1]  Fold "
                f"{fold_idx}/{args.folds} done: {exp_name} "
                f"AUC={float(r['auc']):.4f} "
                f"train_s={float(r['train_time_s']):.1f} "
                f"infer_ms/img={float(r['infer_time_ms_per_image']):.3f}"
            )
            
            if fold_idx == 1:
                if "custom_cnn" in exp_name:
                    filename = "sickle_cnn.keras"
                elif "efficientnetb0" in exp_name:
                    filename = "efficientnetb0_transfer.keras"
                elif "resnet50" in exp_name:
                    filename = "resnet50_transfer.keras"
                elif "mobilenetv2" in exp_name:
                    filename = "mobilenetv2_transfer.keras"
                else:
                    filename = f"{model_key}_transfer.keras"
                
                # Save the 'aug' version if available
                if use_aug or (model_key, True) not in experiments:
                    save_path = preferred_model_path(filename)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    r["model"].save(save_path)
                    print(f"[Option1] Saved model to {save_path}")

            params = r["params"] if params is None else params
            train_times.append(r["train_time_s"])
            train_time_per_epoch.append(r["train_time_s_per_epoch"])
            epochs_trained.append(r["epochs_trained"])
            infer_times.append(r["infer_time_ms_per_image"])
            history_frames.append(r["history_df"].assign(experiment=exp_name, fold=fold_idx))
            cms.append(r["cm"])
            rocs.append(r["roc"])
            metrics_rows.append(
                {
                    "experiment": exp_name,
                    "fold": fold_idx,
                    "accuracy": float(r["accuracy"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                }
            )

            fold_rows.append(
                {
                    "experiment": exp_name,
                    "fold": fold_idx,
                    "auc": float(r["auc"]),
                    "train_time_s": float(r["train_time_s"]),
                    "train_time_s_per_epoch": float(r["train_time_s_per_epoch"]),
                    "epochs_trained": int(r["epochs_trained"]),
                    "infer_time_ms_per_image": float(r["infer_time_ms_per_image"]),
                    "accuracy": float(r["accuracy"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                }
            )

        cm_sum = np.sum(np.stack(cms, axis=0), axis=0)
        _option1_save_confusion(cm_sum, exp_dir / "confusion_matrix.png", title=f"{exp_name} - Confusion (sum over folds)")

        hist_df = pd.concat(history_frames, ignore_index=True)
        hist_df.to_csv(exp_dir / "history_folds.csv", index=False)
        _option1_save_curves(hist_df, exp_dir / "training_validation_curves.png", title=exp_name)

        roc_summary = _option1_summarize_roc(rocs)
        roc_summaries.append((exp_name, roc_summary))
        print(
            f"[Option1] Experiment done: {exp_name} "
            f"AUC={float(roc_summary['auc_mean']):.4f}±{float(roc_summary['auc_std']):.4f}"
        )

        exp_metrics = [m for m in metrics_rows if m["experiment"] == exp_name]
        if exp_metrics:
            import numpy as _np
            accs = _np.array([m["accuracy"] for m in exp_metrics], dtype=float)
            precs = _np.array([m["precision"] for m in exp_metrics], dtype=float)
            recs = _np.array([m["recall"] for m in exp_metrics], dtype=float)
            f1s = _np.array([m["f1"] for m in exp_metrics], dtype=float)
            acc_mean, acc_std = float(accs.mean()), float(accs.std())
            prec_mean, prec_std = float(precs.mean()), float(precs.std())
            rec_mean, rec_std = float(recs.mean()), float(recs.std())
            f1_mean, f1_std = float(f1s.mean()), float(f1s.std())
        else:
            acc_mean = acc_std = prec_mean = prec_std = rec_mean = rec_std = f1_mean = f1_std = float("nan")

        summary_rows.append(
            {
                "experiment": exp_name,
                "model_key": model_key,
                "augmentation": bool(use_aug),
                "params": int(params or 0),
                "auc_mean": float(roc_summary["auc_mean"]),
                "auc_std": float(roc_summary["auc_std"]),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "precision_mean": prec_mean,
                "precision_std": prec_std,
                "recall_mean": rec_mean,
                "recall_std": rec_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "train_time_s_mean": float(np.mean(train_times)),
                "train_time_s_std": float(np.std(train_times)),
                "train_time_s_per_epoch_mean": float(np.mean(train_time_per_epoch)),
                "train_time_s_per_epoch_std": float(np.std(train_time_per_epoch)),
                "epochs_trained_mean": float(np.mean(epochs_trained)),
                "epochs_trained_std": float(np.std(epochs_trained)),
                "infer_ms_per_image_mean": float(np.mean(infer_times)),
                "infer_ms_per_image_std": float(np.std(infer_times)),
            }
        )

    _option1_save_roc_comparison(roc_summaries, out_dir / "roc_comparison.png")

    summary_df = pd.DataFrame(summary_rows).sort_values(["model_key", "augmentation"])
    summary_df.to_csv(out_dir / "cv_summary.csv", index=False)

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_dir / "cv_folds.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "cv_classification_metrics_folds.csv", index=False)

    complexity_df = summary_df[
        [
            "experiment",
            "params",
            "train_time_s_mean",
            "train_time_s_std",
            "train_time_s_per_epoch_mean",
            "train_time_s_per_epoch_std",
            "epochs_trained_mean",
            "epochs_trained_std",
            "infer_ms_per_image_mean",
            "infer_ms_per_image_std",
        ]
    ].copy()
    complexity_df.to_csv(out_dir / "cv_complexity.csv", index=False)

    print("\n=== CV Summary ===")
    print(
        summary_df[
            [
                "experiment",
                "params",
                "auc_mean",
                "auc_std",
                "accuracy_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "train_time_s_mean",
                "infer_ms_per_image_mean",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    import sys

    _option1_main(sys.argv[1:])
