from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


IMG_SIZE = (64, 64)
SEED = 42
TEST_SIZE = 0.2
PCA_COMPONENTS = 100


def _find_class_dirs():
    dataset_dir = Path("dataset")
    if (dataset_dir / "normal").exists() and (dataset_dir / "sickle").exists():
        return {"normal": dataset_dir / "normal", "sickle": dataset_dir / "sickle"}

    alt_dir = Path("sickle cell dataset")
    normal_dir = alt_dir / "Negative" / "Clear"
    sickle_dir = alt_dir / "Positive" / "Labelled"
    if normal_dir.exists() and sickle_dir.exists():
        return {"normal": normal_dir, "sickle": sickle_dir}

    raise RuntimeError(
        "Could not find dataset folders. Expected either:\n"
        "- dataset/normal and dataset/sickle\n"
        "or:\n"
        "- sickle cell dataset/Negative/Clear and sickle cell dataset/Positive/Labelled"
    )


def _list_images(folder: Path):
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext]


def _load_images(paths):
    x = np.zeros((len(paths), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = tf.keras.utils.load_img(str(p), target_size=IMG_SIZE)
        x[i] = tf.keras.utils.img_to_array(img)
    return x


def _score_vector(estimator, x):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(x)
    return None


def _plot_results(results: pd.DataFrame, output_path="ml_comparison_bar.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    model_names = results["Model"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()

    x = np.arange(len(model_names))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = results[metric].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, values)
        ax.set_title(metric)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    class_dirs = _find_class_dirs()
    normal_paths = _list_images(class_dirs["normal"])
    sickle_paths = _list_images(class_dirs["sickle"])

    all_paths = normal_paths + sickle_paths
    y = np.array([0] * len(normal_paths) + [1] * len(sickle_paths), dtype=np.int32)

    train_paths, test_paths, y_train, y_test = train_test_split(
        all_paths,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED,
        shuffle=True,
    )

    x_train_img = _load_images(train_paths)
    x_test_img = _load_images(test_paths)

    x_train = x_train_img.reshape(len(x_train_img), -1)
    x_test = x_test_img.reshape(len(x_test_img), -1)

    n_components = int(min(PCA_COMPONENTS, x_train.shape[0], x_train.shape[1]))

    models = {
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=SEED)),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=SEED)),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED)),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=SEED)),
                ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=SEED)),
                ("clf", DecisionTreeClassifier(random_state=SEED)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=SEED)),
                ("clf", GaussianNB()),
            ]
        ),
    }

    rows = []
    for model_name, pipe in models.items():
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        score_vec = _score_vector(pipe, x_test)
        roc_auc = roc_auc_score(y_test, score_vec) if score_vec is not None else np.nan

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ROC AUC": roc_auc,
            }
        )

    results = pd.DataFrame(rows).sort_values(by="ROC AUC", ascending=False, na_position="last")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    _plot_results(results, output_path="ml_comparison_bar.png")
    print("Saved bar chart to ml_comparison_bar.png")

    # --- SHAP Analysis for the Best Model ---
    best_model_name = results.iloc[0]["Model"]
    print(f"\nPerforming SHAP analysis for the best model: {best_model_name}")
    best_pipe = models[best_model_name]

    # For SHAP, we use a small subset of the test set as background and some samples to explain
    # KernelExplainer is slow and memory-intensive for high-dimensional data (pixels)
    # We'll use a very small subset and limit the number of samples
    bg_size = 20
    test_subset_size = 2
    x_train_bg = x_train[:bg_size]
    x_test_explain = x_test[:test_subset_size]

    # Explain the whole pipeline (raw pixels -> explain)
    # Using a wrapper for predict_proba since KernelExplainer needs a function
    def predict_fn(x):
        return best_pipe.predict_proba(x)

    # We limit nsamples to prevent memory issues
    explainer = shap.KernelExplainer(predict_fn, x_train_bg)
    shap_values = explainer.shap_values(x_test_explain, nsamples=100)

    # shap_values is a list of [neg_class_shap, pos_class_shap] for binary models
    if isinstance(shap_values, list):
        sv = shap_values[1] # Use positive class
    else:
        sv = shap_values

    # Reshape for image plotting if we have 64x64x3 images
    # sv shape is (test_subset_size, 12288) -> (test_subset_size, 64, 64, 3)
    sv_reshaped = sv.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 3)
    x_test_reshaped = x_test_explain.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 3)

    plt.figure(figsize=(10, 5))
    shap.image_plot(sv_reshaped, x_test_reshaped, show=False)
    plt.title(f"SHAP Explanations for {best_model_name}")
    plt.savefig(f"ml_shap_explanation_{best_model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved SHAP image plot for {best_model_name}")


if __name__ == "__main__":
    main()
