"""Option B — Multimodal risk prediction (image + tabular).

Combines the image-derived sickle probability with tabular clinical features
to predict stroke and heart failure risk using the synthetic dataset.
Produces predictions CSV, SHAP explanations, and performance metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_paths import preferred_model_path
from project_paths import preferred_synthetic_csv
from transfer_learning import AppPreprocess


FEATURE_SCHEMA = [
    {"name": "sex", "type": "category", "choices": ["F", "M"]},
    {"name": "age_years", "type": "number"},
    {"name": "hemoglobin_g_dl", "type": "number"},
    {"name": "wbc_10e9_l", "type": "number"},
    {"name": "platelets_10e9_l", "type": "number"},
    {"name": "spo2_percent", "type": "number"},
    {"name": "systolic_bp", "type": "number"},
    {"name": "diastolic_bp", "type": "number"},
    {"name": "smoker", "type": "boolean"},
    {"name": "diabetes", "type": "boolean"},
    {"name": "hypertension", "type": "boolean"},
    {"name": "pulmonary_hypertension", "type": "boolean"},
]


def _build_preprocess(categorical, numeric):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )


def _make_pipeline(name, preprocess, seed):
    key = name.strip().lower()
    if key == "rf":
        clf = RandomForestClassifier(n_estimators=800, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)
        return Pipeline([("preprocess", preprocess), ("model", clf)])
    if key == "hgb":
        clf = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=4, max_iter=350, random_state=seed)
        return Pipeline([("preprocess", preprocess), ("model", clf)])
    if key == "lr":
        clf = LogisticRegression(max_iter=8000, solver="liblinear", class_weight="balanced", random_state=seed)
        return Pipeline([("preprocess", preprocess), ("scaler", StandardScaler(with_mean=False)), ("model", clf)])
    raise ValueError(f"Unknown model name: {name}")


def _choose_threshold(y_val, prob_val):
    thresholds = np.unique(np.round(prob_val, 6))
    if len(thresholds) > 200:
        thresholds = np.unique(np.quantile(prob_val, np.linspace(0.01, 0.99, 99)))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (prob_val >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * pr * rc) / (pr + rc) if (pr + rc) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _load_image(image_path, target_size=(64, 64)):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def _predict_scd_prob(model, image_path, is_custom_cnn):
    img = _load_image(image_path)
    if img is None:
        return np.nan
    x = np.expand_dims(img, 0)
    if is_custom_cnn:
        x = x / 255.0
    prob = float(model.predict(x, verbose=0).reshape(-1)[0])
    return prob


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Option B: multimodal risk prediction")
    parser.add_argument("--csv", default=str(preferred_synthetic_csv("synthetic_scd_heart_stroke_with_images.csv")))
    parser.add_argument("--image-model", default=None, help="Path to .keras for image scoring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Find image classifier
    candidate_models = [
        preferred_model_path("efficientnetb0_transfer.keras"),
        preferred_model_path("sickle_cnn.keras"),
    ]
    if args.image_model:
        img_model_path = Path(args.image_model)
        if not img_model_path.exists():
            img_model_path = preferred_model_path(img_model_path.name)
    else:
        img_model_path = next((p for p in candidate_models if p.exists()), None)

    # Use scd_prob_image from CSV if available, else compute from images
    if "scd_prob_image" not in df.columns and img_model_path and img_model_path.exists() and "image_path" in df.columns:
        print(f"Computing scd_prob_image using {img_model_path.name}...")
        img_model = tf.keras.models.load_model(img_model_path, custom_objects={"AppPreprocess": AppPreprocess})
        is_custom = img_model_path.name.lower() == "sickle_cnn.keras"
        df["scd_prob_image"] = df["image_path"].apply(lambda p: _predict_scd_prob(img_model, p, is_custom))
    elif "scd_prob_image" not in df.columns:
        print("Warning: no scd_prob_image column and no model available; using random values.")
        rng = np.random.default_rng(args.seed)
        df["scd_prob_image"] = rng.beta(2.5, 1.5, size=len(df))

    categorical = ["sex"]
    numeric = ["scd_prob_image"] + [f["name"] for f in FEATURE_SCHEMA if f["name"] != "sex"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = ["stroke", "heart_failure"]
    metrics_rows = []

    for target in targets:
        if target not in df.columns:
            print(f"Target '{target}' not in CSV; skipping.")
            continue

        y = df[target].astype(int).to_numpy()
        x = df[categorical + numeric].copy()

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=args.seed,
            stratify=y if len(np.unique(y)) > 1 else None, shuffle=True,
        )
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.25, random_state=args.seed,
            stratify=y_train if len(np.unique(y_train)) > 1 else None, shuffle=True,
        )

        preprocess = _build_preprocess(categorical, numeric)
        model_names = ["lr", "rf", "hgb"]

        candidates = []
        for name in model_names:
            pipe = _make_pipeline(name, preprocess, seed=args.seed)
            pipe.fit(x_tr, y_tr)
            prob_val = pipe.predict_proba(x_val)[:, 1]
            score = float(average_precision_score(y_val, prob_val)) if len(np.unique(y_val)) > 1 else float("nan")
            candidates.append({"name": name, "pipe": pipe, "score": score, "prob_val": prob_val})

        best = sorted(candidates, key=lambda c: np.nan_to_num(c["score"], nan=-1.0), reverse=True)[0]
        threshold = _choose_threshold(y_val, best["prob_val"])

        prob_test = best["pipe"].predict_proba(x_test)[:, 1]
        pred_test = (prob_test >= threshold).astype(int)

        acc = accuracy_score(y_test, pred_test)
        prec = precision_score(y_test, pred_test, zero_division=0)
        rec = recall_score(y_test, pred_test, zero_division=0)
        f1 = f1_score(y_test, pred_test, zero_division=0)
        auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else float("nan")

        metrics_rows.append({
            "target": target, "model": best["name"],
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "threshold": threshold,
        })

        print(f"[{target}] Best model: {best['name']} | AUC={auc:.4f} F1={f1:.4f}")

        # SHAP analysis
        try:
            import shap
            # Use a small background for SHAP
            x_bg = x_tr.head(100)
            explainer = shap.Explainer(best["pipe"].predict_proba, x_bg)
            shap_values = explainer(x_test.head(50))

            # Summary bar plot
            fig = plt.figure(figsize=(8, 6))
            shap.plots.bar(shap_values[:, :, 1], show=False)
            plt.title(f"{target} - SHAP Summary (model={best['name']})")
            plt.tight_layout()
            plt.savefig(out_dir / f"option_b_{target}_shap_summary_bar.png", dpi=150)
            plt.close(fig)

            # Waterfall for first sample
            fig2 = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            plt.title(f"{target} - SHAP Waterfall")
            plt.tight_layout()
            plt.savefig(out_dir / f"option_b_{target}_shap_waterfall.png", dpi=150)
            plt.close(fig2)
        except Exception as e:
            print(f"  SHAP analysis failed for {target}: {e}")

    # Save predictions and metrics
    pred_df = df.copy()
    pred_df.to_csv(out_dir / "option_b_multimodal_predictions.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "option_b_multimodal_predictions_metrics.csv", index=False)
    print("\n=== Multimodal Risk Metrics ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
