# Project Structure

This project has been organized around a stable public workflow while preserving the original research scripts.

## Stable Entry Points

- `scripts/train_option1.py`
- `scripts/predict_image.py`
- `scripts/run_traditional_ml.py`
- `scripts/run_explainability.py`

These scripts are the recommended commands for demos, documentation, and project report instructions.

## Core Legacy Modules

- `train_cnn.py`: main deep learning workflow for Option 1
- `transfer_learning.py`: transfer-learning model builders and evaluators
- `data_generators.py`: data loading helpers
- `option_a_predict_with_notes.py`: single-image prediction helper
- `traditional_ml.py`: classical machine learning comparison
- `tf_grad_cam.py`, `tf_lime.py`, `tf_shap.py`: explainability workflows

## Recommended Artifact Placement

- `models/`: saved trained models
- `outputs/option1/`: training summaries and cross-validation reports
- `outputs/explainability/`: Grad-CAM, LIME, and SHAP outputs
- `outputs/comparison/`: baseline comparison plots and metrics

## Rationale

The original repository mixed source files, datasets, generated plots, trained models, and experiment outputs in a single directory. The current structure introduces:

- predictable entry points
- lightweight package utilities
- explicit documentation
- dependency and packaging metadata

This improves reproducibility and makes the project easier to submit, explain, and maintain.

