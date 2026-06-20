# Sickle Cell Morphology Detection Project

This is a shareable machine learning project for detecting sickle-cell morphology from peripheral blood smear images. It includes:

- deep learning classification of `normal` vs `sickle` cells
- transfer learning with EfficientNetB0, ResNet50, and MobileNetV2
- a traditional machine learning comparison
- explainability using Grad-CAM, LIME, and SHAP
- optional synthetic tabular data experiments

## What Your Friends Need

- Python 3.10 or newer
- the project folder
- the dataset folder in one of the supported layouts

Install dependencies with:

```powershell
pip install -r requirements.txt
```

## Dataset Layout

The code supports either of these:

```text
dataset/
  normal/
  sickle/
```

or:

```text
sickle cell dataset/
  Negative/
    Clear/
  Positive/
    Labelled/
```

## Clean Project Structure

```text
project/
  data/
    synthetic/            # sample synthetic CSV files
  docs/                   # extra notes
  models/                 # saved .keras models
  outputs/                # generated plots/results
  scripts/                # easiest commands to run
  src/sickle_project/     # small helper package
  sickle cell dataset/    # image dataset
  train_cnn.py            # main training workflow
  traditional_ml.py       # traditional ML comparison
  tf_grad_cam.py          # Grad-CAM
  tf_lime.py              # LIME
  tf_shap.py              # SHAP
```

## Easiest Commands To Run

Train the main image-classification workflow:

```powershell
py -3 scripts\train_option1.py --out-dir outputs\option1
```

Run prediction on one image:

```powershell
py -3 scripts\predict_image.py --image path\to\image.jpg
```

Run traditional ML comparison:

```powershell
py -3 scripts\run_traditional_ml.py
```

Run explainability:

```powershell
py -3 scripts\run_explainability.py gradcam --image path\to\image.jpg
py -3 scripts\run_explainability.py lime --image path\to\image.jpg
py -3 scripts\run_explainability.py shap --image path\to\image.jpg
```

## Important Folders

- `models/` stores trained model files like `sickle_cnn.keras`
- `data/synthetic/` stores synthetic CSV files used by the optional multimodal experiment
- `outputs/` is where generated graphs and reports should go

## If You Want To Share This Further

The easiest options are:

1. Zip the whole project folder and send it.
2. Upload it to Google Drive or OneDrive if the dataset is large.
3. Put it on GitHub if you want version control and easier collaboration.

## Notes

- The repository was cleaned so your friends get source code and important assets, not temporary clutter.
- Some root-level Python files are still kept for compatibility with the original project workflow.
- More structure notes are in [docs/project_structure.md](C:/Users/adith/Documents/project/docs/project_structure.md).
