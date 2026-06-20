from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"


def preferred_model_path(name: str) -> Path:
    model_candidate = MODELS_DIR / name
    if model_candidate.exists():
        return model_candidate
    return PROJECT_ROOT / name


def existing_model_paths(names: list[str]) -> list[Path]:
    return [path for name in names if (path := preferred_model_path(name)).exists()]


def preferred_synthetic_csv(name: str) -> Path:
    csv_candidate = SYNTHETIC_DIR / name
    if csv_candidate.exists():
        return csv_candidate
    return PROJECT_ROOT / name
