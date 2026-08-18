from pathlib import Path

""" Este archivo nos ayuda a no usar rutas hardcodeadas por todo el proyecto."""

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORS = ARTIFACTS_DIR / "preprocessors"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
