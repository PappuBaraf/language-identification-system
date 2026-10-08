"""Project configuration."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "language_dataset.csv"
MODEL_DIR = BASE_DIR / "models" / "saved_models"
ML_MODEL_PATH = MODEL_DIR / "language_classifier.joblib"
MODEL_TYPE = os.getenv("MODEL_TYPE", "ml").strip().lower()
TRANSFORMER_MODEL = os.getenv(
    "TRANSFORMER_MODEL", "papluca/xlm-roberta-base-language-detection"
)
SUPPORTED_LANGUAGES = [
    "English", "Marathi", "Hindi", "Gujarati", "Bengali",
    "Tamil", "Telugu", "Kannada", "Malayalam", "Punjabi",
]
MIN_LETTERS = 2
