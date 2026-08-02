from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw"
CLEANED_DATA = BASE_DIR / "data" / "cleaned"
FINAL_DATA = BASE_DIR / "data" / "final"
OUTPUT_DATA = BASE_DIR / "output"

TRAINING_DATA = BASE_DIR / "data" / "training"
MODELS_DIR = BASE_DIR / "models"