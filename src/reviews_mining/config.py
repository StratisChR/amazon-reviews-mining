"""Paths and settings shared by every step."""
import os
from pathlib import Path

# Short name used in the code and plots -> category name in the dataset.
CATEGORIES = {
    "allBeauty": "All_Beauty",
    "software": "Software",
    "musicalInstruments": "Musical_Instruments",
    "handmadeProducts": "Handmade_Products",
    "giftCards": "Gift_Cards",
}

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
DOWNLOAD_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Everything the pipeline writes goes under DATA_DIR (default: <repo>/data).
# Override with the REVIEWS_DATA_DIR environment variable.
DATA_DIR = Path(os.environ.get("REVIEWS_DATA_DIR",
                               Path(__file__).resolve().parents[2] / "data"))
RAW_DIR = DATA_DIR / "hf"            # raw JSONL files, as downloaded
PROCESSED_DIR = DATA_DIR / "processed"

REVIEW_SAMPLE_SIZE = 100_000   # reviews kept per category for text analysis
SEED = 42


def processed_dir(key: str) -> Path:
    path = PROCESSED_DIR / key
    path.mkdir(parents=True, exist_ok=True)
    return path
