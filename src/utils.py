"""
utils.py - Configuration constants and helper functions for the DistilBERT
           Goodreads genre classification project.

This module contains:
  - Model and data configuration constants
  - Genre URL dictionary for downloading Goodreads reviews
  - Label mapping helpers
  - Metric computation function for the Trainer API
  - Result saving utility
"""

import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
CACHED_MODEL_DIR = "distilbert-reviews-genres"
HF_REPO_NAME = "distilbert-goodreads-genres"

GENRE_URL_DICT = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

# ──────────────────────────────────────────────────────────────
# Label helpers
# ──────────────────────────────────────────────────────────────

def get_label_maps(labels):
    """
    Build label ↔ id mappings from a list of string labels.

    Returns:
        label2id (dict): e.g. {"poetry": 0, "children": 1, ...}
        id2label (dict): e.g. {0: "poetry", 1: "children", ...}
    """
    unique_labels = sorted(set(labels))  # sorted for reproducibility
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(pred):
    """
    Compute accuracy, precision, recall, and F1 for the Trainer API.

    Args:
        pred: EvalPrediction object with .label_ids and .predictions

    Returns:
        dict with accuracy, precision, recall, and f1 keys.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────

def save_results(results: dict, filepath: str):
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")
