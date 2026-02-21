"""
data.py - Data loading, preprocessing, and dataset creation for the
          DistilBERT Goodreads genre classification project.

Pipeline:
  1. Stream & sample reviews from UCSD Goodreads URLs (gzip JSON)
  2. Split into train / test sets
  3. Tokenize with DistilBertTokenizerFast
  4. Wrap in a custom PyTorch Dataset
"""

import gzip
import json
import random

import requests
import torch
from transformers import DistilBertTokenizerFast

from utils import GENRE_URL_DICT, MAX_LENGTH, MODEL_NAME, get_label_maps

# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_reviews(url: str, head: int = 10_000, sample_size: int = 2_000):
    """
    Stream reviews from a gzipped JSON URL and return a random sample.

    Each line in the gzip file is a JSON object with a 'review_text' field.
    We read up to `head` reviews and then randomly sample `sample_size` of them.
    """
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(genre_url_dict: dict = None):
    """
    Load reviews for every genre in the URL dictionary.

    Returns:
        dict: {genre_name: [review_text, ...], ...}
    """
    if genre_url_dict is None:
        genre_url_dict = GENRE_URL_DICT

    genre_reviews = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url)
    return genre_reviews


# ──────────────────────────────────────────────────────────────
# Train / Test split
# ──────────────────────────────────────────────────────────────

def split_data(genre_reviews: dict, train_per_genre: int = 800,
               test_per_genre: int = 200):
    """
    Split genre_reviews into train and test lists.

    For each genre we sample (train_per_genre + test_per_genre) reviews,
    then split them.

    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        sampled = random.sample(reviews, min(train_per_genre + test_per_genre, len(reviews)))
        for review in sampled[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_per_genre:train_per_genre + test_per_genre]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────

class ReviewDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset that wraps tokenizer encodings + labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ──────────────────────────────────────────────────────────────
# End-to-end dataset preparation
# ──────────────────────────────────────────────────────────────

def prepare_datasets(tokenizer=None, max_length: int = None):
    """
    Full pipeline: download → split → tokenize → wrap in Dataset objects.

    Returns:
        train_dataset, test_dataset, label2id, id2label,
        test_texts, test_labels
    """
    if tokenizer is None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    if max_length is None:
        max_length = MAX_LENGTH

    # 1. Download reviews
    genre_reviews = load_all_genres()

    # 2. Split
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

    # 3. Build label maps
    label2id, id2label = get_label_maps(train_labels)

    # 4. Encode labels as integers
    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]

    # 5. Tokenize texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                                max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True,
                               max_length=max_length)

    # 6. Wrap in Dataset objects
    train_dataset = ReviewDataset(train_encodings, train_labels_enc)
    test_dataset = ReviewDataset(test_encodings, test_labels_enc)

    return train_dataset, test_dataset, label2id, id2label, test_texts, test_labels


if __name__ == "__main__":
    # Quick test: load data, print shapes
    train_ds, test_ds, l2id, id2l, _, _ = prepare_datasets()
    print(f"Train dataset length: {len(train_ds)}")
    print(f"Test dataset length:  {len(test_ds)}")
    print(f"Labels: {l2id}")
