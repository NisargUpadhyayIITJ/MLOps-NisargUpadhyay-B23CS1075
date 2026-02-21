"""
eval.py - Evaluation script for DistilBERT Goodreads genre classification.

Usage:
    # Evaluate from a local model directory:
    python src/eval.py --model_path ./distilbert-reviews-genres

    # Evaluate from a HuggingFace Hub repository:
    python src/eval.py --hf_repo NisargUpadhyay/distilbert-goodreads-genres

    # Compare local vs. Hub evaluation:
    python src/eval.py --model_path ./distilbert-reviews-genres \
                       --hf_repo NisargUpadhyay/distilbert-goodreads-genres

What this script does:
    1. Loads the model from the specified source (local or Hub)
    2. Downloads and prepares the test dataset
    3. Runs evaluation (accuracy, F1, precision, recall)
    4. Prints a detailed classification report
    5. Saves results to JSON
    6. (Optional) Compares local vs. Hub metrics side by side
"""

import argparse
import json
import os
import sys

# Add src/ to path so imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import prepare_datasets
from utils import CACHED_MODEL_DIR, MODEL_NAME, compute_metrics, save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DistilBERT on Goodreads genre classification"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a locally saved model directory",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g. NisargUpadhyay/distilbert-goodreads-genres)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def evaluate_model(model_source: str, source_type: str, output_dir: str):
    """
    Evaluate a model from the given source.

    Args:
        model_source: local path or HF repo ID
        source_type: 'local' or 'hub'
        output_dir: where to save results

    Returns:
        dict: evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print(f"Loading model from ({source_type}): {model_source}")

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_source)
    model = DistilBertForSequenceClassification.from_pretrained(model_source).to(device)

    # Get label maps from model config
    id2label = model.config.id2label
    label2id = model.config.label2id
    print(f"Labels: {list(label2id.keys())}")

    # Prepare test dataset
    print("Preparing test dataset...")
    _, test_dataset, _, _, test_texts, test_labels = prepare_datasets(
        tokenizer=tokenizer
    )

    # Set up a minimal Trainer for evaluation
    eval_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    print("Running evaluation...")
    eval_results = trainer.evaluate()

    print(f"\n--- Evaluation Results ({source_type}) ---")
    print(f"  Loss:      {eval_results['eval_loss']:.4f}")
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  F1:        {eval_results['eval_f1']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")

    # Get predictions for classification report
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1).flatten().tolist()
    predicted_label_names = [id2label[str(l)] if str(l) in id2label else id2label.get(l, str(l))
                            for l in predicted_labels]

    print(f"\n--- Classification Report ({source_type}) ---")
    report = classification_report(test_labels, predicted_label_names)
    print(report)

    # Save results
    result_file = os.path.join(output_dir, f"{source_type}_eval_results.json")
    eval_results["source"] = model_source
    eval_results["source_type"] = source_type
    eval_results["classification_report"] = report
    save_results(eval_results, result_file)

    return eval_results


def compare_results(local_results: dict, hub_results: dict):
    """Print a side-by-side comparison of local vs. Hub evaluation."""
    print(f"\n{'='*60}")
    print("  COMPARISON: Local Model vs. HuggingFace Hub Model")
    print(f"{'='*60}")
    metrics = ["eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
    labels = ["Loss", "Accuracy", "F1", "Precision", "Recall"]

    print(f"  {'Metric':<12} {'Local':>10} {'Hub':>10} {'Diff':>10}")
    print(f"  {'-'*42}")
    for metric, label in zip(metrics, labels):
        local_val = local_results.get(metric, 0)
        hub_val = hub_results.get(metric, 0)
        diff = hub_val - local_val
        print(f"  {label:<12} {local_val:>10.4f} {hub_val:>10.4f} {diff:>+10.4f}")
    print()


def main():
    args = parse_args()

    if args.model_path is None and args.hf_repo is None:
        print("ERROR: Please provide --model_path and/or --hf_repo")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    local_results = None
    hub_results = None

    # Evaluate local model
    if args.model_path:
        print(f"\n{'='*60}")
        print("  Evaluating LOCAL model")
        print(f"{'='*60}")
        local_results = evaluate_model(args.model_path, "local", args.output_dir)

    # Evaluate Hub model
    if args.hf_repo:
        print(f"\n{'='*60}")
        print("  Evaluating HUGGINGFACE HUB model")
        print(f"{'='*60}")
        hub_results = evaluate_model(args.hf_repo, "hub", args.output_dir)

    # Compare if both provided
    if local_results and hub_results:
        compare_results(local_results, hub_results)

    print("Evaluation complete! 🎉")


if __name__ == "__main__":
    main()
