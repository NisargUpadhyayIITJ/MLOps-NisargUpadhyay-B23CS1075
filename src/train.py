"""
train.py - Training script for DistilBERT Goodreads genre classification.

Usage:
    python src/train.py                        # Train and save locally
    python src/train.py --push_to_hub          # Train, save, and push to HF Hub
    python src/train.py --hf_token YOUR_TOKEN  # Provide token via CLI

What this script does (step-by-step):
    1. Loads the DistilBERT tokenizer and pre-trained model from HuggingFace
    2. Downloads Goodreads reviews and prepares train/test datasets
    3. Configures training arguments (3 epochs, lr=5e-5, batch=10)
    4. Trains the model using the HuggingFace Trainer API
    5. Evaluates on the test set and saves metrics
    6. Saves the fine-tuned model locally
    7. (Optional) Pushes the model to your HuggingFace Hub profile
"""

import argparse
import json
import os
import sys

# Add src/ to path so imports work when run from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import prepare_datasets
from utils import (
    CACHED_MODEL_DIR,
    HF_REPO_NAME,
    MODEL_NAME,
    compute_metrics,
    save_results,
)

# Disable wandb logging (requires API key we don't need)
os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on Goodreads genre classification"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the trained model to HuggingFace Hub after training",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace write-access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default=None,
        help="HuggingFace username for the repo (e.g. NisargUpadhyay)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CACHED_MODEL_DIR,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device setup ──────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Step 1: Load tokenizer ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 1: Loading tokenizer from '{MODEL_NAME}'")
    print(f"{'='*60}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # ── Step 2: Prepare datasets ──────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2: Downloading data and preparing datasets")
    print(f"{'='*60}")
    train_dataset, test_dataset, label2id, id2label, test_texts, test_labels = (
        prepare_datasets(tokenizer=tokenizer)
    )
    num_labels = len(id2label)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {list(label2id.keys())}")

    # ── Step 3: Load pre-trained model ────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 3: Loading pre-trained DistilBERT model (num_labels={num_labels})")
    print(f"{'='*60}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # ── Step 4: Configure training ────────────────────────────
    print(f"\n{'='*60}")
    print("Step 4: Configuring training arguments")
    print(f"{'='*60}")
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to=[],
    )
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Warmup steps:   100")
    print(f"  Weight decay:   0.01")

    # ── Step 5: Train ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 5: Training the model with HuggingFace Trainer API")
    print(f"{'='*60}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Training loss: {train_result.training_loss:.4f}")

    # ── Step 6: Evaluate ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 6: Evaluating on test set")
    print(f"{'='*60}")
    eval_results = trainer.evaluate()
    print(f"  Eval loss:     {eval_results['eval_loss']:.4f}")
    print(f"  Accuracy:      {eval_results['eval_accuracy']:.4f}")
    print(f"  F1:            {eval_results['eval_f1']:.4f}")
    print(f"  Precision:     {eval_results['eval_precision']:.4f}")
    print(f"  Recall:        {eval_results['eval_recall']:.4f}")

    # Save evaluation results
    save_results(eval_results, "eval_results/train_eval_results.json")

    # Save label maps for later use
    label_info = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    save_results(label_info, "eval_results/label_maps.json")

    # ── Step 7: Save model locally ────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 7: Saving model to '{args.output_dir}'")
    print(f"{'='*60}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Model and tokenizer saved to '{args.output_dir}'")

    # ── Step 8 (optional): Push to HuggingFace Hub ────────────
    if args.push_to_hub:
        print(f"\n{'='*60}")
        print("Step 8: Pushing model to HuggingFace Hub")
        print(f"{'='*60}")
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token is None:
            print("ERROR: No HuggingFace token provided. Use --hf_token or set HF_TOKEN env var.")
            sys.exit(1)

        repo_name = HF_REPO_NAME
        if args.hf_username:
            repo_id = f"{args.hf_username}/{repo_name}"
        else:
            repo_id = repo_name

        print(f"  Pushing to: {repo_id}")
        model.push_to_hub(repo_id, token=token)
        tokenizer.push_to_hub(repo_id, token=token)
        print(f"  ✓ Model pushed to https://huggingface.co/{repo_id}")

    print(f"\n{'='*60}")
    print("All done! 🎉")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
