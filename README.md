# Assignment 3: End-to-End HuggingFace Model Training & Docker Deployment

**Course:** MLOps (ML/DL Ops)  
**Author:** Nisarg Upadhyay (B23CS1075)  
**Model:** [NisargUpadhyay/distilbert-goodreads-genres](https://huggingface.co/NisargUpadhyay/distilbert-goodreads-genres)

---

## 📋 Project Overview

This project fine-tunes a **DistilBERT** model (`distilbert-base-cased`) on **Goodreads book reviews** for **genre classification** across 8 genres:

| #   | Genre                     |
| --- | ------------------------- |
| 0   | Children                  |
| 1   | Comics & Graphic          |
| 2   | Fantasy & Paranormal      |
| 3   | History & Biography       |
| 4   | Mystery, Thriller & Crime |
| 5   | Poetry                    |
| 6   | Romance                   |
| 7   | Young Adult               |

The workflow includes converting a Jupyter notebook into production-ready Python scripts, training with the HuggingFace Trainer API, evaluation, pushing the model to HuggingFace Hub, and containerizing everything with Docker.

---

## 🏗️ Project Structure

```
├── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb  # Original notebook
├── src/
│   ├── data.py          # Data loading, preprocessing, dataset class
│   ├── utils.py         # Config constants, metrics, label maps, helpers
│   ├── train.py         # Training script (Tasks 4-7)
│   └── eval.py          # Evaluation script (Tasks 6, 8)
├── Dockerfile           # Training Docker image
├── Dockerfile.eval      # Production evaluation-only Docker image
├── requirements.txt     # Python dependencies
├── eval_results/        # Saved evaluation metrics (JSON)
├── results/             # Training checkpoints
└── README.md            # This file
```

---

## 🧠 Model Selection & Rationale

**Model:** `distilbert-base-cased` (DistilBERT)

| Criterion                 | Why DistilBERT?                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| **Speed**                 | 60% faster than BERT-base, ideal for iterative training experiments                                |
| **Size**                  | 66M parameters (vs. 110M for BERT-base) — fits easily on any GPU                                   |
| **Performance**           | Retains 97% of BERT's language understanding capability                                            |
| **Cased**                 | Preserves capitalization, which can be meaningful in genre-specific writing styles                 |
| **HuggingFace Ecosystem** | Native support with `Trainer` API, `DistilBertForSequenceClassification`, and one-line push to Hub |

For a classification task with 8 genres at a moderate dataset size (6,400 training samples), DistilBERT provides the best tradeoff between training speed and accuracy.

---

## 📊 Training Summary

### Hyperparameters

| Parameter          | Value |
| ------------------ | ----- |
| Epochs             | 3     |
| Learning Rate      | 5e-5  |
| Batch Size (train) | 10    |
| Batch Size (eval)  | 16    |
| Warmup Steps       | 100   |
| Weight Decay       | 0.01  |
| Max Token Length   | 512   |
| Optimizer          | AdamW |

### Dataset

| Split | Samples | Per Genre |
| ----- | ------- | --------- |
| Train | 6,400   | 800       |
| Test  | 1,600   | 200       |

Data is streamed live from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) — no local data files needed.

---

## 📈 Evaluation Results

### Training Evaluation (after 3 epochs)

| Metric            | Value  |
| ----------------- | ------ |
| **Loss**          | 1.2379 |
| **Accuracy**      | 0.5988 |
| **F1 (weighted)** | 0.6001 |
| **Precision**     | 0.6044 |
| **Recall**        | 0.5988 |

### Post-Training Local Model Evaluation

| Metric            | Value  |
| ----------------- | ------ |
| **Loss**          | 1.2047 |
| **Accuracy**      | 0.6206 |
| **F1 (weighted)** | 0.6202 |
| **Precision**     | 0.6210 |
| **Recall**        | 0.6206 |

### Per-Genre Classification Report

```
                        precision    recall  f1-score   support

              children       0.65      0.71      0.68       200
        comics_graphic       0.79      0.82      0.80       200
    fantasy_paranormal       0.43      0.41      0.42       200
     history_biography       0.61      0.56      0.58       200
mystery_thriller_crime       0.60      0.57      0.59       200
                poetry       0.77      0.78      0.77       200
               romance       0.71      0.65      0.68       200
           young_adult       0.42      0.46      0.44       200

              accuracy                           0.62      1600
             macro avg       0.62      0.62      0.62      1600
          weighted avg       0.62      0.62      0.62      1600
```

### Evaluation Comparison: Local vs. HuggingFace Hub

Since the Hub model is identical to the locally saved model (same weights pushed directly), the evaluation metrics are the same. The `eval.py` script supports comparing both:

```bash
python src/eval.py --model_path ./distilbert-reviews-genres \
                   --hf_repo NisargUpadhyay/distilbert-goodreads-genres
```

| Metric    | Local  | Hub (same model) |
| --------- | ------ | ---------------- |
| Accuracy  | 0.6206 | 0.6206           |
| F1        | 0.6202 | 0.6202           |
| Precision | 0.6210 | 0.6210           |
| Recall    | 0.6206 | 0.6206           |

### Key Observations

- **Best performers:** Comics & Graphic (0.80 F1) and Poetry (0.77 F1) — these genres have distinctive review language
- **Hardest genres:** Fantasy & Paranormal (0.42 F1) and Young Adult (0.44 F1) — likely overlap in review language with other genres
- **Overall:** 62% accuracy across 8 classes (random baseline = 12.5%) — significant improvement

---

## 🚀 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training (with Hub push)

```bash
python src/train.py --push_to_hub \
                    --hf_username NisargUpadhyay \
                    --hf_token YOUR_HF_TOKEN
```

### Training (local only)

```bash
python src/train.py
```

### Evaluation (from local model)

```bash
python src/eval.py --model_path ./distilbert-reviews-genres
```

### Evaluation (from HuggingFace Hub)

```bash
python src/eval.py --hf_repo NisargUpadhyay/distilbert-goodreads-genres
```

---

## 🐳 Docker Instructions

### Training Image

```bash
# Build
docker build -t mlops-train .

# Run (training only)
docker run --gpus all mlops-train

# Run (training + push to Hub)
docker run --gpus all \
  -e HF_TOKEN=your_token \
  mlops-train python src/train.py --push_to_hub --hf_username NisargUpadhyay
```

### Production Evaluation Image

```bash
# Build
docker build -f Dockerfile.eval -t mlops-eval .

# Run (automatically evaluates model from HuggingFace Hub)
docker run mlops-eval
```

---

## 🔗 Links

- **HuggingFace Model:** [NisargUpadhyay/distilbert-goodreads-genres](https://huggingface.co/NisargUpadhyay/distilbert-goodreads-genres)
- **Original Notebook:** [Colab Link](https://colab.research.google.com/drive/1-DhcPi4j3VBVFt9K39e895aIkXyNLJwS?usp=sharing)
- **Dataset Source:** [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html)

---

## 🧩 Challenges & Solutions

| Challenge                                                | Solution                                                             |
| -------------------------------------------------------- | -------------------------------------------------------------------- |
| Large data downloads (~1GB gzip per genre)               | Stream and sample only 10K reviews per genre, then use 1K per genre  |
| Notebook had Colab-specific code (`!pip`, `%matplotlib`) | Removed magic commands, restructured into importable modules         |
| WandB logging requiring API key                          | Disabled via `WANDB_DISABLED` env variable and `report_to=[]`        |
| Model download throttling from HuggingFace               | Used caching and retry logic; local model evaluation as primary      |
| Overlapping genre definitions (Fantasy vs. Young Adult)  | Acknowledged in analysis — real-world limitation of genre boundaries |

---

## 📁 Script Descriptions (What Each File Does)

### `src/data.py`

- **`load_reviews(url)`** — Streams gzipped JSON reviews from UCSD servers, takes first 10K, samples 2K
- **`load_all_genres()`** — Calls `load_reviews()` for all 8 genres
- **`split_data()`** — Splits into 800 train / 200 test per genre
- **`ReviewDataset`** — Custom PyTorch Dataset wrapping tokenized encodings + labels
- **`prepare_datasets()`** — End-to-end pipeline: download → split → tokenize → wrap

### `src/utils.py`

- Configuration constants (`MODEL_NAME`, `MAX_LENGTH`, genre URLs)
- **`get_label_maps()`** — Build `label2id` / `id2label` dicts
- **`compute_metrics()`** — Returns accuracy, F1, precision, recall for Trainer API
- **`save_results()`** — Write metrics dict to JSON file

### `src/train.py`

- Full training pipeline with step-by-step logging
- Loads DistilBERT, creates Trainer, runs 3 epochs
- Saves model locally + optionally pushes to HuggingFace Hub
- Accepts CLI args: `--push_to_hub`, `--hf_token`, `--epochs`, etc.

### `src/eval.py`

- Loads model from local path (`--model_path`) or HuggingFace Hub (`--hf_repo`)
- Runs evaluation metrics + full classification report
- Compares metrics side-by-side if both sources specified
- Saves results to `eval_results/` as JSON
