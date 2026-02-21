# ──────────────────────────────────────────────────────────────
# Dockerfile - Development / Training image
# ──────────────────────────────────────────────────────────────
# This Dockerfile builds an image for training the DistilBERT
# Goodreads genre classification model.
#
# Build:
#   docker build -t mlops-train .
#
# Run (training only):
#   docker run --gpus all mlops-train
#
# Run (training + push to HuggingFace Hub):
#   docker run --gpus all -e HF_TOKEN=your_token mlops-train \
#       python src/train.py --push_to_hub --hf_username NisargUpadhyay
# ──────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy the original notebook (for reference)
COPY ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb .

# Default command: run training
CMD ["python", "src/train.py"]
