FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories used by the scripts
RUN mkdir -p weights/q1 weights/q2 results/q1 results/q2 wandb

# Default to an interactive shell instead of strictly the python interpreter
CMD ["/bin/bash"]
