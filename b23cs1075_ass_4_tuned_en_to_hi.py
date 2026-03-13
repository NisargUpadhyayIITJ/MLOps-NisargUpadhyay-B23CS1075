#!/usr/bin/env python3
"""Assignment 4: English-to-Hindi transformer tuning with Ray Tune + Optuna.

This script keeps the original notebook architecture intact, but refactors the
training loop so it can be used for:

1. Baseline reproduction
2. Hyperparameter tuning with Ray Tune + Optuna + ASHA
3. Final best-model training and saving
4. Submission report generation

The original notebook already contains recorded baseline outputs. Those values
are included below so the report can be scaffolded even before the tuning sweep
is executed in a fully provisioned environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = None

try:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
except ImportError:
    SmoothingFunction = None
    corpus_bleu = None

try:
    from ray import train, tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
except ImportError:
    train = None
    tune = None
    ASHAScheduler = None
    OptunaSearch = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROLLNO = "b23cs1075"
DEFAULT_DATA_PATH = "English-Hindi.tsv"
DEFAULT_BASELINE_MODEL_PATH = "transformer_translation_final.pth"
DEFAULT_BEST_MODEL_PATH = f"{ROLLNO}_ass_4_best_model.pth"
DEFAULT_REPORT_MD_PATH = f"{ROLLNO}_ass_4_report.md"
DEFAULT_SUMMARY_JSON = "artifacts/assignment4/summary.json"
DEFAULT_TUNING_JSON = "artifacts/assignment4/tuning_results.json"
DEFAULT_BEST_CONFIG_JSON = "artifacts/assignment4/best_config.json"
DEFAULT_VOCAB_DIR = "artifacts/assignment4/vocabs"
GITHUB_REPO_URL = "https://github.com/NisargUpadhyayIITJ/MLOps-NisargUpadhyay-B23CS1075"
HF_MODEL_REPO_URL = "https://huggingface.co/NisargUpadhyay/b23cs1075-assignment4-models"

NOTEBOOK_BASELINE = {
    "epochs": 100,
    "training_time_seconds": 3139.0,
    "training_time_minutes": 52.32,
    "final_loss": 0.0974,
    "bleu": 0.5247,
    "bleu_percent": 52.47,
    "source": "Extracted from en_to_hi.ipynb recorded outputs on 2026-03-13",
}

DEFAULT_VAL_PAIRS: List[Tuple[str, str]] = [
    ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?", "आप कैसे हैं?"),
    ("You should sleep.", "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.", "मुझे टॉम को बताने दीजिए।"),
]

SEARCH_SPACE_DESCRIPTION = {
    "learning_rate": "loguniform(1e-5, 1e-3)",
    "batch_size": "choice([16, 32, 64])",
    "num_heads": "choice([4, 8])",
    "d_ff": "choice([1024, 1536, 2048])",
    "dropout": "uniform(0.10, 0.40)",
    "num_layers": "choice([4, 6])",
    "weight_decay": "loguniform(1e-6, 1e-3)",
}


def require_base_dependencies() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if corpus_bleu is None or SmoothingFunction is None:
        missing.append("nltk")
    if tqdm is None:
        missing.append("tqdm")
    if torch is None or nn is None or optim is None:
        missing.append("torch")

    if missing:
        raise ImportError(
            "Missing required packages: "
            + ", ".join(sorted(set(missing)))
            + ". Install assignment4_requirements.txt first."
        )


def require_tuning_dependencies() -> None:
    require_base_dependencies()
    missing = []
    if tune is None or train is None:
        missing.append("ray[tune]")
    if OptunaSearch is None:
        missing.append("optuna")
    if ASHAScheduler is None:
        missing.append("ray[tune]")

    if missing:
        raise ImportError(
            "Missing tuning packages: "
            + ", ".join(sorted(set(missing)))
            + ". Install assignment4_requirements.txt first."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_translation_dataframe(data_path: str) -> "pd.DataFrame":
    require_base_dependencies()
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["id1", "en", "id2", "hi"],
    )
    df = df[["en", "hi"]].dropna().reset_index(drop=True)
    return df


class Vocabulary:
    def __init__(self, freq_threshold: int = 2) -> None:
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def tokenize(self, sentence: str) -> List[str]:
        return sentence.lower().strip().split()

    def build_vocab(self, sentence_list: Iterable[str]) -> None:
        frequencies: Counter[str] = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def numericalize(self, sentence: str) -> List[int]:
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in self.tokenize(sentence)]

    def __len__(self) -> int:
        return len(self.stoi)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])


def build_vocabs(df: "pd.DataFrame") -> Tuple[Vocabulary, Vocabulary]:
    en_vocab = Vocabulary(freq_threshold=2)
    hi_vocab = Vocabulary(freq_threshold=2)
    en_vocab.build_vocab(df["en"].tolist())
    hi_vocab.build_vocab(df["hi"].tolist())
    return en_vocab, hi_vocab


def encode_sentence(sentence: str, vocab: Vocabulary, max_len: int = 50) -> List[int]:
    tokens = (
        [vocab.stoi["<sos>"]]
        + vocab.numericalize(sentence)[: max_len - 2]
        + [vocab.stoi["<eos>"]]
    )
    padding = [vocab.stoi["<pad>"]] * max(0, max_len - len(tokens))
    return tokens + padding


if nn is not None:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.pe[:, : x.size(1)]


    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
            super().__init__()
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads")

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.query_linear = nn.Linear(d_model, d_model)
            self.key_linear = nn.Linear(d_model, d_model)
            self.value_linear = nn.Linear(d_model, d_model)
            self.out_linear = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            q: "torch.Tensor",
            k: "torch.Tensor",
            v: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            batch_size = q.size(0)

            query = self.query_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            key = self.key_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            value = self.value_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attention_weights = torch.softmax(scores, dim=-1)
            attention_output = torch.matmul(self.dropout(attention_weights), value)
            attention_output = (
                attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            )
            return self.out_linear(attention_output)


    class FeedForward(nn.Module):
        def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1) -> None:
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.linear2(self.dropout(self.relu(self.linear1(x))))


    class LayerNorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.zeros(d_model))
            self.eps = eps

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True, unbiased=False)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta


    class EncoderLayer(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.ffn = FeedForward(d_model, d_ff, dropout)
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
            return x


    class DecoderLayer(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.ffn = FeedForward(d_model, d_ff, dropout)
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
            self.norm3 = LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: "torch.Tensor",
            enc_out: "torch.Tensor",
            src_mask: Optional["torch.Tensor"] = None,
            tgt_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
            x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
            x = self.norm3(x + self.dropout(self.ffn(x)))
            return x


    class Encoder(nn.Module):
        def __init__(
            self,
            input_vocab_size: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            max_len: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.embed = nn.Embedding(input_vocab_size, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len)
            self.layers = nn.ModuleList(
                [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            x = self.dropout(self.pos_enc(self.embed(x)))
            for layer in self.layers:
                x = layer(x, mask)
            return x


    class Decoder(nn.Module):
        def __init__(
            self,
            target_vocab_size: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            max_len: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.embed = nn.Embedding(target_vocab_size, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len)
            self.layers = nn.ModuleList(
                [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
            )
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: "torch.Tensor",
            enc_out: "torch.Tensor",
            src_mask: Optional["torch.Tensor"] = None,
            tgt_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            x = self.dropout(self.pos_enc(self.embed(x)))
            for layer in self.layers:
                x = layer(x, enc_out, src_mask, tgt_mask)
            return x


    class Transformer(nn.Module):
        def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            d_ff: int = 2048,
            max_len: int = 100,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
            self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
            self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        def make_pad_mask(self, seq: "torch.Tensor", pad_idx: int) -> "torch.Tensor":
            return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

        def make_subsequent_mask(self, size: int) -> "torch.Tensor":
            device = next(self.parameters()).device
            return torch.tril(torch.ones((size, size), device=device)).bool()

        def forward(
            self,
            src: "torch.Tensor",
            tgt: "torch.Tensor",
            src_pad_idx: int,
            tgt_pad_idx: int,
        ) -> "torch.Tensor":
            src_mask = self.make_pad_mask(src, src_pad_idx)
            tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
            tgt_sub_mask = self.make_subsequent_mask(tgt.size(1))
            tgt_mask = tgt_pad_mask & tgt_sub_mask

            enc_out = self.encoder(src, src_mask)
            dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
            return self.fc_out(dec_out)


    class TranslationDataset(Dataset):
        def __init__(self, df: "pd.DataFrame", en_vocab: Vocabulary, hi_vocab: Vocabulary, max_len: int = 50) -> None:
            self.en_sentences = df["en"].tolist()
            self.hi_sentences = df["hi"].tolist()
            self.en_vocab = en_vocab
            self.hi_vocab = hi_vocab
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.en_sentences)

        def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
            src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
            tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
            return torch.tensor(src), torch.tensor(tgt)

else:
    PositionalEncoding = None
    MultiHeadAttention = None
    FeedForward = None
    LayerNorm = None
    EncoderLayer = None
    DecoderLayer = None
    Encoder = None
    Decoder = None
    Transformer = None
    TranslationDataset = None


def collate_fn(batch: Sequence[Tuple["torch.Tensor", "torch.Tensor"]]) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output


def get_device(force_cpu: bool = False) -> "torch.device":
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def build_train_loader(
    df: "pd.DataFrame",
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
    batch_size: int,
    max_len: int,
) -> "DataLoader":
    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def build_model(config: Dict[str, Any], en_vocab: Vocabulary, hi_vocab: Vocabulary, device: "torch.device") -> "Transformer":
    return Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
        dropout=config["dropout"],
    ).to(device)


def translate_sentence(
    model: "Transformer",
    sentence: str,
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
    src_pad_idx: int,
    tgt_pad_idx: int,
    device: "torch.device",
    max_len: int = 50,
) -> str:
    model.eval()
    src_tokens = encode_sentence(sentence, en_vocab, max_len=max_len)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

    tgt_tokens = [hi_vocab["<sos>"]]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
            output = model(src_tensor, tgt_tensor, src_pad_idx, tgt_pad_idx)
            next_token = output[0, -1].argmax().item()
            tgt_tokens.append(next_token)
            if next_token == hi_vocab["<eos>"]:
                break

    special_tokens = {
        hi_vocab["<pad>"],
        hi_vocab["<sos>"],
        hi_vocab["<eos>"],
    }
    translated = [hi_vocab.itos[idx] for idx in tgt_tokens if idx not in special_tokens]
    return " ".join(translated).strip()


def evaluate_bleu_nltk(
    model: "Transformer",
    dataset: Sequence[Tuple[str, str]],
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
    src_pad_idx: int,
    tgt_pad_idx: int,
    device: "torch.device",
    max_len: int = 50,
) -> float:
    references: List[List[List[str]]] = []
    hypotheses: List[List[str]] = []
    smoothie = SmoothingFunction().method4

    for en_sentence, hi_sentence in dataset:
        pred = translate_sentence(
            model=model,
            sentence=en_sentence,
            en_vocab=en_vocab,
            hi_vocab=hi_vocab,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            device=device,
            max_len=max_len,
        )
        hypotheses.append(pred.split())
        references.append([hi_sentence.split()])

    return corpus_bleu(references, hypotheses, smoothing_function=smoothie)


def run_epoch(
    model: "Transformer",
    train_loader: "DataLoader",
    optimizer: "optim.Optimizer",
    criterion: "nn.Module",
    src_pad_idx: int,
    tgt_pad_idx: int,
    device: "torch.device",
    show_progress: bool,
    epoch_index: int,
    num_epochs: int,
) -> float:
    model.train()
    epoch_loss = 0.0
    loop: Iterable[Any]
    if show_progress and tqdm is not None:
        loop = tqdm(train_loader, desc=f"Epoch [{epoch_index + 1}/{num_epochs}]")
    else:
        loop = train_loader

    for src, tgt_input, tgt_output in loop:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        output = model(src, tgt_input, src_pad_idx, tgt_pad_idx)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if show_progress and tqdm is not None:
            loop.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / max(1, len(train_loader))


@dataclass
class TrainingSummary:
    epochs: int
    training_time_seconds: float
    training_time_minutes: float
    final_loss: float
    best_bleu: float
    best_bleu_percent: float
    best_epoch: int
    epochs_to_target: Optional[int]
    model_path: Optional[str]
    config: Dict[str, Any]


def train_model(
    config: Dict[str, Any],
    df: "pd.DataFrame",
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
    num_epochs: int,
    val_pairs: Sequence[Tuple[str, str]],
    model_path: Optional[str],
    target_bleu: Optional[float],
    report_to_ray: bool = False,
    show_progress: bool = True,
) -> TrainingSummary:
    require_base_dependencies()
    set_seed(config["seed"])
    device = get_device(force_cpu=config.get("force_cpu", False))
    src_pad_idx = en_vocab["<pad>"]
    tgt_pad_idx = hi_vocab["<pad>"]

    train_loader = build_train_loader(
        df=df,
        en_vocab=en_vocab,
        hi_vocab=hi_vocab,
        batch_size=config["batch_size"],
        max_len=config["max_len"],
    )
    model = build_model(config, en_vocab, hi_vocab, device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    best_state: Optional[Dict[str, Any]] = None
    best_bleu = -1.0
    best_epoch = 0
    epochs_to_target: Optional[int] = None
    final_loss = float("nan")
    start_time = time.time()

    for epoch in range(num_epochs):
        final_loss = run_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            device=device,
            show_progress=show_progress,
            epoch_index=epoch,
            num_epochs=num_epochs,
        )

        bleu = evaluate_bleu_nltk(
            model=model,
            dataset=val_pairs,
            en_vocab=en_vocab,
            hi_vocab=hi_vocab,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            device=device,
            max_len=config["max_len"],
        )

        if bleu > best_bleu:
            best_bleu = bleu
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if target_bleu is not None and epochs_to_target is None and bleu >= target_bleu:
            epochs_to_target = epoch + 1

        metrics = {
            "epoch": epoch + 1,
            "loss": final_loss,
            "bleu": bleu,
            "bleu_percent": bleu * 100.0,
        }

        if report_to_ray:
            tune.report(metrics)
        elif show_progress:
            print(
                f"Epoch {epoch + 1:02d}/{num_epochs} | "
                f"loss={final_loss:.4f} | bleu={bleu * 100:.2f}"
            )

    elapsed = time.time() - start_time

    if model_path and best_state is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)

    return TrainingSummary(
        epochs=num_epochs,
        training_time_seconds=elapsed,
        training_time_minutes=elapsed / 60.0,
        final_loss=final_loss,
        best_bleu=best_bleu,
        best_bleu_percent=best_bleu * 100.0,
        best_epoch=best_epoch,
        epochs_to_target=epochs_to_target,
        model_path=model_path,
        config=config,
    )


def build_baseline_config(max_len: int, seed: int, force_cpu: bool) -> Dict[str, Any]:
    return {
        "learning_rate": 1e-4,
        "batch_size": 60,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "num_layers": 6,
        "weight_decay": 0.0,
        "d_model": 512,
        "max_len": max_len,
        "seed": seed,
        "force_cpu": force_cpu,
    }


def build_search_space(max_len: int, tune_epochs: int, seed: int, force_cpu: bool) -> Dict[str, Any]:
    require_tuning_dependencies()
    return {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "num_heads": tune.choice([4, 8]),
        "d_ff": tune.choice([1024, 1536, 2048]),
        "dropout": tune.uniform(0.10, 0.40),
        "num_layers": tune.choice([4, 6]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "d_model": 512,
        "max_len": max_len,
        "seed": seed,
        "force_cpu": force_cpu,
        "epochs": tune_epochs,
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_vocabs(en_vocab: Vocabulary, hi_vocab: Vocabulary, vocab_dir: str) -> None:
    vocab_path = Path(vocab_dir)
    vocab_path.mkdir(parents=True, exist_ok=True)
    with open(vocab_path / "en_vocab.pkl", "wb") as handle:
        pickle.dump(en_vocab, handle)
    with open(vocab_path / "hi_vocab.pkl", "wb") as handle:
        pickle.dump(hi_vocab, handle)


def run_baseline(args: argparse.Namespace, df: "pd.DataFrame", en_vocab: Vocabulary, hi_vocab: Vocabulary) -> Dict[str, Any]:
    baseline_config = build_baseline_config(args.max_len, args.seed, args.force_cpu)
    summary = train_model(
        config=baseline_config,
        df=df,
        en_vocab=en_vocab,
        hi_vocab=hi_vocab,
        num_epochs=args.baseline_epochs,
        val_pairs=DEFAULT_VAL_PAIRS,
        model_path=args.baseline_model_path,
        target_bleu=None,
        report_to_ray=False,
        show_progress=True,
    )
    baseline_payload = asdict(summary)
    write_json("artifacts/assignment4/baseline_metrics.json", baseline_payload)
    save_vocabs(en_vocab, hi_vocab, DEFAULT_VOCAB_DIR)
    return baseline_payload


def train_tune(
    config: Dict[str, Any],
    data_path: str,
    val_pairs: Sequence[Tuple[str, str]],
) -> None:
    require_tuning_dependencies()
    df = load_translation_dataframe(data_path)
    en_vocab, hi_vocab = build_vocabs(df)
    train_model(
        config=config,
        df=df,
        en_vocab=en_vocab,
        hi_vocab=hi_vocab,
        num_epochs=config["epochs"],
        val_pairs=val_pairs,
        model_path=None,
        target_bleu=None,
        report_to_ray=True,
        show_progress=False,
    )


def run_tuning(args: argparse.Namespace) -> Dict[str, Any]:
    require_tuning_dependencies()

    resolved_data_path = str(Path(args.data_path).resolve())
    search_space = build_search_space(args.max_len, args.tune_epochs, args.seed, args.force_cpu)
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="bleu",
        mode="max",
        max_t=args.tune_epochs,
        grace_period=max(1, min(args.tune_epochs, args.tune_epochs // 4)),
        reduction_factor=2,
    )
    search_alg = OptunaSearch(metric="bleu", mode="max")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_tune,
                data_path=resolved_data_path,
                val_pairs=DEFAULT_VAL_PAIRS,
            ),
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.num_samples,
        ),
        run_config=tune.RunConfig(
            name="assignment4_transformer_tuning",
            storage_path=str(Path(args.ray_results_dir).resolve()),
            verbose=1,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="bleu", mode="max")
    best_payload = {
        "best_config": best_result.config,
        "best_metrics": best_result.metrics,
        "num_samples": args.num_samples,
        "tune_epochs": args.tune_epochs,
        "search_space": SEARCH_SPACE_DESCRIPTION,
        "metric": "bleu",
        "mode": "max",
        "scheduler": "ASHAScheduler",
        "search_algorithm": "OptunaSearch",
    }
    write_json(args.tuning_json_path, best_payload)
    write_json(args.best_config_path, best_result.config)
    return best_payload


def recover_tuning_from_ray_results(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.ray_results_dir) / "assignment4_transformer_tuning"
    if not root.exists():
        raise FileNotFoundError(f"Ray results directory not found at {root}")

    state_files = sorted(root.glob("experiment_state-*.json"))
    if not state_files:
        raise FileNotFoundError(f"No Ray experiment state files found under {root}")

    state_file = state_files[-1]
    state_data = json.loads(state_file.read_text(encoding="utf-8"))
    status_counts: Dict[str, int] = {}
    state_rows: Dict[str, Dict[str, Any]] = {}

    for entry in state_data.get("trial_data", []):
        trial_meta = json.loads(entry[0])
        runtime_meta = json.loads(entry[1])
        trial_id = trial_meta["trial_id"]
        status = trial_meta["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        state_rows[trial_id] = {
            "status": status,
            "logdir": trial_meta.get("relative_logdir"),
            "last_result": runtime_meta.get("last_result", {}),
        }

    recovered_rows: List[Dict[str, Any]] = []
    for result_path in root.glob("train_tune_*/result.json"):
        trial_dir = result_path.parent
        params_path = trial_dir / "params.json"
        if not params_path.exists():
            continue

        config = json.loads(params_path.read_text(encoding="utf-8"))
        best_metrics: Optional[Dict[str, Any]] = None
        last_metrics: Optional[Dict[str, Any]] = None

        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            last_metrics = row
            if best_metrics is None or row.get("bleu", float("-inf")) > best_metrics.get("bleu", float("-inf")):
                best_metrics = row

        if best_metrics is None or last_metrics is None:
            continue

        trial_id = best_metrics.get("trial_id") or last_metrics.get("trial_id")
        state_row = state_rows.get(trial_id, {})
        recovered_rows.append(
            {
                "trial_id": trial_id,
                "trial_dir": trial_dir.name,
                "config": config,
                "status": state_row.get("status"),
                "best_metrics": best_metrics,
                "last_metrics": last_metrics,
            }
        )

    if not recovered_rows:
        raise RuntimeError(f"No recoverable trial result files found under {root}")

    best_trial = max(recovered_rows, key=lambda row: row["best_metrics"].get("bleu", float("-inf")))
    best_payload = {
        "best_config": best_trial["config"],
        "best_metrics": {
            "epoch": best_trial["best_metrics"].get("epoch"),
            "loss": best_trial["best_metrics"].get("loss"),
            "bleu": best_trial["best_metrics"].get("bleu"),
            "bleu_percent": best_trial["best_metrics"].get("bleu_percent"),
            "trial_id": best_trial["trial_id"],
            "trial_dir": best_trial["trial_dir"],
            "status": best_trial.get("status"),
            "last_epoch": best_trial["last_metrics"].get("epoch"),
            "last_loss": best_trial["last_metrics"].get("loss"),
            "last_bleu": best_trial["last_metrics"].get("bleu"),
            "last_bleu_percent": best_trial["last_metrics"].get("bleu_percent"),
        },
        "num_samples": len(state_rows),
        "tune_epochs": args.tune_epochs,
        "search_space": SEARCH_SPACE_DESCRIPTION,
        "metric": "bleu",
        "mode": "max",
        "scheduler": "ASHAScheduler",
        "search_algorithm": "OptunaSearch",
        "recovered_from_partial_sweep": True,
        "ray_state_file": state_file.name,
        "ray_status_counts": status_counts,
        "recovery_note": "Recovered from trial result.json files after the Ray driver received SIGTERM before final summary export.",
    }
    write_json(args.tuning_json_path, best_payload)
    write_json(args.best_config_path, best_trial["config"])
    return best_payload


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    with open(path_obj, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_best_model_training(
    args: argparse.Namespace,
    df: "pd.DataFrame",
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
) -> Dict[str, Any]:
    best_config = load_json_if_exists(args.best_config_path)
    if best_config is None:
        raise FileNotFoundError(
            f"Best-config file not found at {args.best_config_path}. Run `--action tune` first."
        )

    # Remove Ray-only bookkeeping if present and keep execution flags explicit.
    best_config = {**best_config, "seed": args.seed, "force_cpu": args.force_cpu, "max_len": args.max_len}

    summary = train_model(
        config=best_config,
        df=df,
        en_vocab=en_vocab,
        hi_vocab=hi_vocab,
        num_epochs=args.final_epochs,
        val_pairs=DEFAULT_VAL_PAIRS,
        model_path=args.best_model_path,
        target_bleu=args.target_bleu,
        report_to_ray=False,
        show_progress=True,
    )
    payload = asdict(summary)
    write_json("artifacts/assignment4/final_metrics.json", payload)
    save_vocabs(en_vocab, hi_vocab, DEFAULT_VOCAB_DIR)
    return payload


def format_metric(value: Optional[float], percent: bool = False) -> str:
    if value is None:
        return "Pending"
    if percent:
        return f"{value * 100:.2f}%" if value <= 1.0 else f"{value:.2f}%"
    return f"{value:.4f}"


def render_report_markdown(summary: Dict[str, Any], report_path: str) -> None:
    baseline = summary.get("baseline") or NOTEBOOK_BASELINE
    tuning = summary.get("tuning")
    best_model = summary.get("best_model")

    best_config = tuning.get("best_config") if tuning else None
    best_metrics = tuning.get("best_metrics") if tuning else None

    tuned_time = best_model.get("training_time_minutes") if best_model else None
    tuned_loss = best_model.get("final_loss") if best_model else None
    tuned_bleu = best_model.get("best_bleu") if best_model else None
    epochs_to_target = best_model.get("epochs_to_target") if best_model else None

    lines = [
        f"# {ROLLNO.upper()} Assignment 4 Report",
        "",
        "## Repository Links",
        f"- GitHub Repository: {GITHUB_REPO_URL}",
        f"- Hugging Face Model Repository: {HF_MODEL_REPO_URL}",
        "",
        "## Baseline Metrics",
        f"- Training Time (100 epochs): {baseline.get('training_time_minutes', NOTEBOOK_BASELINE['training_time_minutes']):.2f} minutes",
        f"- Final Loss: {baseline.get('final_loss', NOTEBOOK_BASELINE['final_loss']):.4f}",
        f"- BLEU Score (after 100 epochs): {baseline.get('bleu_percent', NOTEBOOK_BASELINE['bleu_percent']):.2f}",
        "",
        "## Hyperparameters Tuned",
        "| Hyperparameter | Search Range |",
        "| --- | --- |",
    ]

    for name, search_range in SEARCH_SPACE_DESCRIPTION.items():
        lines.append(f"| {name} | {search_range} |")

    lines.extend(
        [
            "",
            "## Best Configuration Found",
            f"- Configuration: `{json.dumps(best_config, ensure_ascii=False)}`" if best_config else "- Configuration: Pending tuning run",
            (
                f"- Sweep size used for current run: {tuning.get('num_samples')} trials x {tuning.get('tune_epochs')} epochs"
                if tuning
                else "- Sweep size used for current run: Pending tuning run"
            ),
            f"- Best trial BLEU: {best_metrics.get('bleu', 0.0) * 100:.2f}" if best_metrics and "bleu" in best_metrics else "- Best trial BLEU: Pending tuning run",
            f"- Best trial loss: {best_metrics.get('loss', 0.0):.4f}" if best_metrics and "loss" in best_metrics else "- Best trial loss: Pending tuning run",
            "",
            "## Final Metrics of Best Model",
            f"- Training Time: {tuned_time:.2f} minutes" if tuned_time is not None else "- Training Time: Pending final best-model training",
            f"- Final Loss: {tuned_loss:.4f}" if tuned_loss is not None else "- Final Loss: Pending final best-model training",
            f"- BLEU Score: {tuned_bleu * 100:.2f}" if tuned_bleu is not None else "- BLEU Score: Pending final best-model training",
            (
                f"- Epochs required to match/beat baseline: {epochs_to_target}"
                if epochs_to_target is not None
                else (
                    f"- Epochs required to match/beat baseline: Not reached within {best_model.get('epochs')} epochs"
                    if best_model
                    else "- Epochs required to match/beat baseline: Pending final best-model training"
                )
            ),
            "",
            "## Notes",
            f"- Baseline reference source: {baseline.get('source', NOTEBOOK_BASELINE['source'])}",
            "- Ray Tune reports both `loss` and `bleu` each epoch via `tune.report(...)` for compatibility with the installed Ray version.",
            "- ASHA stops weak trials early to reduce compute cost during the Optuna-guided sweep.",
        ]
    )

    if tuning and tuning.get("recovered_from_partial_sweep"):
        lines.append(f"- Recovery note: {tuning.get('recovery_note')}")
        lines.append(f"- Ray status counts at recovery time: `{json.dumps(tuning.get('ray_status_counts', {}), ensure_ascii=False)}`")

    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_payload(
    baseline_payload: Optional[Dict[str, Any]],
    tuning_payload: Optional[Dict[str, Any]],
    best_model_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_for_report = (
        baseline_payload
        if baseline_payload and baseline_payload.get("epochs") == NOTEBOOK_BASELINE["epochs"]
        else NOTEBOOK_BASELINE
    )
    return {
        "baseline": baseline_for_report,
        "tuning": tuning_payload,
        "best_model": best_model_payload,
        "search_space": SEARCH_SPACE_DESCRIPTION,
        "artifacts": {
            "baseline_model": DEFAULT_BASELINE_MODEL_PATH,
            "best_model": DEFAULT_BEST_MODEL_PATH,
            "report_markdown": DEFAULT_REPORT_MD_PATH,
            "summary_json": DEFAULT_SUMMARY_JSON,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["baseline", "tune", "final", "report", "recover", "all"], default="all")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--tune-epochs", type=int, default=20)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-bleu", type=float, default=NOTEBOOK_BASELINE["bleu"])
    parser.add_argument("--cpus-per-trial", type=float, default=2.0)
    parser.add_argument("--gpus-per-trial", type=float, default=1.0 if torch is not None and torch.cuda.is_available() else 0.0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--baseline-model-path", default=DEFAULT_BASELINE_MODEL_PATH)
    parser.add_argument("--best-model-path", default=DEFAULT_BEST_MODEL_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_MD_PATH)
    parser.add_argument("--summary-json-path", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--tuning-json-path", default=DEFAULT_TUNING_JSON)
    parser.add_argument("--best-config-path", default=DEFAULT_BEST_CONFIG_JSON)
    parser.add_argument("--ray-results-dir", default="artifacts/assignment4/ray_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_payload: Optional[Dict[str, Any]] = None
    tuning_payload: Optional[Dict[str, Any]] = None
    best_model_payload: Optional[Dict[str, Any]] = None

    if args.action in {"baseline", "all", "final"}:
        require_base_dependencies()
        df = load_translation_dataframe(args.data_path)
        en_vocab, hi_vocab = build_vocabs(df)
    else:
        df = None
        en_vocab = None
        hi_vocab = None

    if args.action in {"baseline", "all"}:
        baseline_payload = run_baseline(args, df, en_vocab, hi_vocab)
    elif args.action == "report":
        baseline_payload = load_json_if_exists("artifacts/assignment4/baseline_metrics.json") or NOTEBOOK_BASELINE

    if args.action in {"tune", "all"}:
        tuning_payload = run_tuning(args)
    elif args.action == "recover":
        tuning_payload = recover_tuning_from_ray_results(args)
    elif args.action in {"final", "report"}:
        tuning_payload = load_json_if_exists(args.tuning_json_path)

    if args.action in {"final", "all"}:
        if df is None or en_vocab is None or hi_vocab is None:
            require_base_dependencies()
            df = load_translation_dataframe(args.data_path)
            en_vocab, hi_vocab = build_vocabs(df)
        best_model_payload = run_best_model_training(args, df, en_vocab, hi_vocab)
    elif args.action == "report":
        best_model_payload = load_json_if_exists("artifacts/assignment4/final_metrics.json")

    summary_payload = build_summary_payload(baseline_payload, tuning_payload, best_model_payload)
    write_json(args.summary_json_path, summary_payload)
    render_report_markdown(summary_payload, args.report_path)

    print(f"Summary JSON written to {args.summary_json_path}")
    print(f"Report markdown written to {args.report_path}")
    if args.action in {"baseline", "all"}:
        print(f"Baseline model saved to {args.baseline_model_path}")
    if args.action in {"final", "all"}:
        print(f"Best model saved to {args.best_model_path}")


if __name__ == "__main__":
    main()
