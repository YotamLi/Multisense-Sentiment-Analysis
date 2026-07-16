"""
Single-head BERT sentiment baseline.

Fine-tunes a plain `bert-base-uncased` with one 3-class classification head
on the sentiment-labelled rows of the processed corpus. Used as a direct
point of comparison against VADER, Naive Bayes, and the Multi-Head BERT
model so we can measure the incremental value of multi-task learning.

Public API:
    SentimentDataset
    train_bert_sentiment
    eval_bert_sentiment
    extract_sentiment_rows
    load_or_train_bert_sentiment
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.preprocessor import TextPreprocessor

from .metrics import compute_metrics

SENTIMENT_LABELS = ["positive", "negative", "neutral"]


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def extract_sentiment_rows(
    df: pd.DataFrame,
    preprocessor: TextPreprocessor,
) -> Tuple[list[str], list[int]]:
    """Filter rows with a valid sentiment label and clean the text."""
    mask = df["sentiment"].astype(int) != -1
    sub = df.loc[mask]
    texts = [preprocessor.clean(str(t)) for t in sub["text"].tolist()]
    labels = [int(x) for x in sub["sentiment"].tolist()]
    return texts, labels


def train_bert_sentiment(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    device: torch.device,
    encoder_name: str = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
):
    """Fine-tune a single-head BERT classifier. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_name, num_labels=3
    ).to(device)

    train_loader = DataLoader(
        SentimentDataset(train_texts, train_labels, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SentimentDataset(val_texts, val_labels, tokenizer, max_length),
        batch_size=batch_size,
    )

    total_steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"]
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                trues.extend(labels.tolist())

        val_metrics = compute_metrics(trues, preds, label_names=SENTIMENT_LABELS)
        logger.info(
            f"[BERT-sentiment] epoch {epoch} "
            f"train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macroF1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer


@torch.no_grad()
def eval_bert_sentiment(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
) -> dict:
    """Run the fine-tuned sentiment model on (texts, labels) and return metrics."""
    model.eval()
    loader = DataLoader(
        SentimentDataset(texts, labels, tokenizer, max_length),
        batch_size=batch_size,
    )
    preds, trues = [], []
    for batch in loader:
        y = batch["labels"]
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        trues.extend(y.tolist())
    return compute_metrics(trues, preds, label_names=SENTIMENT_LABELS)


def load_or_train_bert_sentiment(
    bert_dir: Path,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    device: torch.device,
    encoder_name: str = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    epochs: int = 3,
    force_retrain: bool = False,
):
    """Load a fine-tuned sentiment model from disk if present, else train and save."""
    bert_dir = Path(bert_dir)
    weights_present = (
        (bert_dir / "pytorch_model.bin").exists()
        or (bert_dir / "model.safetensors").exists()
    )
    if weights_present and not force_retrain:
        logger.info(f"Loading fine-tuned BERT-sentiment from {bert_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(bert_dir)
        ).to(device)
        return model, tokenizer

    logger.info(f"Fine-tuning {encoder_name} on sentiment only")
    model, tokenizer = train_bert_sentiment(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        device,
        encoder_name=encoder_name,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
    )
    bert_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(bert_dir))
    tokenizer.save_pretrained(str(bert_dir))
    logger.info(f"Saved fine-tuned BERT-sentiment to {bert_dir}")
    return model, tokenizer
