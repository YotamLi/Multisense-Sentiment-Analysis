"""PyTorch datasets for heterogeneous multi-task labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from .preprocessor import TextPreprocessor


class MultiTaskDataset(Dataset):
    TASK_NAMES = ("sentiment", "emotion", "intensity", "topic")

    def __init__(
        self,
        data: pd.DataFrame | str,
        tokenizer: BertTokenizer | None = None,
        preprocessor: TextPreprocessor | None = None,
        max_length: int = 128,
        label_maps: dict[str, dict[str, int]] | None = None,
        encoder_name: str = "bert-base-uncased",
    ):
        self.df = pd.read_csv(data) if isinstance(data, (str, Path)) else data.copy()
        if "text" not in self.df.columns:
            raise ValueError("Dataset requires a text column")
        self.max_length = max_length
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(encoder_name)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.label_maps = label_maps or {}

    def __len__(self) -> int:
        return len(self.df)

    def _encode_label(self, task: str, value) -> int:
        if value is None or pd.isna(value):
            return -1

        # Numeric labels in prepared CSV files are already canonical IDs.
        try:
            numeric = int(float(value))
            return numeric
        except (TypeError, ValueError):
            pass

        mapping = self.label_maps.get(task, {})
        key = str(value)
        if key not in mapping:
            raise ValueError(f"Unknown string label for {task}: {value!r}")
        return int(mapping[key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        text = self.preprocessor.clean(str(row["text"]))
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        for task in self.TASK_NAMES:
            item[f"{task}_labels"] = torch.tensor(
                self._encode_label(task, row.get(task, -1)),
                dtype=torch.long,
            )
        return item


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    batch_size: int = 16,
    max_length: int = 128,
    encoder_name: str = "bert-base-uncased",
    num_workers: int = 0,
    label_maps: dict[str, dict[str, int]] | None = None,
) -> dict[str, DataLoader]:
    tokenizer = BertTokenizer.from_pretrained(encoder_name)
    preprocessor = TextPreprocessor()

    def collate(batch):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch]) for key in keys}

    def dataset(frame: pd.DataFrame) -> MultiTaskDataset:
        return MultiTaskDataset(
            frame,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            max_length=max_length,
            label_maps=label_maps,
            encoder_name=encoder_name,
        )

    loaders = {
        "train": DataLoader(
            dataset(train_df),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            dataset(val_df),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    if test_df is not None:
        loaders["test"] = DataLoader(
            dataset(test_df),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
