"""Strictly validate V2 dataset schema, labels and split isolation."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

TASK_RANGES = {
    "sentiment": (0, 2),
    "emotion": (0, 5),
    "intensity": (0, 2),
    "topic": (0, 5),
}
REQUIRED_V2_COLUMNS = {
    "text", "sentiment", "emotion", "intensity", "topic",
    "source", "source_split", "provenance", "normalised_text", "text_hash",
}


@click.command()
@click.option("--data-dir", default="data/processed_v2", show_default=True)
def main(data_dir: str):
    root = Path(data_dir)
    frames = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frames[split] = frame
        missing = REQUIRED_V2_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{split}.csv missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{split}.csv is empty")
        if frame["text"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{split}.csv contains empty text")
        if frame["text_hash"].duplicated().any():
            raise ValueError(f"{split}.csv contains duplicate text hashes")
        if frame[list(TASK_RANGES)].eq(-1).all(axis=1).any():
            raise ValueError(f"{split}.csv contains rows with all labels missing")

        for task, (minimum, maximum) in TASK_RANGES.items():
            values = pd.to_numeric(frame[task], errors="coerce")
            if values.isna().any():
                raise ValueError(f"{split}.{task} contains non-numeric labels")
            invalid = values.ne(-1) & ((values < minimum) | (values > maximum))
            if invalid.any():
                raise ValueError(
                    f"{split}.{task} invalid labels: {sorted(values[invalid].unique())}"
                )
            if values.ne(-1).sum() == 0:
                raise ValueError(f"{split}.{task} has no valid labels")

        print(f"{split.upper()}: {len(frame):,} rows")
        for task in TASK_RANGES:
            print(f"  {task}: {int(frame[task].ne(-1).sum()):,} valid labels")

    hashes = {split: set(frame["text_hash"]) for split, frame in frames.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hashes[left] & hashes[right]
        if overlap:
            raise ValueError(f"{left} vs {right}: {len(overlap)} overlapping texts")
        print(f"{left} vs {right}: 0 overlapping texts")

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
