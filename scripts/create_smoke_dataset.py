"""Create a small class-balanced dataset for one-epoch smoke training."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

TASKS = ("sentiment", "emotion", "intensity", "topic")


def balanced_subset(frame: pd.DataFrame, samples_per_class: int, split: str) -> pd.DataFrame:
    selected = []
    for task_index, task in enumerate(TASKS):
        values = pd.to_numeric(frame[task], errors="coerce").fillna(-1).astype(int)
        valid = frame[values.ne(-1)].copy()
        labels = sorted(valid[task].astype(int).unique())
        if not labels:
            raise ValueError(f"No valid {task} labels found in {split}")
        for label in labels:
            rows = valid[valid[task].astype(int).eq(label)]
            selected.append(
                rows.sample(
                    n=min(samples_per_class, len(rows)),
                    random_state=42 + task_index * 100 + int(label),
                )
            )
    return (
        pd.concat(selected, ignore_index=True)
        .drop_duplicates(subset=["text_hash"] if "text_hash" in frame.columns else ["text"])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


@click.command()
@click.option("--source-dir", default="data/processed_v2", show_default=True)
@click.option("--output-dir", default="data/smoke_v2", show_default=True)
@click.option("--train-per-class", default=32, show_default=True)
@click.option("--eval-per-class", default=12, show_default=True)
def main(source_dir: str, output_dir: str, train_per_class: int, eval_per_class: int):
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        frame = pd.read_csv(source / f"{split}.csv")
        subset = balanced_subset(
            frame,
            train_per_class if split == "train" else eval_per_class,
            split,
        )
        subset.to_csv(output / f"{split}.csv", index=False)
        print(f"{split.upper()}: {len(subset):,} rows")
        for task in TASKS:
            dist = (
                subset.loc[subset[task].ne(-1), task]
                .astype(int)
                .value_counts()
                .sort_index()
                .to_dict()
            )
            print(f"  {task}: {dist}")


if __name__ == "__main__":
    main()
