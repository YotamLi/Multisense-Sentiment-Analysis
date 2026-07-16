"""Audit duplicate texts, leakage and label distributions in dataset splits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import click
import pandas as pd

TASK_COLUMNS = ("sentiment", "emotion", "intensity", "topic")


def normalise_text(text: object) -> str:
    value = str(text or "").casefold().strip()
    value = re.sub(r"https?://\S+|www\.\S+", "<url>", value)
    value = re.sub(r"@\w+", "<user>", value)
    return re.sub(r"\s+", " ", value).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} file: {path}")
    frame = pd.read_csv(path).copy()
    if "text" not in frame.columns:
        raise ValueError(f"{path.name} has no text column")
    frame["split"] = split
    if "text_hash" not in frame.columns:
        frame["normalised_text"] = frame["text"].map(normalise_text)
        frame["text_hash"] = frame["normalised_text"].map(text_hash)
    return frame


def valid_count(frame: pd.DataFrame, task: str) -> int:
    if task not in frame.columns:
        return 0
    values = pd.to_numeric(frame[task], errors="coerce")
    return int((values.notna() & values.ne(-1)).sum())


def distribution(frame: pd.DataFrame, task: str) -> dict[str, int]:
    if task not in frame.columns:
        return {}
    values = pd.to_numeric(frame[task], errors="coerce")
    values = values[values.notna() & values.ne(-1)].astype(int)
    return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}


@click.command()
@click.option("--data-dir", default="data/processed_v2", show_default=True)
@click.option("--output", default="reports/dataset_audit_v2.json", show_default=True)
def main(data_dir: str, output: str):
    data_path = Path(data_dir)
    frames = {
        split: load_split(data_path / f"{split}.csv", split)
        for split in ("train", "val", "test")
    }
    report: dict = {"files": {}, "cross_split_overlap": {}}

    for split, frame in frames.items():
        duplicate_mask = frame.duplicated("text_hash", keep=False)
        report["files"][split] = {
            "rows": int(len(frame)),
            "unique_texts": int(frame["text_hash"].nunique()),
            "duplicate_rows": int(duplicate_mask.sum()),
            "duplicate_groups": int(frame.loc[duplicate_mask, "text_hash"].nunique()),
            "valid_task_labels": {
                task: valid_count(frame, task) for task in TASK_COLUMNS
            },
            "label_distribution": {
                task: distribution(frame, task) for task in TASK_COLUMNS
            },
        }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frames[left]["text_hash"]) & set(frames[right]["text_hash"])
        report["cross_split_overlap"][f"{left}_vs_{right}"] = len(overlap)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)
    for split, details in report["files"].items():
        print(f"\n{split.upper()}: {details['rows']:,} rows")
        print(f"  unique texts: {details['unique_texts']:,}")
        print(f"  duplicate rows: {details['duplicate_rows']:,}")
        for task, count in details["valid_task_labels"].items():
            print(f"  {task}: {count:,} valid labels")
    print("\nCROSS-SPLIT OVERLAP:")
    for name, count in report["cross_split_overlap"].items():
        print(f"  {name}: {count}")
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
