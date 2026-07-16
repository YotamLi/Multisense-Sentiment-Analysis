"""Build a provenance-aware, leakage-safe multi-source V2 dataset."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger

from src.utils.helpers import load_config, setup_logger

MISSING = -1
TASK_COLUMNS = ("sentiment", "emotion", "intensity", "topic")
EMOTION_LABEL_MAP = {
    "joy": 0,
    "anger": 1,
    "sadness": 2,
    "fear": 3,
    "surprise": 4,
    "disgust": 5,
}
VALIDATION_SPLIT_NAMES = {"validation", "valid", "val", "dev"}
TEST_SPLIT_NAMES = {"test"}
SOURCES_WITHOUT_VALIDATION = {"tweet_topic_single"}


def _cap_by_split(frame: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Apply a deterministic cap while retaining every available source split."""
    if max_samples <= 0 or len(frame) <= max_samples:
        return frame.reset_index(drop=True)

    grouped = list(frame.groupby("source_split", sort=True))
    allocations: dict[str, int] = {}
    allocated = 0
    for split_name, group in grouped:
        quota = max(1, int(max_samples * len(group) / len(frame)))
        quota = min(quota, len(group))
        allocations[str(split_name)] = quota
        allocated += quota

    # Allocate remaining capacity to the largest groups with unused rows.
    while allocated < max_samples:
        changed = False
        for split_name, group in sorted(grouped, key=lambda pair: len(pair[1]), reverse=True):
            key = str(split_name)
            if allocations[key] < len(group):
                allocations[key] += 1
                allocated += 1
                changed = True
                if allocated >= max_samples:
                    break
        if not changed:
            break

    sampled = []
    for index, (split_name, group) in enumerate(grouped):
        sampled.append(
            group.sample(
                n=allocations[str(split_name)],
                random_state=42 + index,
            )
        )
    return (
        pd.concat(sampled, ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


def load_sentiment_source(max_samples: int = 20_000) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tweet_eval_to_project = {0: 1, 1: 2, 2: 0}
    rows = []
    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        for item in dataset[split]:
            rows.append(
                {
                    "text": item["text"],
                    "sentiment": tweet_eval_to_project[int(item["label"])],
                    "emotion": MISSING,
                    "intensity": MISSING,
                    "topic": MISSING,
                    "source": "tweet_eval_sentiment",
                    "source_split": split,
                }
            )
    frame = _cap_by_split(pd.DataFrame(rows), max_samples)
    logger.info(
        f"Sentiment source: {len(frame)} samples "
        f"{frame['source_split'].value_counts().to_dict()}"
    )
    return frame


def load_emotion_source(max_samples: int = 20_000) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
    )
    fallback_names = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness",
        "surprise", "neutral",
    ]
    feature = dataset["train"].features["labels"]
    label_names = getattr(getattr(feature, "feature", None), "names", None)
    label_names = label_names or fallback_names
    mapping = {
        "admiration": "joy", "amusement": "joy", "approval": "joy",
        "caring": "joy", "desire": "joy", "excitement": "joy",
        "gratitude": "joy", "joy": "joy", "love": "joy",
        "optimism": "joy", "pride": "joy", "relief": "joy",
        "anger": "anger", "annoyance": "anger", "disapproval": "anger",
        "sadness": "sadness", "disappointment": "sadness",
        "embarrassment": "sadness", "grief": "sadness", "remorse": "sadness",
        "fear": "fear", "nervousness": "fear",
        "confusion": "surprise", "curiosity": "surprise",
        "realization": "surprise", "surprise": "surprise",
        "disgust": "disgust",
    }

    rows = []
    stats = {"empty": 0, "neutral_or_unmapped": 0, "ambiguous": 0, "invalid": 0}
    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        for item in dataset[split]:
            label_ids = item.get("labels", [])
            if not label_ids:
                stats["empty"] += 1
                continue
            mapped = set()
            for label_id in label_ids:
                if not isinstance(label_id, int) or not 0 <= label_id < len(label_names):
                    stats["invalid"] += 1
                    continue
                basic = mapping.get(label_names[label_id])
                if basic is not None:
                    mapped.add(basic)
            if not mapped:
                stats["neutral_or_unmapped"] += 1
                continue
            if len(mapped) > 1:
                stats["ambiguous"] += 1
                continue
            emotion = next(iter(mapped))
            rows.append(
                {
                    "text": item["text"],
                    "sentiment": MISSING,
                    "emotion": EMOTION_LABEL_MAP[emotion],
                    "intensity": MISSING,
                    "topic": MISSING,
                    "source": "go_emotions",
                    "source_split": split,
                }
            )

    frame = _cap_by_split(pd.DataFrame(rows), max_samples)
    logger.info(
        f"Emotion source: {len(frame)} retained; filtering={stats}; "
        f"splits={frame['source_split'].value_counts().to_dict()}"
    )
    return frame


def load_intensity_source(max_samples: int = 20_000) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    repository = "vgaraujov/semeval-2025-task11-track-b"
    emotion_columns = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
    frames = []
    for split in ("train", "dev", "test"):
        filename = f"eng/{split}-00000-of-00001.parquet"
        path = hf_hub_download(
            repo_id=repository,
            filename=filename,
            repo_type="dataset",
        )
        frame = pd.read_parquet(path)
        frame["source_split"] = split
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    maximum_score = raw[list(emotion_columns)].fillna(0).astype(int).max(axis=1)
    keep = maximum_score >= 1
    score_to_label = {3: 0, 2: 1, 1: 2}
    rows = [
        {
            "text": str(text),
            "sentiment": MISSING,
            "emotion": MISSING,
            "intensity": score_to_label[int(score)],
            "topic": MISSING,
            "source": "semeval_2025_intensity",
            "source_split": str(split),
        }
        for text, score, split in zip(
            raw.loc[keep, "text"],
            maximum_score[keep],
            raw.loc[keep, "source_split"],
        )
    ]
    frame = _cap_by_split(pd.DataFrame(rows), max_samples)
    logger.info(
        f"Intensity source: {len(frame)} samples "
        f"{frame['source_split'].value_counts().to_dict()}"
    )
    return frame


def _infer_topic_split(filename: str) -> str:
    lowered = filename.lower()
    if "train" in lowered:
        return "train"
    if any(token in lowered for token in ("validation", "valid", "dev")):
        return "validation"
    if "test" in lowered:
        return "test"
    return "unknown"


def load_topic_source(max_samples: int = 20_000) -> pd.DataFrame:
    from huggingface_hub import HfApi, hf_hub_download

    repository = "cardiffnlp/tweet_topic_single"
    filenames = HfApi().list_repo_files(
        repo_id=repository,
        repo_type="dataset",
    )
    data_files = [
        filename
        for filename in filenames
        if filename.endswith((".jsonl", ".json"))
        and not filename.endswith(("dataset_infos.json", "label.json", "label2id.json"))
    ]
    if not data_files:
        raise RuntimeError(f"No JSON data files found in {repository}")

    def records(path: Path):
        with path.open("r", encoding="utf-8") as handle:
            first = handle.read(1)
            if not first:
                return
            handle.seek(0)
            if first == "[":
                for item in json.load(handle):
                    yield item
                return
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    rows = []
    for filename in data_files:
        local_path = Path(
            hf_hub_download(
                repo_id=repository,
                filename=filename,
                repo_type="dataset",
            )
        )
        source_split = _infer_topic_split(filename)
        for item in records(local_path):
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            label = item.get("label")
            if text is None or label is None:
                continue
            if isinstance(label, list):
                if not label:
                    continue
                label = label[0]
            rows.append(
                {
                    "text": str(text),
                    "sentiment": MISSING,
                    "emotion": MISSING,
                    "intensity": MISSING,
                    "topic": int(label),
                    "source": "tweet_topic_single",
                    "source_split": source_split,
                }
            )

    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["text", "topic", "source_split"]
    )
    frame = _cap_by_split(frame, max_samples)
    logger.info(
        f"Topic source: {len(frame)} samples "
        f"{frame['source_split'].value_counts().to_dict()}"
    )
    return frame


def normalise_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    value = str(text).casefold().strip()
    value = re.sub(r"https?://\S+|www\.\S+", "<url>", value)
    value = re.sub(r"@\w+", "<user>", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def create_text_hash(normalised: str) -> str:
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def merge_duplicate_texts(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["text"] = working["text"].fillna("").astype(str).str.strip()
    working["normalised_text"] = working["text"].map(normalise_text)
    working = working[working["normalised_text"] != ""].copy()
    working["text_hash"] = working["normalised_text"].map(create_text_hash)
    for task in TASK_COLUMNS:
        working[task] = pd.to_numeric(
            working[task], errors="coerce"
        ).fillna(MISSING).astype(int)

    merged_rows = []
    conflicts = {task: 0 for task in TASK_COLUMNS}
    for text_hash, group in working.groupby("text_hash", sort=False):
        representative = max(group["text"].tolist(), key=len)
        provenance = sorted(
            f"{row.source}:{row.source_split}"
            for row in group.itertuples()
        )
        row = {
            "text": representative,
            "normalised_text": group["normalised_text"].iloc[0],
            "text_hash": text_hash,
            "source": "|".join(sorted(set(group["source"].astype(str)))),
            "source_split": "|".join(sorted(set(group["source_split"].astype(str)))),
            "provenance": "|".join(sorted(set(provenance))),
        }
        for task in TASK_COLUMNS:
            labels = sorted(
                {int(value) for value in group[task] if int(value) != MISSING}
            )
            if len(labels) == 1:
                row[task] = labels[0]
            elif len(labels) == 0:
                row[task] = MISSING
            else:
                row[task] = MISSING
                conflicts[task] += 1
        if any(row[task] != MISSING for task in TASK_COLUMNS):
            merged_rows.append(row)

    merged = pd.DataFrame(merged_rows)
    logger.info(
        f"Merged duplicate texts: {len(working)} rows -> {len(merged)} unique texts"
    )
    logger.info(f"Conflicting labels marked missing: {conflicts}")
    return merged


def hash_fraction(text_hash: str) -> float:
    return int(text_hash[:16], 16) / float(16**16)


def parse_provenance(provenance: object) -> list[tuple[str, str]]:
    entries = []
    for item in str(provenance or "").split("|"):
        if ":" not in item:
            continue
        source, split = item.rsplit(":", 1)
        entries.append((source.strip().lower(), split.strip().lower()))
    return entries


def assign_project_split(
    row: pd.Series,
    val_ratio: float,
    test_ratio: float,
) -> str:
    provenance = parse_provenance(row.get("provenance", ""))
    sources = {source for source, _ in provenance}
    source_splits = {split for _, split in provenance}

    # Most conservative priority: any official test occurrence makes the
    # merged text test-only; validation is second; training is last.
    if source_splits & TEST_SPLIT_NAMES:
        return "test"
    if source_splits & VALIDATION_SPLIT_NAMES:
        return "val"

    if sources & SOURCES_WITHOUT_VALIDATION:
        if hash_fraction(str(row["text_hash"])) < val_ratio:
            return "val"

    # Unknown/train-only sources can be divided deterministically.
    if source_splits <= {"train", "unknown"} and "unknown" in source_splits:
        fraction = hash_fraction(str(row["text_hash"]))
        if fraction < test_ratio:
            return "test"
        if fraction < test_ratio + val_ratio:
            return "val"

    return "train"


def unify_and_export(
    dataframes: list[pd.DataFrame],
    output_directory: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> None:
    combined = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Raw combined rows: {len(combined)}")
    combined = merge_duplicate_texts(combined)
    combined["project_split"] = combined.apply(
        assign_project_split,
        axis=1,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    split_frames = {
        split: (
            combined[combined["project_split"] == split]
            .drop(columns=["project_split"])
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        for split in ("train", "val", "test")
    }
    hash_sets = {
        split: set(frame["text_hash"])
        for split, frame in split_frames.items()
    }
    assert not hash_sets["train"] & hash_sets["val"]
    assert not hash_sets["train"] & hash_sets["test"]
    assert not hash_sets["val"] & hash_sets["test"]

    output_directory.mkdir(parents=True, exist_ok=True)
    for split, frame in split_frames.items():
        frame.to_csv(output_directory / f"{split}.csv", index=False)
        logger.info(
            f"{split}: {len(frame)} rows, {frame['text_hash'].nunique()} unique texts"
        )
        for task in TASK_COLUMNS:
            logger.info(
                f"  {task}: {int(frame[task].ne(MISSING).sum())} valid labels"
            )


@click.command()
@click.option("--config", default="config/config.yaml", show_default=True)
@click.option("--max-per-source", default=20_000, type=int, show_default=True)
def main(config: str, max_per_source: int):
    setup_logger()
    configuration = load_config(config)
    data_config = configuration["data"]
    output_directory = Path(data_config["processed_dir"])
    unify_and_export(
        [
            load_sentiment_source(max_per_source),
            load_emotion_source(max_per_source),
            load_intensity_source(max_per_source),
            load_topic_source(max_per_source),
        ],
        output_directory,
        val_ratio=float(data_config.get("val_split", 0.1)),
        test_ratio=float(data_config.get("test_split", 0.1)),
    )


if __name__ == "__main__":
    main()
