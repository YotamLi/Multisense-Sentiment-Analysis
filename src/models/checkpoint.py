"""Checkpoint loading and compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


HEAD_WEIGHT_KEYS = {
    "sentiment": "heads.sentiment.classifier.4.weight",
    "emotion": "heads.emotion.classifier.4.weight",
    "intensity": "heads.intensity.classifier.4.weight",
    "topic": "heads.topic.classifier.4.weight",
}


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded

    if isinstance(loaded, dict):
        return {
            "format_version": 0,
            "model_state_dict": loaded,
        }

    raise TypeError(
        f"Unsupported checkpoint type {type(loaded)!r} in {checkpoint_path}"
    )


def infer_head_sizes(checkpoint: dict) -> dict[str, int]:
    state_dict = checkpoint["model_state_dict"]
    sizes: dict[str, int] = {}

    for task, key in HEAD_WEIGHT_KEYS.items():
        tensor = state_dict.get(key)
        if tensor is not None and getattr(tensor, "ndim", 0) >= 1:
            sizes[task] = int(tensor.shape[0])

    return sizes


def configured_head_sizes(config: dict) -> dict[str, int]:
    heads = config.get("model", {}).get("heads", {})
    return {
        task: int(task_config["num_classes"])
        for task, task_config in heads.items()
    }


def validate_checkpoint_compatibility(
    checkpoint: dict,
    config: dict,
) -> None:
    actual = infer_head_sizes(checkpoint)
    expected = configured_head_sizes(config)

    mismatches = []
    for task, expected_size in expected.items():
        actual_size = actual.get(task)
        if actual_size is not None and actual_size != expected_size:
            mismatches.append(
                f"{task}: checkpoint={actual_size}, config={expected_size}"
            )

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            "Checkpoint/config class-count mismatch. "
            f"{joined}. Use a configuration whose class counts and "
            "label maps match the checkpoint metadata."
        )
