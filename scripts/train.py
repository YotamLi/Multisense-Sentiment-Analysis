"""Train Multi-Head BERT using a selected YAML configuration."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger

from src.data.dataset import build_dataloaders
from src.models.multi_head_bert import MultiHeadBERT
from src.models.trainer import MultiTaskTrainer
from src.utils.helpers import get_device, load_config, set_seed, setup_logger


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@click.command()
@click.option("--config", default="config/config.yaml", show_default=True)
@click.option("--epochs", default=None, type=int)
@click.option("--batch-size", "batch_size", default=None, type=int)
@click.option("--output-dir", default="checkpoints", show_default=True)
def main(
    config: str,
    epochs: int | None,
    batch_size: int | None,
    output_dir: str,
):
    setup_logger()
    cfg = load_config(config)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = get_device()

    training_config = dict(cfg["training"])
    if epochs is not None:
        training_config["epochs"] = epochs
    if batch_size is not None:
        training_config["batch_size"] = batch_size

    model_config = cfg["model"]
    data_config = cfg["data"]
    processed = Path(data_config["processed_dir"])
    train_csv = processed / "train.csv"
    val_csv = processed / "val.csv"
    test_csv = processed / "test.csv"
    missing = [path for path in (train_csv, val_csv, test_csv) if not path.is_file()]
    if missing:
        raise click.ClickException(
            "Missing processed data: " + ", ".join(map(str, missing))
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    label_maps = {
        task: {label: index for index, label in enumerate(task_config["labels"])}
        for task, task_config in model_config["heads"].items()
    }
    loaders = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=int(training_config["batch_size"]),
        max_length=int(model_config["max_length"]),
        encoder_name=model_config["encoder_name"],
        label_maps=label_maps,
    )

    model = MultiHeadBERT(
        encoder_name=model_config["encoder_name"],
        num_sentiment_classes=model_config["heads"]["sentiment"]["num_classes"],
        num_emotion_classes=model_config["heads"]["emotion"]["num_classes"],
        num_intensity_classes=model_config["heads"]["intensity"]["num_classes"],
        num_topic_classes=model_config["heads"]["topic"]["num_classes"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
        loss_weights=training_config["loss_weights"],
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Model parameters: {total_parameters:,}")

    metadata = {
        "model_config": model_config,
        "label_maps": {
            task: task_config["labels"]
            for task, task_config in model_config["heads"].items()
        },
        "data_config": data_config,
        "seed": seed,
        "data_fingerprint": {
            "train_sha256": file_sha256(train_csv),
            "val_sha256": file_sha256(val_csv),
            "test_sha256": file_sha256(test_csv),
        },
        "config_path": str(Path(config).resolve()),
    }
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=training_config,
        device=device,
        output_dir=output_dir,
        metadata=metadata,
    )
    history = trainer.train()

    history_path = Path(output_dir) / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Training history written to: {history_path}")


if __name__ == "__main__":
    main()
