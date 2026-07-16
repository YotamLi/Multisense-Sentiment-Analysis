"""Load a checkpoint and run one pure-BERT four-task prediction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import torch
from transformers import BertTokenizer

from src.models.checkpoint import load_checkpoint, validate_checkpoint_compatibility
from src.models.multi_head_bert import MultiHeadBERT
from src.utils.helpers import load_config


@click.command()
@click.option("--config", default="config/config.yaml", show_default=True)
@click.option("--checkpoint", required=True)
@click.option(
    "--text",
    default="The movie was absolutely brilliant! Best film I have seen this year.",
    show_default=True,
)
def main(config: str, checkpoint: str, text: str):
    cfg = load_config(config)
    model_cfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_checkpoint(checkpoint, map_location=device)
    validate_checkpoint_compatibility(loaded, cfg)

    tokenizer = BertTokenizer.from_pretrained(model_cfg["encoder_name"])
    model = MultiHeadBERT(
        encoder_name=model_cfg["encoder_name"],
        num_sentiment_classes=model_cfg["heads"]["sentiment"]["num_classes"],
        num_emotion_classes=model_cfg["heads"]["emotion"]["num_classes"],
        num_intensity_classes=model_cfg["heads"]["intensity"]["num_classes"],
        num_topic_classes=model_cfg["heads"]["topic"]["num_classes"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        loss_weights=cfg["training"]["loss_weights"],
    )
    model.load_state_dict(loaded["model_state_dict"])
    model.to(device).eval()
    encoded = tokenizer(
        text,
        max_length=model_cfg["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    predictions = model.predict(**encoded)

    print(f"Device: {device}")
    print(f"Input: {text}\n")
    for task, result in predictions.items():
        labels = model_cfg["heads"][task]["labels"]
        index = int(result["predicted_class"])
        print(f"{task.upper()}: {labels[index]} ({result['confidence']:.2%})")


if __name__ == "__main__":
    main()
