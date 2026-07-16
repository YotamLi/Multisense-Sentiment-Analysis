"""Fair evaluation of VADER, Naive Bayes, single-head BERT and MTL BERT."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
import torch
from loguru import logger

from src.data.dataset import build_dataloaders
from src.data.preprocessor import TextPreprocessor
from src.evaluation.baseline import NaiveBayesBaseline, VADERBaseline
from src.evaluation.bert_sentiment import (
    SENTIMENT_LABELS,
    eval_bert_sentiment,
    extract_sentiment_rows,
    load_or_train_bert_sentiment,
)
from src.evaluation.metrics import compute_multitask_metrics
from src.models.checkpoint import load_checkpoint, validate_checkpoint_compatibility
from src.models.multi_head_bert import MultiHeadBERT
from src.utils.helpers import get_device, load_config, set_seed, setup_logger


def tune_naive_bayes(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
) -> float:
    candidates = (0.1, 0.5, 1.0, 2.0)
    best_alpha = candidates[0]
    best_f1 = float("-inf")
    for alpha in candidates:
        model = NaiveBayesBaseline(alpha=alpha)
        model.train(train_texts, train_labels)
        metrics = model.evaluate(
            val_texts,
            val_labels,
            label_names=SENTIMENT_LABELS,
        )
        logger.info(f"Naive Bayes alpha={alpha}: val F1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_alpha = alpha
    return best_alpha


@click.command()
@click.option("--config", default="config/config.yaml", show_default=True)
@click.option("--checkpoint", default=None)
@click.option("--output", default="evaluation_results_v2.json", show_default=True)
@click.option("--bert-dir", default="checkpoints/bert_sentiment_v2", show_default=True)
@click.option("--skip-bert-sentiment", is_flag=True)
@click.option("--force-bert-retrain", is_flag=True)
@click.option("--bert-epochs", default=3, type=int, show_default=True)
@click.option("--bert-batch-size", default=32, type=int, show_default=True)
def main(
    config: str,
    checkpoint: str | None,
    output: str,
    bert_dir: str,
    skip_bert_sentiment: bool,
    force_bert_retrain: bool,
    bert_epochs: int,
    bert_batch_size: int,
):
    setup_logger()
    cfg = load_config(config)
    set_seed(42)
    device = get_device()
    processed = Path(cfg["data"]["processed_dir"])
    paths = {split: processed / f"{split}.csv" for split in ("train", "val", "test")}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise click.ClickException("Missing data: " + ", ".join(map(str, missing)))

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])
    preprocessor = TextPreprocessor()
    train_texts, train_labels = extract_sentiment_rows(train_df, preprocessor)
    val_texts, val_labels = extract_sentiment_rows(val_df, preprocessor)
    test_texts, test_labels = extract_sentiment_rows(test_df, preprocessor)
    results: dict = {}

    logger.info("Evaluating VADER on the held-out test set")
    results["vader_baseline"] = VADERBaseline().evaluate(
        test_texts,
        test_labels,
    )

    logger.info("Tuning Naive Bayes on validation data")
    best_alpha = tune_naive_bayes(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
    )
    nb = NaiveBayesBaseline(alpha=best_alpha)
    nb.train(train_texts + val_texts, train_labels + val_labels)
    nb_metrics = nb.evaluate(
        test_texts,
        test_labels,
        label_names=SENTIMENT_LABELS,
    )
    nb_metrics["selected_alpha"] = best_alpha
    results["naive_bayes_baseline"] = nb_metrics

    if not skip_bert_sentiment:
        logger.info("Evaluating single-head BERT sentiment model")
        bert_model, tokenizer = load_or_train_bert_sentiment(
            bert_dir=Path(bert_dir),
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            device=device,
            encoder_name=cfg["model"]["encoder_name"],
            max_length=int(cfg["model"]["max_length"]),
            batch_size=bert_batch_size,
            epochs=bert_epochs,
            force_retrain=force_bert_retrain,
        )
        results["bert_sentiment"] = eval_bert_sentiment(
            bert_model,
            tokenizer,
            test_texts,
            test_labels,
            device,
            max_length=int(cfg["model"]["max_length"]),
        )

    if checkpoint:
        logger.info("Evaluating Multi-Head BERT on the held-out test set")
        loaded = load_checkpoint(checkpoint, map_location=device)
        validate_checkpoint_compatibility(loaded, cfg)
        model_cfg = cfg["model"]
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
        loaders = build_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=int(cfg["training"]["batch_size"]),
            max_length=int(model_cfg["max_length"]),
            encoder_name=model_cfg["encoder_name"],
        )
        task_predictions = {task: [] for task in MultiHeadBERT.TASK_NAMES}
        task_labels = {task: [] for task in MultiHeadBERT.TASK_NAMES}
        with torch.no_grad():
            for batch in loaders["test"]:
                device_batch = {key: value.to(device) for key, value in batch.items()}
                output_batch = model(**device_batch)
                logits_map = {
                    "sentiment": output_batch.sentiment_logits,
                    "emotion": output_batch.emotion_logits,
                    "intensity": output_batch.intensity_logits,
                    "topic": output_batch.topic_logits,
                }
                for task, logits in logits_map.items():
                    labels = batch[f"{task}_labels"]
                    predictions = logits.argmax(dim=-1).cpu()
                    valid = labels.ne(-1)
                    task_labels[task].extend(labels[valid].tolist())
                    task_predictions[task].extend(predictions[valid].tolist())

        label_maps = {
            task: task_config["labels"]
            for task, task_config in model_cfg["heads"].items()
        }
        results["multi_head_bert"] = compute_multitask_metrics(
            {
                task: (task_labels[task], task_predictions[task])
                for task in MultiHeadBERT.TASK_NAMES
                if task_labels[task]
            },
            label_maps,
        )
        results["multi_head_bert"]["checkpoint_metadata"] = {
            "format_version": loaded.get("format_version", "legacy"),
            "epoch": loaded.get("epoch"),
            "monitor_score": loaded.get("monitor_score"),
        }

    output_path = Path(output)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Evaluation results saved to {output_path}")

    visualizer = Path(__file__).resolve().parent / "visualize_results.py"
    subprocess.run(
        [sys.executable, str(visualizer), "--input", str(output_path)],
        cwd=str(Path(__file__).resolve().parent.parent),
        check=False,
    )


if __name__ == "__main__":
    main()
