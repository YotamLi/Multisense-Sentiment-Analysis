"""
Evaluation metrics for multi-task sentiment analysis.

Provides standard classification metrics (accuracy, precision, recall, F1)
for single-task and multi-task evaluation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from loguru import logger


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
    average: str = "macro",
    ignore_label: int = -1,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Samples where y_true == ignore_label are filtered before computation.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        label_names: Optional human-readable class names.
        average: Averaging strategy for multi-class metrics.
        ignore_label: Sentinel value for missing labels (-1).

    Returns:
        Dict with accuracy, precision, recall, macro/weighted F1,
        per-class metrics, and confusion matrix.
    """
    filtered = [
        (t, p) for t, p in zip(y_true, y_pred) if t != ignore_label
    ]
    if not filtered:
        return {
            "accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
            "macro_f1": 0.0, "weighted_precision": 0.0, "weighted_recall": 0.0,
            "weighted_f1": 0.0, "confusion_matrix": [],
            "classification_report": {}, "num_evaluated": 0,
        }

    y_true_f, y_pred_f = zip(*filtered)
    y_true_f, y_pred_f = list(y_true_f), list(y_pred_f)

    accuracy = accuracy_score(y_true_f, y_pred_f)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, average="weighted", zero_division=0
    )

    unique_labels = sorted(set(y_true_f) | set(y_pred_f))
    cm = confusion_matrix(y_true_f, y_pred_f, labels=unique_labels)

    resolved_names = None
    if label_names is not None:
        resolved_names = [
            label_names[i] if i < len(label_names) else str(i)
            for i in unique_labels
        ]

    report = classification_report(
        y_true_f,
        y_pred_f,
        labels=unique_labels,
        target_names=resolved_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(precision_macro, 4),
        "macro_recall": round(recall_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_precision": round(precision_weighted, 4),
        "weighted_recall": round(recall_weighted, 4),
        "weighted_f1": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "num_evaluated": len(y_true_f),
    }


def compute_multitask_metrics(
    task_results: dict[str, tuple[list[int], list[int]]],
    label_maps: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """
    Compute metrics for all tasks in the multi-task setup.

    Args:
        task_results: {task_name: (y_true, y_pred)} for each task.
        label_maps: Optional {task_name: [label_names]} for readable reports.
    """
    all_metrics = {}
    label_maps = label_maps or {}

    for task_name, (y_true, y_pred) in task_results.items():
        if not y_true:
            continue
        metrics = compute_metrics(
            y_true, y_pred, label_names=label_maps.get(task_name)
        )
        all_metrics[task_name] = metrics
        logger.info(
            f"{task_name}: Acc={metrics['accuracy']:.4f}, "
            f"F1(macro)={metrics['macro_f1']:.4f}, "
            f"F1(weighted)={metrics['weighted_f1']:.4f}"
        )

    avg_macro_f1 = np.mean([
        m["macro_f1"] for m in all_metrics.values()
    ])
    all_metrics["average_macro_f1"] = round(avg_macro_f1, 4)

    return all_metrics
