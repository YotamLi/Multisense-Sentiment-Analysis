"""
Task-specific classification heads for Multi-Task BERT.

Each head operates on the [CLS] token representation from the shared BERT
encoder. The two-layer MLP with intermediate nonlinearity provides sufficient
capacity for task-specific decision boundaries while keeping the head
lightweight relative to the encoder.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    A two-layer MLP classification head with dropout regularization.

    Architecture: Linear → ReLU → Dropout → Linear
    The first projection maps from the encoder hidden size (typically 768)
    to a task-specific intermediate dimension, allowing each head to learn
    its own feature subspace from the shared representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.33),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [CLS] representation from BERT, shape (batch, input_dim).
        Returns:
            Logits of shape (batch, num_classes).
        """
        return self.classifier(pooled_output)


class MultiHeadOutput:
    """Container for multi-task model outputs."""

    def __init__(
        self,
        sentiment_logits: torch.Tensor | None = None,
        emotion_logits: torch.Tensor | None = None,
        intensity_logits: torch.Tensor | None = None,
        topic_logits: torch.Tensor | None = None,
        loss: torch.Tensor | None = None,
        per_task_losses: dict[str, torch.Tensor] | None = None,
    ):
        self.sentiment_logits = sentiment_logits
        self.emotion_logits = emotion_logits
        self.intensity_logits = intensity_logits
        self.topic_logits = topic_logits
        self.loss = loss
        self.per_task_losses = per_task_losses or {}
