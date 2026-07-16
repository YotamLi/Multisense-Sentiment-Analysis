"""
Multi-Head BERT: a hard parameter sharing multi-task model for sentiment analysis.

The architecture follows the standard MTL paradigm where a single pre-trained
BERT encoder is shared across all tasks. Each task has a lightweight
classification head that consumes the [CLS] pooled representation.

Hard parameter sharing acts as an implicit regularizer — the shared encoder must
learn representations useful for all tasks simultaneously, reducing overfitting
risk on low-resource auxiliary tasks (Ruder, 2017).

The total training objective is a weighted sum of per-task cross-entropy losses:
    L_total = Σ_k λ_k · L_k
where λ_k controls how much each task influences the shared encoder gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from .heads import ClassificationHead, MultiHeadOutput


class MultiHeadBERT(nn.Module):
    """
    Multi-task BERT with four classification heads:
        1. Sentiment polarity    (positive / negative / neutral)
        2. Emotion category      (joy / anger / sadness / fear / surprise / disgust)
        3. Emotion intensity     (strong / medium / weak)
        4. Topic classification  (configurable domain labels)
    """

    TASK_NAMES = ("sentiment", "emotion", "intensity", "topic")

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        num_sentiment_classes: int = 3,
        num_emotion_classes: int = 6,
        num_intensity_classes: int = 3,
        num_topic_classes: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        loss_weights: dict[str, float] | None = None,
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size  # 768 for bert-base

        self.heads = nn.ModuleDict(
            {
                "sentiment": ClassificationHead(
                    encoder_dim, hidden_dim, num_sentiment_classes, dropout
                ),
                "emotion": ClassificationHead(
                    encoder_dim, hidden_dim, num_emotion_classes, dropout
                ),
                "intensity": ClassificationHead(
                    encoder_dim, hidden_dim, num_intensity_classes, dropout
                ),
                "topic": ClassificationHead(
                    encoder_dim, hidden_dim, num_topic_classes, dropout
                ),
            }
        )

        self.loss_fns = nn.ModuleDict(
            {task: nn.CrossEntropyLoss(ignore_index=-1) for task in self.TASK_NAMES}
        )

        default_weights = {
            "sentiment": 1.0,
            "emotion": 0.5,
            "intensity": 0.3,
            "topic": 0.3,
        }
        self.loss_weights = loss_weights or default_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        sentiment_labels: torch.Tensor | None = None,
        emotion_labels: torch.Tensor | None = None,
        intensity_labels: torch.Tensor | None = None,
        topic_labels: torch.Tensor | None = None,
    ) -> MultiHeadOutput:
        """
        Forward pass with optional label-conditioned loss computation.

        Labels can be None for tasks without supervision in the current batch,
        enabling heterogeneous batching across tasks with different data sources.
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use the [CLS] token's representation as the sentence embedding.
        # pooler_output applies a linear + tanh on top of [CLS] hidden state.
        pooled = encoder_output.pooler_output

        sentiment_logits = self.heads["sentiment"](pooled)
        emotion_logits = self.heads["emotion"](pooled)
        intensity_logits = self.heads["intensity"](pooled)
        topic_logits = self.heads["topic"](pooled)

        labels_map = {
            "sentiment": (sentiment_labels, sentiment_logits),
            "emotion": (emotion_labels, emotion_logits),
            "intensity": (intensity_labels, intensity_logits),
            "topic": (topic_labels, topic_logits),
        }

        total_loss = None
        per_task_losses = {}

        for task_name, (labels, logits) in labels_map.items():
            if labels is not None:
                valid_mask = labels != -1
                if valid_mask.any():
                    task_loss = self.loss_fns[task_name](logits, labels)
                    per_task_losses[task_name] = task_loss
                    weighted = self.loss_weights[task_name] * task_loss
                    total_loss = (
                        weighted if total_loss is None
                        else total_loss + weighted
                    )

        return MultiHeadOutput(
            sentiment_logits=sentiment_logits,
            emotion_logits=emotion_logits,
            intensity_logits=intensity_logits,
            topic_logits=topic_logits,
            loss=total_loss,
            per_task_losses=per_task_losses,
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, dict]:
        """
        Inference-time prediction returning labels and confidence scores.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask, token_type_ids)

        results = {}
        for task_name, logits in [
            ("sentiment", output.sentiment_logits),
            ("emotion", output.emotion_logits),
            ("intensity", output.intensity_logits),
            ("topic", output.topic_logits),
        ]:
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)
            results[task_name] = {
                "predicted_class": predicted.item(),
                "confidence": confidence.item(),
                "probabilities": probs.squeeze().tolist(),
            }

        return results

    def get_encoder_params(self):
        """Parameters of the shared BERT encoder (lower learning rate)."""
        return self.encoder.parameters()

    def get_head_params(self):
        """Parameters of all classification heads (higher learning rate)."""
        return self.heads.parameters()
