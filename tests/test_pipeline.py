"""Fast unit tests that do not download BERT weights or call external APIs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch


def test_preprocessor_basics():
    from src.data.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean("@john I don't like https://example.com #Bad")
    assert "user" in cleaned
    assert "do not" in cleaned
    assert "https://" not in cleaned
    assert "#" not in cleaned


def test_classification_head_shape_and_gradient():
    from src.models.heads import ClassificationHead

    head = ClassificationHead(16, 8, 3)
    values = torch.randn(4, 16, requires_grad=True)
    output = head(values)
    assert output.shape == (4, 3)
    output.sum().backward()
    assert values.grad is not None


def test_metrics():
    from src.evaluation.metrics import compute_metrics

    metrics = compute_metrics([0, 1, 2, 0], [0, 1, 1, 0])
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_vader_baseline():
    from src.evaluation.baseline import VADERBaseline

    baseline = VADERBaseline()
    assert baseline.predict_one("This is wonderful!")["label"] == "positive"
    assert baseline.predict_one("This is terrible.")["label"] == "negative"


def test_checkpoint_head_size_inference():
    from src.models.checkpoint import infer_head_sizes

    checkpoint = {
        "model_state_dict": {
            "heads.sentiment.classifier.4.weight": torch.zeros(3, 8),
            "heads.emotion.classifier.4.weight": torch.zeros(6, 8),
            "heads.intensity.classifier.4.weight": torch.zeros(3, 8),
            "heads.topic.classifier.4.weight": torch.zeros(6, 8),
        }
    }
    assert infer_head_sizes(checkpoint) == {
        "sentiment": 3,
        "emotion": 6,
        "intensity": 3,
        "topic": 6,
    }


def test_llm_output_validation_rejects_free_form_topic():
    from src.pipeline.chains import LLMReasoningChain

    chain = object.__new__(LLMReasoningChain)
    chain.topic_labels = ("topic_0", "topic_1")
    chain.allow_topic_correction = False
    initial = {
        "sentiment": {"label": "positive", "confidence": 0.8},
        "emotion": {
            "label": "joy",
            "confidence": 0.7,
            "probabilities": {"joy": 1.0},
        },
        "intensity": {"label": "medium", "confidence": 0.6},
        "topic": {"label": "topic_1", "confidence": 0.55},
    }
    raw = {
        "sentiment": {"label": "negative", "confidence": 1.4},
        "emotion": {
            "label": "frustration",
            "confidence": -1,
            "probabilities": {"anger": 5, "sadness": 5},
        },
        "intensity": {"label": "very_strong", "confidence": "bad"},
        "topic": {"label": "transportation", "confidence": 0.99},
        "sarcasm_detected": "yes",
        "explanation": "Sarcasm changes the literal meaning.",
    }
    validated = chain._validate_llm_result(raw, initial)
    assert validated["sentiment"]["confidence"] == 1.0
    assert validated["emotion"]["label"] == "joy"
    assert validated["intensity"]["label"] == "medium"
    assert validated["topic"]["label"] == "topic_1"
    assert validated["sarcasm_detected"] is True
    assert abs(sum(validated["emotion"]["probabilities"].values()) - 1.0) < 0.001


def test_duplicate_merge_and_test_priority():
    from scripts.prepare_multisource_data import (
        assign_project_split,
        merge_duplicate_texts,
    )

    frame = pd.DataFrame(
        [
            {
                "text": "Hello @alice",
                "sentiment": 0,
                "emotion": -1,
                "intensity": -1,
                "topic": -1,
                "source": "source_a",
                "source_split": "train",
            },
            {
                "text": "hello @bob",
                "sentiment": -1,
                "emotion": 1,
                "intensity": -1,
                "topic": -1,
                "source": "source_b",
                "source_split": "test",
            },
        ]
    )
    merged = merge_duplicate_texts(frame)
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["sentiment"] == 0
    assert row["emotion"] == 1
    assert assign_project_split(row, 0.1, 0.1) == "test"
