"""
Baseline models for sentiment analysis comparison.

Provides VADER (rule-based) and Naive Bayes (TF-IDF) baselines.
These serve as lower bounds — the multi-task BERT system should
substantially outperform these simpler approaches to justify
its added complexity.

VADER is particularly strong on social media text due to its
hand-crafted rules for emoji, slang, and punctuation-based
intensity modifiers.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import cross_val_predict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger

from .metrics import compute_metrics


class VADERBaseline:
    """
    Rule-based sentiment analysis using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) uses a
    curated lexicon combined with grammatical heuristics (negation,
    degree modifiers, punctuation, capitalization) to compute a
    compound sentiment score.

    The compound score is mapped to polarity labels using standard
    thresholds: compound >= 0.05 → positive, <= -0.05 → negative,
    otherwise neutral.
    """

    def __init__(
        self,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
    ):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.label_map = {"positive": 0, "negative": 1, "neutral": 2}
        self.label_names = ["positive", "negative", "neutral"]

    def predict_one(self, text: str) -> dict:
        """Analyze a single text, returning polarity and scores."""
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= self.pos_threshold:
            label = "positive"
        elif compound <= self.neg_threshold:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "label_idx": self.label_map[label],
            "compound": compound,
            "scores": scores,
        }

    def predict_batch(self, texts: list[str]) -> list[int]:
        """Predict polarity labels for a batch of texts."""
        return [self.predict_one(t)["label_idx"] for t in texts]

    def evaluate(
        self,
        texts: list[str],
        y_true: list[int],
    ) -> dict:
        """Run VADER on texts and compute metrics against ground truth."""
        y_pred = self.predict_batch(texts)
        metrics = compute_metrics(
            y_true, y_pred, label_names=self.label_names
        )
        logger.info(
            f"VADER Baseline: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['macro_f1']:.4f}"
        )
        return metrics


class NaiveBayesBaseline:
    """
    TF-IDF + Multinomial Naive Bayes baseline.

    Despite its simplicity, this combination remains a competitive
    baseline for text classification. The conditional independence
    assumption of Naive Bayes works surprisingly well in
    high-dimensional sparse feature spaces typical of bag-of-words
    representations.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        alpha: float = 1.0,
    ):
        self.pipeline = SkPipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                ),
            ),
            ("nb", MultinomialNB(alpha=alpha)),
        ])
        self.is_fitted = False

    def train(
        self,
        texts: list[str],
        labels: list[int],
    ):
        """Fit the TF-IDF + NB pipeline."""
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        logger.info(
            f"Naive Bayes trained on {len(texts)} samples"
        )

    def predict(self, texts: list[str]) -> list[int]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.pipeline.predict(texts).tolist()

    def evaluate(
        self,
        texts: list[str],
        y_true: list[int],
        label_names: list[str] | None = None,
    ) -> dict:
        """Predict and compute metrics."""
        y_pred = self.predict(texts)
        metrics = compute_metrics(y_true, y_pred, label_names=label_names)
        logger.info(
            f"Naive Bayes Baseline: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['macro_f1']:.4f}"
        )
        return metrics

    def cross_validate(
        self,
        texts: list[str],
        labels: list[int],
        cv: int = 5,
        label_names: list[str] | None = None,
    ) -> dict:
        """K-fold cross-validation for more robust estimation."""
        y_pred = cross_val_predict(self.pipeline, texts, labels, cv=cv)
        metrics = compute_metrics(
            labels, y_pred.tolist(), label_names=label_names
        )
        logger.info(
            f"Naive Bayes {cv}-fold CV: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['macro_f1']:.4f}"
        )
        return metrics
