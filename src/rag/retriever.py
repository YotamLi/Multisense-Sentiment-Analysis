"""
Lexicon-augmented retrieval component for the RAG pipeline.

Extracts salient sentiment-bearing tokens from the input text, retrieves
matching lexicon entries from the vector store, and aggregates evidence
across multiple lexicon sources (VADER, NRC, SentiWordNet).

The retriever acts as the bridge between the BERT model's predictions and
the LLM explanation generator — it provides interpretable lexicon-based
evidence that grounds the natural language explanation.
"""

from __future__ import annotations

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from loguru import logger

from .knowledge_base import SentimentKnowledgeBase


def _ensure_nltk_data():
    """Download required NLTK resources if not present."""
    for resource in ["punkt", "averaged_perceptron_tagger", "punkt_tab",
                     "averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)


class LexiconRetriever:
    """
    Retrieves sentiment lexicon evidence for a given text.

    Strategy:
    1. POS-tag the input to identify adjectives, adverbs, and verbs —
       the word classes most likely to carry sentiment.
    2. Query the vector store with both the full text and individual
       sentiment-bearing tokens.
    3. Deduplicate and rank results by similarity score.
    """

    SENTIMENT_POS_TAGS = {
        "JJ", "JJR", "JJS",  # adjectives
        "RB", "RBR", "RBS",  # adverbs
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # verbs
    }

    # Negation markers that flip sentiment polarity
    NEGATION_WORDS = {
        "not", "no", "never", "neither", "nobody", "nothing",
        "nowhere", "nor", "cannot", "can't", "won't", "don't",
        "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
        "wouldn't", "shouldn't", "couldn't", "hardly", "barely", "scarcely",
    }

    def __init__(
        self,
        knowledge_base: SentimentKnowledgeBase,
        top_k: int = 5,
    ):
        _ensure_nltk_data()
        self.kb = knowledge_base
        self.top_k = top_k

    def retrieve(self, text: str) -> dict:
        """
        Full retrieval pipeline: extract keywords → query KB → aggregate.

        Returns a structured dict with matched keywords, lexicon scores,
        and detected negation contexts.
        """
        keywords = self._extract_sentiment_keywords(text)
        negations = self._detect_negations(text)

        full_text_results = self.kb.query(text, top_k=self.top_k)

        keyword_results = {}
        for kw in keywords:
            matches = self.kb.query(kw, top_k=3)
            if matches:
                keyword_results[kw] = matches

        aggregated = self._aggregate_evidence(
            full_text_results, keyword_results, negations
        )

        return aggregated

    def _extract_sentiment_keywords(self, text: str) -> list[str]:
        """
        Extract words likely to carry sentiment signal via POS tagging.
        Adjectives, adverbs, and verbs are the primary carriers of
        evaluative meaning in natural language.
        """
        try:
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            keywords = [
                word for word, tag in tagged
                if tag in self.SENTIMENT_POS_TAGS and len(word) > 2
            ]
            return list(dict.fromkeys(keywords))  # deduplicate, preserve order
        except Exception as e:
            logger.warning(f"POS tagging failed, falling back to simple extraction: {e}")
            return self._simple_keyword_extract(text)

    def _simple_keyword_extract(self, text: str) -> list[str]:
        """Fallback: extract all non-stopword tokens over 3 characters."""
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be",
                     "been", "being", "have", "has", "had", "do", "does",
                     "did", "will", "would", "could", "should", "may",
                     "might", "shall", "can", "this", "that", "these",
                     "those", "it", "its"}
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return [t for t in tokens if t not in stopwords]

    def _detect_negations(self, text: str) -> list[dict]:
        """
        Detect negation scopes in text.

        Negation is critical for sentiment analysis — "not good" carries
        opposite polarity to "good". We identify negation words and their
        likely scope (next 1–3 tokens).
        """
        tokens = text.lower().split()
        negations = []

        for i, token in enumerate(tokens):
            if token in self.NEGATION_WORDS:
                scope = tokens[i + 1: i + 4]  # typical negation scope
                negations.append({
                    "negation_word": token,
                    "scope": scope,
                    "position": i,
                })

        return negations

    def _aggregate_evidence(
        self,
        full_text_results: list[dict],
        keyword_results: dict[str, list[dict]],
        negations: list[dict],
    ) -> dict:
        """Merge and structure all retrieval evidence."""
        matched_keywords = list(keyword_results.keys())

        lexicon_scores = {}
        for kw, results in keyword_results.items():
            scores = {}
            for r in results:
                meta = r["metadata"]
                source = meta.get("source", "unknown")
                if source == "vader":
                    scores["vader_compound"] = meta.get("vader_score", 0)
                elif source == "nrc":
                    import json
                    scores["nrc_emotions"] = json.loads(
                        meta.get("emotions", "[]")
                    )
                elif source == "sentiwordnet":
                    scores["swn_pos"] = meta.get("pos_score", 0)
                    scores["swn_neg"] = meta.get("neg_score", 0)
            lexicon_scores[kw] = scores

        return {
            "matched_keywords": matched_keywords,
            "lexicon_scores": lexicon_scores,
            "full_text_matches": [
                {
                    "document": r["document"],
                    "similarity": round(r["similarity"], 4),
                }
                for r in full_text_results
            ],
            "negations": negations,
        }
