"""LangChain-compatible components for the sentiment analysis workflow."""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from loguru import logger
from transformers import BertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..data.preprocessor import TextPreprocessor
from ..models.multi_head_bert import MultiHeadBERT
from ..rag.retriever import LexiconRetriever


class PreprocessChain:
    """Clean and normalise raw user input."""

    def __init__(self, preprocessor: TextPreprocessor | None = None):
        self.preprocessor = preprocessor or TextPreprocessor()

    def as_runnable(self):
        def _run(inputs: dict) -> dict:
            raw = inputs.get("text", "")
            return {
                **inputs,
                "cleaned_text": self.preprocessor.clean(raw),
                "raw_text": raw,
            }

        return RunnableLambda(_run)


class BERTInferenceChain:
    """Run one shared-encoder forward pass and decode all four heads."""

    def __init__(
        self,
        model: MultiHeadBERT,
        tokenizer: BertTokenizer,
        label_maps: dict[str, list[str]],
        device: torch.device | None = None,
        max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
        self.model.to(self.device).eval()

    def as_runnable(self):
        def _run(inputs: dict) -> dict:
            text = inputs.get("cleaned_text", inputs.get("text", ""))
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = encoded.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            predictions = self.model.predict(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            decoded: dict[str, dict] = {}
            for task, info in predictions.items():
                labels = self.label_maps.get(task, [])
                predicted_index = int(info["predicted_class"])
                probabilities = info["probabilities"]
                decoded[task] = {
                    "label": (
                        labels[predicted_index]
                        if predicted_index < len(labels)
                        else str(predicted_index)
                    ),
                    "confidence": round(float(info["confidence"]), 4),
                    "probabilities": {
                        (
                            labels[index]
                            if index < len(labels)
                            else str(index)
                        ): round(float(probability), 4)
                        for index, probability in enumerate(probabilities)
                    },
                }

            return {**inputs, "initial_predictions": decoded}

        return RunnableLambda(_run)


class VADERSentimentChain:
    """Produce rough initial predictions when no BERT checkpoint is used."""

    TOPIC_KEYWORDS = {
        "sports": ["sport", "game", "player", "team", "match", "score"],
        "business": ["business", "market", "stock", "company", "economy"],
        "technology": ["technology", "computer", "software", "internet", "ai"],
        "entertainment": ["movie", "film", "music", "actor", "show", "tv"],
    }

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def as_runnable(self):
        def _run(inputs: dict) -> dict:
            raw = inputs.get("raw_text", inputs.get("text", ""))
            scores = self.vader.polarity_scores(raw)
            compound = scores["compound"]
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            emotion = (
                "joy"
                if sentiment == "positive"
                else "anger"
                if sentiment == "negative"
                else "surprise"
            )
            intensity = (
                "strong"
                if abs(compound) > 0.5
                else "medium"
                if abs(compound) > 0.2
                else "weak"
            )
            emotion_probabilities = {
                label: 1.0 if label == emotion else 0.0
                for label in LLMReasoningChain.EMOTION_LABELS
            }
            predictions = {
                "sentiment": {
                    "label": sentiment,
                    "confidence": round(min(max(abs(compound), 0.35), 0.99), 4),
                    "probabilities": {
                        "positive": round(scores["pos"], 4),
                        "negative": round(scores["neg"], 4),
                        "neutral": round(scores["neu"], 4),
                    },
                },
                "emotion": {
                    "label": emotion,
                    "confidence": 0.4,
                    "probabilities": emotion_probabilities,
                },
                "intensity": {
                    "label": intensity,
                    "confidence": 0.5,
                    "probabilities": {},
                },
                "topic": {
                    "label": self._detect_topic(raw),
                    "confidence": 0.4,
                    "probabilities": {},
                },
            }
            return {
                **inputs,
                "initial_predictions": predictions,
                "vader_scores": scores,
            }

        return RunnableLambda(_run)

    def _detect_topic(self, text: str) -> str:
        lowered = text.lower()
        best_label = "general"
        best_count = 0
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            count = sum(keyword in lowered for keyword in keywords)
            if count > best_count:
                best_label = topic
                best_count = count
        return best_label


class RAGRetrievalChain:
    """Retrieve lexicon evidence from the vector knowledge base."""

    def __init__(self, retriever: LexiconRetriever):
        self.retriever = retriever

    def as_runnable(self):
        def _run(inputs: dict) -> dict:
            text = inputs.get("cleaned_text", inputs.get("text", ""))
            return {
                **inputs,
                "rag_evidence": self.retriever.retrieve(text),
            }

        return RunnableLambda(_run)


class PassthroughReasoningChain:
    """Use the initial model result when LLM reasoning is disabled."""

    def as_runnable(self):
        def _run(inputs: dict) -> dict:
            predictions = inputs.get("initial_predictions", {})
            fallback = LLMReasoningChain._fallback_from_initial(predictions)
            fallback["explanation"] = (
                "LLM reasoning is disabled; showing the initial model predictions."
            )
            return {**inputs, "llm_analysis": fallback}

        return RunnableLambda(_run)


class LLMReasoningChain:
    """Use Gemini for pragmatic reasoning, sarcasm detection and correction."""

    SENTIMENT_LABELS = ("positive", "negative", "neutral")
    EMOTION_LABELS = (
        "joy",
        "anger",
        "sadness",
        "fear",
        "surprise",
        "disgust",
    )
    INTENSITY_LABELS = ("strong", "medium", "weak")

    SYSTEM_PROMPT = (
        "You are a sentiment-analysis adjudicator. Use the raw text, the initial "
        "model predictions, and lexicon evidence to produce a corrected analysis. "
        "Pay particular attention to sarcasm, irony, negation and implied meaning. "
        "Sentiment must be positive, negative or neutral. Emotion must be one of "
        "joy, anger, sadness, fear, surprise or disgust. Intensity must be strong, "
        "medium or weak. Return ONLY one complete valid JSON object with no markdown."
    )

    def __init__(
        self,
        llm: BaseChatModel,
        topic_labels: list[str] | None = None,
        allow_topic_correction: bool = False,
    ):
        self.llm = llm
        self.topic_labels = tuple(topic_labels or ())
        self.allow_topic_correction = allow_topic_correction

    def as_runnable(self):
        def _run(inputs: dict) -> dict:
            predictions = inputs.get("initial_predictions", {})
            rag = inputs.get("rag_evidence", {})
            allowed_topics = ", ".join(self.topic_labels) or "retain initial topic"
            human_prompt = (
                f"Text: {inputs.get('raw_text', '')!r}\n\n"
                f"Initial predictions:\n{json.dumps(predictions, ensure_ascii=False)}\n\n"
                f"RAG evidence:\n{json.dumps(rag, ensure_ascii=False)}\n\n"
                f"Allowed topic labels: {allowed_topics}. "
                f"Topic correction enabled: {self.allow_topic_correction}.\n\n"
                "Return this schema only:\n"
                "{\n"
                '  "sentiment": {"label": "...", "confidence": 0.0},\n'
                '  "emotion": {"label": "...", "confidence": 0.0, '
                '"probabilities": {"joy": 0.0, "anger": 0.0, '
                '"sadness": 0.0, "fear": 0.0, "surprise": 0.0, '
                '"disgust": 0.0}},\n'
                '  "intensity": {"label": "...", "confidence": 0.0},\n'
                '  "topic": {"label": "...", "confidence": 0.0},\n'
                '  "sarcasm_detected": false,\n'
                '  "explanation": "2-4 concise sentences"\n'
                "}"
            )
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ]

            try:
                response = self.llm.invoke(messages)
                raw_text = self._response_text(response)
                parsed = self._parse_llm_json(raw_text)
                llm_result = self._validate_llm_result(parsed, predictions)
                logger.info(
                    "LLM reasoning complete — "
                    f"sentiment={llm_result['sentiment']['label']}, "
                    f"sarcasm={llm_result['sarcasm_detected']}"
                )
            except Exception as exc:
                logger.warning(
                    f"LLM call failed ({exc}), falling back to initial predictions"
                )
                llm_result = self._fallback_from_initial(predictions)

            return {**inputs, "llm_analysis": llm_result}

        return RunnableLambda(_run)

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        parts.append(block_text)
                else:
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str):
                        parts.append(block_text)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _parse_llm_json(raw: str) -> dict:
        cleaned = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        if start >= 0:
            decoder = json.JSONDecoder()
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse LLM output as JSON: {raw[:300]}")

    @staticmethod
    def _safe_confidence(value: Any, fallback: float = 0.5) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(fallback)
        return round(max(0.0, min(1.0, number)), 4)

    @classmethod
    def _normalise_emotion_probabilities(
        cls,
        probabilities: Any,
        selected_label: str,
    ) -> dict[str, float]:
        source = probabilities if isinstance(probabilities, dict) else {}
        cleaned: dict[str, float] = {}
        for label in cls.EMOTION_LABELS:
            try:
                value = float(source.get(label, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            cleaned[label] = max(0.0, value)

        total = sum(cleaned.values())
        if total <= 0:
            return {
                label: 1.0 if label == selected_label else 0.0
                for label in cls.EMOTION_LABELS
            }
        return {
            label: round(value / total, 4)
            for label, value in cleaned.items()
        }

    def _validate_llm_result(self, result: dict, predictions: dict) -> dict:
        if not isinstance(result, dict):
            return self._fallback_from_initial(predictions)

        def validate_task(
            task_name: str,
            allowed_labels: tuple[str, ...],
            default_label: str,
        ) -> dict:
            initial = predictions.get(task_name, {})
            raw_task = result.get(task_name, {})
            if not isinstance(raw_task, dict):
                raw_task = {}
            initial_label = str(
                initial.get("label", default_label)
            ).strip().lower()
            label = str(
                raw_task.get("label", initial_label)
            ).strip().lower()
            if label not in allowed_labels:
                label = (
                    initial_label
                    if initial_label in allowed_labels
                    else default_label
                )
            return {
                "label": label,
                "confidence": self._safe_confidence(
                    raw_task.get("confidence"),
                    initial.get("confidence", 0.5),
                ),
            }

        sentiment = validate_task(
            "sentiment", self.SENTIMENT_LABELS, "neutral"
        )
        emotion = validate_task(
            "emotion", self.EMOTION_LABELS, "surprise"
        )
        raw_emotion = result.get("emotion", {})
        raw_probabilities = (
            raw_emotion.get("probabilities", {})
            if isinstance(raw_emotion, dict)
            else {}
        )
        if not raw_probabilities:
            raw_probabilities = predictions.get("emotion", {}).get(
                "probabilities", {}
            )
        emotion["probabilities"] = self._normalise_emotion_probabilities(
            raw_probabilities,
            emotion["label"],
        )
        intensity = validate_task(
            "intensity", self.INTENSITY_LABELS, "medium"
        )

        initial_topic = predictions.get(
            "topic", {"label": "unknown", "confidence": 0.5}
        )
        topic = dict(initial_topic)
        if self.allow_topic_correction and self.topic_labels:
            raw_topic = result.get("topic", {})
            if isinstance(raw_topic, dict):
                candidate = str(raw_topic.get("label", "")).strip().lower()
                allowed_lookup = {
                    label.lower(): label for label in self.topic_labels
                }
                if candidate in allowed_lookup:
                    topic = {
                        "label": allowed_lookup[candidate],
                        "confidence": self._safe_confidence(
                            raw_topic.get("confidence"),
                            initial_topic.get("confidence", 0.5),
                        ),
                        "probabilities": initial_topic.get("probabilities", {}),
                    }

        sarcasm = result.get("sarcasm_detected", False)
        if isinstance(sarcasm, str):
            sarcasm = sarcasm.strip().lower() in {"true", "yes", "1"}
        else:
            sarcasm = bool(sarcasm)

        explanation = result.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)
        explanation = explanation.strip() or (
            "The final result was produced from validated model and LLM output."
        )

        return {
            "sentiment": sentiment,
            "emotion": emotion,
            "intensity": intensity,
            "topic": topic,
            "sarcasm_detected": sarcasm,
            "explanation": explanation,
        }

    @classmethod
    def _fallback_from_initial(cls, predictions: dict) -> dict:
        emotion = dict(
            predictions.get(
                "emotion",
                {"label": "surprise", "confidence": 0.3},
            )
        )
        emotion["probabilities"] = cls._normalise_emotion_probabilities(
            emotion.get("probabilities", {}),
            str(emotion.get("label", "surprise")),
        )
        return {
            "sentiment": predictions.get(
                "sentiment", {"label": "neutral", "confidence": 0.5}
            ),
            "emotion": emotion,
            "intensity": predictions.get(
                "intensity", {"label": "medium", "confidence": 0.5}
            ),
            "topic": predictions.get(
                "topic", {"label": "unknown", "confidence": 0.5}
            ),
            "sarcasm_detected": False,
            "explanation": (
                "LLM reasoning unavailable; showing initial model predictions."
            ),
        }


class AggregationChain:
    """Merge final analysis, initial predictions and complete RAG evidence."""

    def as_runnable(self):
        def _run(inputs: dict) -> dict:
            llm = inputs.get("llm_analysis", {})
            rag = inputs.get("rag_evidence", {})
            initial = inputs.get("initial_predictions", {})
            return {
                "input_text": inputs.get("raw_text", ""),
                "sentiment": llm.get(
                    "sentiment", initial.get("sentiment", {})
                ),
                "emotion": llm.get(
                    "emotion", initial.get("emotion", {})
                ),
                "intensity": llm.get(
                    "intensity", initial.get("intensity", {})
                ),
                "topic": llm.get("topic", initial.get("topic", {})),
                "sarcasm": {
                    "is_sarcastic": bool(
                        llm.get("sarcasm_detected", False)
                    )
                },
                "rag_evidence": {
                    "matched_keywords": rag.get("matched_keywords", []),
                    "lexicon_scores": rag.get("lexicon_scores", {}),
                    "full_text_matches": rag.get("full_text_matches", []),
                    "negations": rag.get("negations", []),
                },
                "explanation": llm.get("explanation", ""),
                "initial_predictions": initial,
            }

        return RunnableLambda(_run)
