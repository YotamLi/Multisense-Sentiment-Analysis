"""Compose preprocessing, initial models, RAG and Gemini reasoning."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from transformers import BertTokenizer

from ..models.checkpoint import (
    load_checkpoint,
    validate_checkpoint_compatibility,
)
from ..models.multi_head_bert import MultiHeadBERT
from ..rag.knowledge_base import SentimentKnowledgeBase
from ..rag.retriever import LexiconRetriever
from ..utils.helpers import load_config, load_env
from .chains import (
    AggregationChain,
    BERTInferenceChain,
    LLMReasoningChain,
    PassthroughReasoningChain,
    PreprocessChain,
    RAGRetrievalChain,
    VADERSentimentChain,
)


def _build_llm(config: dict):
    """Create Gemini only when LLM reasoning is enabled."""
    load_env()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY was not found. Copy .env.example to .env and "
            "insert a Google AI Studio API key, or run with --disable-llm."
        )

    pipeline_config = config.get("pipeline", {})
    model_name = pipeline_config.get("llm_model", "gemini-3.5-flash")
    kwargs = {
        "model": model_name,
        "google_api_key": api_key,
        "temperature": float(pipeline_config.get("temperature", 0.0)),
        "max_output_tokens": int(pipeline_config.get("max_tokens", 2048)),
    }
    thinking_level = pipeline_config.get("thinking_level")
    if thinking_level:
        kwargs["thinking_level"] = thinking_level

    try:
        llm = ChatGoogleGenerativeAI(**kwargs)
    except TypeError:
        # Backward compatibility with integrations that do not yet expose
        # Gemini thinking_level in the constructor.
        kwargs.pop("thinking_level", None)
        llm = ChatGoogleGenerativeAI(**kwargs)

    logger.info(f"LLM initialised: model={model_name} (Google Gemini)")
    return llm


def _build_rag(
    config: dict,
    device: torch.device | str | None = None,
) -> LexiconRetriever | None:
    """Load a non-empty ChromaDB collection, otherwise disable RAG."""
    try:
        rag_config = config["rag"]
        persist_directory = Path(rag_config["persist_directory"])
        if not persist_directory.exists() or not any(persist_directory.iterdir()):
            logger.info(
                "RAG KB not built — run 'python scripts/build_rag.py' first"
            )
            return None

        knowledge_base = SentimentKnowledgeBase(
            persist_directory=rag_config["persist_directory"],
            collection_name=rag_config["collection_name"],
            embedding_model=rag_config["embedding_model"],
            device=device,
        )
        document_count = int(
            knowledge_base.get_stats().get("total_documents", 0)
        )
        if document_count <= 0:
            logger.warning("RAG collection is empty; RAG disabled")
            return None

        retriever = LexiconRetriever(
            knowledge_base,
            top_k=int(rag_config.get("top_k", 5)),
        )
        logger.info(f"RAG KB loaded ({document_count} docs)")
        return retriever
    except Exception as exc:
        logger.warning(f"RAG initialisation failed: {exc}")
        return None


def _empty_rag_runnable():
    def _run(inputs: dict) -> dict:
        return {
            **inputs,
            "rag_evidence": {
                "matched_keywords": [],
                "lexicon_scores": {},
                "full_text_matches": [],
                "negations": [],
            },
        }

    return RunnableLambda(_run)


class SentimentPipeline:
    """Multi-Head BERT with optional RAG and Gemini reasoning."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_checkpoint: str | None = None,
        device: torch.device | None = None,
        enable_rag: bool = True,
        enable_llm: bool = True,
    ):
        if not model_checkpoint:
            raise ValueError(
                "A BERT checkpoint is required for SentimentPipeline."
            )

        self.config = load_config(config_path)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_config = self.config["model"]
        checkpoint = load_checkpoint(
            model_checkpoint,
            map_location=self.device,
        )
        validate_checkpoint_compatibility(checkpoint, self.config)

        self.tokenizer = BertTokenizer.from_pretrained(
            model_config["encoder_name"]
        )
        self.model = MultiHeadBERT(
            encoder_name=model_config["encoder_name"],
            num_sentiment_classes=model_config["heads"]["sentiment"]["num_classes"],
            num_emotion_classes=model_config["heads"]["emotion"]["num_classes"],
            num_intensity_classes=model_config["heads"]["intensity"]["num_classes"],
            num_topic_classes=model_config["heads"]["topic"]["num_classes"],
            hidden_dim=model_config["hidden_dim"],
            dropout=model_config["dropout"],
            loss_weights=self.config["training"]["loss_weights"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        logger.info(f"Loaded BERT checkpoint: {model_checkpoint}")

        label_maps = {
            task: task_config["labels"]
            for task, task_config in model_config["heads"].items()
        }
        retriever = (
            _build_rag(self.config, device=self.device)
            if enable_rag
            else None
        )
        llm = _build_llm(self.config) if enable_llm else None

        self.initial_model_name = "Multi-Head BERT"
        self.checkpoint_path = str(model_checkpoint)
        self.llm_enabled = llm is not None
        self.llm_model_name = (
            self.config.get("pipeline", {}).get("llm_model")
            if self.llm_enabled
            else "Disabled"
        )
        self.rag_retriever = retriever
        self.rag_enabled = retriever is not None
        self.rag_document_count = (
            int(retriever.kb.get_stats().get("total_documents", 0))
            if retriever is not None
            else 0
        )
        self.pipeline = self._compose(label_maps, llm, retriever)
        logger.info(
            "SentimentPipeline ready "
            f"(BERT, RAG={self.rag_enabled}, LLM={self.llm_enabled})"
        )

    def _compose(
        self,
        label_maps: dict[str, list[str]],
        llm,
        retriever,
    ) -> RunnableSequence:
        preprocess = PreprocessChain().as_runnable()
        bert = BERTInferenceChain(
            self.model,
            self.tokenizer,
            label_maps,
            self.device,
            int(self.config["model"]["max_length"]),
        ).as_runnable()
        rag = (
            RAGRetrievalChain(retriever).as_runnable()
            if retriever is not None
            else _empty_rag_runnable()
        )
        if llm is not None:
            pipeline_config = self.config.get("pipeline", {})
            reasoning = LLMReasoningChain(
                llm,
                topic_labels=label_maps.get("topic", []),
                allow_topic_correction=bool(
                    pipeline_config.get("allow_llm_topic_correction", False)
                ),
            ).as_runnable()
        else:
            reasoning = PassthroughReasoningChain().as_runnable()

        return preprocess | bert | rag | reasoning | AggregationChain().as_runnable()

    def analyze(self, text: str) -> dict:
        return self.pipeline.invoke({"text": text})

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(text) for text in texts]


class HybridPipeline:
    """VADER initial analysis with optional RAG and Gemini reasoning."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        enable_rag: bool = True,
        enable_llm: bool = True,
    ):
        self.config = load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        retriever = (
            _build_rag(self.config, device=self.device)
            if enable_rag
            else None
        )
        llm = _build_llm(self.config) if enable_llm else None

        self.initial_model_name = "VADER"
        self.checkpoint_path = None
        self.llm_enabled = llm is not None
        self.llm_model_name = (
            self.config.get("pipeline", {}).get("llm_model")
            if self.llm_enabled
            else "Disabled"
        )
        self.rag_retriever = retriever
        self.rag_enabled = retriever is not None
        self.rag_document_count = (
            int(retriever.kb.get_stats().get("total_documents", 0))
            if retriever is not None
            else 0
        )

        preprocess = PreprocessChain().as_runnable()
        vader = VADERSentimentChain().as_runnable()
        rag = (
            RAGRetrievalChain(retriever).as_runnable()
            if retriever is not None
            else _empty_rag_runnable()
        )
        if llm is not None:
            topic_labels = self.config.get("model", {}).get("heads", {}).get(
                "topic", {}
            ).get("labels", [])
            reasoning = LLMReasoningChain(
                llm,
                topic_labels=topic_labels,
                allow_topic_correction=False,
            ).as_runnable()
        else:
            reasoning = PassthroughReasoningChain().as_runnable()

        self.pipeline = (
            preprocess | vader | rag | reasoning | AggregationChain().as_runnable()
        )
        logger.info(
            "HybridPipeline ready "
            f"(VADER, RAG={self.rag_enabled}, LLM={self.llm_enabled})"
        )

    def analyze(self, text: str) -> dict:
        return self.pipeline.invoke({"text": text})

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(text) for text in texts]
