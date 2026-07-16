"""
RAG Knowledge Base backed by ChromaDB.

Indexes sentiment lexicons (VADER, NRC Emotion Lexicon, SentiWordNet) as
structured documents in a persistent vector store. Each document represents
a lexicon entry with its valence scores, emotion associations, and example
usage, enabling retrieval-augmented explanation of model predictions.

The embedding model (all-MiniLM-L6-v2) runs on CUDA when available,
accelerating both index-time and query-time encoding.
"""

from __future__ import annotations

import json
from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from loguru import logger


class _GPUEmbeddingFunction:
    """
    ChromaDB-compatible embedding function backed by SentenceTransformer
    with explicit GPU device placement.

    ChromaDB's default embedding path uses a CPU-only ONNX model. This
    wrapper ensures all encode() calls are dispatched to the NVIDIA GPU
    when available, providing ~10-50x speedup on batch embedding.

    Implements the full ChromaDB EmbeddingFunction protocol:
    __call__, embed_documents, embed_query, name.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self._device = device
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(f"Embedding model '{model_name}' loaded on {device.upper()}")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=256,
            show_progress_bar=len(texts) > 500,
            device=self._device,
        )
        return embeddings.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._encode(documents)

    def embed_query(self, input=None, **kwargs) -> list:
        if isinstance(input, str):
            return self._encode([input])[0]
        return self._encode(input)

    def name(self) -> str:
        return f"gpu_st_{self._model_name}"


def _resolve_device(device: str | torch.device | None) -> str:
    """Resolve a device specification to a string for SentenceTransformer."""
    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


class SentimentKnowledgeBase:
    """
    Manages the vector store containing sentiment lexicon entries.
    Supports incremental ingestion, persistence, and semantic retrieval.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "sentiment_lexicon",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | torch.device | None = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        resolved_device = _resolve_device(device)
        self._embedding_fn = _GPUEmbeddingFunction(embedding_model, resolved_device)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

        logger.info(
            f"Knowledge base initialized: {self.collection.count()} documents "
            f"in '{collection_name}' (embeddings on {resolved_device.upper()})"
        )

    def add_vader_lexicon(self, vader_scores: dict[str, float]):
        """
        Ingest VADER lexicon entries.

        Args:
            vader_scores: Mapping from word/phrase to compound valence score
                          (range: -1.0 to +1.0).
        """
        documents, metadatas, ids = [], [], []

        for word, score in vader_scores.items():
            polarity = (
                "positive" if score > 0.05
                else "negative" if score < -0.05
                else "neutral"
            )
            doc = (
                f"Word: {word}. "
                f"VADER compound score: {score:.3f}. "
                f"Polarity: {polarity}. "
                f"Source: VADER sentiment lexicon."
            )
            documents.append(doc)
            metadatas.append({
                "word": word,
                "vader_score": score,
                "polarity": polarity,
                "source": "vader",
            })
            ids.append(f"vader_{word}")

        self._batch_add(documents, metadatas, ids)
        logger.info(f"Added {len(documents)} VADER entries")

    def add_nrc_lexicon(self, nrc_entries: list[dict]):
        """
        Ingest NRC Emotion Lexicon entries.

        Args:
            nrc_entries: List of dicts with keys: word, emotions (list of str),
                         positive (bool), negative (bool).
        """
        documents, metadatas, ids = [], [], []

        for entry in nrc_entries:
            word = entry["word"]
            emotions = entry.get("emotions", [])
            doc = (
                f"Word: {word}. "
                f"Associated emotions: {', '.join(emotions)}. "
                f"Positive: {entry.get('positive', False)}. "
                f"Negative: {entry.get('negative', False)}. "
                f"Source: NRC Emotion Lexicon."
            )
            documents.append(doc)
            metadatas.append({
                "word": word,
                "emotions": json.dumps(emotions),
                "positive": entry.get("positive", False),
                "negative": entry.get("negative", False),
                "source": "nrc",
            })
            ids.append(f"nrc_{word}")

        self._batch_add(documents, metadatas, ids)
        logger.info(f"Added {len(documents)} NRC entries")

    def add_sentiwordnet_lexicon(self, swn_entries: list[dict]):
        """
        Ingest SentiWordNet entries.

        Args:
            swn_entries: List of dicts with keys: word, pos_score, neg_score,
                         obj_score, pos_tag.
        """
        documents, metadatas, ids = [], [], []

        for entry in swn_entries:
            word = entry["word"]
            doc = (
                f"Word: {word}. "
                f"SentiWordNet scores — positive: {entry['pos_score']:.3f}, "
                f"negative: {entry['neg_score']:.3f}, "
                f"objective: {entry['obj_score']:.3f}. "
                f"POS tag: {entry.get('pos_tag', 'unknown')}. "
                f"Source: SentiWordNet."
            )
            documents.append(doc)
            metadatas.append({
                "word": word,
                "pos_score": entry["pos_score"],
                "neg_score": entry["neg_score"],
                "obj_score": entry["obj_score"],
                "source": "sentiwordnet",
            })
            ids.append(f"swn_{word}_{entry.get('pos_tag', 'x')}")

        self._batch_add(documents, metadatas, ids)
        logger.info(f"Added {len(documents)} SentiWordNet entries")

    def add_custom_entries(self, entries: list[dict]):
        """Add custom domain-specific lexicon entries."""
        documents, metadatas, ids = [], [], []

        for entry in entries:
            word = entry["word"]
            doc = (
                f"Word: {word}. "
                f"Domain: {entry.get('domain', 'general')}. "
                f"Sentiment: {entry.get('sentiment', 'unknown')}. "
                f"Description: {entry.get('description', '')}. "
                f"Source: custom lexicon."
            )
            documents.append(doc)
            metadatas.append({
                "word": word,
                "domain": entry.get("domain", "general"),
                "source": "custom",
            })
            ids.append(f"custom_{word}")

        self._batch_add(documents, metadatas, ids)
        logger.info(f"Added {len(documents)} custom entries")

    def _batch_add(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
        batch_size: int = 500,
    ):
        """Add documents in batches to avoid ChromaDB size limits."""
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.upsert(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )

    def query(
        self,
        text: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant lexicon entries for a given text.

        Args:
            text: Query text (typically the input review or extracted keywords).
            top_k: Number of results to return.
            source_filter: Optional filter by source ("vader", "nrc", "sentiwordnet").
        """
        where_filter = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        entries = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                entries.append({
                    "document": doc,
                    "metadata": meta,
                    "similarity": 1 - dist,  # cosine distance → similarity
                })

        return entries

    def get_stats(self) -> dict:
        """Return summary statistics of the knowledge base."""
        return {
            "total_documents": self.collection.count(),
            "persist_directory": str(self.persist_directory),
        }
