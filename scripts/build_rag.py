"""
Build the RAG knowledge base from sentiment lexicons.

Downloads and indexes VADER, NRC, and SentiWordNet into ChromaDB.

Usage:
    python scripts/build_rag.py --config config/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import nltk
import torch
from loguru import logger

from src.utils.helpers import load_config, setup_logger
from src.rag.knowledge_base import SentimentKnowledgeBase


def load_vader_lexicon() -> dict[str, float]:
    """
    Extract the VADER lexicon as a {word: compound_score} dictionary.
    VADER's lexicon is embedded in the package itself.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return dict(analyzer.lexicon)


def load_nrc_lexicon(lexicon_dir: str) -> list[dict]:
    """
    Load the NRC Emotion Lexicon.

    If the NRC file is not found locally, generates a representative
    subset for demonstration purposes. In production, download the
    full NRC lexicon from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
    """
    nrc_path = Path(lexicon_dir) / "NRC-Emotion-Lexicon.txt"

    if nrc_path.exists():
        entries = {}
        with open(nrc_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    word, emotion, assoc = parts
                    if word not in entries:
                        entries[word] = {
                            "word": word,
                            "emotions": [],
                            "positive": False,
                            "negative": False,
                        }
                    if int(assoc) == 1:
                        if emotion == "positive":
                            entries[word]["positive"] = True
                        elif emotion == "negative":
                            entries[word]["negative"] = True
                        else:
                            entries[word]["emotions"].append(emotion)

        return list(entries.values())

    logger.warning(
        f"NRC lexicon not found at {nrc_path}. "
        "Using built-in demo subset. For the full lexicon, download from "
        "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm"
    )
    return _nrc_demo_subset()


def _nrc_demo_subset() -> list[dict]:
    """Minimal NRC-style entries for demonstration."""
    return [
        {"word": "happy", "emotions": ["joy", "anticipation"], "positive": True, "negative": False},
        {"word": "terrible", "emotions": ["anger", "disgust"], "positive": False, "negative": True},
        {"word": "sad", "emotions": ["sadness"], "positive": False, "negative": True},
        {"word": "afraid", "emotions": ["fear"], "positive": False, "negative": True},
        {"word": "wonderful", "emotions": ["joy", "trust"], "positive": True, "negative": False},
        {"word": "angry", "emotions": ["anger"], "positive": False, "negative": True},
        {"word": "surprise", "emotions": ["surprise"], "positive": False, "negative": False},
        {"word": "love", "emotions": ["joy", "trust"], "positive": True, "negative": False},
        {"word": "hate", "emotions": ["anger", "disgust"], "positive": False, "negative": True},
        {"word": "excellent", "emotions": ["joy", "trust"], "positive": True, "negative": False},
        {"word": "awful", "emotions": ["anger", "disgust", "sadness"], "positive": False, "negative": True},
        {"word": "beautiful", "emotions": ["joy"], "positive": True, "negative": False},
        {"word": "disgusting", "emotions": ["disgust"], "positive": False, "negative": True},
        {"word": "exciting", "emotions": ["joy", "anticipation", "surprise"], "positive": True, "negative": False},
        {"word": "boring", "emotions": ["sadness"], "positive": False, "negative": True},
        {"word": "fantastic", "emotions": ["joy", "surprise"], "positive": True, "negative": False},
        {"word": "horrible", "emotions": ["anger", "disgust", "fear"], "positive": False, "negative": True},
        {"word": "brilliant", "emotions": ["joy", "trust"], "positive": True, "negative": False},
        {"word": "disappointed", "emotions": ["sadness"], "positive": False, "negative": True},
        {"word": "thrilling", "emotions": ["joy", "anticipation", "surprise"], "positive": True, "negative": False},
    ]


def load_sentiwordnet(lexicon_dir: str) -> list[dict]:
    """
    Load SentiWordNet entries.

    Falls back to NLTK's SentiWordNet corpus if the raw file is unavailable.
    """
    try:
        nltk.download("sentiwordnet", quiet=True)
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import sentiwordnet as swn

        entries = []
        seen = set()
        for synset in swn.all_senti_synsets():
            word = synset.synset.lemmas()[0].name()
            if word in seen or "_" in word:
                continue
            if synset.pos_score() == 0 and synset.neg_score() == 0:
                continue

            seen.add(word)
            entries.append({
                "word": word,
                "pos_score": synset.pos_score(),
                "neg_score": synset.neg_score(),
                "obj_score": synset.obj_score(),
                "pos_tag": synset.synset.pos(),
            })

            if len(entries) >= 5000:
                break

        logger.info(f"Loaded {len(entries)} SentiWordNet entries via NLTK")
        return entries

    except Exception as e:
        logger.warning(f"SentiWordNet loading failed: {e}. Using demo subset.")
        return [
            {"word": "good", "pos_score": 0.75, "neg_score": 0.0, "obj_score": 0.25, "pos_tag": "a"},
            {"word": "bad", "pos_score": 0.0, "neg_score": 0.75, "obj_score": 0.25, "pos_tag": "a"},
            {"word": "great", "pos_score": 0.75, "neg_score": 0.0, "obj_score": 0.25, "pos_tag": "a"},
            {"word": "terrible", "pos_score": 0.0, "neg_score": 0.875, "obj_score": 0.125, "pos_tag": "a"},
            {"word": "excellent", "pos_score": 0.875, "neg_score": 0.0, "obj_score": 0.125, "pos_tag": "a"},
        ]


@click.command()
@click.option("--config", default="config/config.yaml")
@click.option("--device", default=None, help="Force device (cuda / cpu)")
def main(config: str, device: str | None):
    setup_logger()
    cfg = load_config(config)
    rag_cfg = cfg["rag"]
    data_cfg = cfg["data"]

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"CUDA available — GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA not available — running on CPU (this will be slow)")
    logger.info(f"Embedding device: {resolved.upper()}")

    logger.info("Building RAG knowledge base...")

    kb = SentimentKnowledgeBase(
        persist_directory=rag_cfg["persist_directory"],
        collection_name=rag_cfg["collection_name"],
        embedding_model=rag_cfg["embedding_model"],
        device=resolved,
    )

    logger.info("Loading VADER lexicon...")
    vader_scores = load_vader_lexicon()
    kb.add_vader_lexicon(vader_scores)

    logger.info("Loading NRC lexicon...")
    nrc_entries = load_nrc_lexicon(data_cfg["lexicon_dir"])
    kb.add_nrc_lexicon(nrc_entries)

    logger.info("Loading SentiWordNet...")
    swn_entries = load_sentiwordnet(data_cfg["lexicon_dir"])
    kb.add_sentiwordnet_lexicon(swn_entries)

    stats = kb.get_stats()
    logger.info(f"Knowledge base built: {stats['total_documents']} total documents")
    logger.info(f"Persisted to: {stats['persist_directory']}")


if __name__ == "__main__":
    main()
