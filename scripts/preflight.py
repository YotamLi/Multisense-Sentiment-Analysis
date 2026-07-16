"""Quick environment, data, RAG and checkpoint compatibility checks."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
import torch

from src.models.checkpoint import infer_head_sizes, load_checkpoint
from src.utils.helpers import load_config, load_env


@click.command()
@click.option("--config", default="config/config.yaml", show_default=True)
@click.option("--checkpoint", default=None)
def main(config: str, checkpoint: str | None):
    cfg = load_config(config)
    print("=" * 70)
    print("MULTISENSE PREFLIGHT")
    print("=" * 70)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA build: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    packages = [
        "transformers", "datasets", "langchain", "langchain_google_genai",
        "chromadb", "sentence_transformers", "gradio", "plotly",
    ]
    print("\nCore imports:")
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"  {package}: {version}")
        except Exception as exc:
            print(f"  {package}: FAILED ({exc})")

    data_dir = Path(cfg["data"]["processed_dir"])
    print(f"\nData directory: {data_dir}")
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            print(f"  {split}: {len(frame):,} rows")
        else:
            print(f"  {split}: MISSING")

    rag_dir = Path(cfg["rag"]["persist_directory"])
    print(f"\nRAG directory: {rag_dir} ({'present' if rag_dir.exists() else 'missing'})")

    load_env()
    print(f"Gemini API key detected: {bool(os.getenv('GOOGLE_API_KEY'))}")

    if checkpoint:
        loaded = load_checkpoint(checkpoint)
        print(f"\nCheckpoint: {checkpoint}")
        print(f"  format_version: {loaded.get('format_version', 'legacy')}")
        print(f"  epoch: {loaded.get('epoch', 'unknown')}")
        print(f"  head sizes: {infer_head_sizes(loaded)}")


if __name__ == "__main__":
    main()
