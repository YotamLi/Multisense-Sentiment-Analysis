"""
Utility functions: configuration loading, environment setup, reproducibility.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Environment loaded from {env_path}")
    else:
        load_dotenv()


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_logger(log_file: str | None = None, level: str = "INFO"):
    """Configure loguru with console and optional file output."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
    )

    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="7 days",
        )


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device
