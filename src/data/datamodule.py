"""HuggingFace dataset loading for training."""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk

from src.common.config import AppConfig, ModelSpec, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR

logger = get_logger(__name__)


def load_tokenized_splits(model_spec: ModelSpec, config: AppConfig | None = None) -> dict:
    cfg = config or load_config()
    base = DATA_DIR / "processed" / "tokenized" / model_spec.key
    splits = {}
    for split in ("train", "val", "test"):
        path = base / split
        if not path.exists():
            raise FileNotFoundError(f"Tokenized split missing: {path}. Run tokenization first.")
        splits[split] = load_from_disk(str(path))
        logger.info("Loaded %s/%s: %d rows", model_spec.key, split, len(splits[split]))
    return splits
