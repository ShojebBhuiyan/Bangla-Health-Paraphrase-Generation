"""Sentence-BERT semantic similarity."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.common.logging import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def compute_semantic_similarity(
    sources: list[str],
    predictions: list[str],
    model_name: str,
) -> float:
    if not sources:
        return 0.0
    try:
        model = _get_model(model_name)
        src_emb = model.encode(sources, convert_to_tensor=True, show_progress_bar=False)
        pred_emb = model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(src_emb, pred_emb).diagonal()
        return float(sims.mean().item())
    except Exception as exc:
        logger.warning("Semantic similarity failed for %s: %s", model_name, exc)
        return 0.0
