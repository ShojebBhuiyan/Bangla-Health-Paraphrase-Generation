"""BERTScore evaluation."""

from __future__ import annotations

from bert_score import score as bert_score_fn

from src.common.logging import get_logger

logger = get_logger(__name__)


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "xlm-roberta-large",
) -> float:
    if not predictions:
        return 0.0
    try:
        _, _, f1 = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            lang="multilingual",
            verbose=False,
        )
        return float(f1.mean().item())
    except Exception as exc:
        logger.warning("BERTScore failed: %s", exc)
        return 0.0
