"""Text cleaning and normalization."""

from __future__ import annotations

import unicodedata

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger

logger = get_logger(__name__)


@F.udf(StringType())
def normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    return text.strip()


def _token_count(text_col):
    return F.size(F.split(F.trim(text_col), r"\s+"))


def clean(df: DataFrame, config: AppConfig | None = None) -> DataFrame:
    cfg = config or load_config()
    before = df.count()

    cleaned = (
        df.withColumn("source_sentence", normalize_text(F.col("source_sentence")))
        .withColumn("paraphrased_sentence", normalize_text(F.col("paraphrased_sentence")))
        .filter(F.col("source_sentence").isNotNull())
        .filter(F.col("paraphrased_sentence").isNotNull())
        .filter(F.col("source_sentence") != F.col("paraphrased_sentence"))
    )

    src_tokens = _token_count(F.col("source_sentence"))
    tgt_tokens = _token_count(F.col("paraphrased_sentence"))
    cleaned = cleaned.filter(
        (src_tokens >= cfg.dataset.min_tokens)
        & (tgt_tokens >= cfg.dataset.min_tokens)
        & (src_tokens <= cfg.dataset.max_tokens)
        & (tgt_tokens <= cfg.dataset.max_tokens)
    )

    after = cleaned.count()
    logger.info("Cleaned dataset: %d -> %d rows", before, after)
    return cleaned
