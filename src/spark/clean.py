"""Text cleaning and normalization."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import Column
from pyspark.sql import functions as F

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger

logger = get_logger(__name__)

# Control chars (Cc) and common problematic bytes — handled in JVM, no Python UDF.
_CONTROL_CHARS = r"[\u0000-\u001F\u007F-\u009F]"


def _normalize_col(col: Column) -> Column:
    """Trim, drop control characters, collapse whitespace (Spark SQL only)."""
    stripped = F.trim(F.regexp_replace(col, _CONTROL_CHARS, ""))
    return F.trim(F.regexp_replace(stripped, r"\s+", " "))


def _token_count(text_col: Column) -> Column:
    return F.size(F.split(F.trim(text_col), r"\s+"))


def clean(df: DataFrame, config: AppConfig | None = None) -> DataFrame:
    cfg = config or load_config()
    before = df.count()

    cleaned = (
        df.withColumn("source_sentence", _normalize_col(F.col("source_sentence")))
        .withColumn("paraphrased_sentence", _normalize_col(F.col("paraphrased_sentence")))
        .filter(F.col("source_sentence").isNotNull())
        .filter(F.col("paraphrased_sentence").isNotNull())
        .filter(F.length(F.col("source_sentence")) > 0)
        .filter(F.length(F.col("paraphrased_sentence")) > 0)
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
