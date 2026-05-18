"""Feature engineering and train/val/test splits."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR

logger = get_logger(__name__)

BANLA_UNICODE_PATTERN = r"[\u0980-\u09FF]"


def add_features(df: DataFrame) -> DataFrame:
    src_tokens = F.size(F.split(F.trim(F.col("source_sentence")), r"\s+"))
    tgt_tokens = F.size(F.split(F.trim(F.col("paraphrased_sentence")), r"\s+"))

    src_chars = F.length(F.col("source_sentence"))
    tgt_chars = F.length(F.col("paraphrased_sentence"))

    src_set = F.array_distinct(F.split(F.lower(F.col("source_sentence")), r"\s+"))
    tgt_set = F.array_distinct(F.split(F.lower(F.col("paraphrased_sentence")), r"\s+"))
    overlap = F.size(F.array_intersect(src_set, tgt_set))
    union_size = F.size(F.array_union(src_set, tgt_set))

    return (
        df.withColumn("src_len", src_tokens)
        .withColumn("tgt_len", tgt_tokens)
        .withColumn("src_char_len", src_chars)
        .withColumn("tgt_char_len", tgt_chars)
        .withColumn(
            "len_ratio",
            F.when(tgt_tokens > 0, src_tokens / tgt_tokens).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "word_overlap",
            F.when(union_size > 0, overlap / union_size).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "char_overlap",
            F.levenshtein(F.col("source_sentence"), F.col("paraphrased_sentence"))
            / F.greatest(src_chars, tgt_chars, F.lit(1)),
        )
        .withColumn(
            "bn_char_ratio",
            F.length(F.regexp_replace(F.col("source_sentence"), BANLA_UNICODE_PATTERN, ""))
            / F.greatest(src_chars, F.lit(1)),
        )
    )


def split_and_persist(df: DataFrame, config: AppConfig | None = None) -> dict[str, DataFrame]:
    cfg = config or load_config()
    featured = add_features(df)

    interim_path = DATA_DIR / "interim" / "featured.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    featured.write.mode("overwrite").parquet(str(interim_path))
    featured = featured.sparkSession.read.parquet(str(interim_path))

    weights = [
        cfg.dataset.train_split,
        cfg.dataset.validation_split,
        cfg.dataset.test_split,
    ]
    train_df, val_df, test_df = featured.randomSplit(weights, seed=cfg.experiment.seed)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():
        out = processed_dir / f"{name}.parquet"
        split_df.write.mode("overwrite").parquet(str(out))
        logger.info("%s split: %d rows -> %s", name, split_df.count(), out)

    return splits


def run_feature_engineering(df: DataFrame, config: AppConfig | None = None) -> dict[str, DataFrame]:
    return split_and_persist(df, config)
