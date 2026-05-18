"""Exact and near-duplicate removal."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, MinHashLSH, Tokenizer
from pyspark.sql.window import Window

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger

logger = get_logger(__name__)


def _exact_dedup(df: DataFrame) -> DataFrame:
    pair_hash = F.sha2(
        F.concat_ws("||", F.col("source_sentence"), F.col("paraphrased_sentence")),
        256,
    )
    window = Window.partitionBy(pair_hash).orderBy(F.col("id"))
    return (
        df.withColumn("_pair_hash", pair_hash)
        .withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_pair_hash", "_rn")
    )


def _near_dedup(df: DataFrame, threshold: float) -> DataFrame:
    tokenizer = Tokenizer(inputCol="source_sentence", outputCol="tokens")
    hashing_tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=1024)
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=3)

    tokenized = tokenizer.transform(df)
    featurized = hashing_tf.transform(tokenized)
    model = mh.fit(featurized)
    transformed = model.transform(featurized)

    pairs = model.approxSimilarityJoin(
        transformed,
        transformed,
        threshold,
        distCol="jaccard_dist",
    )
    near_dup_ids = (
        pairs.filter(F.col("datasetA.id") < F.col("datasetB.id"))
        .select(F.col("datasetB.id").alias("dup_id"))
        .distinct()
    )
    return df.join(near_dup_ids, df.id == near_dup_ids.dup_id, "left_anti")


def deduplicate(df: DataFrame, config: AppConfig | None = None) -> DataFrame:
    cfg = config or load_config()
    before = df.count()
    deduped = _exact_dedup(df)

    if cfg.spark.near_dup_enabled:
        try:
            deduped = _near_dedup(deduped, cfg.spark.near_dup_jaccard_threshold)
        except Exception as exc:
            logger.warning("Near-dup step skipped: %s", exc)

    after = deduped.count()
    logger.info("Deduplicated: %d -> %d rows", before, after)
    return deduped
