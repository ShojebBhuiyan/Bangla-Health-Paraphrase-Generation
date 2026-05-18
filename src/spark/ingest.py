"""Dataset ingestion from CSV to Parquet."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR, PROJECT_ROOT

logger = get_logger(__name__)

SCHEMA = StructType(
    [
        StructField("sl", IntegerType(), True),
        StructField("id", IntegerType(), True),
        StructField("source_sentence", StringType(), True),
        StructField("paraphrased_sentence", StringType(), True),
    ]
)


def ingest(config: AppConfig | None = None) -> DataFrame:
    cfg = config or load_config()
    csv_path = PROJECT_ROOT / cfg.dataset.raw_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    from src.spark.session import get_spark

    spark = get_spark(cfg)
    df = (
        spark.read.option("header", True)
        .option("encoding", "UTF-8")
        .schema(SCHEMA)
        .csv(str(csv_path))
    )

    if cfg.dataset.dev_subset:
        df = df.limit(cfg.dataset.dev_subset)
        logger.info("Using dev subset: %d rows", cfg.dataset.dev_subset)

    out_path = DATA_DIR / "raw" / "raw.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(out_path))
    logger.info("Ingested %d rows to %s", df.count(), out_path)
    return spark.read.parquet(str(out_path))
