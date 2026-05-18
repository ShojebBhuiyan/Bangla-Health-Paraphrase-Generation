"""Spark session factory."""

from __future__ import annotations

import os
from functools import lru_cache

from pyspark.sql import SparkSession

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT

logger = get_logger(__name__)


def _setup_hadoop_home() -> None:
    hadoop_home = PROJECT_ROOT / "hadoop"
    if hadoop_home.exists():
        os.environ.setdefault("HADOOP_HOME", str(hadoop_home))
        bin_dir = hadoop_home / "bin"
        if bin_dir.exists():
            os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"


@lru_cache(maxsize=1)
def get_spark(config: AppConfig | None = None) -> SparkSession:
    cfg = config or load_config()
    _setup_hadoop_home()

    log4j = PROJECT_ROOT / "configs" / "log4j2.properties"
    builder = (
        SparkSession.builder.appName(cfg.spark.app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions))
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config("spark.executor.memory", cfg.spark.executor_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", str(cfg.spark.enable_arrow).lower())
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )
    if log4j.exists():
        builder = builder.config("spark.driver.extraJavaOptions", f"-Dlog4j.configurationFile=file:///{log4j.as_posix()}")

    spark = builder.getOrCreate()
    spark.sparkContext.setCheckpointDir(str(PROJECT_ROOT / cfg.outputs.spark_checkpoints))
    logger.info("Spark session started: %s", cfg.spark.app_name)
    return spark


def stop_spark() -> None:
    try:
        spark = SparkSession.getActiveSession()
        if spark is not None:
            spark.stop()
            logger.info("Spark session stopped")
    except Exception:
        pass
    get_spark.cache_clear()
