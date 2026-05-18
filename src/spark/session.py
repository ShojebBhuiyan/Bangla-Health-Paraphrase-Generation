"""Spark session factory."""

from __future__ import annotations

import os
from typing import Optional

from pyspark.sql import SparkSession

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT

logger = get_logger(__name__)
_spark_instance: Optional[SparkSession] = None


def _setup_hadoop_home() -> None:
    hadoop_home = PROJECT_ROOT / "hadoop"
    if hadoop_home.exists():
        os.environ.setdefault("HADOOP_HOME", str(hadoop_home))
        bin_dir = hadoop_home / "bin"
        if bin_dir.exists():
            os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"


def get_spark(config: AppConfig | None = None) -> SparkSession:
    global _spark_instance
    if _spark_instance is not None:
        return _spark_instance

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

    _spark_instance = builder.getOrCreate()
    checkpoint_dir = PROJECT_ROOT / cfg.outputs.spark_checkpoints
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        _spark_instance.sparkContext.setCheckpointDir(str(checkpoint_dir))
    except Exception as exc:
        logger.warning("Could not set Spark checkpoint dir (winutils may be missing): %s", exc)
    logger.info("Spark session started: %s", cfg.spark.app_name)
    return _spark_instance


def stop_spark() -> None:
    global _spark_instance
    try:
        if _spark_instance is not None:
            _spark_instance.stop()
            logger.info("Spark session stopped")
    except Exception:
        pass
    _spark_instance = None
