"""Spark session factory."""

from __future__ import annotations

import os
import sys
from typing import Optional

from src.spark._runtime import configure_spark_runtime, default_master
from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT

configure_spark_runtime()

from pyspark.sql import SparkSession

logger = get_logger(__name__)
_spark_instance: Optional[SparkSession] = None


def _setup_hadoop_home() -> None:
    hadoop_home = PROJECT_ROOT / "hadoop"
    if hadoop_home.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        bin_dir = hadoop_home / "bin"
        if bin_dir.exists():
            os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"


def get_spark(config: AppConfig | None = None) -> SparkSession:
    global _spark_instance
    if _spark_instance is not None:
        return _spark_instance

    spark_home = configure_spark_runtime()
    if spark_home is None:
        raise RuntimeError(
            "SPARK_HOME not found. Install Spark 4.1.x and set SPARK_HOME "
            "(e.g. C:\\Spark), or run from an environment where C:\\Spark exists."
        )

    cfg = config or load_config()
    _setup_hadoop_home()

    python = sys.executable
    log4j = PROJECT_ROOT / "configs" / "log4j2.properties"
    builder = (
        SparkSession.builder.appName(cfg.spark.app_name)
        .master(default_master())
        .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions))
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config("spark.executor.memory", cfg.spark.executor_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", str(cfg.spark.enable_arrow).lower())
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.python.use.daemon", "false")
        .config("spark.pyspark.python", python)
        .config("spark.pyspark.driver.python", python)
    )

    if log4j.exists():
        # as_uri() percent-encodes spaces (e.g. "Study Material") — required on Windows
        log4j_uri = log4j.resolve().as_uri()
        builder = builder.config(
            "spark.driver.extraJavaOptions",
            f"-Dlog4j2.configurationFile={log4j_uri}",
        )

    logger.info("Using system Spark at %s (Python %s)", spark_home, python)
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
