"""Spark package — bootstrap system PySpark before submodule imports."""

from src.spark._runtime import configure_spark_runtime

configure_spark_runtime()
