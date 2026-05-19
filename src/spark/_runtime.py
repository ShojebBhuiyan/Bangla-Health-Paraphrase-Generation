"""Configure PySpark to use the system Spark install (SPARK_HOME), not pip pyspark."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_CONFIGURED = False


def _resolve_spark_home() -> Path | None:
    env = os.environ.get("SPARK_HOME")
    if env:
        home = Path(env)
        if (home / "bin" / "spark-submit.cmd").exists() or (home / "bin" / "spark-submit").exists():
            return home
    default = Path("C:/Spark")
    if (default / "bin" / "spark-submit.cmd").exists():
        return default
    return None


def _prepend_pythonpath(path: Path) -> None:
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def configure_spark_runtime() -> Path | None:
    """Point Python at SPARK_HOME bindings and set worker/driver env vars."""
    global _CONFIGURED
    if _CONFIGURED:
        return _resolve_spark_home()

    spark_home = _resolve_spark_home()
    if spark_home is None:
        return None

    os.environ["SPARK_HOME"] = str(spark_home)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    python = sys.executable
    os.environ["PYSPARK_PYTHON"] = python
    os.environ["PYSPARK_DRIVER_PYTHON"] = python

    spark_python = spark_home / "python"
    _prepend_pythonpath(spark_python)

    lib_dir = spark_python / "lib"
    if lib_dir.is_dir():
        for py4j_zip in sorted(lib_dir.glob("py4j-*-src.zip")):
            _prepend_pythonpath(py4j_zip)

    _CONFIGURED = True
    return spark_home


def default_master() -> str:
    """Single-threaded local mode avoids flaky Python workers on Windows."""
    if sys.platform == "win32":
        return "local[1]"
    return "local[*]"
