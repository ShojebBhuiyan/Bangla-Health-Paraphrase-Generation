"""Reload pipeline modules so notebook kernels pick up code edits."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType


def reload_preprocess_modules() -> ModuleType:
    """Reload Spark preprocess stack (ingest → validate → clean → dedupe → features)."""
    import src.spark.clean
    import src.spark.deduplicate
    import src.spark.feature_engineering
    import src.spark.ingest
    import src.spark.validate
    import src.pipeline.preprocess

    modules = (
        src.spark.validate,
        src.spark.clean,
        src.spark.deduplicate,
        src.spark.ingest,
        src.spark.feature_engineering,
        src.pipeline.preprocess,
    )
    for mod in modules:
        importlib.reload(mod)
    return src.pipeline.preprocess


def assert_preprocess_pipeline() -> None:
    """Fail fast when a Jupyter kernel still has a stale preprocess implementation."""
    from src.pipeline.preprocess import run_preprocess

    source = inspect.getsource(run_preprocess)
    validate_idx = source.find("validate(df, cfg)")
    dedupe_idx = source.find("deduplicate(df, cfg)")
    if "profile_raw" not in source or validate_idx == -1 or validate_idx < dedupe_idx:
        raise RuntimeError(
            "Stale preprocess code in memory. Use Kernel → Restart, run §1 setup, "
            "then re-run the preprocess cell (or call reload_preprocess_modules())."
        )
