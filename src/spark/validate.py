"""Dataset validation and quality checks."""

from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import OUTPUTS_DIR, PROJECT_ROOT

logger = get_logger(__name__)

BANLA_UNICODE_PATTERN = r"[\u0980-\u09FF]"


def validate(df: DataFrame, config: AppConfig | None = None) -> dict:
    cfg = config or load_config()
    total = df.count()
    null_counts = {
        col: df.filter(F.col(col).isNull()).count()
        for col in ["source_sentence", "paraphrased_sentence", "id"]
    }
    duplicate_ids = (
        df.groupBy("id").count().filter(F.col("count") > 1).count()
    )
    bangla_coverage = df.select(
        F.avg(
            F.when(
                F.col("source_sentence").rlike(BANLA_UNICODE_PATTERN),
                1.0,
            ).otherwise(0.0)
        ).alias("bangla_ratio")
    ).collect()[0]["bangla_ratio"]

    summary = {
        "total_rows": total,
        "null_counts": null_counts,
        "duplicate_ids": duplicate_ids,
        "bangla_coverage_ratio": float(bangla_coverage or 0.0),
        "valid": total > 0 and null_counts["source_sentence"] == 0,
    }

    report_path = PROJECT_ROOT / cfg.outputs.reports / "validation_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Validation summary: %s", summary)

    if not summary["valid"]:
        raise ValueError(f"Validation failed: {summary}")

    return summary
