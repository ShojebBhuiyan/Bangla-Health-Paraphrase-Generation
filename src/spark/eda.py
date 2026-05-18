"""Spark-based exploratory data analysis."""

from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR, PROJECT_ROOT
from src.visualization.plots import plot_histogram

logger = get_logger(__name__)


def _collect_numeric(df: DataFrame, col: str, sample_frac: float = 1.0) -> list[float]:
    sample = df.sample(fraction=sample_frac, seed=42) if sample_frac < 1.0 else df
    rows = sample.select(col).dropna().collect()
    return [float(r[col]) for r in rows]


def compute_eda_stats(config: AppConfig | None = None) -> dict:
    cfg = config or load_config()
    from src.spark.session import get_spark

    spark = get_spark(cfg)
    featured_path = DATA_DIR / "interim" / "featured.parquet"
    if not featured_path.exists():
        raise FileNotFoundError("Run preprocessing before EDA")

    df = spark.read.parquet(str(featured_path))
    total = df.count()

    null_counts = {
        c: df.filter(F.col(c).isNull()).count()
        for c in ["source_sentence", "paraphrased_sentence"]
    }

    src_lens = _collect_numeric(df, "src_len")
    tgt_lens = _collect_numeric(df, "tgt_len")
    len_ratios = _collect_numeric(df, "len_ratio")
    word_overlaps = _collect_numeric(df, "word_overlap")

    token_df = (
        df.select(F.explode(F.split(F.col("source_sentence"), r"\s+")).alias("token"))
        .groupBy("token")
        .count()
        .orderBy(F.desc("count"))
    )
    top_tokens = token_df.limit(50).collect()
    vocab_size = token_df.count()

    stats = {
        "total_rows": total,
        "null_counts": null_counts,
        "src_len_mean": sum(src_lens) / max(len(src_lens), 1),
        "tgt_len_mean": sum(tgt_lens) / max(len(tgt_lens), 1),
        "len_ratio_mean": sum(len_ratios) / max(len(len_ratios), 1),
        "word_overlap_mean": sum(word_overlaps) / max(len(word_overlaps), 1),
        "vocab_size": vocab_size,
        "top_tokens": [{"token": r["token"], "count": r["count"]} for r in top_tokens],
    }

    figures_dir = PROJECT_ROOT / cfg.outputs.figures / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_histogram(src_lens, "Source Sentence Length", "Tokens", figures_dir / "src_len_dist")
    plot_histogram(tgt_lens, "Target Sentence Length", "Tokens", figures_dir / "tgt_len_dist")
    plot_histogram(len_ratios, "Length Ratio Distribution", "src/tgt", figures_dir / "len_ratio_dist")
    plot_histogram(word_overlaps, "Word Overlap Distribution", "Jaccard", figures_dir / "word_overlap_dist")

    stats_path = PROJECT_ROOT / cfg.outputs.reports / "eda_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("EDA stats written to %s", stats_path)
    return stats
