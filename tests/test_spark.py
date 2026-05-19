"""Spark preprocessing pipeline tests."""

from pathlib import Path

import pytest

from src.common.config import load_config
from src.common.paths import PROJECT_ROOT
from src.spark.clean import clean
from src.spark.deduplicate import deduplicate
from src.spark.session import get_spark, stop_spark
from src.spark.validate import profile_raw, validate

WINUTILS = PROJECT_ROOT / "hadoop" / "bin" / "winutils.exe"
pytestmark = pytest.mark.skipif(
    not WINUTILS.exists(),
    reason="winutils.exe not installed; run scripts/setup_spark_windows.ps1",
)


@pytest.fixture(scope="module")
def spark():
    cfg = load_config()
    spark = get_spark(cfg)
    yield spark
    stop_spark()


@pytest.fixture(scope="module")
def sample_df(spark, tmp_path_factory):
    fixture = tmp_path_factory.mktemp("fixture") / "sample.csv"
    lines = ["sl,id,source_sentence,paraphrased_sentence"]
    for i in range(100):
        src = f"ডেঙ্গু রোগ {i} হলো একটি স্বাস্থ্য সমস্যা।"
        par = f"ডেঙ্গু {i} একটি স্বাস্থ্য সমস্যা।"
        lines.append(f"{i},{i},{src},{par}")
    fixture.write_text("\n".join(lines), encoding="utf-8")

    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    schema = StructType(
        [
            StructField("sl", IntegerType(), True),
            StructField("id", IntegerType(), True),
            StructField("source_sentence", StringType(), True),
            StructField("paraphrased_sentence", StringType(), True),
        ]
    )
    return (
        spark.read.option("header", True)
        .schema(schema)
        .csv(str(fixture))
    )


def _df_from_csv(spark, rows: list[tuple], header: str = "sl,id,source_sentence,paraphrased_sentence"):
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    import tempfile

    schema = StructType(
        [
            StructField("sl", IntegerType(), True),
            StructField("id", IntegerType(), True),
            StructField("source_sentence", StringType(), True),
            StructField("paraphrased_sentence", StringType(), True),
        ]
    )
    lines = [header] + [",".join(str(v) for v in row) for row in rows]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    return spark.read.option("header", True).schema(schema).csv(path)


def _df_from_parquet(spark, rows: list[dict], tmp_path):
    import pandas as pd

    path = tmp_path / "rows.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return spark.read.parquet(str(path))


def test_normalize_col_strips_control_chars(spark, tmp_path):
    df = _df_from_parquet(
        spark,
        [
            {
                "sl": 1,
                "id": 1,
                "source_sentence": "hello\x00world foo bar",
                "paraphrased_sentence": "hello world variant text",
            }
        ],
        tmp_path,
    )
    cleaned = clean(df)
    row = cleaned.collect()[0]
    assert "\x00" not in row["source_sentence"]


def test_clean_removes_trivial_pairs(spark):
    df = _df_from_csv(
        spark,
        [
            (1, 1, "same text here", "same text here"),
            (2, 2, "ডেঙ্গু রোগ হলো সমস্যা।", "ডেঙ্গু একটি স্বাস্থ্য সমস্যা।"),
        ],
    )
    cleaned = clean(df)
    assert cleaned.count() == 1


def test_deduplicate_exact(sample_df):
    deduped = deduplicate(sample_df)
    assert deduped.count() <= sample_df.count()


def test_profile_raw_allows_nulls(sample_df):
    summary = profile_raw(sample_df)
    assert summary["stage"] == "raw"
    assert summary["valid"] is True


def test_validate_after_clean(sample_df):
    cleaned = deduplicate(clean(sample_df))
    summary = validate(cleaned)
    assert summary["stage"] == "clean"
    assert summary["valid"] is True
    assert summary["null_counts"]["source_sentence"] == 0
    assert summary["null_counts"]["paraphrased_sentence"] == 0


def test_feature_columns(sample_df):
    from src.spark.feature_engineering import add_features

    featured = add_features(clean(sample_df))
    cols = set(featured.columns)
    assert "src_len" in cols
    assert "word_overlap" in cols
