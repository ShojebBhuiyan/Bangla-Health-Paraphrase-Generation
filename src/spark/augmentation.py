"""Bangla token-level EDA augmentation in Spark."""

from __future__ import annotations

import random
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR
from src.common.seed import set_seed

logger = get_logger(__name__)

_wordnet = None


def _get_wordnet():
    global _wordnet
    if _wordnet is None:
        try:
            from bnlp_toolkit import Wordnet

            _wordnet = Wordnet()
        except Exception:
            _wordnet = False
    return _wordnet if _wordnet is not False else None


def _synonym(word: str) -> str:
    wn = _get_wordnet()
    if not wn:
        return word
    try:
        syns = wn.synonyms(word)
        if syns:
            return random.choice(syns)
    except Exception:
        pass
    return word


def _augment_text(
    text: str,
    synonym_prob: float,
    swap_prob: float,
    deletion_prob: float,
    min_tokens: int,
) -> str:
    if not text:
        return text
    tokens = text.split()
    if len(tokens) < min_tokens:
        return text

    new_tokens = []
    for token in tokens:
        r = random.random()
        if r < synonym_prob:
            new_tokens.append(_synonym(token))
        else:
            new_tokens.append(token)

    if len(new_tokens) >= 2 and random.random() < swap_prob:
        i = random.randint(0, len(new_tokens) - 2)
        new_tokens[i], new_tokens[i + 1] = new_tokens[i + 1], new_tokens[i]

    if len(new_tokens) > min_tokens and random.random() < deletion_prob:
        idx = random.randint(0, len(new_tokens) - 1)
        new_tokens.pop(idx)

    return " ".join(new_tokens)


@F.udf(StringType())
def augment_source_udf(
    text: str,
    synonym_prob: float,
    swap_prob: float,
    deletion_prob: float,
    min_tokens: int,
) -> str:
    return _augment_text(text, synonym_prob, swap_prob, deletion_prob, int(min_tokens))


def augment_train(config: AppConfig | None = None, force: bool = False) -> Path:
    cfg = config or load_config()
    set_seed(cfg.experiment.seed)

    if not force and is_done("augmentation", cfg):
        out = DATA_DIR / "augmented" / "train.parquet"
        logger.info("Augmentation already complete: %s", out)
        return out

    from src.spark.session import get_spark

    spark = get_spark(cfg)
    train_path = DATA_DIR / "processed" / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Run preprocessing before augmentation")

    train_df = spark.read.parquet(str(train_path))
    aug = cfg.augmentation

    augmented = train_df.withColumn(
        "source_sentence",
        augment_source_udf(
            F.col("source_sentence"),
            F.lit(aug.synonym_prob),
            F.lit(aug.swap_prob),
            F.lit(aug.deletion_prob),
            F.lit(aug.min_tokens_after_aug),
        ),
    ).withColumn("is_augmented", F.lit(True))

    original = train_df.withColumn("is_augmented", F.lit(False))
    combined = original.unionByName(augmented, allowMissingColumns=True)

    out_path = DATA_DIR / "augmented" / "train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write.mode("overwrite").parquet(str(out_path))
    mark_done("augmentation", {"rows": combined.count()}, cfg)
    logger.info("Augmented train set saved: %s (%d rows)", out_path, combined.count())
    return out_path
